"""
EV + PV + BESS day-ahead scheduling project
Baseline: EV charging as-soon-as-possible (ASAP), non-controllable load
Smart: EV charging is controllable between connection and disconnection

Input files expected in the same folder as this script, or update PATHS below:
- ev_sessions(1).csv
- scaled_largehotel_profile(1).csv
- PV_production_yearly(3).mat
- prices_yearly(3).mat

Run:
    pip install numpy pandas scipy matplotlib
    python ev_pv_bess_smart_charging_project.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# ===================== USER SETTINGS =====================
BASE_DIR = Path(__file__).resolve().parent

EV_FILE = BASE_DIR / "ev_sessions.csv"
LOAD_FILE = BASE_DIR / "scaled_largehotel_profile.csv"
PV_FILE = BASE_DIR / "PV_production_yearly.mat"
PRICE_FILE = BASE_DIR / "prices_yearly.mat"

# PV+BESS+grid system parameters. Change these to your OSOPSS final values if needed.
S_PV_KWP = 64.7          # installed PV size [kWp]
E_PACK_MAX = 16.44       # BESS nominal capacity [kWh]
P_BESS_N = 6.29          # BESS converter power [kW]
P_GRID_MAX = 150.0       # grid connection limit [kW]

# EV charging parameters
P_CHARGER_KW = 22.0      # max charging power per EV [kW]
TD = 0.25                # sampling time [h], 15 min
N = 96                   # samples/day

# Battery parameters
SOC_MIN = 0.20
SOC_MAX = 0.80
ETA_CH = 0.90
ETA_DCH = 0.90
P_BAT_EUR_PER_KWH = 350.0
N_CYC = 6000.0
C_DEG = P_BAT_EUR_PER_KWH / (2.0 * N_CYC)   # EUR/kWh throughput

# Prices and peak penalty
PAYMENT_TO_SUPPLIER_UPSCALING = 1.10
PAYMENT_FROM_SUPPLIER_DOWNSCALING = 0.90
P_GRID_FEE_PLUS = 0.05       # EUR/kWh
P_GRID_FEE_MINUS = -0.01     # EUR/kWh
P_PP = 4.0                   # EUR/kW/month
PEAK_DAILY_FACTOR = 1.0 / 30.0

OUT_DIR = BASE_DIR / "results_ev_pv_bess"
OUT_DIR.mkdir(exist_ok=True)
# =========================================================


def read_inputs():
    ev = pd.read_csv(EV_FILE)
    fixed = pd.read_csv(LOAD_FILE)
    zeta_pv = sio.loadmat(PV_FILE)["zeta_PV"].reshape(-1)
    prices = sio.loadmat(PRICE_FILE)["prices_all_year"].reshape(-1)

    ev["connection_timestamp"] = pd.to_datetime(ev["connection_timestamp"])
    ev["full_timestamp"] = pd.to_datetime(ev["full_timestamp"])
    ev["disconnection_timestamp"] = pd.to_datetime(ev["disconnection_timestamp"])
    fixed["timestamp"] = pd.to_datetime(fixed["timestamp"])

    assert len(fixed) == 8760, "Fixed load must have 8760 hourly rows."
    assert len(zeta_pv) == 8760, "PV profile must have 8760 hourly rows."
    assert len(prices) == 8760, "Price profile must have 8760 hourly rows."
    return ev, fixed, zeta_pv, prices


def tariff_vectors(price_mean_hourly):
    p_supply_plus = np.where(
        price_mean_hourly >= 0,
        PAYMENT_TO_SUPPLIER_UPSCALING * price_mean_hourly,
        PAYMENT_FROM_SUPPLIER_DOWNSCALING * price_mean_hourly,
    )
    p_supply_minus = np.where(
        price_mean_hourly >= 0,
        PAYMENT_FROM_SUPPLIER_DOWNSCALING * price_mean_hourly,
        PAYMENT_TO_SUPPLIER_UPSCALING * price_mean_hourly,
    )
    p_e_plus = p_supply_plus + P_GRID_FEE_PLUS
    p_e_minus = p_supply_minus + P_GRID_FEE_MINUS
    return p_e_plus, p_e_minus


def day_profiles(day, ev, fixed, zeta_pv, prices):
    """Return 96-step fixed load, PV max, price vectors, and sessions for one day."""
    h0 = (day - 1) * 24
    h1 = h0 + 24

    fixed_hourly_kwh = fixed["scaled_energy_kwh"].to_numpy()[h0:h1]
    fixed_15 = np.repeat(fixed_hourly_kwh / 4.0, 4)       # kWh per 15 min

    pvmax_hourly_kwh = zeta_pv[h0:h1] * S_PV_KWP          # kWh per hour because Td=1h in source
    pvmax_15 = np.repeat(pvmax_hourly_kwh / 4.0, 4)       # kWh per 15 min

    price_15 = np.repeat(prices[h0:h1], 4)
    p_plus, p_minus = tariff_vectors(price_15)

    sessions = ev.loc[ev["day"] == day].copy().reset_index(drop=True)
    return fixed_15, pvmax_15, p_plus, p_minus, sessions


def overlap_hours(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def asap_ev_profile(sessions):
    """EV energy profile [kWh/15 min] spread uniformly from connection to full-charge time."""
    e_ev = np.zeros(N)
    for _, s in sessions.iterrows():
        c = float(s["connection_minute"])
        f = float(s["full_minute"])
        energy = float(s["energy_kwh"])
        duration_h = max((f - c) / 60.0, 1e-9)
        power_kw = energy / duration_h
        for k in range(N):
            slot0 = 15.0 * k
            slot1 = 15.0 * (k + 1)
            ov_h = overlap_hours(c, f, slot0, slot1) / 60.0
            e_ev[k] += power_kw * ov_h
    return e_ev


def smart_variable_map(sessions):
    """Create EV decision variables only where a vehicle is connected."""
    pairs = []
    upper = []
    for i, s in sessions.iterrows():
        c = float(s["connection_minute"])
        d = float(s["disconnection_minute"])
        for k in range(N):
            slot0 = 15.0 * k
            slot1 = 15.0 * (k + 1)
            ov_h = overlap_hours(c, d, slot0, slot1) / 60.0
            if ov_h > 1e-12:
                pairs.append((i, k))
                upper.append(P_CHARGER_KW * ov_h)
    return pairs, np.array(upper)


def solve_dispatch(load_15, pvmax_15, p_plus, p_minus, sessions=None, smart_ev=False, ev_fixed_15=None):
    """
    LP variables:
    [Ebat0, EPV(96), Ech(96), Edch(96), EVsmart(M), epsE(96), epsPP]
    If smart_ev=False, EV load is fixed in ev_fixed_15 and M=0.
    """
    if smart_ev:
        pairs, ev_ub = smart_variable_map(sessions)
        M = len(pairs)
    else:
        pairs, ev_ub, M = [], np.array([]), 0
        if ev_fixed_15 is None:
            ev_fixed_15 = np.zeros(N)

    idx_Ebat0 = 0
    idx_PV = 1
    idx_ch = idx_PV + N
    idx_dch = idx_ch + N
    idx_ev = idx_dch + N
    idx_epsE = idx_ev + M
    idx_epsPP = idx_epsE + N
    nvar = idx_epsPP + 1

    bounds = []
    bounds.append((SOC_MIN * E_PACK_MAX, SOC_MAX * E_PACK_MAX))
    bounds += [(0.0, pvmax_15[k]) for k in range(N)]
    bounds += [(0.0, P_BESS_N * TD) for _ in range(N)]
    bounds += [(0.0, P_BESS_N * TD) for _ in range(N)]
    bounds += [(0.0, ev_ub[j]) for j in range(M)]
    bounds += [(None, None) for _ in range(N)]
    bounds.append((0.0, None))

    c = np.zeros(nvar)
    c[idx_ch:idx_ch+N] = C_DEG * ETA_CH
    c[idx_dch:idx_dch+N] = C_DEG / ETA_DCH
    c[idx_epsE:idx_epsE+N] = 1.0
    c[idx_epsPP] = P_PP * PEAK_DAILY_FACTOR

    A_ub, b_ub = [], []
    A_eq, b_eq = [], []

    # Map smart EV variables to time slots and sessions
    ev_vars_by_time = [[] for _ in range(N)]
    ev_vars_by_session = [[] for _ in range(len(sessions))] if smart_ev else []
    for j, (i, k) in enumerate(pairs):
        ev_vars_by_time[k].append(idx_ev + j)
        ev_vars_by_session[i].append(idx_ev + j)

    # EV energy equality for each session
    if smart_ev:
        for i, s in sessions.iterrows():
            row = np.zeros(nvar)
            for col in ev_vars_by_session[i]:
                row[col] = 1.0
            A_eq.append(row)
            b_eq.append(float(s["energy_kwh"]))

    # BESS cyclic equality: sum eta_ch*Ech - Edch/eta_dch = 0
    row = np.zeros(nvar)
    row[idx_ch:idx_ch+N] = ETA_CH
    row[idx_dch:idx_dch+N] = -1.0 / ETA_DCH
    A_eq.append(row)
    b_eq.append(0.0)

    for k in range(N):
        # BESS power converter: Ech + Edch <= Pn*Td
        row = np.zeros(nvar)
        row[idx_ch+k] = 1.0
        row[idx_dch+k] = 1.0
        A_ub.append(row)
        b_ub.append(P_BESS_N * TD)

        # SOC upper/lower after interval k
        row_soc = np.zeros(nvar)
        row_soc[idx_Ebat0] = 1.0
        row_soc[idx_ch:idx_ch+k+1] = ETA_CH
        row_soc[idx_dch:idx_dch+k+1] = -1.0 / ETA_DCH

        A_ub.append(row_soc.copy())
        b_ub.append(SOC_MAX * E_PACK_MAX)
        A_ub.append(-row_soc.copy())
        b_ub.append(-SOC_MIN * E_PACK_MAX)

        # Build grid energy expression:
        # Egrid = fixed load + EV + Ech - Edch - EPV
        base_load = load_15[k]
        if not smart_ev:
            base_load += ev_fixed_15[k]

        row_grid = np.zeros(nvar)
        row_grid[idx_PV+k] = -1.0
        row_grid[idx_ch+k] = 1.0
        row_grid[idx_dch+k] = -1.0
        if smart_ev:
            for col in ev_vars_by_time[k]:
                row_grid[col] = 1.0

        # import limit: Egrid <= P_grid_max*Td
        A_ub.append(row_grid.copy())
        b_ub.append(P_GRID_MAX * TD - base_load)

        # export limit: -Egrid <= P_grid_max*Td
        A_ub.append(-row_grid.copy())
        b_ub.append(P_GRID_MAX * TD + base_load)

        # peak: Egrid/Td <= epsPP  => Egrid - Td*epsPP <= 0
        row = row_grid.copy()
        row[idx_epsPP] = -TD
        A_ub.append(row)
        b_ub.append(-base_load)

        # epsE >= p_plus*Egrid and epsE >= p_minus*Egrid
        row = p_plus[k] * row_grid.copy()
        row[idx_epsE+k] = -1.0
        A_ub.append(row)
        b_ub.append(-p_plus[k] * base_load)

        row = p_minus[k] * row_grid.copy()
        row[idx_epsE+k] = -1.0
        A_ub.append(row)
        b_ub.append(-p_minus[k] * base_load)

    res = linprog(
        c,
        A_ub=np.array(A_ub), b_ub=np.array(b_ub),
        A_eq=np.array(A_eq), b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP infeasible/failed: {res.message}")

    z = res.x
    EPV = z[idx_PV:idx_PV+N]
    Ech = z[idx_ch:idx_ch+N]
    Edch = z[idx_dch:idx_dch+N]
    Ebat = z[idx_Ebat0] + np.cumsum(ETA_CH * Ech - Edch / ETA_DCH)

    if smart_ev:
        EV = np.zeros(N)
        for j, (_, k) in enumerate(pairs):
            EV[k] += z[idx_ev+j]
    else:
        EV = ev_fixed_15.copy()

    Egrid = load_15 + EV + Ech - Edch - EPV
    epsE = np.maximum(p_plus * Egrid, p_minus * Egrid)
    J_E = float(np.sum(epsE))
    J_peak = float(P_PP * PEAK_DAILY_FACTOR * max(0.0, np.max(Egrid / TD)))
    J_deg = float(np.sum(C_DEG * ETA_CH * Ech + C_DEG / ETA_DCH * Edch))
    J_total_check = J_E + J_peak + J_deg

    return {
        "objective": float(res.fun),
        "J_total_check": J_total_check,
        "J_E": J_E,
        "J_peak": J_peak,
        "J_deg": J_deg,
        "EPV": EPV,
        "Ech": Ech,
        "Edch": Edch,
        "Ebat": Ebat,
        "EV": EV,
        "Egrid": Egrid,
        "peak_kw": float(max(0.0, np.max(Egrid / TD))),
        "import_kwh": float(np.sum(np.maximum(Egrid, 0.0))),
        "export_kwh": float(np.sum(np.maximum(-Egrid, 0.0))),
        "pv_used_kwh": float(np.sum(EPV)),
        "ev_energy_kwh": float(np.sum(EV)),
    }


def choose_representative_days(ev):
    daily = ev.groupby("day").agg(
        sessions=("session", "count"),
        ev_energy_kwh=("energy_kwh", "sum"),
        vehicles_generated_day=("vehicles_generated_day", "first"),
    ).reset_index()
    low_day = int(daily.sort_values("sessions").iloc[0]["day"])
    high_day = int(daily.sort_values("sessions").iloc[-1]["day"])
    avg_sessions = daily["sessions"].mean()
    avg_day = int(daily.iloc[(daily["sessions"] - avg_sessions).abs().argmin()]["day"])
    return [low_day, avg_day, high_day], daily


def plot_day(day, time_h, load_15, pvmax_15, baseline, smart):
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.step(time_h, load_15, where="post", label="fixed load")
    plt.step(time_h, baseline["EV"], where="post", label="EV ASAP")
    plt.step(time_h, smart["EV"], where="post", label="EV smart")
    plt.ylabel("Energy [kWh/15min]")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.step(time_h, pvmax_15, where="post", label="PV max")
    plt.step(time_h, smart["EPV"], where="post", label="PV used smart")
    plt.step(time_h, smart["Ech"], where="post", label="BESS ch")
    plt.step(time_h, smart["Edch"], where="post", label="BESS dch")
    plt.ylabel("Energy [kWh/15min]")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.step(time_h, baseline["Egrid"] / TD, where="post", label="grid baseline")
    plt.step(time_h, smart["Egrid"] / TD, where="post", label="grid smart")
    plt.ylabel("Power [kW]")
    plt.xlabel("Time [h]")
    plt.grid(True)
    plt.legend()

    plt.suptitle(f"Day {day}: baseline ASAP vs smart EV charging")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"day_{day}_comparison.png", dpi=200)
    plt.close()


def main():
    ev, fixed, zeta_pv, prices = read_inputs()
    selected_days, daily_info = choose_representative_days(ev)
    time_h = np.arange(N) * TD

    print("Selected representative days:", selected_days)
    print("EV sessions:", len(ev), "Total EV energy [kWh]:", ev["energy_kwh"].sum())
    print("Fixed scaled load energy [kWh]:", fixed["scaled_energy_kwh"].sum())

    rows = []
    for day in selected_days:
        load_15, pvmax_15, p_plus, p_minus, sessions = day_profiles(day, ev, fixed, zeta_pv, prices)
        ev_asap = asap_ev_profile(sessions)

        baseline = solve_dispatch(load_15, pvmax_15, p_plus, p_minus,
                                  smart_ev=False, ev_fixed_15=ev_asap)
        smart = solve_dispatch(load_15, pvmax_15, p_plus, p_minus,
                               sessions=sessions, smart_ev=True)

        row = {
            "day": day,
            "sessions": len(sessions),
            "ev_energy_kwh": sessions["energy_kwh"].sum(),
            "baseline_cost_eur": baseline["J_total_check"],
            "smart_cost_eur": smart["J_total_check"],
            "saving_eur": baseline["J_total_check"] - smart["J_total_check"],
            "saving_percent": 100.0 * (baseline["J_total_check"] - smart["J_total_check"]) / max(abs(baseline["J_total_check"]), 1e-9),
            "baseline_peak_kw": baseline["peak_kw"],
            "smart_peak_kw": smart["peak_kw"],
            "peak_reduction_kw": baseline["peak_kw"] - smart["peak_kw"],
            "baseline_import_kwh": baseline["import_kwh"],
            "smart_import_kwh": smart["import_kwh"],
            "baseline_export_kwh": baseline["export_kwh"],
            "smart_export_kwh": smart["export_kwh"],
        }
        rows.append(row)

        out = pd.DataFrame({
            "time_h": time_h,
            "fixed_load_kwh": load_15,
            "pvmax_kwh": pvmax_15,
            "ev_asap_kwh": baseline["EV"],
            "ev_smart_kwh": smart["EV"],
            "grid_baseline_kw": baseline["Egrid"] / TD,
            "grid_smart_kw": smart["Egrid"] / TD,
            "bess_ch_smart_kwh": smart["Ech"],
            "bess_dch_smart_kwh": smart["Edch"],
            "battery_energy_smart_kwh": smart["Ebat"],
        })
        out.to_csv(OUT_DIR / f"day_{day}_schedule.csv", index=False)
        plot_day(day, time_h, load_15, pvmax_15, baseline, smart)

        print(f"\nDay {day}")
        print(pd.Series(row).to_string())

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "summary_results.csv", index=False)
    daily_info.to_csv(OUT_DIR / "daily_ev_statistics.csv", index=False)
    print("\nSaved results in:", OUT_DIR)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

# ============================================================
# EXTRA EXPLANATION FILE FOR UNDERSTANDING THE PROJECT RESULTS
# ============================================================

def write_explanation_file():
    explanation_path = BASE_DIR / "results_ev_pv_bess" / "project_explanation.txt"

    text = """
EV PV BESS SMART CHARGING PROJECT - RESULT EXPLANATION

1. PROJECT GOAL
The goal of this project is to compare two EV charging strategies in a solar-powered parking lot:

A) Baseline charging:
   EVs charge as soon as possible after they arrive.
   This represents uncontrolled charging.

B) Smart charging:
   EV charging is optimized during the available parking time.
   The optimizer decides when to charge each vehicle before its required full-charge deadline.

The objective is to reduce total operating cost and grid peak power.

------------------------------------------------------------

2. INPUT DATA USED

The project uses:

- ev_sessions.csv
  Contains all generated EV charging sessions for one year.
  Each row represents one EV charging session.

- scaled_largehotel_profile.csv
  Contains the fixed non-EV load profile.
  The annual fixed load energy is scaled to be equal to the annual EV charging energy.

- PV_production_yearly.mat
  Contains normalized hourly PV production.

- prices_yearly.mat
  Contains hourly electricity market prices.

------------------------------------------------------------

3. REPRESENTATIVE DAYS

The code selects three representative days:

- Low-demand day:
  A day with a small number of EV sessions.

- Average-demand day:
  A day close to the average number of EV sessions.

- High-demand day:
  A day with a high number of EV sessions.

This allows the comparison of smart charging performance under different EV demand levels.

------------------------------------------------------------

4. BASELINE CASE

In the baseline case, each EV session is converted into a charging profile where the required energy is spread between:

connection time -> full-charge time

This follows the as-soon-as-possible charging logic.

The EV demand is added directly to the fixed building/load consumption.

The PV and BESS system then operate with this uncontrollable load.

------------------------------------------------------------

5. SMART CHARGING CASE

In the smart charging case, EV charging is treated as a controllable variable.

Each EV must still receive its required energy before its full-charge deadline.

However, the optimizer can shift the charging power within the available time window.

This allows the system to:

- use more PV energy locally
- reduce grid import during expensive periods
- reduce peak grid power
- lower the total operating cost

------------------------------------------------------------

6. COST COMPONENTS

The total daily cost includes:

- Energy cost from grid import/export
- Peak power penalty
- Battery degradation cost

The baseline and smart charging cases are compared using the same economic assumptions.

------------------------------------------------------------

7. MAIN RESULT INTERPRETATION

If savings are small on a low-demand day, this is normal.
There is not much EV charging flexibility available.

If savings increase on average-demand and high-demand days, this shows that smart charging becomes more valuable when more EV sessions are available.

Peak reduction means that smart charging successfully flattened the grid power profile.

------------------------------------------------------------

8. CONCLUSION

The results show that controllable EV charging can reduce operating cost and peak grid demand compared to as-soon-as-possible charging.

The benefit is higher when the number of EV sessions and charging energy are larger.

This confirms that predictive smart charging improves the operation of a PV+BESS-powered parking lot.
"""

    with open(explanation_path, "w", encoding="utf-8") as f:
        f.write(text)

    print("Saved explanation file in:", explanation_path)


write_explanation_file()
# ============================================================
# EXTRA: ADD SEASON LABELS + SIMPLE SAVINGS COMPARISON TABLE
# ============================================================

def get_season_from_day(day):
    """
    Approximate meteorological seasons for the generated year:
    Winter: Dec-Feb
    Spring: Mar-May
    Summer: Jun-Aug
    Autumn: Sep-Nov
    """

    if 1 <= day <= 59:
        return "Winter"
    elif 60 <= day <= 151:
        return "Spring"
    elif 152 <= day <= 243:
        return "Summer"
    elif 244 <= day <= 334:
        return "Autumn"
    else:
        return "Winter"


def add_season_table_and_savings_plot():
    results_dir = BASE_DIR / "results_ev_pv_bess"
    summary_file = results_dir / "summary_results.csv"

    if not summary_file.exists():
        print("summary_results.csv not found. Run the main optimization first.")
        return

    summary = pd.read_csv(summary_file)

    summary["season"] = summary["day"].apply(get_season_from_day)

    summary["day_description"] = summary.apply(
        lambda row: f"Day {int(row['day'])} ({row['season']})",
        axis=1
    )

    summary["cost_reduction_text"] = summary.apply(
        lambda row: f"{row['saving_eur']:.2f} EUR saved ({row['saving_percent']:.2f}%)",
        axis=1
    )

    summary["peak_reduction_text"] = summary.apply(
        lambda row: f"{row['peak_reduction_kw']:.2f} kW peak reduction",
        axis=1
    )

    understanding_table = summary[
        [
            "day_description",
            "sessions",
            "ev_energy_kwh",
            "baseline_cost_eur",
            "smart_cost_eur",
            "saving_eur",
            "saving_percent",
            "baseline_peak_kw",
            "smart_peak_kw",
            "peak_reduction_kw",
            "cost_reduction_text",
            "peak_reduction_text",
        ]
    ].copy()

    understanding_table = understanding_table.rename(
        columns={
            "day_description": "Representative day",
            "sessions": "EV sessions",
            "ev_energy_kwh": "EV energy [kWh]",
            "baseline_cost_eur": "Baseline cost [EUR]",
            "smart_cost_eur": "Smart charging cost [EUR]",
            "saving_eur": "Saving [EUR]",
            "saving_percent": "Saving [%]",
            "baseline_peak_kw": "Baseline peak [kW]",
            "smart_peak_kw": "Smart peak [kW]",
            "peak_reduction_kw": "Peak reduction [kW]",
            "cost_reduction_text": "Cost saving interpretation",
            "peak_reduction_text": "Peak reduction interpretation",
        }
    )

    numeric_cols = [
        "EV energy [kWh]",
        "Baseline cost [EUR]",
        "Smart charging cost [EUR]",
        "Saving [EUR]",
        "Saving [%]",
        "Baseline peak [kW]",
        "Smart peak [kW]",
        "Peak reduction [kW]",
    ]

    for col in numeric_cols:
        understanding_table[col] = understanding_table[col].round(2)

    understanding_csv = results_dir / "understanding_savings_table.csv"
    understanding_txt = results_dir / "understanding_savings_table.txt"

    understanding_table.to_csv(understanding_csv, index=False)

    with open(understanding_txt, "w", encoding="utf-8") as f:
        f.write("SMART CHARGING SAVINGS COMPARISON TABLE\n")
        f.write("=======================================\n\n")
        f.write(understanding_table.to_string(index=False))
        f.write("\n\n")
        f.write("Main interpretation:\n")
        f.write("- Low EV demand gives small savings because there is little charging flexibility.\n")
        f.write("- Higher EV demand gives larger savings because the optimizer has more charging energy to shift.\n")
        f.write("- Peak reduction shows how much smart charging reduces the maximum grid power demand.\n")

    # Plot 1: Baseline cost vs smart charging cost
    plt.figure(figsize=(10, 5))
    x = np.arange(len(summary))
    width = 0.35

    plt.bar(x - width / 2, summary["baseline_cost_eur"], width, label="Baseline ASAP")
    plt.bar(x + width / 2, summary["smart_cost_eur"], width, label="Smart charging")

    plt.xticks(x, summary["day_description"])
    plt.ylabel("Daily cost [EUR]")
    plt.title("Daily operating cost comparison by representative day")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "cost_comparison_by_season.png", dpi=300)
    plt.close()

    # Plot 2: Savings in EUR
    plt.figure(figsize=(10, 5))
    plt.bar(summary["day_description"], summary["saving_eur"])
    plt.ylabel("Saving [EUR/day]")
    plt.title("Daily cost savings from smart charging")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "daily_savings_by_season.png", dpi=300)
    plt.close()

    # Plot 3: Peak reduction
    plt.figure(figsize=(10, 5))
    plt.bar(summary["day_description"], summary["peak_reduction_kw"])
    plt.ylabel("Peak reduction [kW]")
    plt.title("Grid peak reduction from smart charging")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "peak_reduction_by_season.png", dpi=300)
    plt.close()

    print("\nSaved extra understanding files:")
    print(understanding_csv)
    print(understanding_txt)
    print(results_dir / "cost_comparison_by_season.png")
    print(results_dir / "daily_savings_by_season.png")
    print(results_dir / "peak_reduction_by_season.png")


add_season_table_and_savings_plot()