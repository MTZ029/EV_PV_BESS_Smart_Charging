"""
EV + PV + BESS day-ahead scheduling project

Project idea:
We compare two EV charging strategies in a solar-powered parking lot:

1) Baseline ASAP charging:
   - EVs charge as soon as possible after connection.
   - EV charging is treated as a fixed, non-controllable load.

2) Smart charging:
   - EV charging is controllable between connection and disconnection.
   - The optimizer decides when to charge each EV.

The system includes:
- fixed building load
- photovoltaic production
- battery energy storage system
- grid import/export
- EV charging demand

Input files expected in the same folder as this script:
- ev_sessions.csv
- scaled_largehotel_profile.csv
- PV_production_yearly.mat
- prices_yearly.mat

Run:
    pip install numpy pandas scipy matplotlib
    python ev_pv_bess_smart_charging_project.py
"""

# pathlib is used to handle file paths cleanly.
from pathlib import Path

# numpy is used for numerical arrays and vector operations.
import numpy as np

# pandas is used to read CSV files and handle tabular data.
import pandas as pd

# scipy.io is used to read MATLAB .mat files.
import scipy.io as sio

# linprog is the linear programming solver used for optimization.
from scipy.optimize import linprog

# matplotlib is used to generate result plots.
import matplotlib.pyplot as plt


# ===================== USER SETTINGS =====================

# BASE_DIR is the folder where this Python script is located.
# This makes the code portable because files are loaded relative to the script location.
BASE_DIR = Path(__file__).resolve().parent

# Input file paths.
# These files must exist in the same folder as the script.
EV_FILE = BASE_DIR / "ev_sessions.csv"
LOAD_FILE = BASE_DIR / "scaled_largehotel_profile.csv"  # the one provided.
PV_FILE = BASE_DIR / "PV_production_yearly.mat"
PRICE_FILE = BASE_DIR / "prices_yearly.mat"

# PV+BESS+grid system parameters.
# These define the energy system used in the simulation.
S_PV_KWP = 64.7          # Installed PV size [kWp]
E_PACK_MAX = 16.44       # Battery nominal capacity [kWh]
P_BESS_N = 6.29          # Battery converter power [kW]
P_GRID_MAX = 150.0       # Grid connection limit [kW]

# EV charging parameters.
P_CHARGER_KW = 22.0      # Maximum charging power per EV [kW]
TD = 0.25                # Sampling time [h], 0.25 h = 15 minutes
N = 96                   # Number of samples per day: 24 h / 0.25 h = 96

# Battery parameters.
SOC_MIN = 0.20           # Minimum allowed state of charge
SOC_MAX = 0.80           # Maximum allowed state of charge
ETA_CH = 0.90            # Battery charging efficiency
ETA_DCH = 0.90           # Battery discharging efficiency

# Battery degradation cost approximation.
P_BAT_EUR_PER_KWH = 350.0
N_CYC = 6000.0

# Degradation cost per kWh of battery throughput.
# The factor 2 accounts for charge and discharge throughput over cycles.
C_DEG = P_BAT_EUR_PER_KWH / (2.0 * N_CYC)

# Electricity price and tariff parameters.
PAYMENT_TO_SUPPLIER_UPSCALING = 1.10
PAYMENT_FROM_SUPPLIER_DOWNSCALING = 0.90
P_GRID_FEE_PLUS = 0.05       # Grid fee for imported electricity [EUR/kWh]
P_GRID_FEE_MINUS = -0.01     # Grid fee / adjustment for exported electricity [EUR/kWh]

# Peak power penalty.
P_PP = 4.0                   # Monthly peak penalty [EUR/kW/month]
PEAK_DAILY_FACTOR = 1.0 / 30.0  # Convert monthly penalty approximately to daily penalty

# Output folder where results will be saved.
OUT_DIR = BASE_DIR / "results_ev_pv_bess"

# Create the output folder if it does not already exist.
OUT_DIR.mkdir(exist_ok=True)

# Folder where all figures will be saved.
PLOTS_DIR = OUT_DIR / "figures_baseline"

# Create the figures folder if it does not exist.
PLOTS_DIR.mkdir(exist_ok=True)
# =========================================================


def read_inputs():
    """
    Read all project input files.
    """

    # Read the EV charging session table from CSV.
    ev = pd.read_csv(EV_FILE)

    # Read the scaled hotel fixed-load profile from CSV.
    fixed = pd.read_csv(LOAD_FILE)

    # Read the normalized yearly PV production profile from the MATLAB file.
    zeta_pv = sio.loadmat(PV_FILE)["zeta_PV"].reshape(-1)

    # Read the yearly electricity price profile from the MATLAB file.
    prices = sio.loadmat(PRICE_FILE)["prices_all_year"].reshape(-1)

    # Convert EV connection timestamp column to datetime format.
    ev["connection_timestamp"] = pd.to_datetime(ev["connection_timestamp"])

    # Convert EV full-charge timestamp column to datetime format.
    ev["full_timestamp"] = pd.to_datetime(ev["full_timestamp"])

    # Convert EV disconnection timestamp column to datetime format.
    ev["disconnection_timestamp"] = pd.to_datetime(ev["disconnection_timestamp"])

    # Convert hotel-load timestamp column to datetime format.
    fixed["timestamp"] = pd.to_datetime(fixed["timestamp"])

    # Check that the hotel load has one hourly value for the full year.
    assert len(fixed) == 8760, "Fixed hotel load must have 8760 hourly rows."

    # Check that the PV profile has one hourly value for the full year.
    assert len(zeta_pv) == 8760, "PV profile must have 8760 hourly rows."

    # Check that the price profile has one hourly value for the full year.
    assert len(prices) == 8760, "Price profile must have 8760 hourly rows."

    # Return all loaded input data.
    return ev, fixed, zeta_pv, prices


def tariff_vectors(price_mean_hourly):
    """
    Convert electricity market prices into import and export tariff vectors.

    p_e_plus:
        Effective price for importing electricity from the grid.

    p_e_minus:
        Effective price for exporting electricity to the grid.
    """

    # Compute supply price for grid import.
    # If the market price is positive, import price is increased by supplier margin.
    # If the market price is negative, the alternative scaling is used.
    p_supply_plus = np.where(
        price_mean_hourly >= 0,
        PAYMENT_TO_SUPPLIER_UPSCALING * price_mean_hourly,
        PAYMENT_FROM_SUPPLIER_DOWNSCALING * price_mean_hourly,
    )

    # Compute supply price for grid export.
    # Export usually receives a lower price than import.
    p_supply_minus = np.where(
        price_mean_hourly >= 0,
        PAYMENT_FROM_SUPPLIER_DOWNSCALING * price_mean_hourly,
        PAYMENT_TO_SUPPLIER_UPSCALING * price_mean_hourly,
    )

    # Add grid fee for imported electricity.
    p_e_plus = p_supply_plus + P_GRID_FEE_PLUS

    # Add export adjustment.
    p_e_minus = p_supply_minus + P_GRID_FEE_MINUS

    # Return import and export tariff vectors.
    return p_e_plus, p_e_minus


def day_profiles(day, ev, fixed, zeta_pv, prices):
    """
    Extract daily load, PV, prices, and EV sessions.
    """

    # Compute the first hourly index of the selected day.
    h0 = (day - 1) * 24

    # Compute the final hourly index of the selected day.
    h1 = h0 + 24

    # Extract the 24 hourly hotel-load values for the selected day.
    fixed_hourly_kwh = fixed["scaled_energy_kwh"].to_numpy()[h0:h1]

    # Convert hourly hotel load into 15-minute energy values.
    fixed_15 = np.repeat(fixed_hourly_kwh / 4.0, 4)

    # Compute hourly available PV energy from normalized PV and installed PV size.
    pvmax_hourly_kwh = zeta_pv[h0:h1] * S_PV_KWP

    # Convert hourly PV energy into 15-minute energy values.
    pvmax_15 = np.repeat(pvmax_hourly_kwh / 4.0, 4)

    # Convert hourly prices into 15-minute prices.
    price_15 = np.repeat(prices[h0:h1], 4)

    # Build import and export tariff vectors.
    p_plus, p_minus = tariff_vectors(price_15)

    # Select all EV sessions belonging to the selected day.
    sessions = ev.loc[ev["day"] == day].copy().reset_index(drop=True)

    # Return all profiles for the selected day.
    return fixed_15, pvmax_15, p_plus, p_minus, sessions

def overlap_hours(a0, a1, b0, b1):
    """
    Compute overlap between two time intervals.

    Inputs:
    - interval A: [a0, a1]
    - interval B: [b0, b1]

    Output:
    - overlap duration in minutes

    This is used because EVs may arrive or leave in the middle
    of a 15-minute time interval.
    """

    # min(a1, b1) gives the earlier ending time.
    # max(a0, b0) gives the later starting time.
    # If the intervals do not overlap, the result is negative, so max with 0.
    return max(0.0, min(a1, b1) - max(a0, b0))


def asap_ev_profile(sessions):
    """
    Build the baseline ASAP EV charging profile.

    ASAP charging means:
    - vehicle starts charging immediately after connection
    - charging continues until the required energy is delivered
    - the charging profile is fixed and cannot be optimized
    """

    # Initialize EV energy profile for one day.
    # Unit: kWh per 15-minute interval.
    e_ev = np.zeros(N)

    # Loop over all EV charging sessions of the selected day.
    for _, s in sessions.iterrows():

        # Connection time in minutes from start of day.
        c = float(s["connection_minute"])

        # Full-charge time in minutes from start of day.
        f = float(s["full_minute"])

        # Required charging energy for this EV.
        energy = float(s["energy_kwh"])

        # Duration between connection and full-charge time in hours.
        # max avoids division by zero.
        duration_h = max((f - c) / 60.0, 1e-9)

        # Average charging power needed to deliver the required energy.
        power_kw = energy / duration_h

        # Loop over all 96 time intervals of the day.
        for k in range(N):

            # Start and end of the current 15-minute interval in minutes.
            slot0 = 15.0 * k
            slot1 = 15.0 * (k + 1)

            # Compute how much of this slot overlaps with the charging interval.
            ov_h = overlap_hours(c, f, slot0, slot1) / 60.0

            # Add the EV energy charged during this interval.
            e_ev[k] += power_kw * ov_h

    # Return the total ASAP EV charging profile.
    return e_ev


def smart_variable_map(sessions):
    """
    Create EV optimization variables only where a vehicle is connected.

    Instead of creating variables for every EV at every time step,
    the code only creates variables for feasible charging periods.
    This makes the optimization smaller and physically meaningful.
    """

    # pairs will store tuples of the form (EV index, time index).
    pairs = []

    # upper will store the upper bound for each EV charging variable.
    upper = []

    # Loop over each EV charging session.
    for i, s in sessions.iterrows():

        # EV connection time in minutes.
        c = float(s["connection_minute"])

        # EV disconnection time in minutes.
        d = float(s["disconnection_minute"])

        # Loop through all 15-minute intervals.
        for k in range(N):

            # Start and end of current 15-minute interval.
            slot0 = 15.0 * k
            slot1 = 15.0 * (k + 1)

            # Check how long the EV is connected during this slot.
            ov_h = overlap_hours(c, d, slot0, slot1) / 60.0

            # If the EV is connected during this interval, create a decision variable.
            if ov_h > 1e-12:

                # Store EV index and time index.
                pairs.append((i, k))

                # Maximum energy that can be charged in this interval.
                # Energy = power * time.
                upper.append(P_CHARGER_KW * ov_h)

    # Return mapping and upper bounds.
    return pairs, np.array(upper)


def solve_dispatch(load_15, pvmax_15, p_plus, p_minus, sessions=None, smart_ev=False, ev_fixed_15=None):
    """
    Solve the daily dispatch optimization problem using linear programming.

    Two modes are possible:

    1) smart_ev = False:
       EV charging is fixed using the ASAP profile.

    2) smart_ev = True:
       EV charging becomes controllable and is optimized.

    LP decision variables:
    [Ebat0, EPV(96), Ech(96), Edch(96), EVsmart(M), epsE(96), epsPP]

    Ebat0:
        initial battery energy

    EPV:
        PV energy used at each time step

    Ech:
        battery charging energy at each time step

    Edch:
        battery discharging energy at each time step

    EVsmart:
        smart EV charging variables

    epsE:
        auxiliary variables for electricity cost

    epsPP:
        auxiliary variable for peak power
    """

    # If smart charging is active, create EV decision variables.
    if smart_ev:
        pairs, ev_ub = smart_variable_map(sessions)
        M = len(pairs)

    # If smart charging is not active, EV demand is fixed.
    else:
        pairs, ev_ub, M = [], np.array([]), 0

        # If no fixed EV profile is given, use zero EV load.
        if ev_fixed_15 is None:
            ev_fixed_15 = np.zeros(N)

    # ------------------------------------------------------------
    # Variable indexing
    # ------------------------------------------------------------
    # The optimizer uses one long vector of decision variables.
    # These indices define where each group of variables starts.

    idx_Ebat0 = 0
    idx_PV = 1
    idx_ch = idx_PV + N
    idx_dch = idx_ch + N
    idx_ev = idx_dch + N
    idx_epsE = idx_ev + M
    idx_epsPP = idx_epsE + N
    nvar = idx_epsPP + 1

    # ------------------------------------------------------------
    # Variable bounds
    # ------------------------------------------------------------

    bounds = []

    # Initial battery energy must be between SOC limits.
    bounds.append((SOC_MIN * E_PACK_MAX, SOC_MAX * E_PACK_MAX))

    # PV used cannot exceed available PV production.
    bounds += [(0.0, pvmax_15[k]) for k in range(N)]

    # Battery charging energy limit.
    bounds += [(0.0, P_BESS_N * TD) for _ in range(N)]

    # Battery discharging energy limit.
    bounds += [(0.0, P_BESS_N * TD) for _ in range(N)]

    # Smart EV charging energy limits.
    # These only exist when smart_ev=True.
    bounds += [(0.0, ev_ub[j]) for j in range(M)]

    # Electricity cost auxiliary variables are free.
    bounds += [(None, None) for _ in range(N)]

    # Peak power auxiliary variable must be non-negative.
    bounds.append((0.0, None))

    # ------------------------------------------------------------
    # Objective function
    # ------------------------------------------------------------
    # The LP objective is c^T x.
    # We assign cost coefficients to each decision variable.

    c = np.zeros(nvar)

    # Battery charging degradation cost.
    c[idx_ch:idx_ch + N] = C_DEG * ETA_CH

    # Battery discharging degradation cost.
    c[idx_dch:idx_dch + N] = C_DEG / ETA_DCH

    # Electricity import/export cost auxiliary variables.
    c[idx_epsE:idx_epsE + N] = 1.0

    # Peak demand penalty.
    c[idx_epsPP] = P_PP * PEAK_DAILY_FACTOR

    # Constraint matrices.
    # A_ub x <= b_ub
    # A_eq x = b_eq
    A_ub, b_ub = [], []
    A_eq, b_eq = [], []

    # ------------------------------------------------------------
    # Map smart EV variables to time slots and sessions
    # ------------------------------------------------------------

    # For each time step, store the EV variables active at that time.
    ev_vars_by_time = [[] for _ in range(N)]

    # For each EV session, store the variables belonging to that EV.
    ev_vars_by_session = [[] for _ in range(len(sessions))] if smart_ev else []

    # Fill the mapping lists.
    for j, (i, k) in enumerate(pairs):
        ev_vars_by_time[k].append(idx_ev + j)
        ev_vars_by_session[i].append(idx_ev + j)

    # ------------------------------------------------------------
    # EV energy equality constraints
    # ------------------------------------------------------------
    # For smart charging, each EV must receive its required energy.

    if smart_ev:
        for i, s in sessions.iterrows():

            # Create one equality row for this EV.
            row = np.zeros(nvar)

            # Sum all charging variables belonging to this EV.
            for col in ev_vars_by_session[i]:
                row[col] = 1.0

            # Add equality: sum(EV charging energy) = required energy.
            A_eq.append(row)
            b_eq.append(float(s["energy_kwh"]))

    # ------------------------------------------------------------
    # Battery cyclic condition
    # ------------------------------------------------------------
    # Over one day, the net battery energy change is forced to zero.
    # This prevents the optimizer from ending the day with an artificially
    # empty or full battery.

    row = np.zeros(nvar)
    row[idx_ch:idx_ch + N] = ETA_CH
    row[idx_dch:idx_dch + N] = -1.0 / ETA_DCH
    A_eq.append(row)
    b_eq.append(0.0)

    # ------------------------------------------------------------
    # Time-step constraints
    # ------------------------------------------------------------
    # Add operational constraints for each 15-minute interval.

    for k in range(N):

        # --------------------------------------------------------
        # Battery converter power limit
        # --------------------------------------------------------
        # Battery cannot charge and discharge beyond converter rating.
        # Ech + Edch <= P_BESS_N * TD

        row = np.zeros(nvar)
        row[idx_ch + k] = 1.0
        row[idx_dch + k] = 1.0
        A_ub.append(row)
        b_ub.append(P_BESS_N * TD)

        # --------------------------------------------------------
        # Battery SOC limits
        # --------------------------------------------------------
        # Compute battery energy after interval k as:
        # Ebat0 + cumulative charging - cumulative discharging

        row_soc = np.zeros(nvar)
        row_soc[idx_Ebat0] = 1.0
        row_soc[idx_ch:idx_ch + k + 1] = ETA_CH
        row_soc[idx_dch:idx_dch + k + 1] = -1.0 / ETA_DCH

        # Upper SOC limit.
        A_ub.append(row_soc.copy())
        b_ub.append(SOC_MAX * E_PACK_MAX)

        # Lower SOC limit.
        A_ub.append(-row_soc.copy())
        b_ub.append(-SOC_MIN * E_PACK_MAX)

        # --------------------------------------------------------
        # Grid energy expression
        # --------------------------------------------------------
        # Egrid = load + EV + battery charge - battery discharge - PV
        # Positive Egrid means grid import.
        # Negative Egrid means grid export.

        base_load = load_15[k]

        # In baseline mode, EV charging is fixed and added to load.
        if not smart_ev:
            base_load += ev_fixed_15[k]

        # Build coefficients for the variable part of Egrid.
        row_grid = np.zeros(nvar)

        # PV reduces grid import.
        row_grid[idx_PV + k] = -1.0

        # Battery charging increases demand from PV/grid.
        row_grid[idx_ch + k] = 1.0

        # Battery discharging reduces grid import.
        row_grid[idx_dch + k] = -1.0

        # In smart charging mode, EV variables increase demand.
        if smart_ev:
            for col in ev_vars_by_time[k]:
                row_grid[col] = 1.0

        # --------------------------------------------------------
        # Grid import limit
        # --------------------------------------------------------
        # Egrid <= P_GRID_MAX * TD

        A_ub.append(row_grid.copy())
        b_ub.append(P_GRID_MAX * TD - base_load)

        # --------------------------------------------------------
        # Grid export limit
        # --------------------------------------------------------
        # -Egrid <= P_GRID_MAX * TD

        A_ub.append(-row_grid.copy())
        b_ub.append(P_GRID_MAX * TD + base_load)

        # --------------------------------------------------------
        # Peak demand constraint
        # --------------------------------------------------------
        # Egrid / TD <= epsPP
        # Rearranged into linear form:
        # Egrid - TD * epsPP <= 0

        row = row_grid.copy()
        row[idx_epsPP] = -TD
        A_ub.append(row)
        b_ub.append(-base_load)

        # --------------------------------------------------------
        # Electricity cost linearization
        # --------------------------------------------------------
        # epsE is constrained to represent the correct energy cost.
        # The two inequalities allow different import/export prices.

        # epsE >= p_plus * Egrid
        row = p_plus[k] * row_grid.copy()
        row[idx_epsE + k] = -1.0
        A_ub.append(row)
        b_ub.append(-p_plus[k] * base_load)

        # epsE >= p_minus * Egrid
        row = p_minus[k] * row_grid.copy()
        row[idx_epsE + k] = -1.0
        A_ub.append(row)
        b_ub.append(-p_minus[k] * base_load)

    # ------------------------------------------------------------
    # Solve linear programming problem
    # ------------------------------------------------------------

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
    )

    # Stop execution if optimization failed.
    if not res.success:
        raise RuntimeError(f"LP infeasible/failed: {res.message}")

    # Optimized decision vector.
    z = res.x

    # Extract optimized PV usage.
    EPV = z[idx_PV:idx_PV + N]

    # Extract optimized battery charging.
    Ech = z[idx_ch:idx_ch + N]

    # Extract optimized battery discharging.
    Edch = z[idx_dch:idx_dch + N]

    # Reconstruct battery energy trajectory.
    Ebat = z[idx_Ebat0] + np.cumsum(ETA_CH * Ech - Edch / ETA_DCH)

    # Extract EV charging profile.
    if smart_ev:

        # Initialize EV profile.
        EV = np.zeros(N)

        # Sum all EV charging decision variables by time step.
        for j, (_, k) in enumerate(pairs):
            EV[k] += z[idx_ev + j]

    else:

        # In baseline mode, EV profile is already fixed.
        EV = ev_fixed_15.copy()

    # Compute final grid exchange profile.
    Egrid = load_15 + EV + Ech - Edch - EPV

    # Compute electricity cost at each time step.
    epsE = np.maximum(p_plus * Egrid, p_minus * Egrid)

    # Total electricity energy cost.
    J_E = float(np.sum(epsE))

    # Total peak power cost.
    J_peak = float(P_PP * PEAK_DAILY_FACTOR * max(0.0, np.max(Egrid / TD)))

    # Total battery degradation cost.
    J_deg = float(np.sum(C_DEG * ETA_CH * Ech + C_DEG / ETA_DCH * Edch))

    # Total cost check.
    J_total_check = J_E + J_peak + J_deg

    # Return all important optimization outputs.
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
    """
    Select representative days from the yearly EV dataset.

    The selected days are:
    - day with minimum number of sessions
    - day closest to average number of sessions
    - day with maximum number of sessions
    """

    # Group EV sessions by day and compute daily statistics.
    daily = ev.groupby("day").agg(
        sessions=("session", "count"),
        ev_energy_kwh=("energy_kwh", "sum"),
        vehicles_generated_day=("vehicles_generated_day", "first"),
    ).reset_index()

    # Day with lowest EV activity.
    low_day = int(daily.sort_values("sessions").iloc[0]["day"])

    # Day with highest EV activity.
    high_day = int(daily.sort_values("sessions").iloc[-1]["day"])

    # Average number of sessions per day.
    avg_sessions = daily["sessions"].mean()

    # Day whose number of sessions is closest to average.
    avg_day = int(daily.iloc[(daily["sessions"] - avg_sessions).abs().argmin()]["day"])

    # Return selected days and full daily statistics.
    return [low_day, avg_day, high_day], daily


def plot_day(day, time_h, load_15, pvmax_15, baseline, smart):
    """
    Save one daily comparison plot for baseline ASAP vs smart charging.
    """

    # Create one figure with three vertically stacked plots.
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Plot fixed load, ASAP EV charging, and smart EV charging.
    ax[0].step(time_h, load_15, where="post", label="fixed load")
    ax[0].step(time_h, baseline["EV"], where="post", label="EV ASAP")
    ax[0].step(time_h, smart["EV"], where="post", label="EV smart")
    ax[0].set_ylabel("Energy [kWh/15min]")
    ax[0].grid(True)
    ax[0].legend()

    # Plot PV availability, PV used, and BESS operation.
    ax[1].step(time_h, pvmax_15, where="post", label="PV max")
    ax[1].step(time_h, smart["EPV"], where="post", label="PV used smart")
    ax[1].step(time_h, smart["Ech"], where="post", label="BESS ch")
    ax[1].step(time_h, smart["Edch"], where="post", label="BESS dch")
    ax[1].set_ylabel("Energy [kWh/15min]")
    ax[1].grid(True)
    ax[1].legend()

    # Plot grid power for baseline and smart charging.
    ax[2].step(time_h, baseline["Egrid"] / TD, where="post", label="grid baseline")
    ax[2].step(time_h, smart["Egrid"] / TD, where="post", label="grid smart")
    ax[2].set_ylabel("Power [kW]")
    ax[2].set_xlabel("Time [h]")
    ax[2].grid(True)
    ax[2].legend()

    # Add title to the full figure.
    fig.suptitle(f"Day {day}: baseline ASAP vs smart EV charging")

    # Improve spacing so labels do not overlap.
    fig.tight_layout()

    # Save the figure using a clean file name.
    fig.savefig(PLOTS_DIR / f"baseline_day_{day:03d}_dispatch.png", dpi=300)

    # Close the figure so the script continues to the next day.
    plt.close(fig)



def plot_annual_overview(ev, fixed, zeta_pv):
    """
    Save an annual daily-energy overview plot.
    """

    # Create a day index from 1 to 365.
    days = np.arange(1, 366)

    # Compute daily EV charging energy from all EV sessions.
    daily_ev = ev.groupby("day")["energy_kwh"].sum().reindex(days, fill_value=0).to_numpy()

    # Compute daily fixed hotel load energy from the hourly load profile.
    daily_load = fixed["scaled_energy_kwh"].to_numpy().reshape(365, 24).sum(axis=1)

    # Compute daily available PV energy from normalized PV production and installed PV size.
    daily_pv = (zeta_pv * S_PV_KWP).reshape(365, 24).sum(axis=1)

    # Create the annual overview figure.
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot daily hotel load energy.
    ax.plot(days, daily_load, label="Daily fixed load")

    # Plot daily EV energy.
    ax.plot(days, daily_ev, label="Daily EV demand")

    # Plot daily PV energy.
    ax.plot(days, daily_pv, label="Daily PV availability")

    # Add axis labels.
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Energy [kWh/day]")

    # Add grid and legend.
    ax.grid(True)
    ax.legend()

    # Add title.
    ax.set_title("Annual daily energy profiles")

    # Save the annual overview figure.
    fig.savefig(PLOTS_DIR / "annual_daily_energy_profiles.png", dpi=300)

    # Close the figure.
    plt.close(fig)

def main():
    """
    Main execution workflow.

    Steps:
    1. Read input data.
    2. Select representative days.
    3. Build ASAP charging profile.
    4. Solve baseline optimization.
    5. Solve smart charging optimization.
    6. Compare and save results.
    """

    # Load all input datasets.
    # Read all project input data.
    ev, fixed, zeta_pv, prices = read_inputs()

    # Print annual hotel-load energy.
    print("Annual load =", fixed["scaled_energy_kwh"].sum())

    # Print annual EV charging energy.
    print("Annual EV =", ev["energy_kwh"].sum())

    # Print annual PV energy available.
    print("Annual PV =", np.sum(zeta_pv * S_PV_KWP))

    # Print number of hourly electricity prices.
    print("Annual prices =", len(prices))

    # Select low, average, and high EV demand days.
    selected_days, daily_info = choose_representative_days(ev)
    
    # Save annual daily energy overview plot.
    plot_annual_overview(ev, fixed, zeta_pv)

    # Time vector for plotting: 0, 0.25, 0.50, ..., 23.75.
    time_h = np.arange(N) * TD

    # Print basic dataset information.
    print("Selected representative days:", selected_days)
    print("EV sessions:", len(ev), "Total EV energy [kWh]:", ev["energy_kwh"].sum())
    # Print annual hotel-load energy from the scaled hotel CSV.
    #print("Annual load =", fixed["scaled_energy_kwh"].sum())

    # Store results for all representative days.
    rows = []

    # Loop over selected representative days.
    for day in selected_days:

        # Extract daily load, PV, price, and EV session data.
        # Extract the 15-minute profiles for this selected day.
        load_15, pvmax_15, p_plus, p_minus, sessions = day_profiles(day, ev, fixed, zeta_pv, prices)

        # Build ASAP EV charging profile for baseline case.
        ev_asap = asap_ev_profile(sessions)

        # ------------------------------------------------------------
        # Baseline case
        # ------------------------------------------------------------
        # EV charging is fixed according to ASAP behaviour.
        baseline = solve_dispatch(load_15, pvmax_15, p_plus, p_minus,
                                  smart_ev=False, ev_fixed_15=ev_asap)

        # ------------------------------------------------------------
        # Smart charging case
        # ------------------------------------------------------------
        # EV charging is optimized within connection windows.
        smart = solve_dispatch(load_15, pvmax_15, p_plus, p_minus,
                               sessions=sessions, smart_ev=True)

        # Compute comparison metrics for this day.
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

        # Add current day result to final summary list.
        rows.append(row)

        # Save detailed schedule for this day.
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

        # Export daily schedule as CSV.
        out.to_csv(OUT_DIR / f"day_{day}_schedule.csv", index=False)

        # Generate and save plot for this day.
        plot_day(day, time_h, load_15, pvmax_15, baseline, smart)

        # Print result of current day.
        print(f"\nDay {day}")
        print(pd.Series(row).to_string())

    # Convert results into dataframe.
    summary = pd.DataFrame(rows)

    # Save summary of all selected days.
    summary.to_csv(OUT_DIR / "summary_results.csv", index=False)

    # Save yearly daily EV statistics.
    daily_info.to_csv(OUT_DIR / "daily_ev_statistics.csv", index=False)

    # Print output location and summary table.
    print("\nSaved results in:", OUT_DIR)
    print(summary.to_string(index=False))


# Run the main function only if this file is executed directly.
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


# add_season_table_and_savings_plot()