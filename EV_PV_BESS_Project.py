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
P_CHARGER_KW = 22.0

# Sampling time [h]. Use 1.0 for hourly optimization.
TD = 1.0

# Number of time steps per day.
N = int(24 / TD)

# Slot length in minutes.
SLOT_MIN = 60.0 * TD

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

# Output folder where all project results will be stored.
OUT_DIR = BASE_DIR / "Project_Results"

# Create the folder if it does not exist.
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Folder where all figures will be saved.
PLOTS_DIR = OUT_DIR / "Figures"

# Create the figures folder.
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Folder where scenario-analysis results will be saved.
SCENARIO_DIR = OUT_DIR / "Scenario_Analysis"

# Create the scenario-analysis folder.
SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
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
    # fixed_15 = np.repeat(fixed_hourly_kwh / 4.0, 4)
    fixed_15 = np.repeat(fixed_hourly_kwh / (1.0 / TD), int(1.0 / TD))
    # Compute hourly available PV energy from normalized PV and installed PV size.
    pvmax_hourly_kwh = zeta_pv[h0:h1] * S_PV_KWP

    # Convert hourly PV energy into 15-minute energy values.
    # pvmax_15 = np.repeat(pvmax_hourly_kwh / 4.0, 4)
    pvmax_15 = np.repeat(pvmax_hourly_kwh / (1.0 / TD), int(1.0 / TD))

    # Convert hourly prices into 15-minute prices.
    # price_15 = np.repeat(prices[h0:h1], 4)
    price_15 = np.repeat(prices[h0:h1], int(1.0 / TD))
    
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
            slot0 = SLOT_MIN * k
            slot1 = SLOT_MIN * (k + 1)
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
            slot0 = SLOT_MIN * k
            slot1 = SLOT_MIN * (k + 1)

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
    Save a clean daily EV scheduling figure.

    This figure only shows the variables needed to explain
    how smart charging changes the EV charging profile.
    """

    # Convert building load from kWh per 15 minutes to kW.
    fixed_load_kw = load_15 / TD

    # Convert available PV from kWh per 15 minutes to kW.
    pv_available_kw = pvmax_15 / TD

    # Convert ASAP EV charging from kWh per 15 minutes to kW.
    ev_asap_kw = baseline["EV"] / TD

    # Convert smart EV charging from kWh per 15 minutes to kW.
    ev_smart_kw = smart["EV"] / TD

    # Create one clean figure.
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the fixed building load.
    ax.step(time_h, fixed_load_kw, where="post", label="Building load")

    # Plot available PV power.
    ax.step(time_h, pv_available_kw, where="post", label="PV availability")

    # Plot baseline ASAP EV charging.
    ax.step(time_h, ev_asap_kw, where="post", label="EV ASAP")

    # Plot optimized smart EV charging.
    ax.step(time_h, ev_smart_kw, where="post", label="EV smart")

    # Set x-axis label.
    ax.set_xlabel("Time [h]")

    # Set y-axis label.
    ax.set_ylabel("Power [kW]")

    # Set title.
    ax.set_title(f"Day {day}: EV charging schedule comparison")

    # Add grid.
    ax.grid(True)

    # Add legend.
    ax.legend()

    # Make layout compact.
    fig.tight_layout()

    # Save figure with clear name.
    fig.savefig(PLOTS_DIR / f"Fig_EV_schedule_day_{day:03d}.png", dpi=300)

    # Close figure after saving.
    plt.close(fig)

def plot_grid_comparison(day, time_h, baseline, smart):
    """
    Save a clean grid-power comparison figure.

    This figure shows how smart charging changes the grid power profile.
    """

    # Convert baseline grid exchange from kWh per 15 minutes to kW.
    grid_baseline_kw = baseline["Egrid"] / TD

    # Convert smart grid exchange from kWh per 15 minutes to kW.
    grid_smart_kw = smart["Egrid"] / TD

    # Create one clean figure.
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot baseline grid power.
    ax.step(time_h, grid_baseline_kw, where="post", label="Grid baseline")

    # Plot smart charging grid power.
    ax.step(time_h, grid_smart_kw, where="post", label="Grid smart")

    # Add horizontal zero line to separate import and export.
    ax.axhline(0, linewidth=1)

    # Set x-axis label.
    ax.set_xlabel("Time [h]")

    # Set y-axis label.
    ax.set_ylabel("Grid power [kW]")

    # Set title.
    ax.set_title(f"Day {day}: grid power comparison")

    # Add grid.
    ax.grid(True)

    # Add legend.
    ax.legend()

    # Make layout compact.
    fig.tight_layout()

    # Save figure.
    fig.savefig(PLOTS_DIR / f"Fig_Grid_day_{day:03d}.png", dpi=300)

    # Close figure after saving.
    plt.close(fig)

    
def plot_battery_operation(day, time_h, smart):
    """
    Save a clean battery operation figure for the smart charging case.

    This figure shows how the BESS supports the optimized EV schedule.
    """

    # Convert battery charging energy from kWh per 15 minutes to kW.
    bess_charge_kw = smart["Ech"] / TD

    # Convert battery discharging energy from kWh per 15 minutes to kW.
    bess_discharge_kw = smart["Edch"] / TD

    # Convert battery energy to SOC in percent.
    soc_percent = 100.0 * smart["Ebat"] / E_PACK_MAX

    # Create one figure with two vertical subplots.
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot BESS charging power.
    ax[0].step(time_h, bess_charge_kw, where="post", label="BESS charge")

    # Plot BESS discharging power.
    ax[0].step(time_h, bess_discharge_kw, where="post", label="BESS discharge")

    # Set y-axis label for power subplot.
    ax[0].set_ylabel("Power [kW]")

    # Add grid to power subplot.
    ax[0].grid(True)

    # Add legend to power subplot.
    ax[0].legend()

    # Plot battery SOC.
    ax[1].step(time_h, soc_percent, where="post", label="Battery SOC")

    # Set x-axis label.
    ax[1].set_xlabel("Time [h]")

    # Set y-axis label for SOC subplot.
    ax[1].set_ylabel("SOC [%]")

    # Add grid to SOC subplot.
    ax[1].grid(True)

    # Add legend to SOC subplot.
    ax[1].legend()

    # Set full figure title.
    fig.suptitle(f"Day {day}: battery operation under smart charging")

    # Make layout compact.
    fig.tight_layout()

    # Save figure.
    fig.savefig(PLOTS_DIR / f"Fig_Battery_day_{day:03d}.png", dpi=300)

    # Close figure after saving.
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






def compute_day_metrics(day, sessions, load_15, pvmax_15, baseline, smart):
    """
    Compute numerical performance metrics for one representative day.
    """

    # Compute total fixed building load energy for the day.
    building_load_kwh = float(np.sum(load_15))

    # Compute total EV charging energy required on this day.
    ev_energy_kwh = float(sessions["energy_kwh"].sum())

    # Compute total available PV energy on this day.
    pv_available_kwh = float(np.sum(pvmax_15))

    # Compute total PV energy used in the baseline case.
    baseline_pv_used_kwh = float(baseline["pv_used_kwh"])

    # Compute total PV energy used in the smart charging case.
    smart_pv_used_kwh = float(smart["pv_used_kwh"])

    # Compute PV utilization percentage for the baseline case.
    baseline_pv_util_percent = 100.0 * baseline_pv_used_kwh / max(pv_available_kwh, 1e-9)

    # Compute PV utilization percentage for the smart case.
    smart_pv_util_percent = 100.0 * smart_pv_used_kwh / max(pv_available_kwh, 1e-9)

    # Compute total battery throughput for smart charging.
    smart_battery_throughput_kwh = float(np.sum(smart["Ech"] + smart["Edch"]))

    # Compute approximate equivalent battery cycles for smart charging.
    smart_equivalent_cycles = smart_battery_throughput_kwh / max(2.0 * E_PACK_MAX, 1e-9)

    # Compute grid import reduction in kWh.
    import_reduction_kwh = baseline["import_kwh"] - smart["import_kwh"]

    # Compute grid import reduction in percent.
    import_reduction_percent = 100.0 * import_reduction_kwh / max(baseline["import_kwh"], 1e-9)

    # Compute peak grid power reduction in kW.
    peak_reduction_kw = baseline["peak_kw"] - smart["peak_kw"]

    # Compute peak grid power reduction in percent.
    peak_reduction_percent = 100.0 * peak_reduction_kw / max(baseline["peak_kw"], 1e-9)

    # Compute total daily cost saving in EUR.
    cost_saving_eur = baseline["J_total_check"] - smart["J_total_check"]

    # Compute total daily cost saving in percent.
    cost_saving_percent = 100.0 * cost_saving_eur / max(abs(baseline["J_total_check"]), 1e-9)

    # Return all computed metrics as one dictionary.
    return {
        "day": day,
        "sessions": len(sessions),
        "building_load_kwh": building_load_kwh,
        "ev_energy_kwh": ev_energy_kwh,
        "pv_available_kwh": pv_available_kwh,
        "baseline_cost_eur": baseline["J_total_check"],
        "smart_cost_eur": smart["J_total_check"],
        "cost_saving_eur": cost_saving_eur,
        "cost_saving_percent": cost_saving_percent,
        "baseline_import_kwh": baseline["import_kwh"],
        "smart_import_kwh": smart["import_kwh"],
        "import_reduction_kwh": import_reduction_kwh,
        "import_reduction_percent": import_reduction_percent,
        "baseline_export_kwh": baseline["export_kwh"],
        "smart_export_kwh": smart["export_kwh"],
        "baseline_peak_kw": baseline["peak_kw"],
        "smart_peak_kw": smart["peak_kw"],
        "peak_reduction_kw": peak_reduction_kw,
        "peak_reduction_percent": peak_reduction_percent,
        "baseline_pv_used_kwh": baseline_pv_used_kwh,
        "smart_pv_used_kwh": smart_pv_used_kwh,
        "baseline_pv_util_percent": baseline_pv_util_percent,
        "smart_pv_util_percent": smart_pv_util_percent,
        "smart_battery_throughput_kwh": smart_battery_throughput_kwh,
        "smart_equivalent_cycles": smart_equivalent_cycles,
    }


def run_scenario_analysis(ev, fixed, zeta_pv, prices, selected_days):
    """
    Run additional scenario analysis for the representative days.
    """

        # Define scenario list.
    scenarios = [
        {
            "name": "High_EV_Demand_150",
            "ev_multiplier": 1.50,
            "pv_multiplier": 1.00,
            "battery_multiplier": 1.00,
            "price_multiplier": 1.00,
        },
        {
            "name": "Low_PV_70",
            "ev_multiplier": 1.00,
            "pv_multiplier": 0.70,
            "battery_multiplier": 1.00,
            "price_multiplier": 1.00,
        },
        {
            "name": "High_PV_130",
            "ev_multiplier": 1.00,
            "pv_multiplier": 1.30,
            "battery_multiplier": 1.00,
            "price_multiplier": 1.00,
        },
        {
            "name": "Large_Battery_2x",
            "ev_multiplier": 1.00,
            "pv_multiplier": 1.00,
            "battery_multiplier": 2.00,
            "price_multiplier": 1.00,
        },
        {
            "name": "No_Battery",
            "ev_multiplier": 1.00,
            "pv_multiplier": 1.00,
            "battery_multiplier": 0.00,
            "price_multiplier": 1.00,
        },
        {
            "name": "High_Electricity_Price_2x",
            "ev_multiplier": 1.00,
            "pv_multiplier": 1.00,
            "battery_multiplier": 1.00,
            "price_multiplier": 2.00,
        },
    ]

    # Store original global battery values.
    original_e_pack_max = E_PACK_MAX
    original_p_bess_n = P_BESS_N

    # Store all scenario results.
    scenario_rows = []

    # Loop through all scenarios.
    for scenario in scenarios:

        # Print scenario currently being solved.
        print("\nRunning scenario:", scenario["name"])

        # Scale PV profile for this scenario.
        zeta_pv_scenario = zeta_pv * scenario["pv_multiplier"]
        
        # Scale electricity prices for this scenario.
        prices_scenario = prices * scenario["price_multiplier"]

        # Create a copy of EV sessions for this scenario.
        ev_scenario = ev.copy()

        # Scale EV energy demand for this scenario.
        ev_scenario["energy_kwh"] = ev_scenario["energy_kwh"] * scenario["ev_multiplier"]

        # Temporarily modify global battery values.
        globals()["E_PACK_MAX"] = original_e_pack_max * scenario["battery_multiplier"]
        globals()["P_BESS_N"] = original_p_bess_n * scenario["battery_multiplier"]

        # Avoid exact zero battery capacity because SOC calculations divide by E_PACK_MAX.
        if scenario["battery_multiplier"] == 0.00:
            globals()["E_PACK_MAX"] = 1e-6
            globals()["P_BESS_N"] = 0.0

        # Loop over representative days.
        for day in selected_days:

            # Extract daily profiles using scenario-scaled PV and price data.
            load_15, pvmax_15, p_plus, p_minus, sessions = day_profiles(
                day,
                ev_scenario,
                fixed,
                zeta_pv_scenario,
                prices_scenario,
            )

            # Build ASAP EV charging profile.
            ev_asap = asap_ev_profile(sessions)

            # Solve baseline ASAP case.
            baseline = solve_dispatch(
                load_15,
                pvmax_15,
                p_plus,
                p_minus,
                smart_ev=False,
                ev_fixed_15=ev_asap,
            )

            # Solve smart charging case.
            smart = solve_dispatch(
                load_15,
                pvmax_15,
                p_plus,
                p_minus,
                sessions=sessions,
                smart_ev=True,
            )

            # Compute metrics for this scenario and day.
            row = compute_day_metrics(day, sessions, load_15, pvmax_15, baseline, smart)

            # Add scenario name.
            row["scenario"] = scenario["name"]

            # Add scenario multipliers.
            row["ev_multiplier"] = scenario["ev_multiplier"]
            row["pv_multiplier"] = scenario["pv_multiplier"]
            row["battery_multiplier"] = scenario["battery_multiplier"]

            # Store electricity price multiplier for this scenario.
            row["price_multiplier"] = scenario["price_multiplier"]

            # Add row to scenario results.
            scenario_rows.append(row)

    # Restore original battery values.
    globals()["E_PACK_MAX"] = original_e_pack_max
    globals()["P_BESS_N"] = original_p_bess_n

    # Convert scenario results to DataFrame.
    scenario_summary = pd.DataFrame(scenario_rows)

    # Save detailed scenario results.
    scenario_summary.to_csv(
        SCENARIO_DIR / "Scenario_Analysis_Detailed_Metrics.csv",
        index=False,
    )

    # Save rounded scenario results.
    scenario_summary.round(2).to_csv(
        SCENARIO_DIR / "Scenario_Analysis_Report_Table.csv",
        index=False,
    )

    # Print confirmation.
    print("\nSaved scenario analysis results in:", SCENARIO_DIR)

    # Return results in case we want to plot later.
    return scenario_summary



def plot_scenario_comparison():
    """
    Create comparison figures for the scenario analysis.

    Only Day 153 is used because it has the highest PV production
    and therefore best illustrates the effect of each scenario.
    """

    # Read the scenario analysis table.
    scenario_file = SCENARIO_DIR / "Scenario_Analysis_Report_Table.csv"

    if not scenario_file.exists():
        print("Scenario table not found.")
        return

    scenario = pd.read_csv(scenario_file)

    # Only keep Day 153.
    scenario = scenario[scenario["day"] == 153].copy()

    # Shorter names for plotting.
    rename = {
        "High_EV_Demand_150": "EV +50%",
        "Low_PV_70": "PV -30%",
        "High_PV_130": "PV +30%",
        "Large_Battery_2x": "Battery ×2",
        "No_Battery": "No Battery",
        "High_Electricity_Price_2x": "Price ×2",
    }

    scenario["Scenario"] = scenario["scenario"].map(rename)

    ####################################################################
    # Figure 14
    ####################################################################

    plt.figure(figsize=(8,5))

    plt.bar(
        scenario["Scenario"],
        scenario["cost_saving_eur"]
    )

    plt.ylabel("Cost saving [EUR/day]")
    plt.title("Scenario comparison - Daily cost savings")
    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "Figure_14_Scenario_Cost_Saving.png",
        dpi=300
    )

    plt.close()

    ####################################################################
    # Figure 15
    ####################################################################

    plt.figure(figsize=(8,5))

    plt.bar(
        scenario["Scenario"],
        scenario["import_reduction_percent"]
    )

    plt.ylabel("Grid import reduction [%]")
    plt.title("Scenario comparison - Grid import reduction")
    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "Figure_15_Scenario_Grid_Import.png",
        dpi=300
    )

    plt.close()

    ####################################################################
    # Figure 16
    ####################################################################

    plt.figure(figsize=(8,5))

    plt.bar(
        scenario["Scenario"],
        scenario["peak_reduction_kw"]
    )

    plt.ylabel("Peak reduction [kW]")
    plt.title("Scenario comparison - Peak reduction")
    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "Figure_16_Scenario_Peak_Reduction.png",
        dpi=300
    )

    plt.close()

    ####################################################################
    # Figure 17
    ####################################################################

    plt.figure(figsize=(8,5))

    plt.bar(
        scenario["Scenario"],
        scenario["smart_cost_eur"]
    )

    plt.ylabel("Operating cost [EUR/day]")
    plt.title("Scenario comparison - Smart charging operating cost")
    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "Figure_17_Scenario_Operating_Cost.png",
        dpi=300
    )

    plt.close()

    print("\nScenario comparison figures saved.")




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
                # Compute all performance metrics for this representative day.
        row = compute_day_metrics(day, sessions, load_15, pvmax_15, baseline, smart)

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
        # Save the detailed schedule for this representative day.
        out.to_csv(OUT_DIR / f"Representative_Day_{day:03d}_Schedule.csv",index=False)

        # Generate and save plot for this day.
        # Save clean EV scheduling figure for this day.
        plot_day(day, time_h, load_15, pvmax_15, baseline, smart)

        # Save clean grid comparison figure for this day.
        plot_grid_comparison(day, time_h, baseline, smart)

        # Save clean battery operation figure for this day.
        plot_battery_operation(day, time_h, smart)

        # Print result of current day.
        print(f"\nDay {day}")
        print(pd.Series(row).to_string())

    # Convert results into dataframe.
    summary = pd.DataFrame(rows)

    # Save summary of all selected days.
        # Save full numerical results.
    # Save all calculated performance indicators.
    summary.to_csv(OUT_DIR / "Representative_Days_Performance_Metrics.csv",index=False,)

    # Save rounded report-ready results.
    # Save rounded values ready to copy directly into the report.
    summary.round(2).to_csv(OUT_DIR / "Representative_Days_Performance_Metrics_Report.csv",index=False,)


    # Save yearly daily EV statistics.
    daily_info.to_csv(OUT_DIR / "Yearly_EV_Demand_Statistics.csv", index=False)

    # Run scenario analysis after the base representative-day analysis.
    run_scenario_analysis(ev, fixed, zeta_pv, prices, selected_days)

    # Generate comparison figures for the scenario analysis.
    plot_scenario_comparison()

    # Generate one 3-day EV schedule figure for each scenario.
    plot_ev_schedule_scenarios_all_days()

    # Print output location and summary table.
    print("\nSaved results in:", OUT_DIR)
    print(summary.to_string(index=False))


def plot_ev_schedule_scenarios_all_days():
    """
    Create one EV schedule figure per scenario.

    Each figure contains 3 subplots:
    - Day 23
    - Day 153
    - Day 183

    Each subplot shows:
    - Building Load
    - PV Generation
    - ASAP Charging
    - Smart Charging

    A small text box gives:
    - Cost saving
    - Peak reduction
    """

    # Define scenario list in the order we want to plot.
    scenarios = [
        {
            "name": "High_EV_Demand_150",
            "label": "EV +50%",
            "ev_multiplier": 1.50,
            "pv_multiplier": 1.00,
            "battery_multiplier": 1.00,
            "price_multiplier": 1.00,
        },
        {
            "name": "Low_PV_70",
            "label": "PV -30%",
            "ev_multiplier": 1.00,
            "pv_multiplier": 0.70,
            "battery_multiplier": 1.00,
            "price_multiplier": 1.00,
        },
        {
            "name": "High_PV_130",
            "label": "PV +30%",
            "ev_multiplier": 1.00,
            "pv_multiplier": 1.30,
            "battery_multiplier": 1.00,
            "price_multiplier": 1.00,
        },
        {
            "name": "Large_Battery_2x",
            "label": "Battery x2",
            "ev_multiplier": 1.00,
            "pv_multiplier": 1.00,
            "battery_multiplier": 2.00,
            "price_multiplier": 1.00,
        },
        {
            "name": "No_Battery",
            "label": "No Battery",
            "ev_multiplier": 1.00,
            "pv_multiplier": 1.00,
            "battery_multiplier": 0.00,
            "price_multiplier": 1.00,
        },
        {
            "name": "High_Electricity_Price_2x",
            "label": "Price x2",
            "ev_multiplier": 1.00,
            "pv_multiplier": 1.00,
            "battery_multiplier": 1.00,
            "price_multiplier": 2.00,
        },
    ]

    # Read all input datasets.
    ev, fixed, zeta_pv, prices = read_inputs()

    # Days to show in each scenario figure.
    days_to_plot = [23, 153, 183]

    # Time vector for plotting.
    time_h = np.arange(N) * TD

    # Store original battery values.
    original_e_pack_max = E_PACK_MAX
    original_p_bess_n = P_BESS_N

    # Loop through each scenario.
    for scenario in scenarios:

        # Scale EV demand for this scenario.
        ev_scenario = ev.copy()
        ev_scenario["energy_kwh"] = ev_scenario["energy_kwh"] * scenario["ev_multiplier"]

        # Scale PV production for this scenario.
        zeta_pv_scenario = zeta_pv * scenario["pv_multiplier"]

        # Scale electricity prices for this scenario.
        prices_scenario = prices * scenario["price_multiplier"]

        # Temporarily scale battery capacity and power.
        globals()["E_PACK_MAX"] = original_e_pack_max * scenario["battery_multiplier"]
        globals()["P_BESS_N"] = original_p_bess_n * scenario["battery_multiplier"]

        # Disable battery safely for the no-battery case.
        if scenario["battery_multiplier"] == 0.00:
            globals()["E_PACK_MAX"] = 1e-6
            globals()["P_BESS_N"] = 0.0

        # Create one figure with 3 vertical subplots.
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

        # Add global title.
        fig.suptitle(
            f"Scenario: {scenario['label']} - EV Charging Schedule Comparison",
            fontsize=18,
            fontweight="bold",
        )

        # Loop through the 3 representative days.
        for i, day in enumerate(days_to_plot):

            # Select current subplot.
            ax = axes[i]

            # Extract daily profiles.
            load_15, pvmax_15, p_plus, p_minus, sessions = day_profiles(
                day,
                ev_scenario,
                fixed,
                zeta_pv_scenario,
                prices_scenario,
            )

            # Build baseline ASAP EV profile.
            ev_asap = asap_ev_profile(sessions)

            # Solve baseline dispatch.
            baseline = solve_dispatch(
                load_15,
                pvmax_15,
                p_plus,
                p_minus,
                smart_ev=False,
                ev_fixed_15=ev_asap,
            )

            # Solve smart charging dispatch.
            smart = solve_dispatch(
                load_15,
                pvmax_15,
                p_plus,
                p_minus,
                sessions=sessions,
                smart_ev=True,
            )

            # Compute metrics for the text box.
            metrics = compute_day_metrics(
                day,
                sessions,
                load_15,
                pvmax_15,
                baseline,
                smart,
            )

            # Convert profiles from kWh per step to kW.
            building_kw = load_15 / TD
            pv_kw = pvmax_15 / TD
            ev_asap_kw = baseline["EV"] / TD
            ev_smart_kw = smart["EV"] / TD

            # Plot building load.
            ax.step(
                time_h,
                building_kw,
                where="post",
                label="Building Load",
                linewidth=2.3,
            )

            # Plot PV generation.
            ax.step(
                time_h,
                pv_kw,
                where="post",
                label="PV Generation",
                linewidth=2.3,
            )

            # Plot ASAP charging.
            ax.step(
                time_h,
                ev_asap_kw,
                where="post",
                label="ASAP Charging",
                linewidth=2.3,
            )

            # Plot smart charging.
            ax.step(
                time_h,
                ev_smart_kw,
                where="post",
                label="Smart Charging",
                linewidth=2.3,
            )

            # Add metrics text box.
            # Compute peak change using smart peak minus baseline peak.

            # Negative value means smart charging reduced the peak.
            # Positive value means smart charging increased the peak.
            delta_peak_kw = metrics["smart_peak_kw"] - metrics["baseline_peak_kw"]

            # Create the annotation text shown inside each subplot.
            textstr = (
                f"Saving: {metrics['cost_saving_eur']:.2f} EUR/day\n"
                rf"$\Delta P_{{\mathrm{{peak}}}}$: {delta_peak_kw:+.1f} kW")

            ax.text(
                0.02,
                0.95,
                textstr,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="gray",
                ),
            )

            # Set subplot title.
            ax.set_title(f"Day {day}", fontsize=14)

            # Set y-axis label.
            ax.set_ylabel("Power [kW]", fontsize=12)

            # Set grid.
            ax.grid(True, alpha=0.35)

            # Set axis tick size.
            ax.tick_params(axis="both", labelsize=10)

            # Add legend only to first subplot.
            if i == 0:
                ax.legend(loc="upper right", fontsize=10)

        # Set x-axis label only on the bottom subplot.
        axes[-1].set_xlabel("Time [h]", fontsize=12)

        # Improve spacing.
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Create safe filename.
        safe_name = scenario["name"]

        # Save figure.
        fig.savefig(
            PLOTS_DIR / f"Figure_EV_Schedule_{safe_name}_All_Days.png",
            dpi=300,
        )

        # Close figure.
        plt.close(fig)

    # Restore original battery values.
    globals()["E_PACK_MAX"] = original_e_pack_max
    globals()["P_BESS_N"] = original_p_bess_n

    # Print confirmation.
    print("Saved scenario EV schedule figures for all representative days.")


# ============================================================
# EXTRA EXPLANATION FILE FOR UNDERSTANDING THE PROJECT RESULTS
# ============================================================

def write_explanation_file():
    """
    Save a short text explanation of the project results.
    """

    # Define the explanation file path inside the main output folder.
    explanation_path = OUT_DIR / "Project_Result_Explanation.txt"

    # Text that will be saved in the explanation file.
    text = """
EV PV BESS SMART CHARGING PROJECT - RESULT EXPLANATION

The project compares two EV charging strategies:

1. Baseline ASAP charging:
EVs charge as soon as possible after connection.

2. Smart charging:
EV charging is optimized within the vehicle connection window.

The system includes:
- fixed hotel load,
- PV generation,
- battery energy storage,
- grid import/export,
- EV charging demand.

The main result files are:
- Representative_Days_Performance_Metrics.csv
- Representative_Days_Performance_Metrics_Report.csv
- Yearly_EV_Demand_Statistics.csv
- Representative_Day_XXX_Schedule.csv

The main figures are saved inside:
- Project_Results/Figures
"""

    # Write the explanation text file.
    with open(explanation_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Print confirmation.
    print("Saved explanation file in:", explanation_path)


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
    """
    Create additional report-ready savings tables and plots.
    """

    # Use the main output folder.
    results_dir = OUT_DIR

    # Read the renamed performance metrics file.
    summary_file = results_dir / "Representative_Days_Performance_Metrics.csv"

    # Stop if the metrics file does not exist yet.
    if not summary_file.exists():
        print("Representative_Days_Performance_Metrics.csv not found. Run main() first.")
        return

    # Read the summary metrics table.
    summary = pd.read_csv(summary_file)

    # Add season label to each representative day.
    summary["season"] = summary["day"].apply(get_season_from_day)

    # Create readable day labels.
    summary["day_description"] = summary.apply(
        lambda row: f"Day {int(row['day']):03d} ({row['season']})",
        axis=1
    )

    # Create cost saving interpretation text.
    summary["cost_reduction_text"] = summary.apply(
        lambda row: f"{row['cost_saving_eur']:.2f} EUR saved ({row['cost_saving_percent']:.2f}%)",
        axis=1
    )

    # Create peak reduction interpretation text.
    summary["peak_reduction_text"] = summary.apply(
        lambda row: f"{row['peak_reduction_kw']:.2f} kW peak reduction ({row['peak_reduction_percent']:.2f}%)",
        axis=1
    )

    # Select the most useful columns for the report.
    report_table = summary[
        [
            "day_description",
            "sessions",
            "building_load_kwh",
            "ev_energy_kwh",
            "pv_available_kwh",
            "baseline_cost_eur",
            "smart_cost_eur",
            "cost_saving_eur",
            "cost_saving_percent",
            "baseline_import_kwh",
            "smart_import_kwh",
            "import_reduction_kwh",
            "import_reduction_percent",
            "baseline_peak_kw",
            "smart_peak_kw",
            "peak_reduction_kw",
            "peak_reduction_percent",
            "smart_battery_throughput_kwh",
            "smart_equivalent_cycles",
            "cost_reduction_text",
            "peak_reduction_text",
        ]
    ].copy()

    # Rename columns for readability.
    report_table = report_table.rename(
        columns={
            "day_description": "Representative day",
            "sessions": "EV sessions",
            "building_load_kwh": "Building load [kWh]",
            "ev_energy_kwh": "EV energy [kWh]",
            "pv_available_kwh": "PV available [kWh]",
            "baseline_cost_eur": "Baseline cost [EUR]",
            "smart_cost_eur": "Smart cost [EUR]",
            "cost_saving_eur": "Cost saving [EUR]",
            "cost_saving_percent": "Cost saving [%]",
            "baseline_import_kwh": "Baseline import [kWh]",
            "smart_import_kwh": "Smart import [kWh]",
            "import_reduction_kwh": "Import reduction [kWh]",
            "import_reduction_percent": "Import reduction [%]",
            "baseline_peak_kw": "Baseline peak [kW]",
            "smart_peak_kw": "Smart peak [kW]",
            "peak_reduction_kw": "Peak reduction [kW]",
            "peak_reduction_percent": "Peak reduction [%]",
            "smart_battery_throughput_kwh": "Battery throughput [kWh]",
            "smart_equivalent_cycles": "Equivalent battery cycles",
            "cost_reduction_text": "Cost interpretation",
            "peak_reduction_text": "Peak interpretation",
        }
    )

    # Round numerical columns.
    report_table = report_table.round(2)

    # Save report-ready CSV.
    report_csv = results_dir / "Smart_Charging_Savings_Summary.csv"
    report_table.to_csv(report_csv, index=False)

    # Save report-ready text file.
    report_txt = results_dir / "Smart_Charging_Savings_Summary.txt"
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("SMART CHARGING SAVINGS SUMMARY\n")
        f.write("==============================\n\n")
        f.write(report_table.to_string(index=False))

    # Plot cost comparison.
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary))
    width = 0.35

    ax.bar(x - width / 2, summary["baseline_cost_eur"], width, label="Baseline ASAP")
    ax.bar(x + width / 2, summary["smart_cost_eur"], width, label="Smart charging")

    ax.set_xticks(x)
    ax.set_xticklabels(summary["day_description"])
    ax.set_ylabel("Daily cost [EUR]")
    ax.set_title("Daily operating cost comparison")
    ax.grid(axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "Figure_11_Cost_Comparison.png", dpi=300)
    plt.close(fig)

    # Plot daily cost savings.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary["day_description"], summary["cost_saving_eur"])
    ax.set_ylabel("Cost saving [EUR/day]")
    ax.set_title("Daily cost savings from smart charging")
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "Figure_12_Daily_Cost_Savings.png", dpi=300)
    plt.close(fig)

    # Plot peak reduction.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary["day_description"], summary["peak_reduction_kw"])
    ax.set_ylabel("Peak reduction [kW]")
    ax.set_title("Grid peak reduction from smart charging")
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "Figure_13_Peak_Reduction.png", dpi=300)
    plt.close(fig)

    # Print confirmation.
    print("\nSaved extra report tables and plots:")
    print(report_csv)
    print(report_txt)
    print(PLOTS_DIR / "Figure_11_Cost_Comparison.png")
    print(PLOTS_DIR / "Figure_12_Daily_Cost_Savings.png")
    print(PLOTS_DIR / "Figure_13_Peak_Reduction.png")


# ============================================================
# RUN COMPLETE PROJECT
# ============================================================

if __name__ == "__main__":
    main()
    write_explanation_file()
    add_season_table_and_savings_plot()