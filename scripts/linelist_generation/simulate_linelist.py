#!/usr/bin/env python3
"""
simulate_linelist.py
End-to-end script:
- loads people, infection, household, and RUCC files
- merges them into a single frame
- computes test probabilities
- simulates who tests positive
- writes the positive-testing linelist CSV

Usage:
    python simulate_linelist.py \
      --people va_persontrait_epihiper.txt \
      --infection run_03_vadelta_rate_limited_ticks.metadata.fixed_dates.tsv \
      --households va_household.csv \
      --rucc Ruralurbancontinuumcodes2023.csv \
      --out simulated_test_positive_linelist.csv \
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from rucc_utils import load_and_pivot_rucc
from testing_prob import compute_testing_probability
from ascertainment_module import (
load_ascertainment_parameters,
preprocess_for_ascertainment,
compute_ascertainment_probability
)

def process_epihiper(
    epihiper_path: str,
    persontrait_path: str,
    household_path: str,
    rucc_path: str,
    start_date: str,
    start_tick: int,
    prefix_filter: tuple
) -> pd.DataFrame:
    """
    Loads raw EpiHiper output, filters for relevant infectious states (A, P, I, dm, hM),
    sorts chronologically, and decorates with person/household/location data.

    Returns:
        A single, sorted DataFrame of all potential ascertainment events.
    """
    print("--- Starting EpiHiper Processing (Model B: First Ascertained Event) ---")

    # 1. Load all required data files
    print(f"Loading EpiHiper data from {epihiper_path}...")
    epi_df = pd.read_csv(epihiper_path)
    print(f"Loading persontrait data from {persontrait_path}...")
    person_df = pd.read_csv(persontrait_path) # No index needed for a standard merge
    print(f"Loading household data from {household_path}...")
    household_df = pd.read_csv(household_path)
    print(f"Loading and pivoting RUCC data from {rucc_path}...")
    rucc_df = load_and_pivot_rucc(rucc_path) # Assumes load_and_pivot_rucc is available

    # 2. Filter for relevant infectious states (EXCLUDING 'E' states)
    relevant_prefixes = prefix_filter
    initial_count = len(epi_df)
    epi_df = epi_df[epi_df['exit_state'].str.startswith(relevant_prefixes)].copy()
    print(f"Filtered to {len(epi_df)} relevant infectious events (from {initial_count} total).")
    
    if epi_df.empty:
        print("Warning: No relevant infectious events found after filtering. Returning empty DataFrame.")
        return pd.DataFrame()

    # 3. Sort events chronologically by tick
    epi_df.sort_values(by='tick', inplace=True)
    print("Sorted all events chronologically by tick.")

    # 4. Decorate the event data by merging with other files
    # Merge persontrait first to get 'hid' and 'county_fips'
    decorated_df = epi_df.merge(person_df, on='pid', how='left')
    
    # Merge household data (requires 'hid' from the persontrait file)
    decorated_df = decorated_df.merge(household_df, on='hid', how='left')

    # Merge RUCC data (requires 'county_fips' from persontrait file)
    # Ensure FIPS codes are correctly formatted 5-digit strings
    decorated_df["county_fips"] = decorated_df["county_fips"].astype(str).str.zfill(5)
    rucc_df["FIPS"] = rucc_df["FIPS"].astype(str).str.zfill(5)
    decorated_df = decorated_df.merge(
        rucc_df[["FIPS", "rucc_code"]],
        left_on="county_fips",
        right_on="FIPS",
        how="left",
    )
    print("Successfully decorated data with person, household, and RUCC info.")
    
    # 5. Calculate the 'date' column from the tick
    base_date = pd.to_datetime(start_date)
    decorated_df['date'] = decorated_df['tick'].apply(
        lambda x: base_date + pd.Timedelta(days=(x - start_tick))
    )

    # 6. Final cleanup of columns before returning
    final_df = decorated_df.drop(columns=['FIPS'], errors='ignore')
    
    print("--- EpiHiper Processing Complete ---")
    return final_df

def _normalize_age_group_column(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Unicode dashes with ASCII hyphens and trim whitespace."""
    if "age_group" in df.columns:
        df["age_group"] = (
            df["age_group"]
            .astype(str)
            .str.replace(r"[–—−]", "-", regex=True)
            .str.strip()
        )
    return df


def load_inputs(people_path: str, infection_path: str, households_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Load people
    people_df = pd.read_csv(people_path, sep=",", skiprows=1)
    # standardize id column name used downstream
    people_df = people_df.rename(columns={"pid": "sim_pid"})

    # Load infection; the original code skipped only row 1
    infection_df = pd.read_csv(infection_path, sep="\t", skiprows=[1])

    # Bring in county_fips, hid, occupation_socp from people
    cols_to_merge = []
    for c in ("sim_pid", "county_fips", "hid", "occupation_socp"):
        if c in people_df.columns:
            cols_to_merge.append(c)
    if "sim_pid" not in cols_to_merge:
        raise ValueError("people_df must include 'pid' or 'sim_pid' for joining.")

    infection_df = infection_df.merge(people_df[cols_to_merge], on="sim_pid", how="left")

    # Load households
    household_df = pd.read_csv(households_path)

    return people_df, infection_df, household_df


def merge_all(infection_df: pd.DataFrame, household_df: pd.DataFrame, rucc_wide: pd.DataFrame) -> pd.DataFrame:
    # Ensure 5-char FIPS strings
    if "county_fips" in infection_df.columns:
        infection_df["county_fips"] = infection_df["county_fips"].astype(str).str.zfill(5)
    rucc_wide["FIPS"] = rucc_wide["FIPS"].astype(str).str.zfill(5)

    # Merge RUCC code
    merged = infection_df.merge(
        rucc_wide[["FIPS", "rucc_code"]],
        left_on="county_fips",
        right_on="FIPS",
        how="left",
    )

    # Merge households on hid
    if "hid" in merged.columns and "hid" in household_df.columns:
        merged = merged.merge(household_df, on="hid", how="left")

    # Normalize age_group column if present
    merged = _normalize_age_group_column(merged)

    return merged



def simulate(events_df: pd.DataFrame, params: dict, seed: int | None = None) -> pd.DataFrame:
    """
    NEW: Iterates through chronologically sorted events, performing a Bernoulli trial
    for each until an individual is ascertained for the first time.
    """
    if events_df.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    
    # Prepare the DataFrame with necessary columns for the ascertainment model
    # This adds 'symptom_severity', etc., based on 'exit_state'
    events_df = preprocess_for_ascertainment(events_df)

    # Calculate probabilities for ALL potential events at once (vectorized and fast)
    events_df["ascertainment_prob"] = events_df.apply(
        lambda row: compute_ascertainment_probability(row, params), axis=1
    )

    ascertained_pids = set()
    line_list_rows = []

    print(f"Simulating ascertainment for {len(events_df)} potential events...")
    # Iterate through the events in chronological order
    for index, event in events_df.iterrows():
        pid = event['pid']
        
        # If this person has already been found, skip to the next event
        if pid in ascertained_pids:
            continue
        
        # Perform the Bernoulli trial for this event
        prob = event['ascertainment_prob']
        if rng.random() < prob:
            # Success! Add this event to our line list and track the PID
            line_list_rows.append(event)
            ascertained_pids.add(pid)
    
    if not line_list_rows:
        return pd.DataFrame()
        
    # Convert the list of successful rows back into a DataFrame
    return pd.DataFrame(line_list_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate test-positive linelist from inputs.")
    
    # REMOVE THIS LINE
    # p.add_argument("--infection", required=True, help="Path to run_03_vadelta_rate_limited_ticks.metadata.fixed_dates.tsv.")
    
    # ADD THESE LINES
    p.add_argument("--epihiper", required=True, help="Path to raw epihiper ABM output CSV.")
    p.add_argument("--start_date", default="2021-01-01", help="The calendar date corresponding to the start_tick.")
    p.add_argument("--start_tick", type=int, default=0, help="The simulation tick that corresponds to the start_date.")
    
    # Keep the other arguments
    p.add_argument("--people", required=True, dest="persontrait_file", help="Path to va_persontrait_epihiper.txt.")
    p.add_argument("--households", required=True, help="Path to va_household.csv.")
    p.add_argument("--rucc", required=True, help="Path to Ruralurbancontinuumcodes2023.csv.")
    p.add_argument("--ascertain", required=True, help="Path to ascertainment_parameters.yaml file.")
    p.add_argument("--out", default="simulated_test_positive_linelist.csv", help="Output CSV path.")
    p.add_argument("--seed", type=int, default=None, help="Base random seed for reproducibility.")
    p.add_argument("--n_seeds", type=int, default=1, help="Number of different seeds to generate linelists with.")
    p.add_argument("--prefix_override", nargs="+", default=['A', 'P', 'I', 'dm', 'hM'], help="List of state prefixes to include as infectious (e.g., A P I dm hM).")
    return p.parse_args()



def main():
    args = parse_args()
    
    # Single call to the new data processing function
    events_df = process_epihiper(
        epihiper_path=args.epihiper,
        persontrait_path=args.persontrait_file,
        household_path=args.households,
        rucc_path=args.rucc,
        start_date=args.start_date,
        start_tick=args.start_tick,
        prefix_filter=tuple(args.prefix_override)
    )

    # Load the ascertainment model parameters from the YAML file
    params = load_ascertainment_parameters(args.ascertain)

    base_seed = args.seed if args.seed is not None else 0
    seeds = [base_seed + i for i in range(args.n_seeds)]

    for s in seeds:
        print(f"\n--- Running simulation for seed {s} ---")
        # The 'simulate' function now contains the core Model B logic
        linelist_df = simulate(events_df, params=params, seed=s)

        if args.n_seeds > 1:
            out_path = args.out.replace(".csv", f"_seed{s}.csv")
        else:
            out_path = args.out

        linelist_df.to_csv(out_path, index=False)
        print(f"Wrote {len(linelist_df):,} rows to {out_path} for seed {s}")


if __name__ == "__main__":
    main()
