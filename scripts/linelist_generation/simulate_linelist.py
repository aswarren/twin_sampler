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


def simulate(df: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # If asymptomatic not present, simulate per original logic (35%)
    if "asymptomatic" not in df.columns:
        df = df.copy()
        df["asymptomatic"] = rng.random(len(df)) < 0.35

    # Compute testing probabilities row-wise
    df["test_prob"] = df.apply(compute_testing_probability, axis=1)

    # Simulate tested_positive ~ Bernoulli(test_prob)
    # Note: rng.binomial supports vectorized p
    df["tested_positive"] = rng.binomial(1, df["test_prob"].to_numpy())

    # Return only positives as linelist
    linelist_df = df.loc[df["tested_positive"] == 1].copy()
    return linelist_df

def simulate_ascertainment(df: pd.DataFrame, params: dict, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # The new ascertainment model derives asymptomatic status from 'exit_state',
    # so the original simulation of 'asymptomatic' is no longer needed.

    # 1. Prepare the DataFrame with necessary columns for the ascertainment model
    df = preprocess_for_ascertainment(df)

    # 2. Compute ascertainment probabilities row-wise using the new function
    #    We use a lambda to pass the extra `params` argument.
    df["ascertainment_prob"] = df.apply(
        lambda row: compute_ascertainment_probability(row, params), axis=1
    )

    # 3. Simulate ascertained_case ~ Bernoulli(ascertainment_prob)
    df["ascertained_case"] = rng.binomial(1, df["ascertainment_prob"].to_numpy())

    # 4. Return only ascertained cases as the linelist
    linelist_df = df.loc[df["ascertained_case"] == 1].copy()
    return linelist_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate test-positive linelist from inputs.")
    p.add_argument("--people", required=True, help="Path to va_persontrait_epihiper.txt (CSV with header on row 2).")
    p.add_argument("--infection", required=True, help="Path to run_03_vadelta_rate_limited_ticks.metadata.fixed_dates.tsv.")
    p.add_argument("--households", required=True, help="Path to va_household.csv.")
    p.add_argument("--rucc", required=True, help="Path to Ruralurbancontinuumcodes2023.csv (long format).")
    p.add_argument("--out", default="simulated_test_positive_linelist.csv", help="Output CSV path.")
    p.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility.")
    p.add_argument("--n_seeds", type=int, default=1, help="Number of different seeds to generate linelists with.")
    p.add_argument("--ascertain", required=True, help="Path to ascertainment_parameters.yaml file.")

    return p.parse_args()


def main():
    args = parse_args()

    people_df, infection_df, household_df = load_inputs(
        people_path=args.people,
        infection_path=args.infection,
        households_path=args.households,
    )

    rucc_wide = load_and_pivot_rucc(args.rucc)
    df = merge_all(infection_df, household_df, rucc_wide)
    if args.acertain:
        params = load_ascertainment_parameters(args.params)

        base_seed = args.seed if args.seed is not None else 0
        seeds = [base_seed + i for i in range(args.n_seeds)]
    
        for s in seeds:
            # Pass the merged df AND the params to the simulate function
            linelist_df = simulate_ascertainment(df, params=params, seed=s)

            if args.n_seeds > 1:
                out_path = args.out.replace(".csv", f"_seed{s}.csv")
            else:
                out_path = args.out

            linelist_df.to_csv(out_path, index=False)
            print(f"Wrote {len(linelist_df):,} rows to {out_path}")
    else:
        # If user provides --seed, use it as a starting seed
        # Otherwise, default to 0
        base_seed = args.seed if args.seed is not None else 0
        seeds = [base_seed + i for i in range(args.n_seeds)]

        for s in seeds:
            linelist_df = simulate(df, seed=s)

            # Add suffix if generating multiple outputs
            if args.n_seeds > 1:
                out_path = args.out.replace(".csv", f"_seed{s}.csv")
            else:
                out_path = args.out

            linelist_df.to_csv(out_path, index=False)
            print(f"Wrote {len(linelist_df):,} rows to {out_path}")


if __name__ == "__main__":
    main()
