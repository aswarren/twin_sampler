#!/usr/bin/env python3
"""
plot_age_race.py
Plots age_group x smh_race distributions as grouped bars with log-scale y-axis.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt

AGE_ORDER = [
    "Preschool (0-4)",
    "Student (5-17)",
    "Adult (18-49)",
    "Older adult (50-64)",
    "Senior (65+)",
]

def norm_age(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(r"[–—−]", "-", regex=True)
         .str.strip()
    )

def main():
    ap = argparse.ArgumentParser(description="Plot grouped bar chart of smh_race × age_group.")
    ap.add_argument("--csv", required=True, help="Path to simulated_test_positive_linelist.csv")
    ap.add_argument("--title", default="SMH Race Distribution by Age Group (Log Scale)", help="Plot title")
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    df["age_group"] = norm_age(df["age_group"])
    df["age_group"] = pd.Categorical(df["age_group"], categories=AGE_ORDER, ordered=True)

    # Cross-tabulate counts
    counts = pd.crosstab(df["smh_race"], df["age_group"]).reindex(columns=AGE_ORDER, fill_value=0)

    # Plot
    ax = counts.plot(
        kind="bar",
        logy=True,
        figsize=(12, 6),
    )
    ax.set_xlabel("SMH Race")
    ax.set_ylabel("Number of People (log scale)")
    ax.set_title(args.title)
    plt.legend(title="Age Group")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
