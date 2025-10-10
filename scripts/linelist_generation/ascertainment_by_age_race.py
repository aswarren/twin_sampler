# ascertainment_by_age_race.py
from pathlib import Path
import pandas as pd
import numpy as np

# ---- Config ----
ROOT = Path("/project/bii_nssac/biocomplexity/c4gc_asw3xp/LineList/asw_test/twin_sampler/scripts/linelist_generation/results")  # point this at the directory containing replicate_* folders
REPLICATE_GLOB = "replicate_*"
LINE_LIST_NAME = "linelist.csv.xz"
ALL_EVENTS_NAME = "linelist_allevents.csv.xz"
OUT_PER_REP = "ascertainment_per_replicate.csv"
OUT_SUMMARY = "ascertainment_summary_ci.csv"

AGE_GROUP_MAP = {
    "p": "Preschool (0-4)",
    "s": "Student (5-17)",
    "a": "Adult (18-49)",
    "o": "Older adult (50-64)",
    "g": "Senior (65+)",
}

# ---- Helpers ----
def read_xz(path: Path, usecols=None):
    return pd.read_csv(path, compression="xz", low_memory=False, usecols=usecols)

def find_replicates(root: Path) -> list[Path]:
    return sorted([p for p in root.glob(REPLICATE_GLOB) if p.is_dir()])

def compute_per_replicate(rep_dir: Path, fips_code: int | None = None) -> pd.DataFrame:
    """
    Returns a tidy dataframe with columns:
      replicate, age_group, smh_race, n_infected, n_ascertained, ascertainment
    """
    ll_path = rep_dir / LINE_LIST_NAME
    ae_path = rep_dir / ALL_EVENTS_NAME
    if not ll_path.exists() or not ae_path.exists():
        # Gracefully skip if a replicate is incomplete
        return pd.DataFrame(columns=[
            "replicate","age_group","smh_race","n_infected","n_ascertained","ascertainment"
        ])

    # Define the base columns needed for grouping
    g_vars = ["age_group", "smh_race"]
    
    # Determine the full set of columns to load based on whether we are filtering
    usecols_ll = g_vars + ["tested_positive"]
    usecols_inf = g_vars
    if fips_code is not None:
        usecols_ll.append("county_fips")
        usecols_inf.append("county_fips")

    # Load data with the necessary columns
    try:
        ll = read_xz(ll_path, usecols=usecols_ll)
    except ValueError:
        usecols_ll_fallback = g_vars + (["county_fips"] if fips_code else [])
        ll = read_xz(ll_path, usecols=usecols_ll_fallback)
        ll["tested_positive"] = 1
        
    infections = read_xz(ae_path, usecols=usecols_inf)

    # --- APPLY THE FILTER (if provided) ---
    if fips_code is not None:
        ll = ll[ll["county_fips"] == fips_code]
        infections = infections[infections["county_fips"] == fips_code]
        
        # Check if any data remains after filtering
        if ll.empty and infections.empty:
            print(f"  - Warning: No data found for FIPS {fips_code} in {rep_dir.name}. Skipping.")
            return pd.DataFrame()

    # Treat ascertained cases as those who tested positive
    ascertained = ll[ll["tested_positive"] == 1]
    
    # Count by stratum (g_vars is already defined)
    inf_counts = infections.value_counts(g_vars).rename("n_infected").reset_index()
    asc_counts = ascertained.value_counts(g_vars).rename("n_ascertained").reset_index()

    # Full outer merge to keep strata present in either side
    df = pd.merge(inf_counts, asc_counts, on=g_vars, how="outer").fillna(0)
    df["n_infected"] = df["n_infected"].astype(int)
    df["n_ascertained"] = df["n_ascertained"].astype(int)
    # Avoid div-by-zero
    df["ascertainment"] = np.where(
        df["n_infected"] > 0,
        df["n_ascertained"] / df["n_infected"],
        np.nan
    )
    df.insert(0, "replicate", rep_dir.name)
    return df

def summarize_across_replicates(per_rep: pd.DataFrame) -> pd.DataFrame:
    """
    Produces summary stats and 95% percentile CI across replicates for each stratum.
    """

    # --- pooled counts across replicates (for overall rate) ---
    agg_counts = (
        per_rep.groupby(["age_group", "smh_race"], as_index=False)[["n_infected", "n_ascertained"]]
        .sum()
    )
    agg_counts["ascertainment_pooled"] = np.where(
        agg_counts["n_infected"] > 0,
        agg_counts["n_ascertained"] / agg_counts["n_infected"],
        np.nan
    )

    # --- replicate-level statistics (for CI) ---
    # MODIFICATION 1: The function now accepts a DataFrame `group`
    def pct_ci(group: pd.DataFrame) -> pd.Series:
        # We select the column to operate on INSIDE the function
        s = group["ascertainment"].dropna()
        if len(s) == 0:
            return pd.Series(
                {"asc_mean": np.nan, "asc_median": np.nan,
                 "asc_p2p5": np.nan, "asc_p97p5": np.nan, "n_reps": 0}
            )
        return pd.Series(
            {
                "asc_mean": s.mean(),
                "asc_median": s.median(),
                "asc_p2p5": np.quantile(s, 0.025),
                "asc_p97p5": np.quantile(s, 0.975),
                "n_reps": len(s)
            }
        )

    # MODIFICATION 2: Apply `pct_ci` to the DataFrameGroupBy object
    # The `["ascertainment"]` part is removed from this chain.
    rep_stats = (
        per_rep
        .groupby(["age_group", "smh_race"])
        .apply(pct_ci)
        .reset_index()
    )

    # --- merge pooled counts and CI summaries ---
    summary = pd.merge(rep_stats, agg_counts, on=["age_group", "smh_race"], how="outer")

    # --- add readable labels and tidy columns ---
    summary["age_group_label"] = summary["age_group"].map(AGE_GROUP_MAP).fillna(summary["age_group"])

    cols = [
        "age_group", "age_group_label", "smh_race", "n_reps",
        "asc_mean", "asc_median", "asc_p2p5", "asc_p97p5",
        "n_infected", "n_ascertained", "ascertainment_pooled"
    ]
    summary = summary[[c for c in cols if c in summary.columns]].sort_values(
        ["age_group", "smh_race"]
    )
    return summary.reset_index(drop=True)


def main():

    parser = argparse.ArgumentParser(
        description="Calculate ascertainment rates by age and race, with an optional county filter."
    )
    parser.add_argument(
        "--county_fips",
        type=int,
        default=None,
        help="If provided, restricts the calculation to a specific county FIPS code."
    )
    args = parser.parse_args()
    
    # Announce if a filter is being used
    if args.county_fips is not None:
        print(f"--- Restricting analysis to County FIPS: {args.county_fips} ---")
    reps = find_replicates(ROOT)
    # === Skip broken replicates ===
    broken = {4, 11, 19}
    reps = [r for r in reps if not any(r.name.endswith(f"_{n}") for n in broken)]

    if not reps:
        raise SystemExit("No valid replicate_* folders found under ROOT.")

    per_rep_frames = []
    for rep in reps:
        df_rep = compute_per_replicate(rep, fips_code=args.county_fips)
        if not df_rep.empty:
            per_rep_frames.append(df_rep)

    if not per_rep_frames:
        raise SystemExit("No data loaded from any replicate folder.")

    per_rep_all = pd.concat(per_rep_frames, ignore_index=True)
    summary = summarize_across_replicates(per_rep_all)

    # --- Determine final output filenames ---
    if args.county_fips is not None:
        # If a FIPS filter was used, add it to the output filenames
        fips = args.county_fips
        out_per_rep_path = f"ascertainment_per_replicate_fips{fips}.csv"
        out_summary_path = f"ascertainment_summary_ci_fips{fips}.csv"
    else:
        # Otherwise, use the default global filenames
        out_per_rep_path = OUT_PER_REP
        out_summary_path = OUT_SUMMARY

    # --- Save final output files ---
    per_rep_all.to_csv(out_per_rep_path, index=False)
    summary.to_csv(out_summary_path, index=False)

    print(f"\nSkipped broken replicates: {sorted(broken)}")
    print(f"\nSaved per-replicate data to: {out_per_rep_path}")
    print(f"Saved summary data to: {out_summary_path}\n")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
