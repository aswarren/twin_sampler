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

def compute_per_replicate(rep_dir: Path) -> pd.DataFrame:
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

    # Load minimal columns we need (speed!)
    usecols = ["age_group", "smh_race", "tested_positive"]
    try:
        ll = read_xz(ll_path, usecols=usecols)
    except ValueError:
        # If tested_positive is missing in your line list, fall back to assuming
        # all rows in linelist are ascertained positives.
        usecols_fallback = ["age_group", "smh_race"]
        ll = read_xz(ll_path, usecols=usecols_fallback)
        ll["tested_positive"] = 1

    # Treat ascertained cases as those who tested positive
    ascertained = ll[ll["tested_positive"] == 1]

    # Infections: all rows in allevents represent infections (sim output)
    # (If your allevents include non-infection events, refine here.)
    inf_cols = ["age_group", "smh_race"]
    infections = read_xz(ae_path, usecols=inf_cols)

    # Count by stratum
    g_vars = ["age_group", "smh_race"]
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
    # Keep track of pooled counts too (can be useful)
    agg_counts = per_rep.groupby(["age_group", "smh_race"], as_index=False)[["n_infected","n_ascertained"]].sum()
    agg_counts["ascertainment_pooled"] = np.where(
        agg_counts["n_infected"] > 0,
        agg_counts["n_ascertained"] / agg_counts["n_infected"],
        np.nan
    )

    # Replicate-level CIs (percentile method)
    def pct_ci(s: pd.Series, alpha=0.05):
        s = s.dropna()
        if len(s) == 0:
            return pd.Series({"asc_mean": np.nan, "asc_median": np.nan, "asc_p2p5": np.nan, "asc_p97p5": np.nan, "n_reps": 0})
        return pd.Series({
            "asc_mean": s.mean(),
            "asc_median": s.median(),
            "asc_p2p5": np.quantile(s, 0.025),
            "asc_p97p5": np.quantile(s, 0.975),
            "n_reps": s.size
        })

    rep_stats = (
        per_rep
        .groupby(["age_group","smh_race"])["ascertainment"]
        .apply(pct_ci)
        .reset_index()
    )

    out = pd.merge(rep_stats, agg_counts, on=["age_group","smh_race"], how="outer")

    # Add readable labels
    out["age_group_label"] = out["age_group"].map(AGE_GROUP_MAP).fillna(out["age_group"])
    # Arrange columns nicely
    cols = [
        "age_group","age_group_label","smh_race",
        "n_reps",
        "asc_mean","asc_median","asc_p2p5","asc_p97p5",
        "n_infected","n_ascertained","ascertainment_pooled"
    ]
    # Keep any that exist (in case of empty data)
    cols = [c for c in cols if c in out.columns]
    return out[cols].sort_values(["age_group","smh_race"]).reset_index(drop=True)

def main():
    reps = find_replicates(ROOT)
    # === Skip broken replicates ===
    broken = {4, 11, 19}
    reps = [r for r in reps if not any(r.name.endswith(f"_{n}") for n in broken)]

    if not reps:
        raise SystemExit("No valid replicate_* folders found under ROOT.")

    per_rep_frames = []
    for rep in reps:
        df_rep = compute_per_replicate(rep)
        if not df_rep.empty:
            per_rep_frames.append(df_rep)

    if not per_rep_frames:
        raise SystemExit("No data loaded from any replicate folder.")

    per_rep_all = pd.concat(per_rep_frames, ignore_index=True)
    summary = summarize_across_replicates(per_rep_all)

    per_rep_all.to_csv(OUT_PER_REP, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"Skipped broken replicates: {sorted(broken)}")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
