#!/usr/bin/env python3
# run_all_scenarios.py (history-pool + no-replacement + budgets + AUC + 1x3 figs)
from __future__ import annotations
import argparse
import time
from collections import deque
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scenarios_config import (
    SCENARIOS,
    GROUP_FEATURES,
    DATE_FIELD_DEFAULT,
    START_DATE_DEFAULT,
    MINIMUM_POOL_SIZE_DEFAULT,
)
from sampling_algorithms import make_group, kl_dist, ALGORITHMS


# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Run scenarios 1-8; save 3 images (each 1x3). Also outputs AUC rankings. Infections required."
    )
    ap.add_argument("--linelist", required=True, help="Path to simulated_test_positive_linelist.csv")
    ap.add_argument("--population", required=True, help="Path to va_persontrait_epihiper.txt (skiprows=1)")
    ap.add_argument("--infections", required=True, help="Path to infections TSV")
    ap.add_argument("--date-field", default=DATE_FIELD_DEFAULT, help=f"Linelist date column (default: {DATE_FIELD_DEFAULT})")
    ap.add_argument("--start-date", default=str(START_DATE_DEFAULT.date()), help=f"Week slicing anchor date (default: {START_DATE_DEFAULT.date()})")
    ap.add_argument("--min-pool", type=int, default=MINIMUM_POOL_SIZE_DEFAULT, help=f"Minimum weekly pool size (default: {MINIMUM_POOL_SIZE_DEFAULT})")
    ap.add_argument("--outdir", default="result", help="Output directory for CSVs/plots (default: result)")
    ap.add_argument("--seed", type=int, default=42, help="Global random seed (default: 42)")
    ap.add_argument("--roll-win-inf", type=int, default=4, help="Rolling window (weeks) for infections Plot 3 (default: 4)")

    # ---- Sampling budget overrides ----
    ap.add_argument("--batch-size", type=int,
                    help="Fixed weekly sampling budget N. If set, overrides fraction/cap for all scenarios.")
    ap.add_argument("--batch-frac", type=float,
                    help="Override scenario batch_frac (0.0–1.0) for all scenarios.")
    ap.add_argument("--batch-cap", type=int,
                    help="Override scenario batch_cap for all scenarios.")
    ap.add_argument("--min-per-group", type=int,
                    help="Override scenario min_per_group for all scenarios.")
    ap.add_argument("--min-coverage-frac", type=float,
                    help="Fractional min coverage per group for the 'Uniform Random' sampler (0<frac<=1). Default: 0.05")
    # Add a flag to disable plots
    ap.add_argument("--no-plots", action="store_true",
                    help="If set, disables the generation of all PNG plot files.")

    # Add a flag to enable saving the selected samples
    ap.add_argument("--save-samples", action="store_true",
                    help="If set, saves the full metadata for selected samples for each scenario and algorithm.")

    # ---- No-replacement across weeks (per algorithm) ----
    ap.add_argument("--no-replacement", action="store_true",
                    help="If set, do not re-sample the same row across weeks (per algorithm).")

    return ap.parse_args()

# --------- shared helpers ---------
# Map short codes to long labels; keep existing long labels untouched.
AGE_GROUP_MAP = {
    "p": "Preschool (0-4)",
    "s": "Student (5-17)",
    "a": "Adult (18-49)",
    "o": "Older adult (50-64)",
    "g": "Senior (65+)",
}

def normalize_age_group_col(df, col="age_group"):
    """Map age_group codes (p/s/a/o/g) to long labels; leave long labels as-is."""
    if col in df.columns:
        raw = df[col]
        # case-insensitive match on single-letter codes
        mapped = (
            raw.astype(str).str.strip().str.lower()
            .map(AGE_GROUP_MAP)
        )
        # keep original values where no mapping applies (already long labels or NaN)
        df[col] = mapped.where(mapped.notna(), raw)
    return df

# ----------------- load & preprocess -----------------
def load_linelist_and_population(linelist_path, population_path, date_field, start_date, min_pool):
    line_df = pd.read_csv(linelist_path, parse_dates=[date_field])
    pop_df  = pd.read_csv(population_path, skiprows=1)

    # Normalize age_group in both population and linelist (handles codes or long labels)
    pop_df  = normalize_age_group_col(pop_df,  "age_group")
    line_df = normalize_age_group_col(line_df, "age_group")

    pop_df = pop_df.rename(columns={"gender": "sex"})
    pop_df["sex"]      = pop_df["sex"].astype(str).map({"1": "male", "2": "female"})
    pop_df["smh_race"] = pop_df["smh_race"].astype(str).map({
        "W": "White", "B": "Black", "L": "Latino", "A": "Asian", "O": "Other"
    })

    line_df = make_group(line_df, GROUP_FEATURES)
    pop_df  = make_group(pop_df,  GROUP_FEATURES)
    pop_dist_static = pop_df["group"].value_counts(normalize=True).sort_index()

    # weekly linelist history
    weekly_ll_hist = []
    cur = start_date
    while True:
        prev_mon = cur - timedelta(days=7)
        prev_sun = cur - timedelta(days=1)
        wk = line_df[(line_df[date_field] >= prev_mon) & (line_df[date_field] <= prev_sun)]
        if len(wk) < min_pool:
            break
        weekly_ll_hist.append(wk["group"].value_counts())
        cur += timedelta(weeks=1)

    return line_df, pop_df, pop_dist_static, weekly_ll_hist


def build_weekly_infections(infections_path, pop_df, start_date, num_weeks_ref, date_col: str = "date"):
    """
    Build weekly infections history aligned to linelist slicing.
    Now requires a real date column in the infections file (default: 'date').
    """
    # Let pandas sniff the delimiter (comma, tab, etc.) and avoid skipping header rows.
    inf = pd.read_csv(infections_path, sep=None, engine="python")
    inf.columns = [c.strip() for c in inf.columns]

    inf = normalize_age_group_col(inf, "age_group")
    if date_col not in inf.columns:
        # try case-insensitive match (e.g., 'Date', 'DATE')
        ci_map = {c.lower(): c for c in inf.columns}
        if date_col.lower() in ci_map:
            date_col = ci_map[date_col.lower()]
        else:
            raise ValueError(
                f"Infections file must include a '{date_col}' column "
                f"(case-insensitive). Found columns: {list(inf.columns)}"
            )

    # Parse dates
    inf[date_col] = pd.to_datetime(inf[date_col], errors="coerce")
    if inf[date_col].isna().all():
        raise ValueError(f"Unable to parse any dates in infections column '{date_col}'.")

    # Map pid -> group using population file
    pid_col = "pid" if "pid" in inf.columns else ("sim_pid" if "sim_pid" in inf.columns else None)
    if pid_col is None:
        raise ValueError("Infections file must contain 'pid' or 'sim_pid' to map to demographic groups.")

    if "pid" not in pop_df.columns:
        raise ValueError("Population file must have a 'pid' column to map infections to 'group'.")

    pid_group_map = pop_df[["pid", "group"]].dropna()
    inf = inf.merge(pid_group_map, left_on=pid_col, right_on="pid", how="left")
    inf = inf.dropna(subset=["group"])

    # Weekly counts aligned to linelist weeks
    weekly_inf_hist = []
    cur = start_date
    for _ in range(num_weeks_ref):
        prev_mon = cur - timedelta(days=7)
        prev_sun = cur - timedelta(days=1)
        mask = (inf[date_col] >= prev_mon) & (inf[date_col] <= prev_sun)
        weekly_inf_hist.append(inf.loc[mask, "group"].value_counts())
        cur += timedelta(weeks=1)

    return weekly_inf_hist


# ----------------- evaluation helpers -----------------
def cum_kl_vs_linelist(weekly_sample_hist, weekly_ll_hist):
    cum_s, cum_l = pd.Series(dtype=float), pd.Series(dtype=float)
    out = []
    n = min(len(weekly_sample_hist), len(weekly_ll_hist))
    for i in range(n):
        cum_s = cum_s.add(weekly_sample_hist[i], fill_value=0)
        cum_l = cum_l.add(weekly_ll_hist[i],     fill_value=0)
        out.append(kl_dist(cum_s / cum_s.sum(), cum_l / cum_l.sum()))
    return out

def cum_kl_vs_population(weekly_sample_hist, pop_dist):
    cum_s = pd.Series(dtype=float); out = []
    for wk in weekly_sample_hist:
        cum_s = cum_s.add(wk, fill_value=0)
        out.append(kl_dist(cum_s / cum_s.sum(), pop_dist))
    return out

def roll_kl_vs_linelist(weekly_sample_hist, weekly_ll_hist, window_weeks=4):
    out = []
    n = min(len(weekly_sample_hist), len(weekly_ll_hist))
    for i in range(n):
        s = pd.Series(dtype=float); l = pd.Series(dtype=float)
        start = max(0, i - window_weeks + 1)
        for j in range(start, i + 1):
            s = s.add(weekly_sample_hist[j], fill_value=0)
            l = l.add(weekly_ll_hist[j],     fill_value=0)
        out.append(kl_dist(s / s.sum(), l / l.sum()))
    return out

def linelist_dist_at_week(weekly_ll_hist, week_idx, mode="cumulative", window_weeks=4):
    counts = pd.Series(dtype=float)
    if mode == "cumulative":
        rng = range(0, week_idx + 1)
    else:
        start = max(0, week_idx - window_weeks + 1)
        rng = range(start, week_idx + 1)
    for j in rng:
        if 0 <= j < len(weekly_ll_hist):
            counts = counts.add(weekly_ll_hist[j], fill_value=0)
    return counts / counts.sum() if counts.sum() > 0 else counts

def blended_target(linelist_dist, pop_dist, alpha=0.5):
    if linelist_dist is None or linelist_dist.empty:
        return pop_dist
    tgt = linelist_dist.mul(alpha).add(pop_dist.mul(1 - alpha), fill_value=0.0)
    s = tgt.sum()
    return tgt / s if s > 0 else tgt

def series_auc(ys):
    """Trapezoidal AUC over weeks for a KL series; ignores NaNs. Lower is better."""
    y = np.asarray(list(ys), dtype=float)
    x = np.arange(1, len(y) + 1, dtype=float)
    m = np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    return float(np.trapz(y[m], x[m]))

SCEN_LABELS = {
    1: "CS-C(LL)", 2: "RS-R(LL)", 3: "RS-C(LL)",
    4: "CS-C(LL,P)", 5: "RS-R(LL,P)", 6: "RS-C(LL,P)",
    7: "CS-P", 8: "RS-P",
}


# ----------------- scenario runner (seeded) -----------------
def run_one_scenario(line_df, date_field, pop_dist_static, weekly_ll_hist,
                     scfg, rng_master, start_date, min_pool, overrides=None):
    """
    overrides: dict with optional keys:
      - batch_size_fixed: int
      - batch_frac: float
      - batch_cap: int
      - min_per_group: int
      - no_replacement: bool
    """
    overrides = overrides or {}
    ov_fixed = overrides.get("batch_size_fixed", None)
    ov_frac  = overrides.get("batch_frac", None)
    ov_cap   = overrides.get("batch_cap", None)
    ov_mpg   = overrides.get("min_per_group", None)
    ov_norep = bool(overrides.get("no_replacement", False))

    weekly_hist = {algo: [] for algo in ALGORITHMS.keys()}
    weekly_samples = {algo: [] for algo in ALGORITHMS.keys()}

    per_algo_eval, per_algo_time = {}, {}

    # per-algorithm child RNG (stable split)
    algo_rngs = {name: np.random.default_rng(rng_master.integers(0, 2**63 - 1)) for name in ALGORITHMS.keys()}

    for algo_name, sampler in ALGORITHMS.items():
        t0 = time.perf_counter()
        state = {}

        # For no-replacement: track used base indices (from line_df) per algorithm
        used_idx: set[int] = set()

        dec_win = scfg.get("decision_window_weeks", None)
        recent = deque(maxlen=max(0, (dec_win or 1) - 1))
        current_week = start_date
        week_idx_for_target = 0
        rng = algo_rngs[algo_name]

        # starting bound for "history" pool (all past weeks up to current)
        first_window_start = start_date - pd.Timedelta(days=7)

        while True:
            prev_mon = current_week - timedelta(days=7)
            prev_sun = current_week - timedelta(days=1)
            week_df = line_df[(line_df[date_field] >= prev_mon) & (line_df[date_field] <= prev_sun)]

            # Progress the weekly clock only if the *weekly* pool is viable (unchanged behavior)
            if len(week_df) < min_pool:
                break

            # ----- choose the sampling pool -----
            if scfg.get("pool_mode") == "history":
                # all rows from the first window start through end of current week
                first_window_start = start_date - pd.Timedelta(days=7)
                pool_df = line_df[(line_df[date_field] >= first_window_start) & (line_df[date_field] <= prev_sun)]

            elif scfg.get("pool_mode") == "rolling":
                w = int(scfg.get("pool_window_weeks", 4))
                pool_start = current_week - pd.Timedelta(weeks=w)
                pool_df = line_df[(line_df[date_field] >= pool_start) & (line_df[date_field] <= prev_sun)]

            else:
                # default: current week's pool only
                pool_df = week_df

            # No-replacement: drop rows already used by this algorithm in previous weeks
            if ov_norep or scfg.get("no_replacement", False):
                if len(used_idx) > 0:
                    pool_df = pool_df.drop(index=list(used_idx), errors="ignore")

            # ----- Effective sampling knobs (apply overrides) -----
            eff_frac = ov_frac if ov_frac is not None else scfg["batch_frac"]
            eff_cap  = ov_cap  if ov_cap  is not None else scfg["batch_cap"]
            eff_mpg  = ov_mpg  if ov_mpg  is not None else scfg["min_per_group"]

            if ov_fixed is not None:
                batch_size = int(min(max(0, ov_fixed), len(pool_df)))
            else:
                batch_size = int(min(eff_frac * len(pool_df), eff_cap))

            min_per_group = int(max(0, eff_mpg))

            # If pool exhausted (e.g., due to no-replacement), stop this algorithm gracefully
            if batch_size <= 0 or len(pool_df) == 0:
                break

            # ----- target distribution for this week -----
            if scfg.get("target_type") == "blend":
                ll_mode   = scfg.get("target_linelist_mode", "cumulative")
                ll_window = scfg.get("target_linelist_window", 4)
                alpha     = scfg.get("blend_alpha", 0.5)
                ll_dist   = linelist_dist_at_week(weekly_ll_hist, week_idx_for_target, ll_mode, ll_window)
                target_dist = blended_target(ll_dist, pop_dist_static, alpha)
            elif scfg.get("target_mode") == "linelist_dynamic":
                ll_mode   = scfg.get("target_linelist_mode", "cumulative")
                ll_window = scfg.get("target_linelist_window", 4)
                target_dist = linelist_dist_at_week(weekly_ll_hist, week_idx_for_target, ll_mode, ll_window)
            else:
                target_dist = pop_dist_static

            # ----- prior groups -----
            if dec_win is None:
                prior_groups = []
                for s in weekly_hist[algo_name]:
                    for g, cnt in s.items():
                        prior_groups.extend([g] * int(cnt))
            else:
                prior_groups = [g for lst in list(recent) for g in lst]

            # ----- sample (seeded) FROM CHOSEN POOL -----
            sample_df = sampler(pool_df, target_dist, batch_size, min_per_group, prior_groups, state, rng)

            # Update used indices for no-replacement
            if ov_norep or scfg.get("no_replacement", False):
                used_idx.update(sample_df.index.tolist())

            weekly_hist[algo_name].append(sample_df["group"].value_counts())
            weekly_samples[algo_name].append(sample_df)

            if dec_win is not None:
                recent.append(sample_df["group"].tolist())

            current_week += timedelta(weeks=1)
            week_idx_for_target += 1

        per_algo_time[algo_name] = time.perf_counter() - t0

    # evaluation series (for final plots 1a–c)
    for algo, wh in weekly_hist.items():
        metric = scfg.get("eval_metric", "kl_vs_linelist_cum")
        if metric == "kl_vs_linelist_cum":
            ys = cum_kl_vs_linelist(wh, weekly_ll_hist)
        elif metric == "kl_vs_population_cum":
            ys = cum_kl_vs_population(wh, pop_dist_static)
        elif metric == "kl_vs_linelist_rolling":
            win = scfg.get("eval_window_weeks", 4)
            ys = roll_kl_vs_linelist(wh, weekly_ll_hist, window_weeks=win)
        elif metric == "mean_kl_cum":
            a = cum_kl_vs_linelist(wh, weekly_ll_hist)
            b = cum_kl_vs_population(wh, pop_dist_static)
            ys = [(ai + bi) / 2.0 for ai, bi in zip(a, b)]
        else:
            raise ValueError(f"Unknown eval_metric: {metric}")
        per_algo_eval[algo] = ys

    return weekly_hist, per_algo_eval, per_algo_time, weekly_samples



# ----------------- main -----------------
def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # identifiers so you can aggregate across many runs
    linelist_id = Path(args.linelist).stem
    run_id = f"{linelist_id}__seed{args.seed}"

    start_date = pd.to_datetime(args.start_date)
    rng_master = np.random.default_rng(args.seed)

    # Build overrides once
    overrides = {
        "batch_size_fixed": args.batch_size,
        "batch_frac": args.batch_frac,
        "batch_cap": args.batch_cap,
        "min_per_group": args.min_per_group,
        "no_replacement": args.no_replacement,
    }

    # Load core inputs
    line_df, pop_df, POP_DIST_STATIC, weekly_ll_hist = load_linelist_and_population(
        args.linelist, args.population, args.date_field, start_date, args.min_pool
    )

    # Run scenarios
    scenario_series = {algo: {} for algo in ALGORITHMS.keys()}
    total_algo_time = {algo: 0.0 for algo in ALGORITHMS.keys()}
    count_algo_runs = {algo: 0 for algo in ALGORITHMS.keys()}
    kl_rows = []  # accumulate per-week KL points across all panels (A/B/C)
    all_weekly_hist = {} # This will be populated to replace the replay loop



    for scfg in SCENARIOS:
        print(f"\n=== Running {scfg['name']} ===")
        weekly_hist, per_algo_eval, per_algo_time, weekly_samples = run_one_scenario(
            line_df, args.date_field, POP_DIST_STATIC, weekly_ll_hist,
            scfg, rng_master, start_date, args.min_pool, overrides
        )

        all_weekly_hist[scfg["id"]] = weekly_hist

        # save per-scenario CSV + collect for final plots
        rows = []
        for algo, ys in per_algo_eval.items():
            if scfg["id"] in (4, 5, 6):
                label = f"{algo} Mean KL"
            elif scfg["eval_metric"] == "kl_vs_linelist_rolling":
                label = f"{algo} (Rolling {scfg.get('eval_window_weeks',4)}-Week KL)"
            elif scfg["eval_metric"] == "kl_vs_population_cum":
                label = f"{algo} vs. Population"
            else:
                label = f"{algo} vs. Line List"
            for i, v in enumerate(ys, start=1):
                rows.append({"scenario": scfg["id"], "label": label, "week": i, "kl": float(v)})
                # Panel A ("targets"): save KL per week
                kl_rows.append({
                    "run_id": run_id,
                    "linelist_id": linelist_id,
                    "algorithm": algo,
                    "scenario_id": scfg["id"],
                    "scenario_label": label,
                    "eval_type": "A_targets",
                    "roll_window": None,
                    "week": i,
                    "kl": float(v),
                })
            scenario_series[algo][scfg["id"]] = (list(range(1, len(ys)+1)), ys)
            total_algo_time[algo] += per_algo_time.get(algo, 0.0)
            count_algo_runs[algo] += 1

        for algo, secs in per_algo_time.items():
            print(f"  {algo} time: {secs:.2f}s")

        if args.save_samples:
            print(f"  Saving selected samples for {scfg['name']}...")
            for algo_name, sample_weeks_list in weekly_samples.items():
                # Check if the list of weekly samples is not empty
                if not sample_weeks_list:
                    print(f"    - No samples generated for algorithm '{algo_name}', skipping.")
                    continue
                
                # Combine all weekly sample DataFrames into one large DataFrame
                full_sample_df = pd.concat(sample_weeks_list, ignore_index=True)
                
                # Construct a unique, descriptive filename using the run_id
                sample_out_path = outdir / f"{run_id}_scenario{scfg['id']}_{algo_name}_samples.csv.xz"
                
                # Save the combined DataFrame to a CSV with XZ compression
                full_sample_df.to_csv(sample_out_path, index=False, compression="xz")
                print(f"    - Saved {len(full_sample_df)} samples for '{algo_name}' to {sample_out_path}")

    # ---------- build infections weekly history ----------
    weekly_inf_hist = build_weekly_infections(
        args.infections, pop_df, start_date, num_weeks_ref=len(weekly_ll_hist), date_col="date"
    )
    marker_map = {1:"o", 2:"s", 3:"D", 4:"^", 5:"v", 6:">", 7:"P", 8:"X"}
    algo_list  = list(ALGORITHMS.keys())
    n_algo    = len(algo_list)
    if False: #this isn't needed now that all_weekly_hist is populated in the original run
        print("\nReplaying samples for plotting (seeded) …")
        marker_map = {1:"o", 2:"s", 3:"D", 4:"^", 5:"v", 6:">", 7:"P", 8:"X"}
        algo_list  = list(ALGORITHMS.keys())
        all_weekly_hist = {}
        rng_master2 = np.random.default_rng(args.seed)
        for scfg in SCENARIOS:
            wh, _, _ = run_one_scenario(
                line_df, args.date_field, POP_DIST_STATIC, weekly_ll_hist,
                scfg, rng_master2, start_date, args.min_pool, overrides
            )
            all_weekly_hist[scfg["id"]] = wh  # {algo -> [Series]}

    # Small helpers for infections-based y-series
    def _cum_kl_vs(hist_list, ref_hist_list):
        cum_s = pd.Series(dtype=float); cum_r = pd.Series(dtype=float); out = []
        n = min(len(hist_list), len(ref_hist_list))
        for i in range(n):
            cum_s = cum_s.add(hist_list[i], fill_value=0)
            cum_r = cum_r.add(ref_hist_list[i], fill_value=0)
            if cum_s.sum() == 0 or cum_r.sum() == 0: out.append(np.nan); continue
            out.append(kl_dist(cum_s / cum_s.sum(), cum_r / cum_r.sum()))
        return out

    def _roll_kl_vs(hist_list, ref_hist_list, win=4):
        out = []; n = min(len(hist_list), len(ref_hist_list))
        for i in range(n):
            s = pd.Series(dtype=float); r = pd.Series(dtype=float)
            start = max(0, i - win + 1)
            for j in range(start, i + 1):
                s = s.add(hist_list[j], fill_value=0)
                r = r.add(ref_hist_list[j], fill_value=0)
            if s.sum() == 0 or r.sum() == 0: out.append(np.nan); continue
            out.append(kl_dist(s / s.sum(), r / r.sum()))
        return out

    def _axes_for_algos(n_algo: int, figsize_per_col=(7, 6)):
        """
        Create a 1 x n_algo row of axes, sharing Y.
        Returns: fig, [axes...]
        """
        fig, axes = plt.subplots(1, n_algo, figsize=(figsize_per_col[0]*n_algo, figsize_per_col[1]), sharey=True)
        if n_algo == 1:
            axes = [axes]
        return fig, list(axes)


    auc_rows = []  # dicts: eval_type, algorithm, scenario_id, scenario_label, weeks, auc

    if not args.no_plots:
        print("\nGenerating plots...")
        marker_map = {1:"o", 2:"s", 3:"D", 4:"^", 5:"v", 6:">", 7:"P", 8:"X"}
        algo_list  = list(ALGORITHMS.keys())
        rng_master2 = np.random.default_rng(args.seed)
        # =================== FIGURE A: targets (1×3) ===================
        figA, axesA = _axes_for_algos(n_algo)
        for ax, algo in zip(axesA, algo_list):
            ax.set_title(f"{algo}: Table-3 Scenarios")
            ax.set_xlabel("Week"); ax.set_ylabel("KL" if ax is axesA[0] else "")
            for scn in range(1, 9):
                x, y = scenario_series[algo][scn]
                label = SCEN_LABELS[scn]
                ax.plot(x, y, marker=marker_map.get(scn, "o"), linestyle="-", label=label)
                auc_rows.append({
                    "eval_type": "A_targets",
                    "algorithm": algo,
                    "scenario_id": scn,
                    "scenario_label": label,
                    "weeks": len(y),
                    "auc": series_auc(y),
                    })
            ax.grid(True, linestyle="--", alpha=0.6); ax.legend(ncol=4, fontsize=8); ax.set_xlim(left=0.9)
        figA.tight_layout()
        outA = outdir / "A_table3_targets_1xN.png"
        plt.savefig(outA, dpi=150); print(f"Saved: {outA}")
        plt.close(figA)

        # =================== FIGURE B: cumulative infections (1×3) ===================
        figB, axesB = _axes_for_algos(n_algo)
        for ax, algo in zip(axesB, algo_list):
            ax.set_title(f"{algo}: KL vs Cumulative Infections")
            ax.set_xlabel("Week"); ax.set_ylabel("KL" if ax is axesB[0] else "")
            for scn in range(1, 9):
                ys = _cum_kl_vs(all_weekly_hist[scn][algo], weekly_inf_hist)
                if not ys:
                    continue
                x = list(range(1, len(ys) + 1)); label = SCEN_LABELS[scn]
                ax.plot(x, ys, marker=marker_map.get(scn, "o"), linestyle="-", label=label)
                # Panel B: save KL per week
                for i, v in enumerate(ys, start=1):
                    kl_rows.append({
                        "run_id": run_id,
                        "linelist_id": linelist_id,
                        "algorithm": algo,
                        "scenario_id": scn,
                        "scenario_label": label,
                        "eval_type": "B_cumulative_infections",
                        "roll_window": None,
                        "week": i,
                        "kl": float(v),
                        })
                auc_rows.append({
                    "eval_type": "B_cumulative_infections",
                    "algorithm": algo,
                    "scenario_id": scn,
                    "scenario_label": label,
                    "weeks": len(ys),
                    "auc": series_auc(ys),
                    })
            ax.grid(True, linestyle="--", alpha=0.6); ax.legend(ncol=4, fontsize=8); ax.set_xlim(left=0.9)
        figB.tight_layout()
        outB = outdir / "B_vs_cumulative_infections_1xN.png"
        plt.savefig(outB, dpi=150); print(f"Saved: {outB}")
        plt.close(figB)

        # =================== FIGURE C: rolling infections (1×N) ===================
        figC, axesC = _axes_for_algos(n_algo)
        for ax, algo in zip(axesC, algo_list):
            ax.set_title(f"{algo}: KL vs {args.roll_win_inf}-Week Rolling Infections")
            ax.set_xlabel("Week"); ax.set_ylabel("KL" if ax is axesC[0] else "")
            for scn in range(1, 9):
                ys = _roll_kl_vs(all_weekly_hist[scn][algo], weekly_inf_hist, win=args.roll_win_inf)
                if not ys:
                    continue
                x = list(range(1, len(ys) + 1)); label = SCEN_LABELS[scn]
                ax.plot(x, ys, marker=marker_map.get(scn, "o"), linestyle="-", label=label)
                # Panel C: save KL per week (store rolling window separately)
                for i, v in enumerate(ys, start=1):
                    kl_rows.append({
                        "run_id": run_id,
                        "linelist_id": linelist_id,
                        "algorithm": algo,
                        "scenario_id": scn,
                        "scenario_label": label,
                        "eval_type": "C_rolling_infections",
                        "roll_window": int(args.roll_win_inf),
                        "week": i,
                        "kl": float(v),
                        })
                auc_rows.append({
                    "eval_type": f"C_rolling_infections_w{args.roll_win_inf}",
                    "algorithm": algo,
                    "scenario_id": scn,
                    "scenario_label": label,
                    "weeks": len(ys),
                    "auc": series_auc(ys),
                    })
            ax.grid(True, linestyle="--", alpha=0.6); ax.legend(ncol=4, fontsize=8); ax.set_xlim(left=0.9)
        figC.tight_layout()
        outC = outdir / f"C_vs_rolling{args.roll_win_inf}_infections_1xN.png"
        plt.savefig(outC, dpi=150); print(f"Saved: {outC}")
        plt.close(figC)
    
    
        # =================== FIGURE D: Weekly ratios (3 bars per week) ===================
        # Definitions:
        # - pool_size = LineList size in the current week
        # - infections_size = infections size in the current week
        # - sampled_per_week = samples actually drawn in the replay (pick one scenario+algorithm)
        #
        scenario_for_sampled = 1
        algo_for_sampled = list(ALGORITHMS.keys())[0]
        
        # Build week-wise counts
        weeks_n = len(weekly_ll_hist)
        pool_weekly = [int(weekly_ll_hist[i].sum()) for i in range(weeks_n)]
        inf_weekly  = [int(weekly_inf_hist[i].sum()) for i in range(weeks_n)]
        
        # Use the replayed samples we already computed: all_weekly_hist[scenario_id][algo] -> list[Series]
        sampled_hist_list = all_weekly_hist.get(scenario_for_sampled, {}).get(algo_for_sampled, [])
        sampled_weekly = [int(s.sum()) if i < len(sampled_hist_list) else 0 for i, s in enumerate(sampled_hist_list + [pd.Series(dtype=float)]*max(0, weeks_n - len(sampled_hist_list)))]
        
        # Safe division helpers
        def _safe_div(num, den):
            return (num / den) if (den is not None and den != 0) else float("nan")
        
        ratio_pool_over_inf     = [_safe_div(pool_weekly[i], inf_weekly[i]) for i in range(weeks_n)]
        ratio_sampled_over_inf  = [_safe_div(sampled_weekly[i], inf_weekly[i]) for i in range(weeks_n)]
        ratio_sampled_over_pool = [_safe_div(sampled_weekly[i], pool_weekly[i]) for i in range(weeks_n)]
        
        # Plot grouped bars
        figD, axD = plt.subplots(figsize=(14, 6))
        x = np.arange(weeks_n) + 1  # week numbers starting at 1
        bar_w = 0.25
        axD.bar(x - bar_w, ratio_pool_over_inf,     width=bar_w, label="pool_size / infections_size")
        axD.bar(x,           ratio_sampled_over_inf, width=bar_w, label="sampled / infections_size")
        axD.bar(x + bar_w,   ratio_sampled_over_pool,width=bar_w, label="sampled / pool_size")
        
        axD.set_title(f"Weekly Ratios (Scenario {scenario_for_sampled}, Algo: {algo_for_sampled})")
        axD.set_xlabel("Week")
        axD.set_ylabel("Ratio")
        axD.set_xlim(0.5, weeks_n + 0.5)
        axD.grid(True, linestyle="--", alpha=0.6)
        axD.legend()
        
        figD.tight_layout()
        outD = outdir / "D_weekly_sampling_ratios.png"
        plt.savefig(outD, dpi=150)
        print(f"Saved: {outD}")
        plt.close(figD)
    else:
        print("\n--no-plots flag detected. Skipping plot generation.")


    # ------------------- Save per-week KL series for uncertainty bands -------------------
    kl_df = pd.DataFrame(kl_rows)
    kl_out = outdir / "KL_series.csv"
    kl_df.to_csv(kl_out, index=False)
    print(f"Saved KL series: {kl_out}")

    # =================== AUC summary CSV (ranked) ===================
    auc_df = pd.DataFrame(auc_rows)
    # lower AUC is better
    auc_df["rank_overall"] = auc_df.groupby("eval_type")["auc"].rank(method="dense", ascending=True)
    auc_df["rank_within_algo"] = auc_df.groupby(["eval_type", "algorithm"])["auc"].rank(method="dense", ascending=True)

    auc_out = outdir / "AUC_rankings.csv"
    auc_df.sort_values(["eval_type", "rank_overall", "algorithm", "scenario_id"]).to_csv(auc_out, index=False)
    print(f"\nSaved AUC rankings: {auc_out}")

    # print top results per evaluation to console
    for et in auc_df["eval_type"].unique():
        top = (auc_df[auc_df["eval_type"] == et]
               .sort_values(["rank_overall", "algorithm", "scenario_id"])
               .head(8))
        print(f"\nTop AUCs for {et} (lower is better):")
        for _, r in top.iterrows():
            print(f"  #{int(r['rank_overall'])}: {r['algorithm']} - {r['scenario_label']} "
                  f"(AUC={r['auc']:.4f}, weeks={int(r['weeks'])})")

    print("\n=== Average running time across 8 scenarios (per algorithm) ===")
    for algo in ALGORITHMS.keys():
        n = max(1, count_algo_runs[algo])
        avg_secs = total_algo_time[algo] / n
        print(f"{algo}: {avg_secs:.2f}s on average over {n} scenarios")


if __name__ == "__main__":
    main()
