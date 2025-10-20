#!/usr/bin/env python3
# sampling_algorithms.py
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import entropy

# ----------------- helpers -----------------
def make_group(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in features:
        df[col] = df[col].astype(str).str.strip()
    df["group"] = df[features].agg("_".join, axis=1)
    return df

def kl_dist(p: pd.Series, q: pd.Series) -> float:
    idx = p.index.union(q.index)
    p = p.reindex(idx, fill_value=1e-9)
    q = q.reindex(idx, fill_value=1e-9)
    return float(entropy(p, q))

def _rint(rng: np.random.Generator) -> int:
    """Draw a fresh 32-bit random integer for pandas' random_state."""
    # pandas accepts an int seed; use full 32-bit range for variety
    return int(rng.integers(0, 2**32 - 1, dtype=np.uint32))

# ----------------- samplers -----------------
def reward_function(group, all_groups, target_dist, eps=1e-9):
    current = pd.Series(all_groups).value_counts(normalize=True)
    curr = max(current.get(group, eps), eps)
    targ = max(target_dist.get(group, eps), eps)
    return -np.log(curr / targ)

def gittins_ucb_sampler(
    line_df: pd.DataFrame,
    target_dist: pd.Series,
    batch_size: int,
    min_per_group: int,             # kept for compatibility, ignored
    pulls: dict[str, int],
    rewards: dict[str, float],
    prior_groups: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    UCB1-style bandit:
        G_g = mu_g + sqrt( 2 * log(sum_h pulls[h] + 1) / (pulls[g] + 1) )
    where mu_g = rewards[g] / max(1, pulls[g]).
    - No min-coverage bootstrap.
    - Samples without replacement.
    - Respects `batch_size`.
    """
    if batch_size <= 0 or line_df.empty:
        return line_df.iloc[0:0].copy()

    df = line_df.copy()
    group_counts = df["group"].value_counts().to_dict()
    sampled_groups: list[str] = []
    used_idx = pd.Index([])

    for _ in range(batch_size):
        # Compute UCB index per available group
        total_pulls = int(sum(pulls.values()))
        best_score = -np.inf
        best_groups: list[str] = []

        for g, c in group_counts.items():
            if c <= 0:
                continue
            p_g = int(pulls.get(g, 0))
            r_g = float(rewards.get(g, 0.0))

            mu = r_g / (p_g if p_g > 0 else 1.0)  # mean reward
            explore = np.sqrt(2.0 * np.log(total_pulls + 1.0) / (p_g + 1.0))  # your formula
            score = mu + explore

            if score > best_score + 1e-9:
                best_score = score
                best_groups = [g]
            elif abs(score - best_score) <= 1e-9:
                best_groups.append(g)

        if not best_groups:
            break

        # Tie-break uniformly at random among best groups
        best_g = rng.choice(np.array(best_groups, dtype=object))

        # Draw ONE row from the chosen group without replacement
        pool_g = df[df["group"] == best_g].drop(index=used_idx, errors="ignore")
        if pool_g.empty:
            group_counts[best_g] = 0
            continue

        take = pool_g.sample(1, replace=False, random_state=_rint(rng))
        used_idx = used_idx.union(take.index)

        # Update bandit stats with shaped reward
        sampled_groups.append(best_g)
        pulls[best_g]   = pulls.get(best_g, 0) + 1
        rewards[best_g] = rewards.get(best_g, 0.0) + reward_function(
            best_g, prior_groups + sampled_groups, target_dist
        )

        group_counts[best_g] -= 1
        if group_counts[best_g] < 0:
            group_counts[best_g] = 0

    return df.loc[used_idx].copy()


def greedy_kl_sampler(
    line_df: pd.DataFrame,
    target_dist: pd.Series,
    batch_size: int,
    min_per_group: int,      # kept for compatibility, ignored
    prior_groups: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Greedy KL-divergence sampler.

    Iteratively chooses the group that minimizes the KL divergence
    between the empirical group distribution (prior + current batch)
    and the target distribution.
    """
    df = line_df.copy()
    group_counts = df["group"].value_counts().to_dict()
    picked = []

    for _ in range(batch_size):
        best_g, best_kl = None, float("inf")

        for g, c in group_counts.items():
            if c <= 0:
                continue

            trial = prior_groups + picked + [g]
            p = pd.Series(trial).value_counts(normalize=True).reindex(target_dist.index, fill_value=0.0)
            q = target_dist.reindex(p.index)
            kl = kl_dist(p + 1e-9, q + 1e-9)

            if kl < best_kl:
                best_kl, best_g = kl, g

        if best_g is None:
            break  # no more available groups
        picked.append(best_g)
        group_counts[best_g] -= 1

    # Materialize actual rows for each chosen group
    counts = pd.Series(picked).value_counts()
    parts = []
    for g, cnt in counts.items():
        pool = df[df["group"] == g]
        parts.append(pool.sample(min(cnt, len(pool)), random_state=_rint(rng)))

    return pd.concat(parts) if parts else pd.DataFrame()


DEFAULT_MIN_COVERAGE_FRAC = 0.05  # 5% default

def uniform_sampler_with_min_coverage(
    line_df: pd.DataFrame,
    target_dist: pd.Series,
    batch_size: int,
    min_per_group,                 # kept for backward compat but ignored unless used as a fraction (0<val<=1)
    prior_groups: list[str],       # unused
    state: dict,                   # read min_coverage_frac here if provided by driver
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Uniform sampling with *fractional* minimum coverage per group.

    Behavior:
      - Determine min_coverage as a FRACTION:
          1) If state["min_coverage_frac"] is set (0<frac<=1), use that.
          2) Else if min_per_group is a float in (0,1], treat as fraction (forward compat).
          3) Else fall back to DEFAULT_MIN_COVERAGE_FRAC (0.05).
      - For each target group g, sample ceil(frac * count(g)) (capped by available).
      - Fill any remaining budget with uniform random from the remaining pool.

    Notes:
      - If the required per-group minimum exceeds the batch_size in aggregate, this
        function will return more than batch_size rows (same as previous behavior).
    """
    df = line_df.copy()
    if len(df) == 0 or batch_size <= 0:
        return df.iloc[0:0].copy()

    # Resolve fraction
    frac = None
    if isinstance(state, dict):
        frac = state.get("min_coverage_frac", None)
    if frac is None and isinstance(min_per_group, float) and (0.0 < min_per_group <= 1.0):
        frac = float(min_per_group)
    if frac is None:
        frac = DEFAULT_MIN_COVERAGE_FRAC

    # Work off target groups (as before)
    group_counts = df["group"].value_counts()
    targets = target_dist.index.tolist()

    parts = []
    for g in targets:
        if g in group_counts and group_counts[g] > 0:
            need = int(np.ceil(frac * group_counts[g]))
            if need > 0:
                take = min(need, group_counts[g])
                parts.append(df[df["group"] == g].sample(take, random_state=_rint(rng)))

    initial = pd.concat(parts) if parts else pd.DataFrame()

    # Fill remainder uniformly
    remaining = batch_size - len(initial)
    if remaining <= 0:
        return initial

    rest_pool = df.drop(index=initial.index, errors="ignore")
    take = min(remaining, len(rest_pool))
    rand = rest_pool.sample(take, replace=False, random_state=_rint(rng)) if take > 0 else pd.DataFrame()
    return pd.concat([initial, rand])


def pure_uniform_sampler(
    line_df: pd.DataFrame,
    target_dist: pd.Series,        # unused
    batch_size: int,
    min_per_group: int,            # unused
    prior_groups: list[str],       # unused
    state: dict,                   # unused
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Pure uniform random sampling over the provided pool_df/line_df.
    Ignores target_dist, min_per_group, prior_groups, and state.
    No internal replacement; caller controls cross-week no-replacement.
    """
    if len(line_df) == 0 or batch_size <= 0:
        return line_df.iloc[0:0].copy()
    take = min(batch_size, len(line_df))
    return line_df.sample(take, replace=False, random_state=_rint(rng)).copy()


# mapping used by the driver
ALGORITHMS = {
    "Uniform (pure)": pure_uniform_sampler,
    #"Uniform Random": uniform_sampler_with_min_coverage,
    "Greedy KL-Divergence": lambda df, td, bs, mpg, prior, state, rng: greedy_kl_sampler(df, td, bs, mpg, prior, rng),
    "RL (UCB)":   lambda df, td, bs, mpg, prior, state, rng: gittins_ucb_sampler(
    df, td, bs, mpg, state.setdefault("pulls", {}), state.setdefault("rewards", {}), prior, rng
    ),
}
