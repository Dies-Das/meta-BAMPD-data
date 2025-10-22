#!/usr/bin/env python3
"""
Builds per-trial features for the new bandit-style dataset.

Input CSV must contain these columns (exact names):
subj_id, stage, block_idx, reward_value, horizon, p0, p1,
trial_idx, arm, reward, rt, voc_bound, greedy_gain, optimal_gain,
computational, number_of_computations

Output CSV will include the same columns first (same order as above),
then append (in this exact order):
a0, b0, a1, b1, steps_left, opt_act, greedy_act, took_opt,
abs_dmu, var0, var1, sum_var, diff_var, switch,
prev_reward, prev_surprise, lag_log_rt
"""

import argparse
import itertools
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd


# ----------------------------
# Dynamic programming utilities (Bayes-optimal vs greedy)
# ----------------------------

def states_by_steps_left(T: int):
    """
    Enumerate all Beta-Bernoulli posterior states reachable with s_left steps remaining.
    State encoding uses +1 Beta prior: (a0, b0, a1, b1) where a=1+s, b=1+f.
    """
    states = [[] for _ in range(T + 1)]
    for n in range(T + 1):
        s_left = T - n
        for s0, f0, s1 in itertools.product(range(n + 1), repeat=3):
            f1 = n - (s0 + f0 + s1)
            if f1 < 0:
                continue
            states[s_left].append((1 + s0, 1 + f0, 1 + s1, 1 + f1))
    return states


@dataclass(frozen=True)
class PolicyPack:
    T: int
    opt_by_s: dict   # key: (steps_left, state) -> {0,1,-1}
    greedy: dict     # key: state -> {0,1,-1}


@lru_cache(maxsize=None)
def compute_policies_for_horizon(T: int) -> PolicyPack:
    """
    Backward induction for finite-horizon Bayes-optimal policy.
    Also computes the step-independent greedy policy (posterior mean).
    """
    S = states_by_steps_left(T)
    V = [defaultdict(float) for _ in range(T + 1)]
    opt_by_s = {}

    # base values
    for st in S[0]:
        V[0][st] = 0.0

    # backward induction
    for s in range(1, T + 1):
        for st in S[s]:
            a0, b0, a1, b1 = st
            p0 = a0 / (a0 + b0)
            p1 = a1 / (a1 + b1)

            succ0 = (a0 + 1, b0, a1, b1)
            fail0 = (a0, b0 + 1, a1, b1)
            succ1 = (a0, b0, a1 + 1, b1)
            fail1 = (a0, b0, a1, b1 + 1)

            Q0 = p0 * (1 + V[s - 1][succ0]) + (1 - p0) * V[s - 1][fail0]
            Q1 = p1 * (1 + V[s - 1][succ1]) + (1 - p1) * V[s - 1][fail1]

            V[s][st] = max(Q0, Q1)
            opt_by_s[(s, st)] = 0 if Q0 > Q1 else 1 if Q1 > Q0 else -1

    # greedy (posterior mean) is step-independent
    greedy = {}
    for lst in S:
        for st in lst:
            a0, b0, a1, b1 = st
            m0 = a0 / (a0 + b0)
            m1 = a1 / (a1 + b1)
            greedy[st] = 0 if m0 > m1 else 1 if m1 > m0 else -1

    return PolicyPack(T=T, opt_by_s=opt_by_s, greedy=greedy)


# ----------------------------
# Feature reconstruction
# ----------------------------

BASE_COLS = [
    "subj_id", "stage", "block_idx", "reward_value", "horizon", "p0", "p1",
    "trial_idx", "arm", "reward", "rt", "voc_bound", "greedy_gain",
    "optimal_gain", "computational", "number_of_computations"
]

NEW_COLS = [
    "a0", "b0", "a1", "b1", "steps_left", "opt_act", "greedy_act", "took_opt",
    "abs_dmu", "var0", "var1", "sum_var", "diff_var", "switch",
    "prev_reward", "prev_surprise", "lag_log_rt"
]


def reconstruct_features(df_main: pd.DataFrame) -> pd.DataFrame:
    df = df_main.copy()

    # Preserve original row order
    df["orig_row"] = np.arange(len(df), dtype=np.int64)

    # Ensure required base columns exist
    missing = [c for c in BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    # Coerce numeric where needed for sequencing/state updates
    for col in ["block_idx", "horizon", "trial_idx", "arm", "reward"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["p0", "p1", "rt"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows that lack essentials for state reconstruction
    needed = ["subj_id", "stage", "block_idx", "horizon", "trial_idx", "arm", "reward", "rt", "p0", "p1"]
    df = df.dropna(subset=needed).copy()

    # Cast to ints where appropriate
    df["block_idx"] = df["block_idx"].astype(int)
    df["horizon"] = df["horizon"].astype(int)
    df["trial_idx"] = df["trial_idx"].astype(int)
    df["arm"] = df["arm"].astype(int)
    df["reward"] = df["reward"].astype(int)

    # Guard against non-binary arms
    bad_arms = df.loc[~df["arm"].isin([0, 1])]
    if not bad_arms.empty:
        raise ValueError("Found arm values outside {0,1}. Ensure arms are encoded as 0/1.")

    # Precompute policy packs per horizon
    policies = {int(T): compute_policies_for_horizon(int(T)) for T in sorted(df["horizon"].unique())}

    rows = []

    # Group by (subj_id, stage, block_idx) to keep episodes distinct
    for (sid, stage, block), g in df.groupby(["subj_id", "stage", "block_idx"], sort=False):
        # Compute states chronologically within the episode
        g_sorted = g.sort_values("trial_idx")

        T = int(g_sorted["horizon"].iloc[0])
        ppack = policies[T]

        # Sufficient statistics for Beta posteriors
        s = {0: 0, 1: 0}
        f = {0: 0, 1: 0}

        prev_rew = 0
        prev_surprise = 0.0  # SIGNED surprise carried to next trial
        prev_arm = None
        prev_rt = None

        # Enumerate trials in chronological order for state updates
        for t_pos, r in enumerate(g_sorted.itertuples(index=False), start=0):
            # State BEFORE action
            a0, b0 = 1 + s[0], 1 + f[0]
            a1, b1 = 1 + s[1], 1 + f[1]
            st = (a0, b0, a1, b1)

            # steps_left cannot go below 0
            steps_left = max(T - t_pos, 0)

            # Posterior means and Beta variances
            m0 = a0 / (a0 + b0)
            m1 = a1 / (a1 + b1)
            abs_dmu = abs(m0 - m1)

            var0 = (a0 * b0) / (((a0 + b0) ** 2) * (a0 + b0 + 1))
            var1 = (a1 * b1) / (((a1 + b1) ** 2) * (a1 + b1 + 1))
            sum_var = var0 + var1
            diff_var = abs(var0 - var1)

            # Policies
            opt_act = ppack.opt_by_s.get((steps_left, st), -1)
            greedy_act = ppack.greedy.get(st, -1)

            # Whether the taken action equals optimal (NaN if tie)
            took_opt = np.nan if opt_act == -1 else int(r.arm == opt_act)

            # Switch indicator
            switch = 1 if (prev_arm is not None and int(r.arm) != prev_arm) else 0

            # lag log RT
            lag_log_rt = np.log(prev_rt) if (prev_rt is not None and prev_rt > 0) else 0.0

            # Record row with base columns first, then new features, plus orig_row to restore order
            base_vals = {c: getattr(r, c) for c in BASE_COLS}
            rows.append({
                **base_vals,
                "a0": a0, "b0": b0, "a1": a1, "b1": b1,
                "steps_left": steps_left,
                "opt_act": int(opt_act),
                "greedy_act": int(greedy_act),
                "took_opt": took_opt,
                "abs_dmu": abs_dmu,
                "var0": var0, "var1": var1,
                "sum_var": sum_var, "diff_var": diff_var,
                "switch": switch,
                "prev_reward": prev_rew,
                "prev_surprise": prev_surprise,  # SIGNED surprise from previous trial
                "lag_log_rt": lag_log_rt,
                "orig_row": getattr(r, "orig_row"),
            })

            # --- Update for next trial (apply outcome) ---
            mean_chosen_pre = m0 if int(r.arm) == 0 else m1
            if int(r.arm) == 0:
                if int(r.reward) == 1:
                    s[0] += 1
                else:
                    f[0] += 1
            else:
                if int(r.reward) == 1:
                    s[1] += 1
                else:
                    f[1] += 1

            prev_rew = int(r.reward)
            prev_surprise = (prev_rew - mean_chosen_pre)  # SIGNED
            prev_arm = int(r.arm)
            prev_rt = float(r.rt)

    feat = pd.DataFrame(rows)

    # Restore the original input row order to keep episodes exactly as in the input
    feat = feat.sort_values("orig_row", kind="stable").drop(columns=["orig_row"]).reset_index(drop=True)

    # Reorder columns: base first, then new (exact order)
    feat = feat[BASE_COLS + NEW_COLS]
    return feat


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute trial-level features for the new bandit dataset.")
    ap.add_argument("--input", "-i", required=True, help="Path to input CSV")
    ap.add_argument("--output", "-o", required=True, help="Path to output CSV to write")
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep=None, engine="python")
    out = reconstruct_features(df)
    out.to_csv(args.output, index=False)
    print(f"Wrote {out.shape[0]} rows and {out.shape[1]} columns to: {args.output}")


if __name__ == "__main__":
    main()
