# bayes_rt_with_reward_and_mstate.py
# ============================================================
# End-to-end:
#   1) Reconstruct trial-level features with Bayes-optimal vs greedy (is_mismatch)
#   2) Add reward_value (centered) and RT transform
#   3) Fit Bayesian mixed-effects model with reward_value moderation
#
# Expected raw CSV columns:
#   subj_id, stage, block_idx, horizon, p0, p1, trial_idx, arm, reward, rt, reward_value
# ============================================================

import os
import warnings
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
import arviz as az
import bambi as bmb


# ----------------------------
# Utilities
# ----------------------------

def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _posterior_term(post, name: str):
    """
    Fetch a fixed-effect term robustly from idata.posterior.
    Works for both post['term'] and post['beta'].sel(term=...).
    Returns None if not found.
    """
    if name in post:
        return post[name]
    if ("beta" in post) and ("term" in post["beta"].coords):
        terms = [str(t) for t in post["beta"].coords["term"].values]
        if name in terms:
            return post["beta"].sel(term=name)
    return None

def _get_coef_array(post, *candidates: str) -> np.ndarray:
    for nm in candidates:
        arr = _posterior_term(post, nm)
        if arr is not None:
            return arr.values.reshape(-1)
    raise KeyError(f"Coefficient not found in posterior: {candidates}")

def _hdi_summary(x: np.ndarray, hdi_prob: float = 0.95) -> Dict[str, float]:
    lo, hi = az.hdi(x, hdi_prob=hdi_prob)
    return {"mean": float(np.mean(x)), f"hdi_{(1-hdi_prob)/2*100:.1f}%": float(lo), f"hdi_{(1+(hdi_prob))/2*100:.1f}%": float(hi)}


# ----------------------------
# Dynamic programming (Bayes-optimal vs greedy)
# ----------------------------

def states_by_steps_left(T: int):
    states = [[] for _ in range(T + 1)]
    for n in range(T + 1):
        s_left = T - n
        for s1, f1, s2 in itertools.product(range(n + 1), repeat=3):
            f2 = n - (s1 + f1 + s2)
            if f2 < 0:
                continue
            states[s_left].append((1 + s1, 1 + f1, 1 + s2, 1 + f2))
    return states

class PolicyPackS:
    def __init__(self, T, opt_by_s, greedy, mismatch_by_s, qgap_by_s):
        self.T = T
        self.opt_by_s = opt_by_s
        self.greedy = greedy
        self.mismatch_by_s = mismatch_by_s
        self.qgap_by_s = qgap_by_s

@lru_cache(maxsize=None)
def compute_policies_for_horizon(T: int) -> PolicyPackS:
    """
    Finite-horizon Bayes-optimal policy vs greedy for Bernoulli arms with unit reward.
    (Multiplying all rewards by a positive constant like `reward_value` does not change argmax.)
    """
    S = states_by_steps_left(T)
    V = [defaultdict(float) for _ in range(T + 1)]
    opt_by_s = {}
    qgap_by_s = {}

    # base
    for st in S[0]:
        V[0][st] = 0.0

    # backward induction
    for s in range(1, T + 1):
        for st in S[s]:
            a1, b1, a2, b2 = st
            p1 = a1 / (a1 + b1)
            p2 = a2 / (a2 + b2)

            succ1 = (a1 + 1, b1, a2, b2)
            fail1 = (a1, b1 + 1, a2, b2)
            succ2 = (a1, b1, a2 + 1, b2)
            fail2 = (a1, b1, a2, b2 + 1)  # FIXED: increment b2 on failure

            Q1 = p1 * (1 + V[s - 1][succ1]) + (1 - p1) * V[s - 1][fail1]
            Q2 = p2 * (1 + V[s - 1][succ2]) + (1 - p2) * V[s - 1][fail2]

            V[s][st] = max(Q1, Q2)
            opt_by_s[(s, st)] = 0 if Q1 > Q2 else 1 if Q2 > Q1 else -1
            qgap_by_s[(s, st)] = abs(Q1 - Q2)

    # greedy (step-independent)
    greedy = {}
    for lst in S:
        for st in lst:
            a1, b1, a2, b2 = st
            m1 = a1 / (a1 + b1)
            m2 = a2 / (a2 + b2)
            greedy[st] = 0 if m1 > m2 else 1 if m2 > m1 else -1

    mismatch_by_s = {(s, st) for (s, st), oa in opt_by_s.items() if oa != greedy.get(st, -1)}
    return PolicyPackS(T, opt_by_s, greedy, mismatch_by_s, qgap_by_s)


# ----------------------------
# Feature reconstruction (WITH is_mismatch)
# ----------------------------

def reconstruct_states_and_labels(df_main: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct per-trial state, is_mismatch, took_opt, uncertainties, and lags.
    Groups by (subj_id, stage, block_idx) to keep stages independent.
    """
    df = df_main.copy()
    df["arm"] = df["arm"].astype(int)

    # Sort stable for lag features
    df = df.sort_values(["subj_id", "stage", "block_idx", "trial_idx"], kind="mergesort").reset_index(drop=True)

    policies = {int(T): compute_policies_for_horizon(int(T)) for T in sorted(df["horizon"].unique())}

    rows = []
    for (sid, stage, block), g in df.groupby(["subj_id", "stage", "block_idx"], sort=False):
        g = g.sort_values("trial_idx", kind="mergesort")
        T = int(g["horizon"].iloc[0])
        ppack = policies[T]

        s = {0: 0, 1: 0}
        f = {0: 0, 1: 0}
        prev_rew = 0
        prev_switch = 0
        prev_mstate = 0
        prev_surprise = 0.0
        prev_arm = None
        prev_rt = None

        for i, r in enumerate(g.itertuples(index=False), start=1):
            # state BEFORE action
            a0, b0 = 1 + s[0], 1 + f[0]
            a1, b1 = 1 + s[1], 1 + f[1]
            st = (a0, b0, a1, b1)
            steps_left = T - (i - 1)

            m0 = a0 / (a0 + b0)
            m1 = a1 / (a1 + b1)
            abs_dmu = abs(m0 - m1)

            # Beta variances (posterior uncertainty)
            var0 = (a0 * b0) / (((a0 + b0) ** 2) * (a0 + b0 + 1))
            var1 = (a1 * b1) / (((a1 + b1) ** 2) * (a1 + b1 + 1))
            sum_var = var0 + var1
            diff_var = abs(var1 - var0)

            # Optimal/greedy & mismatch
            opt_act = ppack.opt_by_s.get((steps_left, st), -1)
            greedy_act = ppack.greedy.get(st, -1)
            is_mismatch = int((steps_left, st) in ppack.mismatch_by_s)
            mismatch_strength = ppack.qgap_by_s.get((steps_left, st), 0.0)

            # took_opt (NaN if tie)
            took_opt = np.nan if opt_act == -1 else int(r.arm == opt_act)

            # switch & lag log rt
            switch = 1 if (prev_arm is not None and r.arm != prev_arm) else 0
            lag_log_rt = np.log(prev_rt) if (prev_rt is not None and prev_rt > 0) else 0.0

            rows.append(dict(
                subj_id=r.subj_id, stage=r.stage, block_idx=r.block_idx,
                horizon=T, p0=float(r.p0), p1=float(r.p1),
                trial_idx=int(r.trial_idx), arm=int(r.arm),
                reward=int(r.reward), rt=float(r.rt),

                a0=a0, b0=b0, a1=a1, b1=b1, steps_left=steps_left,
                opt_act=opt_act, greedy_act=greedy_act,
                is_mismatch=is_mismatch, took_opt=took_opt,
                mismatch_strength=mismatch_strength,

                abs_dmu=abs_dmu, abs_dtrue=abs(float(r.p0) - float(r.p1)),
                var0=var0, var1=var1, sum_var=sum_var, diff_var=diff_var,

                switch=switch, prev_switch=prev_switch, prev_mstate=prev_mstate,
                prev_reward=prev_rew, prev_surprise=prev_surprise,
                lag_log_rt=lag_log_rt
            ))

            # --- Update for next trial ---
            mean_chosen_pre = m0 if r.arm == 0 else m1
            if r.arm == 0:
                if r.reward == 1: s[0] += 1
                else:             f[0] += 1
            else:
                if r.reward == 1: s[1] += 1
                else:             f[1] += 1

            prev_mstate = is_mismatch
            prev_switch = switch
            prev_rew = int(r.reward)
            prev_arm = int(r.arm)
            prev_rt = float(r.rt)
            prev_surprise = abs(prev_rew - mean_chosen_pre)

    feat = pd.DataFrame(rows)
    return feat


# ----------------------------
# Build features (add reward_value and RT transform)
# ----------------------------

def build_features_from_raw(raw: pd.DataFrame, rt_cap_ms: int = 5000) -> pd.DataFrame:
    """
    1) Reconstruct features (incl. is_mismatch) via DP.
    2) Attach reward_value by (subj_id, stage, block_idx, trial_idx).
    3) Add capped + log RT, and centered reward_value.
    """
    required = ["subj_id","stage","block_idx","horizon","p0","p1","trial_idx","arm","reward","rt","reward_value"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw CSV: {missing}")

    # Hygiene & types
    raw = raw.copy()
    raw["subj_id"] = raw["subj_id"].astype(str)
    raw = _ensure_numeric(raw, ["block_idx","horizon","p0","p1","trial_idx","arm","reward","rt","reward_value"])
    raw = raw.dropna(subset=["subj_id","stage","block_idx","horizon","p0","p1","trial_idx","arm","reward","rt","reward_value"])
    raw["block_idx"] = raw["block_idx"].astype(int)
    raw["horizon"] = raw["horizon"].astype(int)
    raw["trial_idx"] = raw["trial_idx"].astype(int)
    raw["arm"] = raw["arm"].astype(int)
    raw["reward"] = raw["reward"].astype(int)

    # Reconstruct (adds is_mismatch, took_opt, uncertainties, etc.)
    feat = reconstruct_states_and_labels(raw)

    # Attach reward_value row-wise (keyed by subj_id, stage, block_idx, trial_idx)
    key = ["subj_id","stage","block_idx","trial_idx"]
    rv = raw[key + ["reward_value"]]
    feat = feat.merge(rv, on=key, how="left")

    # RT transform: cap & log(+1 ms)
    feat["rt_capped"] = np.minimum(feat["rt"].values, float(rt_cap_ms))
    feat["log_rt"] = np.log(feat["rt_capped"] + 1.0)

    # Center reward_value (median center for stability)
    med_rv = float(np.nanmedian(feat["reward_value"].values))
    feat["c_reward_value"] = feat["reward_value"] - med_rv
    feat["reward_value_median"] = med_rv

    # Final numeric enforcement for the model’s predictors
    model_cols = ["is_mismatch","took_opt","trial_idx","abs_dmu","prev_reward","sum_var","diff_var","c_reward_value","log_rt"]
    feat = _ensure_numeric(feat, model_cols)
    return feat


def write_features_csv(
    raw_csv: str,
    out_csv: str = "trial_level_features_with_reward.csv",
    rt_cap_ms: int = 5000
) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv)
    feat = build_features_from_raw(raw, rt_cap_ms=rt_cap_ms)
    feat.to_csv(out_csv, index=False)
    print(f"Features written to: {out_csv}")
    # Quick check: ensure is_mismatch varies somewhere
    if feat["is_mismatch"].nunique() <= 1:
        warnings.warn("`is_mismatch` has no variation in the built features. "
                      "Verify horizon, p0/p1, and block boundaries.", RuntimeWarning)
    return feat

def _first_present_term(post, *names):
    """
    Return the first posterior DataArray found among `names`, else None.
    """
    for nm in names:
        arr = _posterior_term(post, nm)
        if arr is not None:
            return arr
    return None

# ----------------------------
# Model building & fitting
# ----------------------------

def _has_variation(s: pd.Series) -> bool:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s.nunique(dropna=True) > 1

def _build_formula(df: pd.DataFrame) -> str:
    """
    Construct the fixed/random effects safely.
    Baseline: is_mismatch * took_opt + trial_idx + abs_dmu + prev_reward + sum_var + diff_var
    Moderator: c_reward_value
    Moderation: c_reward_value with each baseline term (incl. is_mismatch:took_opt)
    Random effects: intercept + slopes for baseline main effects + c_reward_value (by subj_id)
    """
    # Baseline candidates with variation checks
    baseline_terms: List[str] = []
    if "is_mismatch" in df and _has_variation(df["is_mismatch"]): baseline_terms.append("is_mismatch")
    if "took_opt" in df and _has_variation(df["took_opt"]):       baseline_terms.append("took_opt")
    interaction_terms: List[str] = []
    if all(t in baseline_terms for t in ["is_mismatch","took_opt"]):
        interaction_terms.append("is_mismatch:took_opt")
    for term in ["trial_idx","abs_dmu","prev_reward","sum_var","diff_var"]:
        if term in df and _has_variation(df[term]): baseline_terms.append(term)
    # dedupe preserving order
    seen = set(); baseline_terms = [t for t in baseline_terms if not (t in seen or seen.add(t))]

    if "c_reward_value" not in df or not _has_variation(df["c_reward_value"]):
        raise ValueError("`c_reward_value` missing or has no variation; cannot fit moderation model.")

    if baseline_terms or interaction_terms:
        baseline_str = " + ".join(baseline_terms + interaction_terms)
        moderation = " + ".join([f"c_reward_value:{t}" for t in (baseline_terms + interaction_terms)])
        fixed = f"{baseline_str} + c_reward_value + {moderation}"
    else:
        fixed = "c_reward_value"

    # Random effects (omit interaction slopes for stability)
    re_terms = ["1"] + [t for t in baseline_terms if t != "is_mismatch:took_opt"] + ["c_reward_value"]
    re_unique = []
    for t in re_terms:
        if t not in re_unique:
            re_unique.append(t)
    re = "(" + " + ".join(re_unique) + " | subj_id)"

    return f"log_rt ~ {fixed} + {re}"

def fit_bayesian_mixed_rt_with_reward(
    df: pd.DataFrame,
    draws: int = 3000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.99,
    random_seed: int = 205,
    out_prefix: str = "bayes_rt_with_reward",
):
    """
    Fit Gaussian mixed-effects on log-RT with reward_value moderation.
    Includes robust post-processing that:
      - avoids boolean operations on xarray.DataArray
      - computes means for interaction terms as the mean of products
    """
    # Drop trials with undefined optimal action (ties => took_opt NaN)
    X = df.dropna(subset=["took_opt"]).copy()

    # ---- Build model formula safely (uses your existing _build_formula) ----
    formula = _build_formula(X)
    print(f"\nModel formula:\n  {formula}\n")

    mu_intercept = float(X["log_rt"].mean())
    priors = {
        "Intercept": bmb.Prior("Normal", mu=mu_intercept, sigma=1.0),
        "Common":    bmb.Prior("Normal", mu=0.0, sigma=0.5),
        "Sigma":     bmb.Prior("HalfNormal", sigma=1.0),
    }

    model = bmb.Model(formula, X, family="gaussian", priors=priors)
    idata = model.fit(
        draws=draws, tune=tune, chains=chains,
        target_accept=target_accept, random_seed=random_seed,
    )

    # -------- Helpers for robust post-processing --------
    post = idata.posterior

    def _first_present_term(post, *names):
        """Return the first posterior DataArray present among `names`, else None."""
        for nm in names:
            arr = _posterior_term(post, nm)
            if arr is not None:
                return arr
        return None

    def b_int(term: str):
        """
        Get the c_reward_value interaction with `term` (whatever the ordering).
        Returns None if not present.
        """
        arr = _first_present_term(post, f"c_reward_value:{term}", f"{term}:c_reward_value")
        return None if arr is None else arr.values.reshape(-1)

    def _term_mean(df_: pd.DataFrame, term_name: str) -> float:
        """
        Mean of a main term, or mean of the product for interaction terms like 'a:b[:c]'.
        """
        if ":" not in term_name:
            return float(pd.to_numeric(df_[term_name], errors="coerce").mean())
        parts = term_name.split(":")
        prod = np.ones(len(df_), dtype=float)
        for p in parts:
            prod *= pd.to_numeric(df_[p], errors="coerce").values
        return float(np.nanmean(prod))

    # -------- Identify included fixed terms (those we can summarize) --------
    included_terms = []
    for nm in [
        "is_mismatch","took_opt","is_mismatch:took_opt",
        "trial_idx","abs_dmu","prev_reward","sum_var","diff_var"
    ]:
        if _posterior_term(post, nm) is not None:
            included_terms.append(nm)

    # -------- AME of reward_value at dataset means of other covariates --------
    # Mean for interactions is computed as the mean of the product.
    mean_vals = {t: _term_mean(X, t) for t in included_terms}

    beta_cRV = _get_coef_array(post, "c_reward_value")
    ame_draws = beta_cRV.copy()
    for t in included_terms:
        arr = b_int(t)
        if arr is not None:
            ame_draws = ame_draws + mean_vals[t] * arr

    ame_rt_ratio = np.exp(ame_draws)
    ame_summary = {
        "d_logRT_d_reward_value": _hdi_summary(ame_draws, 0.95),
        "RT_ratio_per_unit_reward_value": _hdi_summary(ame_rt_ratio, 0.95),
    }

    # -------- How coefficients change across reward_value (q10/q50/q90) --------
    q10, q50, q90 = np.percentile(X["reward_value"].values, [10, 50, 90])
    med = float(X["reward_value_median"].iloc[0] if "reward_value_median" in X else np.median(X["reward_value"].values))
    levels = {"q10": q10 - med, "q50": q50 - med, "q90": q90 - med}

    coeff_changes = {}
    for term in included_terms:
        beta = _get_coef_array(post, term)
        beta_int = b_int(term)
        out = {}
        for lab, cval in levels.items():
            slope = beta + (0 if beta_int is None else beta_int * cval)
            ratio = np.exp(slope)
            out[lab] = {
                "logRT_slope": _hdi_summary(slope, 0.95),
                "RT_ratio_per_unit_of_term": _hdi_summary(ratio, 0.95),
            }
        coeff_changes[term] = out

    # -------- Simple effect: M-state when optimal across reward_value --------
    simple_effect_by_rv = {}
    if ("is_mismatch" in included_terms) and ("is_mismatch:took_opt" in included_terms):
        beta_m     = _get_coef_array(post, "is_mismatch")
        beta_mopt  = _get_coef_array(post, "is_mismatch:took_opt")
        beta_m_cRV = _get_coef_array(post, "c_reward_value:is_mismatch", "is_mismatch:c_reward_value")

        tri = _first_present_term(
            post,
            "c_reward_value:is_mismatch:took_opt",
            "is_mismatch:c_reward_value:took_opt",
            "is_mismatch:took_opt:c_reward_value",
            "took_opt:is_mismatch:c_reward_value",
            "took_opt:c_reward_value:is_mismatch",
            "c_reward_value:took_opt:is_mismatch",
        )
        beta_tri = None if tri is None else tri.values.reshape(-1)

        for lab, cval in levels.items():
            delta = (beta_m + beta_mopt) + (beta_m_cRV + (0 if beta_tri is None else beta_tri)) * cval
            ratio = np.exp(delta)
            dlo, dhi = az.hdi(delta, hdi_prob=0.95)
            rlo, rhi = az.hdi(ratio, hdi_prob=0.95)
            simple_effect_by_rv[lab] = {
                "delta_logRT_mean": float(delta.mean()),
                "delta_logRT_hdi_2.5%": float(dlo),
                "delta_logRT_hdi_97.5%": float(dhi),
                "p(Δ>0)": float((delta > 0).mean()),
                "RT_ratio_mean": float(ratio.mean()),
                "RT_ratio_hdi_2.5%": float(rlo),
                "RT_ratio_hdi_97.5%": float(rhi),
            }
    else:
        warnings.warn("`is_mismatch` or `is_mismatch:took_opt` absent in fixed effects; skipping simple-effect summary.", RuntimeWarning)

    # -------- Save artifacts & print summaries --------
    os.makedirs("bayes_outputs", exist_ok=True)
    summ = az.summary(idata, round_to=4)
    summ.to_csv(os.path.join("bayes_outputs", f"{out_prefix}_posterior_summary.csv"))
    idata.to_netcdf(os.path.join("bayes_outputs", f"{out_prefix}.nc"))

    print("\nPosterior summary (top rows):")
    print(summ.head(25).to_string())
    print("\nAverage marginal effect of reward_value on log-RT (and RT ratio per +1 unit):")
    print(ame_summary)
    print("\nHow coefficients change across reward_value (q10, q50, q90):")
    for k, v in coeff_changes.items():
        print(f"\n  {k}:")
        for lvl, stats in v.items():
            print(f"    {lvl}: {stats}")
    if simple_effect_by_rv:
        print("\nSimple effect: M-state when optimal across reward_value:")
        for lvl, stats in simple_effect_by_rv.items():
            print(f"  {lvl}: {stats}")

    summaries = {
        "ame_reward_value": ame_summary,
        "coefficients_vs_reward_value": coeff_changes,
        "simple_effect_mismatch_when_optimal_by_reward_value": simple_effect_by_rv,
        "reward_value_levels": {"q10": float(q10), "q50": float(q50), "q90": float(q90), "median_used_for_centering": med},
        "formula": formula,
    }
    return model, idata, summaries


# ----------------------------
# Runners
# ----------------------------

def run_end_to_end(
    raw_csv: str = "bandit-stakes-v0.1.1.csv",
    features_csv: str = "trial_level_features_with_reward.csv",
    draws: int = 3000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.99,
    seed: int = 2025,
    out_prefix: str = "bayes_rt_with_reward",
    rt_cap_ms: int = 5000,
):
    feat = write_features_csv(raw_csv, out_csv=features_csv, rt_cap_ms=rt_cap_ms)
    return fit_bayesian_mixed_rt_with_reward(
        feat,
        draws=draws, tune=tune, chains=chains,
        target_accept=target_accept, random_seed=seed, out_prefix=out_prefix,
    )

def run_from_features_csv(
    features_csv: str = "trial_level_features_with_reward.csv",
    draws: int = 3000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.99,
    seed: int = 2025,
    out_prefix: str = "bayes_rt_with_reward",
):
    feat = pd.read_csv(features_csv)
    return fit_bayesian_mixed_rt_with_reward(
        feat,
        draws=draws, tune=tune, chains=chains,
        target_accept=target_accept, random_seed=seed, out_prefix=out_prefix,
    )


if __name__ == "__main__":
    # Example:
    # run_end_to_end(raw_csv="your_new_data.csv")
    run_end_to_end()

