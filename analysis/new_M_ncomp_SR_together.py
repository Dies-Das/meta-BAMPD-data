#!/usr/bin/env python3
"""
Pooled Bayesian RT analyses with reward_value interactions (FULL dataset),
now making the "number of computations" fit **analogous** to the first fit:

- Model A (Compute × Optimal × Reward):
    log_rt ~ computational * took_opt * reward_value
             + trial_idx + abs_dmu + prev_reward + sum_var + diff_var
           + (1 + computational + took_opt + computational:took_opt
                + trial_idx + abs_dmu + prev_reward + sum_var + diff_var | subj_id)

  We extract, for rv ∈ {1,5}, the simple effect at took_opt=1:
      Δ_rv = β_comp + β_comp:took_opt
             + [β_comp:rv] + [β_comp:took_opt:rv]
  and plot exp(Δ_rv).

- Model B (Number of computations × Optimal × Reward, **analogous** to Model A):
  We allow a computations slope that is **active only when the trial was marked computational**,
  and we interact it with took_opt and reward_value:
    log_rt ~ ncomp_comp * took_opt * reward_value
             + trial_idx + abs_dmu + prev_reward + sum_var + diff_var
           + (1 + ncomp_comp + took_opt + ncomp_comp:took_opt
                + trial_idx + abs_dmu + prev_reward + sum_var + diff_var | subj_id)

  Here ncomp_comp = number_of_computations if computational==1 else 0 (no effect if not computational).
  We then extract, for rv ∈ {1,5}, the **slope per computation when took_opt=1**:
      slope_rv_opt = β_ncomp + β_ncomp:took_opt
                     + [β_ncomp:rv] + [β_ncomp:took_opt:rv]
  and plot exp(slope_rv_opt).

Usage:
  python plan_rt_bayes_pooled_by_reward_analogous.py --input trial_level_features.csv
"""

import argparse
import numpy as np
import pandas as pd
import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt


# -------------------------------------------------
# Preprocess
# -------------------------------------------------

def prepare(feat: pd.DataFrame, rt_cap_ms: int = 5000) -> pd.DataFrame:
    """
    - Cap RT at 5000 ms; define log_rt = log(RT_capped + 1).
    - Ensure numeric predictors; drop rows with missing key predictors.
    - took_opt to 0/1; computational to 0/1.
    - Build ncomp_comp = number_of_computations if computational==1 else 0.
      (This guarantees "no computations effect" on non-computational trials.)
    - Treat reward_value as categorical to estimate level-specific deviations.
    """
    df = feat.copy()

    required = [
        "subj_id", "reward_value", "rt", "took_opt", "computational",
        "number_of_computations", "trial_idx", "abs_dmu", "prev_reward",
        "sum_var", "diff_var"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["subj_id"] = df["subj_id"].astype(str)

    # RT transform
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
    df = df.dropna(subset=["rt"])
    df["rt_capped"] = np.minimum(df["rt"].values, float(rt_cap_ms))
    df["log_rt"] = np.log(df["rt_capped"] + 1.0)

    # Coerce predictors
    for c in [
        "reward_value", "took_opt", "computational", "number_of_computations",
        "trial_idx", "abs_dmu", "prev_reward", "sum_var", "diff_var"
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[
        "reward_value", "took_opt", "computational", "number_of_computations",
        "trial_idx", "abs_dmu", "prev_reward", "sum_var", "diff_var"
    ])

    # Cast binaries
    df["took_opt"] = df["took_opt"].astype(int)                  # 0/1
    df["computational"] = (df["computational"] > 0).astype(int)  # 0/1

    # Computations slope only available on computational trials
    df["ncomp_comp"] = pd.to_numeric(df["number_of_computations"], errors="coerce").fillna(0.0)
    df.loc[df["computational"] == 0, "ncomp_comp"] = 0.0

    # reward_value as categorical
    df["reward_value"] = df["reward_value"].astype(int).astype("category")

    return df


# -------------------------------------------------
# Helpers to extract fixed-effect draws
# -------------------------------------------------

def _fixed_keys(idata):
    return [k for k in idata.posterior.keys()
            if "|" not in k and not k.endswith(("_sigma", "_offset"))]

def _draws_or_zero(idata, name):
    post = idata.posterior
    if name in post:
        return post[name].values.reshape(-1)
    return np.zeros_like(post["Intercept"].values.reshape(-1))

def _find_key_contains(idata, must_include):
    keys = _fixed_keys(idata)
    for k in keys:
        if all(s in k for s in must_include):
            return k
    return None

def _reward_level_suffixes(rv):
    return [f"[T.{rv}]", f"[{rv}]"]


# ----- Model A: Δ_rv (computational effect when took_opt=1) -----

def _get_comp_effect_at_rv(idata, rv):
    base = _draws_or_zero(idata, "computational") + _draws_or_zero(idata, "computational:took_opt")
    delta = base.copy()
    for suf in _reward_level_suffixes(rv):
        k2 = _find_key_contains(idata, ["computational", "reward_value", suf])
        if k2:
            delta += _draws_or_zero(idata, k2)
        k3 = _find_key_contains(idata, ["computational", "took_opt", "reward_value", suf])
        if k3:
            delta += _draws_or_zero(idata, k3)
    return delta


# ----- Model B: slope_rv_opt (per-computation slope when took_opt=1) -----

def _get_ncomp_slope_at_rv_opt(idata, rv):
    """
    slope_rv_opt = β_ncomp_comp + β_ncomp_comp:took_opt
                   + [β_ncomp_comp:rv] + [β_ncomp_comp:took_opt:rv]
    """
    base = _draws_or_zero(idata, "ncomp_comp") + _draws_or_zero(idata, "ncomp_comp:took_opt")
    slope = base.copy()
    for suf in _reward_level_suffixes(rv):
        k2 = _find_key_contains(idata, ["ncomp_comp", "reward_value", suf])
        if k2:
            slope += _draws_or_zero(idata, k2)
        k3 = _find_key_contains(idata, ["ncomp_comp", "took_opt", "reward_value", suf])
        if k3:
            slope += _draws_or_zero(idata, k3)
    return slope


# -------------------------------------------------
# Models
# -------------------------------------------------

def fit_model_A_pooled(df: pd.DataFrame,
                       draws=3000, tune=1000, chains=4,
                       target_accept=0.99, random_seed=205):
    """
    log_rt ~ computational * took_opt * reward_value
             + trial_idx + abs_dmu + prev_reward + sum_var + diff_var
           + (1 + computational + took_opt + computational:took_opt
                + trial_idx + abs_dmu + prev_reward + sum_var + diff_var | subj_id)
    """
    fixed = "computational * took_opt * reward_value + trial_idx + abs_dmu " #+ prev_reward + sum_var + diff_var
    random = ("(1 + computational + took_opt + computational:took_opt "
              "+ trial_idx + abs_dmu | subj_id)") #+ prev_reward + sum_var + diff_var 
    formula = f"log_rt ~ {fixed} + {random}"

    mu_intercept = float(df["log_rt"].mean())
    priors = {
        "Intercept": bmb.Prior("Normal", mu=mu_intercept, sigma=1.0),
        "Common":    bmb.Prior("Normal", mu=0.0, sigma=0.5),
        "Sigma":     bmb.Prior("HalfNormal", sigma=1.0),
    }

    model = bmb.Model(formula, df, family="gaussian", priors=priors)
    idata = model.fit(draws=draws, tune=tune, chains=chains,
                      target_accept=target_accept, random_seed=random_seed)
    return idata


def fit_model_B_pooled_analogous(df: pd.DataFrame,
                                 draws=3000, tune=1000, chains=4,
                                 target_accept=0.99, random_seed=205):
    """
    **Analogous to Model A** but for computations amount:
    log_rt ~ ncomp_comp * took_opt * reward_value
             + trial_idx + abs_dmu + prev_reward + sum_var + diff_var
           + (1 + ncomp_comp + took_opt + ncomp_comp:took_opt
                + trial_idx + abs_dmu + prev_reward + sum_var + diff_var | subj_id)

    ncomp_comp is zero when not computational, so the computations slope is constrained
    to computational trials, and via the interaction we read the slope specifically at took_opt=1.
    """
    fixed = "ncomp_comp * took_opt * reward_value + trial_idx + abs_dmu " #+ prev_reward + sum_var + diff_var
    random = ("(1 + ncomp_comp + took_opt + ncomp_comp:took_opt "
              "+ trial_idx + abs_dmu  | subj_id)") #+ prev_reward + sum_var + diff_var
    formula = f"log_rt ~ {fixed} + {random}"

    mu_intercept = float(df["log_rt"].mean())
    priors = {
        "Intercept": bmb.Prior("Normal", mu=mu_intercept, sigma=1.0),
        "Common":    bmb.Prior("Normal", mu=0.0, sigma=0.5),
        "Sigma":     bmb.Prior("HalfNormal", sigma=1.0),
    }

    model = bmb.Model(formula, df, family="gaussian", priors=priors)
    idata = model.fit(draws=draws, tune=tune, chains=chains,
                      target_accept=target_accept, random_seed=random_seed)
    return idata


# -------------------------------------------------
# Plotting
# -------------------------------------------------

def plot_posteriors_ratio(ratios_by_label, title, ref_val=1.0):
    """
    ratios_by_label: list of tuples (label, ratio_array)
    """
    k = len(ratios_by_label)
    fig, axes = plt.subplots(k, 1, figsize=(8, 2.8 * k), constrained_layout=True)
    if k == 1:
        axes = [axes]
    for ax, (lbl, ratio) in zip(axes, ratios_by_label):
        az.plot_posterior(ratio, ref_val=ref_val, hdi_prob=0.95, ax=ax)
        p_gt1 = float((ratio > ref_val).mean())
        mean = float(np.mean(ratio))
        hdi = az.hdi(ratio, hdi_prob=0.95)
        ax.set_title(f"{title} — reward_value={lbl}\n"
                     f"mean={mean:.3f}, 95% HDI=[{hdi[0]:.3f}, {hdi[1]:.3f}], P(ratio>{ref_val})={p_gt1:.3f}")
        ax.set_xlabel("Multiplicative effect on (RT + 1)")
    return fig, axes


# -------------------------------------------------
# Runner
# -------------------------------------------------

def run_all_and_plot(
    trial_level_csv: str,
    target_reward_levels=(1, 5),
    draws=3000, tune=1000, chains=4,
    target_accept=0.99, seed=2025,
):
    # Load & prepare
    feat = pd.read_csv(trial_level_csv)
    df = prepare(feat)

    # Pooled models across ALL reward levels
    idata_A = fit_model_A_pooled(df, draws, tune, chains, target_accept, seed)
    idata_B = fit_model_B_pooled_analogous(df, draws, tune, chains, target_accept, seed)

    # Build per-reward effects from the pooled models
    ratios_A = []
    ratios_B = []
    for rv in target_reward_levels:
        # Model A: effect of being computational WHEN optimal at reward=rv
        delta_rv = _get_comp_effect_at_rv(idata_A, rv)
        ratio_A = np.exp(delta_rv)
        ratios_A.append((rv, ratio_A))

        # Model B: slope per additional computation WHEN optimal at reward=rv
        slope_rv_opt = _get_ncomp_slope_at_rv_opt(idata_B, rv)
        ratio_B = np.exp(slope_rv_opt)
        ratios_B.append((rv, ratio_B))

    # Plot
    figA, _ = plot_posteriors_ratio(ratios_A, title="Computational when optimal (exp(Δ_rv))")
    figB, _ = plot_posteriors_ratio(ratios_B, title="Per additional computation when optimal (exp(slope_rv_opt))")

    plt.show()
    return figA, figB


# -------------------------------------------------
# CLI
# -------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Pooled Bayesian RT analyses with reward interactions; analogous computations model.")
    ap.add_argument("--input", "-i", required=True, help="Path to trial-level features CSV")
    ap.add_argument("--draws", type=int, default=3000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target_accept", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    run_all_and_plot(
        trial_level_csv=args.input,
        target_reward_levels=(1, 5),  # report for 1 and 5
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
