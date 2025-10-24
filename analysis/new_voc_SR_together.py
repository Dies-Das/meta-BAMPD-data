#!/usr/bin/env python3
"""
Bayesian RT analysis with VOC (value of computation) when VOC is mostly 0 and
small-positive when present.

Idea
----
Because VOC is zero for most trials and tiny (<0.1) when positive, we split its
influence into two parts:

1) Presence effect (any VOC vs none):  voc_present = 1 if VOC>0 else 0
2) Magnitude effect among positives:   voc_mag01 = VOC / 0.01 if VOC>0 else 0
   (so the slope is "per +0.01 VOC", which keeps coefficients well-scaled)

We estimate how each part scales with reward stakes by interacting with
reward_value, while keeping your other controls.

Model (pooled across all rewards):
  log_rt ~ (voc_present * reward_value) + (voc_mag01 * reward_value)
           + trial_idx + abs_dmu + prev_reward + sum_var + diff_var
         + (1 + voc_present + voc_mag01
              + trial_idx + abs_dmu + prev_reward + sum_var + diff_var | subj_id)

We then extract, for rv ∈ {1,5}:
  - Presence effect: Δ_present_rv = β_voc_present + [β_voc_present:rv]
  - Magnitude slope per +0.01 VOC: slope_mag_rv = β_voc_mag01 + [β_voc_mag01:rv]

We plot exp(Δ_present_rv) and exp(slope_mag_rv), i.e., multiplicative changes
in (RT+1). No computational / n_comps / took_opt terms are used.

Usage
-----
python rt_voc_hurdle_by_reward.py --input trial_level_features.csv --voc_col voc_bound --reward_levels 1 5
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

def prepare(
    feat: pd.DataFrame,
    voc_col: str = "voc_bound",
    rt_cap_ms: int = 5000
) -> pd.DataFrame:
    """
    - Cap RT at 5000 ms; log_rt = log(RT_capped + 1).
    - Keep controls: trial_idx, abs_dmu, prev_reward, sum_var, diff_var.
    - reward_value as categorical.
    - Build two VOC-derived regressors:
        voc_present = 1 if VOC>0 else 0
        voc_mag01   = VOC/0.01 if VOC>0 else 0  (slope is "per +0.01 VOC")
    """
    df = feat.copy()

    required = [
        "subj_id", "reward_value", "rt",
        "trial_idx", "abs_dmu", "prev_reward", "sum_var", "diff_var", voc_col
    ]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    df["subj_id"] = df["subj_id"].astype(str)

    # RT transform
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
    df = df.dropna(subset=["rt"])
    df["rt_capped"] = np.minimum(df["rt"].values, float(rt_cap_ms))
    df["log_rt"] = np.log(df["rt_capped"] + 1.0)

    # Coerce predictors
    for c in ["reward_value", "trial_idx", "abs_dmu", "prev_reward", "sum_var", "diff_var", voc_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["reward_value", "trial_idx", "abs_dmu", "prev_reward", "sum_var", "diff_var", voc_col])

    # reward as categorical (keeps 0/1/5 as explicit levels if present)
    df["reward_value"] = df["reward_value"].astype(int).astype("category")

    # Standard VOC name + two-part encoding
    if voc_col != "voc":
        df = df.rename(columns={voc_col: "voc"})

    # Presence indicator and magnitude (per 0.01) among positives
    df["voc_present"] = (df["voc"] > 0).astype(int)
    df["voc_mag01"] = 0.0
    pos = df["voc"] > 0
    df.loc[pos, "voc_mag01"] = df.loc[pos, "voc"] / 0.01  # each unit = 0.01 VOC

    return df


# -------------------------------------------------
# Fixed-effect extraction helpers
# -------------------------------------------------

def _fixed_keys(idata):
    return [k for k in idata.posterior.keys()
            if "|" not in k and not k.endswith(("_sigma", "_offset"))]

def _draws_or_zero(idata, name):
    post = idata.posterior
    if name in post:
        return post[name].values.reshape(-1)
    # zeros with same shape as Intercept draws
    return np.zeros_like(post["Intercept"].values.reshape(-1))

def _find_key_contains(idata, must_include):
    keys = _fixed_keys(idata)
    for k in keys:
        if all(s in k for s in must_include):
            return k
    return None

def _reward_level_suffixes(rv):
    # patsy/bambi typically names: reward_value[T.5]
    return [f"[T.{rv}]", f"[{rv}]"]

def _get_voc_presence_effect_at_rv(idata, rv):
    """
    Δ_present_rv = β_voc_present + [β_voc_present:reward_value[T.rv]]
    """
    base = _draws_or_zero(idata, "voc_present")
    delta = base.copy()
    for suf in _reward_level_suffixes(rv):
        k = _find_key_contains(idata, ["voc_present", "reward_value", suf])
        if k:
            delta += _draws_or_zero(idata, k)
    return delta

def _get_voc_mag_slope_at_rv(idata, rv):
    """
    slope_mag_rv (per +0.01 VOC) = β_voc_mag01 + [β_voc_mag01:reward_value[T.rv]]
    """
    base = _draws_or_zero(idata, "voc_mag01")
    slope = base.copy()
    for suf in _reward_level_suffixes(rv):
        k = _find_key_contains(idata, ["voc_mag01", "reward_value", suf])
        if k:
            slope += _draws_or_zero(idata, k)
    return slope


# -------------------------------------------------
# Model
# -------------------------------------------------

def fit_model_voc_hurdle(
    df: pd.DataFrame,
    draws=3000, tune=1000, chains=4,
    target_accept=0.99, random_seed=205
):
    """
    log_rt ~ (voc_present * reward_value) + (voc_mag01 * reward_value)
             + trial_idx + abs_dmu + prev_reward + sum_var + diff_var
           + (1 + voc_present + voc_mag01
                + trial_idx + abs_dmu + prev_reward + sum_var + diff_var | subj_id)

    Notes:
    - Narrower "Common" prior helps stabilize slopes since voc_mag01 is well-scaled.
    - No computational / n_comps / took_opt terms.
    """
    fixed = (
        "voc_present * reward_value + voc_mag01 * reward_value + "
        "trial_idx + abs_dmu + prev_reward + sum_var + diff_var"
    )
    random = (
        "(1 + voc_present + voc_mag01 + "
        "trial_idx + abs_dmu + prev_reward + sum_var + diff_var | subj_id)"
    )
    formula = f"log_rt ~ {fixed} + {random}"

    mu_intercept = float(df["log_rt"].mean())
    priors = {
        "Intercept": bmb.Prior("Normal", mu=mu_intercept, sigma=1.0),
        # modest shrinkage on all fixed slopes
        "Common":    bmb.Prior("Normal", mu=0.0, sigma=0.3),
        "Sigma":     bmb.Prior("HalfNormal", sigma=1.0),
    }

    model = bmb.Model(formula, df, family="gaussian", priors=priors)
    idata = model.fit(draws=draws, tune=tune, chains=chains,
                      target_accept=target_accept, random_seed=random_seed)
    return idata


# -------------------------------------------------
# Plotting
# -------------------------------------------------

def plot_posteriors_ratio(ratios_by_label, title, ref_val=1.0, xlab="Multiplicative effect on (RT + 1)"):
    """
    ratios_by_label: list of tuples (label, ratio_array)
    """
    k = len(ratios_by_label)
    fig, axes = plt.subplots(k, 1, figsize=(8, 2.8 * k), constrained_layout=True)
    if k == 1:
        axes = [axes]
    for ax, (lbl, ratio) in zip(axes, ratios_by_label):
        ratio = np.asarray(ratio)
        ratio = ratio[np.isfinite(ratio)]
        az.plot_posterior(ratio, ref_val=ref_val, hdi_prob=0.95, ax=ax)
        p_gt1 = float((ratio > ref_val).mean())
        mean = float(np.mean(ratio))
        hdi = az.hdi(ratio, hdi_prob=0.95)
        ax.set_title(f"{title} — reward_value={lbl}\n"
                     f"mean={mean:.3f}, 95% HDI=[{hdi[0]:.3f}, {hdi[1]:.3f}], P(ratio>{ref_val})={p_gt1:.3f}")
        ax.set_xlabel(xlab)
    return fig, axes


# -------------------------------------------------
# Runner
# -------------------------------------------------

def run_all_and_plot(
    trial_level_csv: str,
    voc_col: str = "voc_bound",
    target_reward_levels=(1, 5),
    draws=3000, tune=1000, chains=4,
    target_accept=0.99, seed=2025,
):
    # Load & prepare
    feat = pd.read_csv(trial_level_csv)
    df = prepare(feat, voc_col=voc_col)

    # Fit pooled hurdle-like model
    idata = fit_model_voc_hurdle(df, draws, tune, chains, target_accept, seed)

    # Build per-reward VOC effects from the pooled model
    ratios_presence = []
    ratios_mag = []

    for rv in target_reward_levels:
        # Presence effect (any VOC vs none)
        delta_present = _get_voc_presence_effect_at_rv(idata, rv)  # log scale
        ratio_present = np.exp(delta_present)
        ratios_presence.append((rv, ratio_present))

        # Magnitude effect per +0.01 VOC among positives
        slope_mag = _get_voc_mag_slope_at_rv(idata, rv)            # log scale per +0.01 VOC
        ratio_mag = np.exp(slope_mag)
        ratios_mag.append((rv, ratio_mag))

        # Console summaries
        hdi_p = az.hdi(ratio_present, hdi_prob=0.95)
        hdi_m = az.hdi(ratio_mag, hdi_prob=0.95)
        print(f"[VOC presence] reward_value={rv}: mean={ratio_present.mean():.3f}, "
              f"HDI95%=[{hdi_p[0]:.3f}, {hdi_p[1]:.3f}], P(ratio>1)={(ratio_present>1).mean():.3f}")
        print(f"[VOC magnitude per +0.01] reward_value={rv}: mean={ratio_mag.mean():.3f}, "
              f"HDI95%=[{hdi_m[0]:.3f}, {hdi_m[1]:.3f}], P(ratio>1)={(ratio_mag>1).mean():.3f}")

    # Plot
    fig1, _ = plot_posteriors_ratio(
        ratios_presence,
        title="Effect of having any VOC (exp(Δ_present_rv))",
        xlab="Multiplicative effect on (RT + 1) for VOC>0 vs VOC=0"
    )
    fig2, _ = plot_posteriors_ratio(
        ratios_mag,
        title="Effect per +0.01 VOC among positives (exp(slope_mag_rv))",
        xlab="Multiplicative effect on (RT + 1) per +0.01 VOC"
    )

    plt.show()
    return fig1, fig2


# -------------------------------------------------
# CLI
# -------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Bayesian RT analysis with VOC (presence + magnitude) by reward_value.")
    ap.add_argument("--input", "-i", required=True, help="Path to trial-level features CSV")
    ap.add_argument("--voc_col", type=str, default="voc_bound", help="Column name to use as VOC (default: voc_bound)")
    ap.add_argument("--reward_levels", nargs="+", type=int, default=[1, 5],
                    help="Reward values to report (default: 1 5)")
    ap.add_argument("--draws", type=int, default=3000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target_accept", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    run_all_and_plot(
        trial_level_csv=args.input,
        voc_col=args.voc_col,
        target_reward_levels=tuple(args.reward_levels),
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
