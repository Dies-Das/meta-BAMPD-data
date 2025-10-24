#!/usr/bin/env python3
"""
Bayesian RT analyses with plots (no CSVs) for planning/computation signatures.

We run two models, separately for reward_value = 1 and reward_value = 5:

Model A (Compute×Optimal):
  log_rt ~ computational * took_opt + trial_idx + abs_dmu + prev_reward + sum_var + diff_var
  Random effects: (1 + all fixed slopes | subj_id)
  Report/posterior-plot the simple effect of "computational" at took_opt=1:
      Δ = β_computational + β_computational:took_opt
  We display the posterior of exp(Δ), i.e., the multiplicative change in (RT+1).

Model B (Amount of Computation):
  Subset to trials where computational==1 AND took_opt==1, same reward_value.
  log_rt ~ number_of_computations + trial_idx + abs_dmu + prev_reward + sum_var + diff_var
  Random effects: (1 + all fixed slopes | subj_id)
  Report/posterior-plot exp(β_number_of_computations), i.e., multiplicative change per computation.

Usage:
  python plan_rt_bayes_plots.py --input trial_level_features.csv
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
    RT handling (no z-scoring):
      - cap RT at 5000 ms
      - add 1 ms to avoid log(0)
      - log-transform to log_rt
    Keep predictors on original scales.
    Drop rows where took_opt is NaN (ties with undefined optimal action).
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
    df["log_rt"] = np.log(df["rt_capped"] + 1.0)  # +1 ms to avoid log(0)

    # Coerce predictors
    for c in [
        "reward_value", "took_opt", "computational", "number_of_computations",
        "trial_idx", "abs_dmu", "prev_reward", "sum_var", "diff_var"
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with missing predictors; drop ties where took_opt is NaN
    df = df.dropna(subset=[
        "reward_value", "took_opt", "computational", "number_of_computations",
        "trial_idx", "abs_dmu", "prev_reward", "sum_var", "diff_var"
    ])

    # Ensure took_opt is 0/1
    df["took_opt"] = df["took_opt"].astype(int)

    # Ensure computational is 0/1 int
    df["computational"] = (df["computational"] > 0).astype(int)

    return df


def _get_fixed_draws(idata, name: str):
    """
    Return posterior draws for a fixed (population-level) coefficient `name`.
    Raises a helpful error listing available fixed-effect names if missing.
    """
    post = idata.posterior
    if name in post:
        return post[name].values.reshape(-1)
    # Helpful debug if someone changes names/pats
    available = [k for k in post.keys() if "|" not in k and not k.endswith(("_sigma", "_offset"))]
    raise KeyError(f"Fixed effect '{name}' not found. Fixed effects present: {available}")

# -------------------------------------------------
# Model A: Compute×Optimal across all trials at a given reward_value
# -------------------------------------------------

def fit_model_A_compute_when_optimal(
    df,
    reward_value: int,
    draws: int = 3000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.99,
    random_seed: int = 205,
):
    """
    Gaussian mixed model on log-RT for a specific reward_value:
      FIXED:  computational * took_opt + trial_idx + abs_dmu + prev_reward + sum_var + diff_var
      RANDOM: (1 + computational + took_opt + computational:took_opt
                  + trial_idx + abs_dmu + prev_reward + sum_var + diff_var | subj_id)
    """
    df_rv = df[df["reward_value"] == reward_value].copy()
    if df_rv.empty:
        raise ValueError(f"No rows for reward_value={reward_value} after preprocessing.")

    # ✅ Include BOTH fixed and random parts
    fixed = "computational * took_opt + trial_idx + abs_dmu " #+ prev_reward + sum_var + diff_var
    random = "(1 + computational + took_opt + computational:took_opt + trial_idx + abs_dmu  | subj_id)" #+ prev_reward + sum_var + diff_var
    formula = f"log_rt ~ {fixed} + {random}"

    mu_intercept = float(df_rv["log_rt"].mean())
    priors = {
        "Intercept": bmb.Prior("Normal", mu=mu_intercept, sigma=1.0),
        "Common":    bmb.Prior("Normal", mu=0.0, sigma=0.5),
        "Sigma":     bmb.Prior("HalfNormal", sigma=1.0),
    }

    model = bmb.Model(formula, df_rv, family="gaussian", priors=priors)
    idata = model.fit(
        draws=draws, tune=tune, chains=chains,
        target_accept=target_accept, random_seed=random_seed,
    )

    # Simple effect of computational when took_opt=1: Δ = β_comp + β_comp:took_opt
    beta_comp = _get_fixed_draws(idata, "computational")
    beta_int  = _get_fixed_draws(idata, "computational:took_opt")
    delta = beta_comp + beta_int
    ratio = np.exp(delta)  # multiplicative effect on (RT+1)

    info = {
        "reward_value": reward_value,
        "N_rows": int(df_rv.shape[0]),
        "p(ratio>1)": float((ratio > 1.0).mean()),
        "ratio_mean": float(ratio.mean()),
        "ratio_hdi_2.5%": float(az.hdi(ratio, hdi_prob=0.95)[0]),
        "ratio_hdi_97.5%": float(az.hdi(ratio, hdi_prob=0.95)[1]),
    }
    print(f"[Model A] reward_value={reward_value}: ratio mean={info['ratio_mean']:.3f}, p>1={info['p(ratio>1)']:.3f}")
    return idata, ratio, info

# -------------------------------------------------
# Model B: Among computational==1 & took_opt==1, slope of number_of_computations
# -------------------------------------------------

def fit_model_B_num_computations(
    df,
    reward_value: int,
    draws: int = 3000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.99,
    random_seed: int = 205,
):
    """
    Subset to computational==1 & took_opt==1, specific reward_value.
      FIXED:  number_of_computations + trial_idx + abs_dmu + prev_reward + sum_var + diff_var
      RANDOM: (1 + number_of_computations + trial_idx + abs_dmu + prev_reward + sum_var + diff_var | subj_id)
    """
    sub = df[(df["reward_value"] == reward_value) & (df["computational"] == 1) & (df["took_opt"] == 1)].copy()
    if sub.empty:
        raise ValueError(f"No rows for reward_value={reward_value} with computational==1 and took_opt==1.")

    sub["number_of_computations"] = pd.to_numeric(sub["number_of_computations"], errors="coerce").fillna(0.0)

    fixed = "number_of_computations + trial_idx + abs_dmu " #+ prev_reward + sum_var + diff_var
    random = "(1 + number_of_computations + trial_idx + abs_dmu  | subj_id)" #+ prev_reward + sum_var + diff_var
    formula = f"log_rt ~ {fixed} + {random}"

    mu_intercept = float(sub["log_rt"].mean())
    priors = {
        "Intercept": bmb.Prior("Normal", mu=mu_intercept, sigma=1.0),
        "Common":    bmb.Prior("Normal", mu=0.0, sigma=0.5),
        "Sigma":     bmb.Prior("HalfNormal", sigma=1.0),
    }

    model = bmb.Model(formula, sub, family="gaussian", priors=priors)
    idata = model.fit(
        draws=draws, tune=tune, chains=chains,
        target_accept=target_accept, random_seed=random_seed,
    )

    beta_nc = _get_fixed_draws(idata, "number_of_computations")
    ratio = np.exp(beta_nc)

    info = {
        "reward_value": reward_value,
        "N_rows": int(sub.shape[0]),
        "p(ratio>1)": float((ratio > 1.0).mean()),
        "ratio_mean": float(ratio.mean()),
        "ratio_hdi_2.5%": float(az.hdi(ratio, hdi_prob=0.95)[0]),
        "ratio_hdi_97.5%": float(az.hdi(ratio, hdi_prob=0.95)[1]),
    }
    print(f"[Model B] reward_value={reward_value}: ratio mean={info['ratio_mean']:.3f}, p>1={info['p(ratio>1)']:.3f}")
    return idata, ratio, info

# -------------------------------------------------
# Plotting
# -------------------------------------------------

def plot_posteriors_ratio(ratios_by_rv, title, ref_val=1.0):
    """
    ratios_by_rv: list of tuples (reward_value, ratio_array)
    Creates one figure with one posterior panel per reward_value.
    """
    k = len(ratios_by_rv)
    fig, axes = plt.subplots(k, 1, figsize=(8, 2.8 * k), constrained_layout=True)
    if k == 1:
        axes = [axes]
    for ax, (rv, ratio) in zip(axes, ratios_by_rv):
        az.plot_posterior(ratio, ref_val=ref_val, hdi_prob=0.95, ax=ax)
        p_gt1 = float((ratio > ref_val).mean())
        mean = float(np.mean(ratio))
        hdi = az.hdi(ratio, hdi_prob=0.95)
        ax.set_title(f"{title} — reward_value={rv}\n"
                     f"mean={mean:.3f}, 95% HDI=[{hdi[0]:.3f}, {hdi[1]:.3f}], P(ratio>{ref_val})={p_gt1:.3f}")
        ax.set_xlabel("Multiplicative effect on (RT + 1)")
    return fig, axes


# -------------------------------------------------
# Runner
# -------------------------------------------------

def run_all_and_plot(
    trial_level_csv: str,
    reward_values=(1, 5),
    draws=3000,
    tune=1000,
    chains=4,
    target_accept=0.99,
    seed=2025,
):
    feat = pd.read_csv(trial_level_csv)
    df = prepare(feat)

    # Model A
    ratios_A = []
    for rv in reward_values:
        _, ratio, _ = fit_model_A_compute_when_optimal(
            df,
            reward_value=rv,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
        )
        ratios_A.append((rv, ratio))
    figA, _ = plot_posteriors_ratio(ratios_A, title="Effect of being in a computational state when optimal (exp(Δ))")

    # Model B
    ratios_B = []
    for rv in reward_values:
        _, ratio, _ = fit_model_B_num_computations(
            df,
            reward_value=rv,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
        )
        ratios_B.append((rv, ratio))
    figB, _ = plot_posteriors_ratio(ratios_B, title="Effect per additional computation (exp(β_num_comp))")

    plt.show()
    return figA, figB


# -------------------------------------------------
# CLI
# -------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Bayesian mixed-effects RT analyses with plots.")
    ap.add_argument("--input", "-i", required=True, help="Path to trial-level features CSV")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target_accept", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    run_all_and_plot(
        trial_level_csv=args.input,
        reward_values=(1, 5),  # ignore 0 as requested
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
