import os
import numpy as np
import pandas as pd
import arviz as az
import bambi as bmb

# -------------------------------------------------
# Preprocess (now includes sum_var, diff_var)
# -------------------------------------------------

def prepare(feat: pd.DataFrame, rt_cap_ms: int = 5000) -> pd.DataFrame:
    """
    Paper-like RT handling but no z-scoring or scaling:
      - cap RT at 5000 ms
      - add 1 ms to avoid log(0)
      - log-transform
      - (currently) rows with took_opt == NaN are dropped; took_opt == -1 is kept
    Predictors kept on their original scales:
      is_mismatch ∈ {0,1}
      took_opt    ∈ {-1,0,1} or {0,1}
      trial_idx   as-is
      abs_dmu     as-is (0..1)
      prev_reward ∈ {0,1}
      sum_var     ≥ 0 (total uncertainty)
      diff_var    ≥ 0 (relative uncertainty)
    """
    df = feat.copy()
    req = [
        "subj_id", "rt",
        "is_mismatch", "took_opt",
        "trial_idx", "abs_dmu", "prev_reward",
        "sum_var", "diff_var"    # <— NEW required fields
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["subj_id"] = df["subj_id"].astype(str)

    # RT transform
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
    df = df.dropna(subset=["rt"])
    df["rt_capped"] = np.minimum(df["rt"].values, float(rt_cap_ms))
    df["log_rt"] = np.log(df["rt_capped"] + 1.0)  # +1 ms

    # Ensure predictors are numeric
    for c in ["is_mismatch","took_opt","trial_idx","abs_dmu","prev_reward","sum_var","diff_var"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # NOTE:
    # - This drops rows where took_opt is NaN (ties encoded as NaN will be dropped).
    # - took_opt == -1 is kept (not NaN).
    df = df.dropna(subset=["is_mismatch","took_opt","trial_idx","abs_dmu","prev_reward","sum_var","diff_var"])

    return df


# -------------------------------------------------
# Bayesian mixed-effects fit (adds sum_var & diff_var)
# -------------------------------------------------

def fit_bayesian_mixed_rt_nostd(
    df: pd.DataFrame,
    draws: int = 3000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.99,
    random_seed: int = 205,
    out_prefix: str = "bayes_rt_nostd",
):
    """
    Gaussian mixed model on log-RT:
      Fixed effects:  is_mismatch * took_opt + trial_idx + abs_dmu + prev_reward + sum_var + diff_var
      Random effects: maximal by subject (intercept + slopes for all fixed effects)
    """
    formula = (
       
        "log_rt ~ (1 + is_mismatch + took_opt + is_mismatch:took_opt + trial_idx + abs_dmu + prev_reward + sum_var + diff_var | subj_id)"
    )

    mu_intercept = float(df["log_rt"].mean())

    priors = {
        "Intercept": bmb.Prior("Normal", mu=mu_intercept, sigma=1.0),
        "Common":    bmb.Prior("Normal", mu=0.0, sigma=0.5),  # weakly-informative on log-RT slopes
        "Sigma":     bmb.Prior("HalfNormal", sigma=1.0),
        # Group-specific priors left at Bambi defaults (reasonable shrinkage)
    }

    model = bmb.Model(formula, df, family="gaussian", priors=priors)
    idata = model.fit(
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=random_seed,
    )

    # Simple effect: M-state when optimal (0→1 at took_opt=1) = β_mismatch + β_interaction
    post = idata.posterior
    beta_m   = post["is_mismatch"].values
    beta_int = post["is_mismatch:took_opt"].values
    delta = (beta_m + beta_int).reshape(-1)

    hdi = az.hdi(delta, hdi_prob=0.95)
    prob_gt0 = float((delta > 0).mean())

    ratio = np.exp(delta)  # ≈ multiplicative change in (RT+1)
    hdi_ratio = az.hdi(ratio, hdi_prob=0.95)

    simple_effect = {
        "delta_logRT_mean": float(delta.mean()),
        "delta_logRT_hdi_2.5%": float(hdi[0]),
        "delta_logRT_hdi_97.5%": float(hdi[1]),
        "p(Δ>0)": prob_gt0,
        "RT_ratio_mean": float(ratio.mean()),
        "RT_ratio_hdi_2.5%": float(hdi_ratio[0]),
        "RT_ratio_hdi_97.5%": float(hdi_ratio[1]),
    }

    os.makedirs("bayes_outputs", exist_ok=True)
    summ = az.summary(idata, round_to=4)
    summ.to_csv(os.path.join("bayes_outputs", f"{out_prefix}_posterior_summary.csv"))
    idata.to_netcdf(os.path.join("bayes_outputs", f"{out_prefix}.nc"))

    print("\nPosterior summary (top rows):")
    print(summ.head(20).to_string())
    print("\nSimple effect: M-state when optimal (is_mismatch 0→1 at took_opt=1)")
    print(simple_effect)

    return model, idata, simple_effect



# -------------------------------------------------
# Convenience runner from your features CSV
# -------------------------------------------------

def run_from_feat_csv(
    trial_level_csv="trial_level_features.csv",
    draws=3000,
    tune=1000,
    chains=4,
    target_accept=0.99,
    seed=2025,
    out_prefix="bayes_rt_nostd",
):
    feat = pd.read_csv(trial_level_csv)

    # If you prefer to rebuild from raw trials, drop your reconstruct here:
    # raw = pd.read_csv("bandit-data-aug24.csv")
    # feat = reconstruct_states_and_labels(raw)

    df = prepare(feat)
    return fit_bayesian_mixed_rt_nostd(
        df,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=seed,
        out_prefix=out_prefix,
    )


if __name__ == "__main__":
    run_from_feat_csv()

