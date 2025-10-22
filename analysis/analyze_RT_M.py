# bandit_mstate_rt_analysis.py
import itertools
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

# Statsmodels for inference & p-values
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ----------------------------
# Dynamic programming (Bayes-optimal vs greedy) utilities
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
        self.opt_by_s = opt_by_s          # dict[(s_left, st)] -> {0,1,-1}
        self.greedy = greedy              # dict[st] -> {0,1,-1}
        self.mismatch_by_s = mismatch_by_s  # set of (s_left, st)
        self.qgap_by_s = qgap_by_s        # dict[(s_left, st)] -> |Q1 - Q2|



@lru_cache(maxsize=None)
def compute_policies_for_horizon(T: int) -> PolicyPackS:
    """
    Bayes-optimal (finite-horizon) + greedy, indexed by steps_left.
    Returns a PolicyPackS that matches what reconstruct_states_and_labels() expects.
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
            fail2 = (a1, b1, a2, b2 + 1)   

            Q1 = p1 * (1 + V[s - 1][succ1]) + (1 - p1) * V[s - 1][fail1]
            Q2 = p2 * (1 + V[s - 1][succ2]) + (1 - p2) * V[s - 1][fail2]

            V[s][st] = max(Q1, Q2)
            opt_by_s[(s, st)] = 0 if Q1 > Q2 else 1 if Q2 > Q1 else -1
            qgap_by_s[(s, st)] = abs(Q1 - Q2)

    # greedy is step-independent
    greedy = {}
    for lst in S:
        for st in lst:
            a1, b1, a2, b2 = st
            m1 = a1 / (a1 + b1)
            m2 = a2 / (a2 + b2)
            greedy[st] = 0 if m1 > m2 else 1 if m2 > m1 else -1

    mismatch_by_s = {(s, st) for (s, st), oa in opt_by_s.items() if oa != greedy.get(st, -1)}
    return PolicyPackS(T, opt_by_s, greedy, mismatch_by_s, qgap_by_s)


#This creates the features and the annotated dataset
def reconstruct_states_and_labels(df_main: pd.DataFrame) -> pd.DataFrame:
    df = df_main.copy()
    df["arm"] = df["arm"].astype(int)
    df = df.sort_values(["subj_id","block_idx","trial_idx"]).reset_index(drop=True)

    policies = {int(T): compute_policies_for_horizon(int(T)) for T in sorted(df["horizon"].unique())}

    rows = []
    for (sid, block), g in df.groupby(["subj_id","block_idx"], sort=False):
        g = g.sort_values("trial_idx")
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
        prev_mean_chosen = None  # mean of chosen arm BEFORE outcome at t-1

        for i, r in enumerate(g.itertuples(index=False), start=1):
            # state BEFORE action
            a0, b0 = 1 + s[0], 1 + f[0]
            a1, b1 = 1 + s[1], 1 + f[1]
            st = (a0, b0, a1, b1)

            steps_left = T - (i - 1)

            m0 = a0 / (a0 + b0)
            m1 = a1 / (a1 + b1)
            abs_dmu = abs(m0 - m1)

            # Uncertainty (Beta variances)
            var0 = (a0 * b0) / (((a0 + b0) ** 2) * (a0 + b0 + 1))
            var1 = (a1 * b1) / (((a1 + b1) ** 2) * (a1 + b1 + 1))
            sum_var = var0 + var1
            diff_var = abs(var0 - var1)

            # Optimal/greedy, mismatch and its strength (Q-gap) at this steps_left
            opt_act = ppack.opt_by_s.get((steps_left, st), -1)
            greedy_act = ppack.greedy.get(st, -1)
            is_mismatch = int((steps_left, st) in ppack.mismatch_by_s)
            mismatch_strength = ppack.qgap_by_s.get((steps_left, st), 0.0)

            # Did the subject take the optimal action? (NaN if tie)
            took_opt = np.nan if opt_act == -1 else int(r.arm == opt_act)

            # Switch indicator for this trial (depends on previous arm)
            switch = 1 if (prev_arm is not None and r.arm != prev_arm) else 0

            # lag log RT
            lag_log_rt = np.log(prev_rt) if (prev_rt is not None and prev_rt > 0) else 0.0

            rows.append(dict(
                subj_id=r.subj_id,
                block_idx=r.block_idx,
                horizon=T,
                p0=float(r.p0), p1=float(r.p1),
                trial_idx=int(r.trial_idx),
                arm=int(r.arm), reward=int(r.reward), rt=float(r.rt),
                # state
                a0=a0,b0=b0,a1=a1,b1=b1,
                steps_left=steps_left,
                # labels
                opt_act=opt_act, greedy_act=greedy_act,
                is_mismatch=is_mismatch, took_opt=took_opt,
                mismatch_strength=mismatch_strength,
                # covariates
                abs_dmu=abs_dmu, abs_dtrue=abs(float(r.p0)-float(r.p1)),
                var0=var0, var1=var1, sum_var=sum_var, diff_var=diff_var,
                switch=switch, prev_switch=prev_switch, prev_mstate=prev_mstate,
                prev_reward=prev_rew, prev_surprise=prev_surprise,
                lag_log_rt=lag_log_rt
            ))

            # --- Update for next trial ---
            # Mean of CHOSEN arm BEFORE outcome (used for surprise next trial)
            mean_chosen_pre = m0 if r.arm == 0 else m1

            # Update posterior counts with current outcome
            if r.arm == 0:
                if r.reward == 1: s[0] += 1
                else:             f[0] += 1
            else:
                if r.reward == 1: s[1] += 1
                else:             f[1] += 1

            # carryover trackers
            prev_mstate = is_mismatch
            prev_switch = switch
            prev_rew = int(r.reward)
            prev_arm = int(r.arm)
            prev_rt = float(r.rt)
            # surprise from this trial applied on the next one
            prev_surprise = abs(prev_rew - mean_chosen_pre)
            prev_mean_chosen = mean_chosen_pre

    feat = pd.DataFrame(rows)
    return feat



# ----------------------------
# Per-subject regression & hypothesis tests
# ----------------------------

import statsmodels.formula.api as smf

def fit_subject_models(feat: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Per-subject OLS (clustered by block), no block fixed effects.
    Adds availability counts:
      n_M, n_M_opt1, n_M_opt0, n_opt1, n_opt0
    """
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf

    results = []
    feat = feat.copy()
    feat = feat[~feat["rt"].isna()]
    feat["log_rt"] = np.log(feat["rt"].clip(lower=1e-6))
    feat["trial_idx_sq"] = feat["trial_idx"] ** 1
    
    # --- NEW: drop the top 5% RTs per subject (based on raw RT) ---
    q95 = feat.groupby("subj_id")["rt"].quantile(0.95).rename("rt_q95")
    feat = feat.join(q95, on="subj_id")
    feat = feat[feat["rt"] < feat["rt_q95"]].copy()   # use "<" ; switch to ">=" to be stricter
    feat.drop(columns=["rt_q95"], inplace=True)

    # Drop tie trials (opt undefined) for "took_opt"
    model_df = feat[~feat["took_opt"].isna()].copy()

    for sid, g in model_df.groupby("subj_id", sort=False):
        # ---- Availability counts for diagnostics
        n_obs      = int(g.shape[0])
        n_M        = int((g["is_mismatch"] == 1).sum())
        n_M_opt1   = int(((g["is_mismatch"] == 1) & (g["took_opt"] == 1)).sum())
        n_M_opt0   = int(((g["is_mismatch"] == 1) & (g["took_opt"] == 0)).sum())
        n_opt1     = int((g["took_opt"] == 1).sum())
        n_opt0     = int((g["took_opt"] == 0).sum())

        if g["is_mismatch"].nunique() < 2:
            results.append(dict(
                subj_id=sid, n_obs=n_obs,
                n_M=n_M, n_M_opt1=n_M_opt1, n_M_opt0=n_M_opt0, n_opt1=n_opt1, n_opt0=n_opt0,
                beta_M=np.nan, p_M=np.nan,
                beta_inter=np.nan, p_inter=np.nan,
                beta_M_plus_inter=np.nan, p_M_plus_inter=np.nan,
                supported=False,
                note="No variation in M-state for this subject"
            ))
            continue

        formula = (
            "log_rt ~ is_mismatch + took_opt + is_mismatch:took_opt + sum_var + diff_var"
            "+ trial_idx + abs_dmu + prev_reward"   
        )

        try:
            model = smf.ols(formula, data=g).fit()
        except Exception as e:
            results.append(dict(
                subj_id=sid, n_obs=n_obs,
                n_M=n_M, n_M_opt1=n_M_opt1, n_M_opt0=n_M_opt0, n_opt1=n_opt1, n_opt0=n_opt0,
                beta_M=np.nan, p_M=np.nan,
                beta_inter=np.nan, p_inter=np.nan,
                beta_M_plus_inter=np.nan, p_M_plus_inter=np.nan,
                supported=False,
                note=f"Model failed: {e}"
            ))
            continue

        params, pvals = model.params, model.pvalues

        beta_M = params.get("is_mismatch", np.nan)
        p_M = pvals.get("is_mismatch", np.nan)

        beta_inter = params.get("is_mismatch:took_opt", np.nan)
        p_inter = pvals.get("is_mismatch:took_opt", np.nan)

        # Simple effect in optimal trials: beta_M + beta_inter
        L = np.zeros(len(params))
        name_to_idx = {name: i for i, name in enumerate(params.index)}
        if "is_mismatch" in name_to_idx:
            L[name_to_idx["is_mismatch"]] = 1.0
        if "is_mismatch:took_opt" in name_to_idx:
            L[name_to_idx["is_mismatch:took_opt"]] = 1.0

        try:
            # future-proof & silence warning
            wald = model.wald_test(L, scalar=True)
            p_M_plus_inter = float(wald.pvalue)
        except Exception:
            p_M_plus_inter = np.nan

        beta_M_plus_inter = (
            (beta_M if pd.notna(beta_M) else 0.0)
            + (beta_inter if pd.notna(beta_inter) else 0.0)
        )

        supported = (beta_M_plus_inter > 0) and (pd.notna(p_M_plus_inter) and p_M_plus_inter < alpha)

        results.append(dict(
            subj_id=sid,
            n_obs=n_obs,
            n_M=n_M, n_M_opt1=n_M_opt1, n_M_opt0=n_M_opt0, n_opt1=n_opt1, n_opt0=n_opt0,
            beta_M=float(beta_M) if pd.notna(beta_M) else np.nan,
            p_M=float(p_M) if pd.notna(p_M) else np.nan,
            beta_inter=float(beta_inter) if pd.notna(beta_inter) else np.nan,
            p_inter=float(p_inter) if pd.notna(p_inter) else np.nan,
            beta_M_plus_inter=float(beta_M_plus_inter) if pd.notna(beta_M_plus_inter) else np.nan,
            p_M_plus_inter=float(p_M_plus_inter) if pd.notna(p_M_plus_inter) else np.nan,
            supported=bool(supported),
            note=""
        ))

    return pd.DataFrame(results)
    
    

# ----------------------------
# Main entry point
# ----------------------------

def main(input_csv="bandit-data-aug24.csv",
         stage_filter="main",
         alpha=0.10,
         out_feat_csv="trial_level_features.csv",
         out_results_csv="subject_level_results.csv"):
    # Load and filter
    df = pd.read_csv(input_csv, sep=None, engine="python")
    # Expected cols:
    # subj_id, stage, block_idx, horizon, p0, p1, trial_idx, arm, reward, rt

    df = df[df["stage"] == stage_filter].copy()

    # Basic hygiene & types
    for col in ["block_idx", "horizon", "trial_idx", "arm", "reward"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["p0", "p1", "rt"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["subj_id", "block_idx", "horizon", "trial_idx", "arm", "reward", "rt", "p0", "p1"]).copy()
    df["block_idx"] = df["block_idx"].astype(int)
    df["horizon"] = df["horizon"].astype(int)
    df["trial_idx"] = df["trial_idx"].astype(int)
    df["arm"] = df["arm"].astype(int)
    df["reward"] = df["reward"].astype(int)

    # Reconstruct decision states & labels
    feat = reconstruct_states_and_labels(df)

    # Save per-trial features (useful for auditing)
    feat.to_csv(out_feat_csv, index=False)

    # Fit per-subject models
    results = fit_subject_models(feat, alpha=alpha)

    # Save results
    results.to_csv(out_results_csv, index=False)

    # Console summary
    n_subj = results["subj_id"].nunique()
    n_supported = int(results["supported"].sum())
    print(f"Analyzed {n_subj} subjects. Theory supported (M-state effect when optimal) for {n_supported}/{n_subj} subjects at alpha={alpha}.")
    print("\nTop 10 rows of results:")
    print(results.head(35).to_string(index=False))
    
    


if __name__ == "__main__":
    # You can change filenames here if needed
    main()




















'''
## Other functions for regression

def fit_subject_models_extended(
    feat: pd.DataFrame,
    alpha: float = 0.05,
    min_mopt1: int = 5,      # require at least this many M&Opt=1 trials
    outlier_z: float | None = 3.5,  # set to None to disable MAD filter
    top_pct: float = 0.05,    # exclude top 5% RTs PER SUBJECT
) -> pd.DataFrame:
    """
    Plain OLS per subject (no clustering, no block fixed effects).
    Filters:
      1) Drop trials with undefined optimal action (ties).
      2) Exclude per-subject top `top_pct` (default 5%) of RTs (raw RT).
      3) Optionally exclude log-RT outliers via MAD (if outlier_z is not None).

    Model:
      log_rt ~ trial_idx + trial_idx_sq + abs_dmu
               + horizon + abs_dtrue
               + switch + sum_var + diff_var
               + mismatch_strength
               + is_mismatch + took_opt + is_mismatch:took_opt
               + prev_reward + prev_surprise + prev_switch + prev_mstate + lag_log_rt
    """
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf

    # --- Base features
    X = feat.copy()
    X = X[~X["rt"].isna()].copy()
    X["log_rt"] = np.log(X["rt"].clip(lower=1e-6))
    X["trial_idx_sq"] = X["trial_idx"] ** 2

    # 0) Remove trials where optimal action is undefined (ties)
    X = X[~X["took_opt"].isna()].copy()

    # 1) Exclude top 5% RTs per subject (raw RT)
    q = X.groupby("subj_id")["rt"].quantile(1.0 - top_pct).rename("rt_q95")
    X = X.join(q, on="subj_id")
    X["is_top5"] = X["rt"] >= X["rt_q95"]
    n_obs_before = X.groupby("subj_id").size().to_dict()
    n_top5 = X.groupby("subj_id")["is_top5"].sum().astype(int).to_dict()
    X = X[~X["is_top5"]].copy()
    X.drop(columns=["rt_q95", "is_top5"], inplace=True)

    # 2) Optional MAD-based log-RT outlier filter (per subject)
    if outlier_z is not None:
        subj_stats = X.groupby("subj_id")["log_rt"].agg(
            med="median",
            mad=lambda s: (s - s.median()).abs().median()
        )
        subj_stats["madn"] = 1.4826 * subj_stats["mad"]
        X = X.join(subj_stats, on="subj_id")

        diff = (X["log_rt"] - X["med"]).abs()
        X["is_outlier"] = (X["madn"] > 0) & ((diff / X["madn"]) > float(outlier_z))
        n_outliers = X.groupby("subj_id")["is_outlier"].sum().astype(int).to_dict()

        X = X[~X["is_outlier"]].copy()
        X.drop(columns=["med", "mad", "madn", "is_outlier"], inplace=True)
    else:
        n_outliers = {sid: 0 for sid in X["subj_id"].unique()}

    # --- Regression (no clustering)
    formula = (
        "log_rt ~ trial_idx + trial_idx_sq + abs_dmu "
        "+ horizon + abs_dtrue "
        "+ switch + sum_var + diff_var "
        "+ mismatch_strength "
        "+ is_mismatch + took_opt + is_mismatch:took_opt "
        "+ prev_reward + prev_surprise + prev_switch + prev_mstate + lag_log_rt"
    )


    rows = []
    for sid, g in X.groupby("subj_id", sort=False):
        n_before = int(n_obs_before.get(sid, 0))
        n_top = int(n_top5.get(sid, 0))
        n_out = int(n_outliers.get(sid, 0))
        n_obs = int(g.shape[0])

        # Availability after filtering
        n_M = int((g["is_mismatch"] == 1).sum())
        n_M_opt1 = int(((g["is_mismatch"] == 1) & (g["took_opt"] == 1)).sum())
        n_M_opt0 = int(((g["is_mismatch"] == 1) & (g["took_opt"] == 0)).sum())
        n_opt1 = int((g["took_opt"] == 1).sum())
        n_opt0 = int((g["took_opt"] == 0).sum())

        # Inclusion rule
        if n_M_opt1 < min_mopt1:
            rows.append(dict(
                subj_id=sid, included=False,
                n_obs_before=n_before, n_top5=n_top, n_outliers=n_out, n_obs=n_obs,
                n_M=n_M, n_M_opt1=n_M_opt1, n_M_opt0=n_M_opt0, n_opt1=n_opt1, n_opt0=n_opt0,
                beta_inter=np.nan, p_inter=np.nan,
                beta_M_plus_inter=np.nan, p_M_plus_inter=np.nan,
                supported=False,
                note=f"Excluded: M&Opt=1 trials < {min_mopt1}"
            ))
            continue

        try:
            model = smf.ols(formula, data=g).fit()

            beta_int = model.params.get("is_mismatch:took_opt", np.nan)
            p_int = model.pvalues.get("is_mismatch:took_opt", np.nan)
            beta_M  = model.params.get("is_mismatch", np.nan)

            # Simple effect of M when took_opt=1: beta_M + beta_int
            import numpy as np
            L = np.zeros(len(model.params))
            idx = {nm:i for i, nm in enumerate(model.params.index)}
            if "is_mismatch" in idx: L[idx["is_mismatch"]] = 1.0
            if "is_mismatch:took_opt" in idx: L[idx["is_mismatch:took_opt"]] = 1.0
            wald = model.wald_test(L, scalar=True)
            p_simple = float(wald.pvalue)
            beta_simple = float(
                (0.0 if pd.isna(beta_M) else beta_M) +
                (0.0 if pd.isna(beta_int) else beta_int)
            )
            supported = (beta_simple > 0) and (p_simple < alpha)
            note = ""
        except Exception as e:
            beta_int = p_int = beta_simple = p_simple = np.nan
            supported = False
            note = f"Model failed: {e}"

        rows.append(dict(
            subj_id=sid, included=True,
            n_obs_before=n_before, n_top5=n_top, n_outliers=n_out, n_obs=n_obs,
            n_M=n_M, n_M_opt1=n_M_opt1, n_M_opt0=n_M_opt0, n_opt1=n_opt1, n_opt0=n_opt0,
            beta_inter=beta_int if pd.notna(beta_int) else np.nan,
            p_inter=p_int if pd.notna(p_int) else np.nan,
            beta_M_plus_inter=beta_simple if pd.notna(beta_simple) else np.nan,
            p_M_plus_inter=p_simple if pd.notna(p_simple) else np.nan,
            supported=bool(supported),
            note=note
        ))

    return pd.DataFrame(rows)





def fit_subject_models_minimal(feat: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Per-subject OLS of log(rt) on:
      trial_idx, trial_idx_sq, abs_dmu, and mstate_opt1 (1 if M-state & took_opt=1).
    No block FE, no prev_reward. SEs clustered by block. First trial per block removed.
    Returns subject-level coefficients, p-values, and availability counts.
    """
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf

    out = []
    X = feat.copy()

    # basic features
    X = X[~X["rt"].isna()].copy()
    X["log_rt"] = np.log(X["rt"].clip(lower=1e-6))
    X["trial_idx_sq"] = X["trial_idx"] ** 1

    # drop first trial per block (redundant with tie-drop, but explicit)
    first_in_block = X.groupby(["subj_id", "block_idx"])["trial_idx"].transform("min")
    X = X[X["trial_idx"] != first_in_block].copy()

    # drop tie trials (opt undefined)
    X = X[~X["took_opt"].isna()].copy()

    # target regressor: M-state & optimal
    X["mstate_opt1"] = ((X["is_mismatch"] == 1) & (X["took_opt"] == 1)).astype(int)

    for sid, g in X.groupby("subj_id", sort=False):
        n_obs = int(g.shape[0])
        n_mopt1 = int(g["mstate_opt1"].sum())

        # Need variation in mstate_opt1 (both 0 and 1) to estimate its coefficient
        varies = g["mstate_opt1"].nunique() == 2

        if not varies:
            out.append(dict(
                subj_id=sid, n_obs=n_obs, n_mopt1=n_mopt1,
                beta_mopt1=np.nan, p_mopt1=np.nan, supported=False,
                note="No variation in M&Opt=1 regressor"
            ))
            continue

        try:
            model = smf.ols(
                "log_rt ~ trial_idx + trial_idx_sq + abs_dmu + mstate_opt1",
                data=g
            ).fit(cov_type="cluster", cov_kwds={"groups": g["block_idx"]})
            beta = model.params.get("mstate_opt1", np.nan)
            pval = model.pvalues.get("mstate_opt1", np.nan)
            supported = (beta > 0) and (pd.notna(pval) and pval < alpha)
            note = ""
        except Exception as e:
            beta, pval, supported, note = np.nan, np.nan, False, f"Model failed: {e}"

        out.append(dict(
            subj_id=sid,
            n_obs=n_obs,
            n_mopt1=n_mopt1,
            beta_mopt1=float(beta) if pd.notna(beta) else np.nan,
            p_mopt1=float(pval) if pd.notna(pval) else np.nan,
            supported=bool(supported),
            note=note
        ))

    return pd.DataFrame(out)
'''
