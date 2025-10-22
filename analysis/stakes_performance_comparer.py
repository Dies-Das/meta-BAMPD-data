import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm, kruskal, ttest_ind, t

# ==========================================
# Config
# ==========================================
FEATURES_CSV = "trial_level_features_with_reward.csv"  # <-- set your path
RV_COL = "reward_value"
ISM_COL = "is_mismatch"
TOOK_COL = "took_opt"   # 1 if optimal, 0 if not, NaN if tie/undefined
H_COL = "horizon"
RT_COL = "rt"           # average raw RT as requested

MAX_UNIQUE_AS_LEVELS = 12  # if <= this many unique reward_values, treat as discrete levels
N_QUANTILE_BINS = 5        # otherwise, use this many quantile bins
ALPHA = 0.05

# ==========================================
# Helpers
# ==========================================
def wilson_ci(x, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = x / n
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    half = (z / denom) * np.sqrt(p*(1-p)/n + z**2/(4*n**2))
    return (center - half, center + half)

def two_prop_ztest(x1, n1, x2, n2):
    if min(n1, n2) == 0:
        return np.nan, np.nan
    p1, p2 = x1/n1, x2/n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return np.nan, np.nan
    z = (p1 - p2) / se
    p = 2 * norm.sf(abs(z))
    return z, p

def holm_bonferroni_adjust(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    sorted_p = np.asarray(pvals)[order]
    adj_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, p in enumerate(sorted_p, start=1):
        adj = (m - rank + 1) * p
        running_max = max(running_max, adj)
        adj_sorted[rank - 1] = min(1.0, running_max)
    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    return adj

def mean_ci_t(x, alpha=0.05):
    """Mean ± 95% t CI (two-sided). Returns (mean, sd, se, lo, hi)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan
    if n > 1:
        tcrit = t.ppf(1 - alpha/2, df=n-1)
        lo = mu - tcrit * se
        hi = mu + tcrit * se
    else:
        lo = hi = np.nan
    return (mu, sd, se, lo, hi)

def pairwise_welch_t(groups_dict):
    """Pairwise Welch t-tests (equal_var=False) + Holm correction."""
    labels = list(groups_dict.keys())
    rows = []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            a, b = np.asarray(groups_dict[labels[i]]), np.asarray(groups_dict[labels[j]])
            a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
            if len(a) < 2 or len(b) < 2:
                stat = np.nan; p = np.nan
            else:
                stat, p = ttest_ind(a, b, equal_var=False)
            rows.append({
                "bin_i": labels[i], "bin_j": labels[j],
                "n_i": len(a), "n_j": len(b),
                "mean_i": np.mean(a) if len(a) else np.nan,
                "mean_j": np.mean(b) if len(b) else np.nan,
                "diff": (np.mean(a) - np.mean(b)) if len(a) and len(b) else np.nan,
                "t_welch": stat, "p_raw": p
            })
    dfp = pd.DataFrame(rows)
    if not dfp.empty:
        dfp["p_holm"] = holm_bonferroni_adjust(dfp["p_raw"].fillna(1.0).to_numpy())
        dfp["significant_(holm_0.05)"] = dfp["p_holm"] < ALPHA
    return dfp

# ==========================================
# Load & global binning
# ==========================================
df = pd.read_csv(FEATURES_CSV)
for c in [RV_COL, ISM_COL, TOOK_COL, H_COL, RT_COL]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Define reward_value bins on ALL trials (global), so all analyses use identical bins
rv_values = df[RV_COL].dropna().to_numpy()
uniq = np.sort(np.unique(rv_values))
if len(uniq) <= MAX_UNIQUE_AS_LEVELS:
    rv_bins = pd.Categorical(df[RV_COL], categories=uniq, ordered=True)
else:
    rv_bins = pd.qcut(df[RV_COL], q=N_QUANTILE_BINS, duplicates="drop")
df = df.assign(rv_bin=rv_bins)

# ==========================================
# A) Optimal choice proportion within MISMATCH, stratified by horizon
# ==========================================
X_mis = df[(df[ISM_COL] == 1) & (~df[TOOK_COL].isna())].copy()
X_mis = X_mis[(X_mis[TOOK_COL] == 0) | (X_mis[TOOK_COL] == 1)].copy()
if X_mis.empty:
    print("\n[No mismatch trials with defined optimal action found.]")
else:
    horizons = sorted(X_mis[H_COL].dropna().unique().astype(int))
    for h in horizons:
        g_h = X_mis[X_mis[H_COL] == h].copy()
        if g_h.empty:
            continue
        gb = g_h.groupby("rv_bin", observed=True, sort=True)
        rows = []
        for level, g in gb:
            n = int(g.shape[0])
            x = int((g[TOOK_COL] == 1).sum())
            p_hat = x / n if n > 0 else np.nan
            sd_sample = float(g[TOOK_COL].std(ddof=1)) if n > 1 else np.nan
            sd_binom = np.sqrt(p_hat * (1 - p_hat)) if n > 0 else np.nan
            se = sd_binom / np.sqrt(n) if n > 0 else np.nan
            lo, hi = wilson_ci(x, n, z=1.96)
            rows.append(dict(
                rv_bin=str(level), n=n, n_optimal=x,
                prop_optimal=p_hat, sd_sample=sd_sample,
                sd_binom=sd_binom, se=se, ci95_lo=lo, ci95_hi=hi
            ))
        summary_h = pd.DataFrame(rows)

        with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
            print(f"\n=== Mismatch trials: Proportion optimal by reward_value bin — Horizon = {h} ===")
            print(summary_h.to_string(index=False))

        # Overall chi-square across reward bins
        if summary_h.shape[0] >= 2:
            succ = summary_h["n_optimal"].to_numpy()
            n = summary_h["n"].to_numpy()
            fail = n - succ
            contingency = np.vstack([succ, fail])
            from scipy.stats import chi2_contingency
            chi2, p_chi2, dof, _ = chi2_contingency(contingency, correction=False)
            print(f"\n[H={h}] Overall across reward bins (optimal-in-mismatch): chi2({dof}) = {chi2:.3f}, p = {p_chi2:.4g}")
        else:
            print(f"\n[H={h}] Overall chi-square not applicable (fewer than 2 bins).")

        # Pairwise across reward bins
        labels = summary_h["rv_bin"].tolist()
        K = len(labels)
        pairs = []
        for i in range(K):
            for j in range(i+1, K):
                x1, n1 = int(succ[i]), int(n[i])
                x2, n2 = int(succ[j]), int(n[j])
                z = (x1/n1 - x2/n2) / np.sqrt(((x1+x2)/(n1+n2))*(1 - (x1+x2)/(n1+n2))*(1/n1 + 1/n2))
                p_raw = 2 * norm.sf(abs(z))
                pairs.append({
                    "bin_i": labels[i], "bin_j": labels[j],
                    "n_i": n1, "n_j": n2,
                    "prop_i": x1/n1 if n1>0 else np.nan,
                    "prop_j": x2/n2 if n2>0 else np.nan,
                    "diff": (x1/n1 - x2/n2) if (n1>0 and n2>0) else np.nan,
                    "z": z, "p_raw": p_raw
                })
        pairs_df = pd.DataFrame(pairs)
        if not pairs_df.empty:
            pairs_df["p_holm"] = holm_bonferroni_adjust(pairs_df["p_raw"].fillna(1.0).to_numpy())
            pairs_df["significant_(holm_0.05)"] = pairs_df["p_holm"] < ALPHA
            with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
                print(f"\n[H={h}] Pairwise two-proportion z-tests (Holm-corrected):")
                print(pairs_df.sort_values("p_holm").to_string(index=False))
        else:
            print(f"\n[H={h}] No pairwise comparisons available.")

# ==========================================
# B) Average RT across ALL trials — overall (by reward_value bins)
# ==========================================
X_all = df.dropna(subset=[RT_COL, RV_COL]).copy()
gb_all = X_all.groupby("rv_bin", observed=True, sort=True)

rows = []
groups_for_tests = {}
for level, g in gb_all:
    rt = pd.to_numeric(g[RT_COL], errors="coerce").to_numpy()
    rt = rt[~np.isnan(rt)]
    n = int(rt.size)
    mu, sd, se, lo, hi = mean_ci_t(rt, alpha=0.05)
    rows.append(dict(rv_bin=str(level), n=n, mean_rt=mu, sd_rt=sd, se_rt=se, ci95_lo=lo, ci95_hi=hi))
    groups_for_tests[str(level)] = rt

summary_rt_overall = pd.DataFrame(rows)
with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
    print("\n=== ALL trials: Average RT by reward_value bin (overall across horizons) ===")
    print(summary_rt_overall.to_string(index=False))

# Overall difference across bins (Kruskal–Wallis; robust to non-normality)
valid_groups = [v for v in groups_for_tests.values() if len(v) > 0]
if len(valid_groups) >= 2:
    stat, p_kw = kruskal(*valid_groups)
    print(f"\n[Overall] Kruskal–Wallis across reward bins (RT): H = {stat:.3f}, p = {p_kw:.4g}")
else:
    print("\n[Overall] Kruskal–Wallis not applicable (fewer than 2 non-empty bins).")

# Pairwise Welch t-tests with Holm correction
pw_overall = pairwise_welch_t(groups_for_tests)
if not pw_overall.empty:
    with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
        print("\n[Overall] Pairwise Welch t-tests on RT (Holm-corrected):")
        print(pw_overall.sort_values("p_holm").to_string(index=False))

# ==========================================
# C) Average RT across ALL trials — stratified by horizon
# ==========================================
horizons_all = sorted(X_all[H_COL].dropna().unique().astype(int))
for h in horizons_all:
    g_h = X_all[X_all[H_COL] == h].copy()
    gb_h = g_h.groupby("rv_bin", observed=True, sort=True)

    rows_h = []
    groups_h = {}
    for level, g in gb_h:
        rt = pd.to_numeric(g[RT_COL], errors="coerce").to_numpy()
        rt = rt[~np.isnan(rt)]
        n = int(rt.size)
        mu, sd, se, lo, hi = mean_ci_t(rt, alpha=0.05)
        rows_h.append(dict(rv_bin=str(level), n=n, mean_rt=mu, sd_rt=sd, se_rt=se, ci95_lo=lo, ci95_hi=hi))
        groups_h[str(level)] = rt

    summary_rt_h = pd.DataFrame(rows_h)
    with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
        print(f"\n=== ALL trials: Average RT by reward_value bin — Horizon = {h} ===")
        print(summary_rt_h.to_string(index=False))

    # Kruskal–Wallis within-horizon
    valid_h = [v for v in groups_h.values() if len(v) > 0]
    if len(valid_h) >= 2:
        stat, p_kw = kruskal(*valid_h)
        print(f"\n[H={h}] Kruskal–Wallis across reward bins (RT): H = {stat:.3f}, p = {p_kw:.4g}")
    else:
        print(f"\n[H={h}] Kruskal–Wallis not applicable (fewer than 2 non-empty bins).")

    # Pairwise Welch t-tests within-horizon
    pw_h = pairwise_welch_t(groups_h)
    if not pw_h.empty:
        with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
            print(f"\n[H={h}] Pairwise Welch t-tests on RT (Holm-corrected):")
            print(pw_h.sort_values("p_holm").to_string(index=False))

