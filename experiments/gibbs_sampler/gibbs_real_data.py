"""Finite Bayesian Plackett-Luce Gibbs sampler on real MovieLens top-10 lists
(Caron, Teh & Murphy, Sec. 3.1).

Inputs:
    - data/processed/preference_lists_top10.csv
    - data/processed/catalogue.csv

Outputs are saved under ``results/gibbs_sampler/real/``.
"""

import ast
import csv
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")  # Works in terminals/WSL without graphical windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. SETTINGS AND REAL DATA LOADING
# ============================================================
alpha_prior = 10.0
tau_prior = 1.0  # Gamma rate, NOT scale (cancels out upon normalization)

n_chains = 4
n_iters = 6000
burn_in = 2000
n_predictive_replicates = 200
top_k_forest = 25
top_k_trace = 6
top_k_acf = 8

PROJECT_ROOT = Path(__file__).resolve().parents[2]
data_dir = PROJECT_ROOT / "data" / "processed"
preferences_path = data_dir / "preference_lists_top10.csv"
catalogue_path = data_dir / "catalogue.csv"

output_dir = PROJECT_ROOT / "results" / "gibbs_sampler" / "real"
output_dir.mkdir(parents=True, exist_ok=True)

# Load preference lists and map non-contiguous movieIds -> 0..M-1
prefs_df = pd.read_csv(preferences_path)
parsed_lists = prefs_df["top_movies"].apply(ast.literal_eval).tolist()

unique_movie_ids = np.array(
    sorted({int(mid) for row in parsed_lists for mid in row}), dtype=np.int64
)
id_to_idx = {int(mid): idx for idx, mid in enumerate(unique_movie_ids)}

rankings = np.array(
    [[id_to_idx[int(mid)] for mid in row] for row in parsed_lists], dtype=np.int64
)
L, m = rankings.shape
M = len(unique_movie_ids)

# Load movie titles and metadata from catalogue.csv
cat_df = pd.read_csv(catalogue_path).drop_duplicates(subset=["movieId"])
cat_indexed = cat_df.set_index("movieId")

titles = []
years = []
genres_list = []
short_labels = []

for mid in unique_movie_ids:
    mid_int = int(mid)
    if mid_int in cat_indexed.index:
        row = cat_indexed.loc[mid_int]
        title = str(row["title"])
        year = str(int(row["year"])) if pd.notna(row["year"]) else "N/A"
        genres = str(row["genres"]) if pd.notna(row["genres"]) else ""
    else:
        title = f"Movie {mid_int}"
        year = "N/A"
        genres = ""

    titles.append(title)
    years.append(year)
    genres_list.append(genres)

    clean_title = title if len(title) <= 30 else title[:27] + "..."
    short_labels.append(f"{clean_title} ({year})" if year != "N/A" else clean_title)

print(f"Loaded rankings shape: {rankings.shape} (L={L} users, top-m={m}, M={M} unique movies)")


# ============================================================
# 2. VECTORIZED FINITE GIBBS SAMPLER
# ============================================================
t_start = time.perf_counter()
counts = np.bincount(rankings.ravel(), minlength=M)
posterior_shapes = alpha_prior / M + counts
samples = np.empty((n_chains, n_iters, M), dtype=np.float64)


def gibbs_chain(rng):
    current_w = rng.lognormal(mean=-0.7, sigma=0.7, size=M)
    chain = np.empty((n_iters, M), dtype=np.float64)
    tiny = np.finfo(np.float64).tiny

    for it in range(n_iters):
        # Z_li | w, rankings ~ Exp(sum of weights still available at step i)
        selected_w = current_w[rankings]
        previous_w = np.concatenate(
            [np.zeros((L, 1)), np.cumsum(selected_w[:, :-1], axis=1)], axis=1
        )
        total_w = current_w.sum()
        rates_Z = total_w - previous_w

        # Cancellation guard: recompute tiny remaining sums in log-space.
        fragile = rates_Z <= np.finfo(np.float64).eps * total_w
        if np.any(fragile):
            for ell, i in np.argwhere(fragile):
                active = np.ones(M, dtype=bool)
                active[rankings[ell, :i]] = False
                rates_Z[ell, i] = np.exp(np.logaddexp.reduce(np.log(current_w[active])))
        if np.any(rates_Z <= 0) or not np.all(np.isfinite(rates_Z)):
            raise FloatingPointError("Invalid exponential rate in Z update")

        Z = rng.exponential(scale=1.0 / rates_Z)

        # Sum_{l,i} delta_lik * Z_li via weighted bincount.
        future_Z = np.cumsum(Z[:, ::-1], axis=1)[:, ::-1] - Z
        excluded_Z = np.bincount(
            rankings.ravel(), weights=future_Z.ravel(), minlength=M
        )
        exposure = np.maximum(Z.sum() - excluded_Z, 0.0)

        # w_k | Z, rankings ~ Gamma(alpha/M + n_k, tau + exposure_k)
        rates_w = tau_prior + exposure
        current_w = np.maximum(
            rng.gamma(posterior_shapes, scale=1.0 / rates_w), tiny
        )
        if not np.all(np.isfinite(current_w)):
            raise FloatingPointError("Invalid sampled item weight")

        chain[it] = current_w / current_w.sum()  # Save identifiable probabilities

    return chain


for c, seed in enumerate(np.random.SeedSequence(1234).spawn(n_chains)):
    samples[c] = gibbs_chain(np.random.default_rng(seed))
    print(f"Chain {c + 1}/{n_chains} completed ({time.perf_counter() - t_start:.1f}s)")

post = samples[:, burn_in:, :]
pooled = post.reshape(-1, M)
p_mean = pooled.mean(axis=0)
p_low, p_high = np.quantile(pooled, [0.025, 0.975], axis=0)


# ============================================================
# 3. CONVERGENCE DIAGNOSTICS (SPLIT R-HAT AND APPROXIMATE ESS)
# ============================================================
def split_rhat(chains):
    """Classical split R-hat calculated on post-burn-in normalized weights."""
    n = chains.shape[1] // 2
    halves = np.concatenate([chains[:, :n, :], chains[:, -n:, :]], axis=0)
    W = halves.var(axis=1, ddof=1).mean(axis=0)
    B = n * halves.mean(axis=1).var(axis=0, ddof=1)
    var_hat = ((n - 1) / n) * W + B / n
    return np.sqrt(var_hat / W)


def autocorr(x, max_lag):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = np.dot(x, x)
    acf = np.ones(max_lag + 1)
    if denom <= 0:
        acf[1:] = 0.0
        return acf
    for lag in range(1, max_lag + 1):
        acf[lag] = np.dot(x[:-lag], x[lag:]) / denom
    return acf


def approximate_ess(chains, max_lag=200):
    """Approximate effective sample size using the initial-positive-pairs rule."""
    c, n, k = chains.shape
    limit = min(max_lag, n - 1)
    ess = np.empty(k)
    for item in range(k):
        rho = np.mean([autocorr(chains[j, :, item], limit) for j in range(c)], axis=0)
        rho_sum = 0.0
        for lag in range(1, limit, 2):
            pair = rho[lag] + rho[lag + 1]
            if pair <= 0:
                break
            rho_sum += pair
        tau_int = max(1.0, 1.0 + 2.0 * rho_sum)
        ess[item] = min(c * n, c * n / tau_int)
    return ess


rhat = split_rhat(post)
ess = approximate_ess(post)
rank_order = np.argsort(p_mean)[::-1]

print("\nTop 15 movies by posterior mean probability:")
print("Rank | movieId | Count | Post. mean | 95% CI              | R-hat  | ESS   | Title")
for rank, k in enumerate(rank_order[:15], start=1):
    print(
        f"{rank:4d} | {unique_movie_ids[k]:7d} | {counts[k]:5d} | "
        f"{p_mean[k]:.5f}    | [{p_low[k]:.5f}, {p_high[k]:.5f}] | "
        f"{rhat[k]:.4f} | {ess[k]:5.0f} | {short_labels[k]}"
    )

print(f"\nLargest split R-hat across all {M} items: {rhat.max():.4f} (target < 1.01)")
print(f"Smallest approximate ESS across all {M} items: {ess.min():.0f} (median: {np.median(ess):.0f})")

with (output_dir / "diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "rank",
            "movieId",
            "title",
            "year",
            "genres",
            "observed_count",
            "observed_freq",
            "posterior_mean",
            "ci_2_5",
            "ci_97_5",
            "split_rhat",
            "ess_approx",
        ]
    )
    for rank, k in enumerate(rank_order, start=1):
        writer.writerow(
            [
                rank,
                int(unique_movie_ids[k]),
                titles[k],
                years[k],
                genres_list[k],
                int(counts[k]),
                counts[k] / L,
                p_mean[k],
                p_low[k],
                p_high[k],
                rhat[k],
                ess[k],
            ]
        )

np.savez_compressed(
    output_dir / "posterior_samples.npz",
    post_burnin_samples=post.astype(np.float32),
    unique_movie_ids=unique_movie_ids,
    titles=np.array(titles, dtype=object),
    rankings=rankings,
    burn_in=burn_in,
    alpha_prior=alpha_prior,
    tau_prior=tau_prior,
)


# ============================================================
# 4. LEGIBLE PLOTS WITH MOVIE TITLES
# ============================================================
# 4.1 Trace plots for Top-6 movies (3x2 grid)
top_trace_idx = rank_order[:top_k_trace]
fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
for ax, k in zip(axes.ravel(), top_trace_idx):
    for c in range(n_chains):
        ax.plot(samples[c, :, k], alpha=0.6, linewidth=0.6, label=f"Chain {c + 1}")
    ax.axvline(burn_in, color="black", linestyle="--", linewidth=1.0, label="Burn-in")
    ax.axhline(p_mean[k], color="crimson", linestyle=":", linewidth=1.2, label="Post. mean")
    ax.set_title(short_labels[k], fontsize=10, fontweight="bold")
    ax.set_ylabel("p")
    ax.grid(alpha=0.2)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=n_chains + 2, fontsize=8.5, frameon=True)
axes[-1, 0].set_xlabel("Iteration")
axes[-1, 1].set_xlabel("Iteration")
fig.suptitle("Trace Plots: Top-6 Movies by Posterior Mean Weight", fontsize=13, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(output_dir / "01_trace_plots.png", dpi=180)
plt.close(fig)

# 4.2 Horizontal forest plot of Top-25 movies (mean + 95% credible intervals)
top_forest_idx = rank_order[:top_k_forest][::-1]  # Bottom-to-top so #1 is at top
y_pos = np.arange(top_k_forest)

fig, ax = plt.subplots(figsize=(11, 8.5))
ax.errorbar(
    p_mean[top_forest_idx],
    y_pos,
    xerr=[
        p_mean[top_forest_idx] - p_low[top_forest_idx],
        p_high[top_forest_idx] - p_mean[top_forest_idx],
    ],
    fmt="o",
    color="#1f77b4",
    ecolor="#1f77b4",
    elinewidth=1.4,
    capsize=3.5,
    markersize=5.5,
    label="Posterior mean + 95% credible interval",
)
ax.set_yticks(y_pos)
ax.set_yticklabels(
    [f"#{top_k_forest - i}  {short_labels[k]}" for i, k in enumerate(top_forest_idx)],
    fontsize=9,
)
ax.set_xlabel("Normalized Plackett-Luce weight (p)")
ax.set_title(f"Top-{top_k_forest} Movies: Posterior Mean & 95% Credible Intervals")
ax.grid(axis="x", alpha=0.25)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(output_dir / "02_posterior_intervals.png", dpi=180)
plt.close(fig)

# 4.3 Post-burn-in autocorrelation for Top-8 movies
top_acf_idx = rank_order[:top_k_acf]
max_lag = min(100, post.shape[1] - 1)
cmap = plt.get_cmap("tab10")

fig, ax = plt.subplots(figsize=(11, 5.5))
for i, k in enumerate(top_acf_idx):
    rho = np.mean([autocorr(post[c, :, k], max_lag) for c in range(n_chains)], axis=0)
    ax.plot(
        np.arange(max_lag + 1),
        rho,
        linewidth=1.4,
        color=cmap(i),
        label=f"#{i + 1} {short_labels[k]}",
    )
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Lag")
ax.set_ylabel("Mean autocorrelation across chains")
ax.set_title(f"Post-Burn-In Autocorrelation (Top-{top_k_acf} Movies)")
ax.grid(alpha=0.25)
ax.legend(fontsize=8.5, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig(output_dir / "03_autocorrelation.png", dpi=180)
plt.close(fig)

# 4.4 Posterior predictive check (Top-25 horizontal + Full catalog calibration)
rng_ppc = np.random.default_rng(2026)
rep_indices = rng_ppc.choice(len(pooled), size=n_predictive_replicates, replace=True)
ppc_rates = np.empty((n_predictive_replicates, M), dtype=np.float64)

for r, idx in enumerate(rep_indices):
    log_p = np.log(pooled[idx])
    scores = log_p[None, :] + rng_ppc.gumbel(size=(L, M))
    top_m_sim = np.argpartition(scores, kth=M - m, axis=1)[:, -m:]
    ppc_rates[r] = np.bincount(top_m_sim.ravel(), minlength=M) / L

observed_rate = counts / L
ppc_mean = ppc_rates.mean(axis=0)
ppc_low, ppc_high = np.quantile(ppc_rates, [0.025, 0.975], axis=0)

fig, (ax_left, ax_right) = plt.subplots(
    1, 2, figsize=(15, 8.5), gridspec_kw={"width_ratios": [1.45, 1.0]}
)

# Left panel: Top-25 movies observed vs predictive 95% interval
ax_left.errorbar(
    ppc_mean[top_forest_idx],
    y_pos,
    xerr=[
        ppc_mean[top_forest_idx] - ppc_low[top_forest_idx],
        ppc_high[top_forest_idx] - ppc_mean[top_forest_idx],
    ],
    fmt="o",
    color="#1f77b4",
    capsize=3.5,
    markersize=5,
    label="Posterior predictive mean + 95% interval",
)
ax_left.scatter(
    observed_rate[top_forest_idx],
    y_pos,
    marker="x",
    color="crimson",
    s=55,
    linewidths=1.6,
    zorder=4,
    label="Observed top-10 frequency",
)
ax_left.set_yticks(y_pos)
ax_left.set_yticklabels(
    [f"#{top_k_forest - i}  {short_labels[k]}" for i, k in enumerate(top_forest_idx)],
    fontsize=9,
)
ax_left.set_xlabel("Fraction of users with movie in top-10")
ax_left.set_title(f"Top-{top_k_forest} Movies: Observed vs. Predictive")
ax_left.grid(axis="x", alpha=0.25)
ax_left.legend(loc="lower right", fontsize=8.5)

# Right panel: Full-catalog (M=2095) calibration scatter
ax_right.scatter(observed_rate, ppc_mean, alpha=0.35, s=16, color="#2ca02c", edgecolors="none")
max_val = max(observed_rate.max(), ppc_mean.max()) * 1.05
ax_right.plot([0, max_val], [0, max_val], "k--", linewidth=1.0, label="Ideal calibration (y = x)")
ax_right.set_xlabel("Observed top-10 inclusion frequency")
ax_right.set_ylabel("Posterior predictive mean frequency")
ax_right.set_title(f"Full-Catalog Calibration (M = {M} movies)")
ax_right.grid(alpha=0.25)
ax_right.legend(loc="upper left", fontsize=8.5)

fig.suptitle("Posterior Predictive Check: Top-10 Inclusion Frequencies", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(output_dir / "04_posterior_predictive.png", dpi=180)
plt.close(fig)

print(f"\nCompleted in {time.perf_counter() - t_start:.1f}s. Results saved in: {output_dir}")
for file in sorted(output_dir.iterdir()):
    print("  -", file.name)