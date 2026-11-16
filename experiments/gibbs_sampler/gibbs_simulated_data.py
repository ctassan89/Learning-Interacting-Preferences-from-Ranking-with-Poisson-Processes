"""Finite Bayesian Plackett-Luce Gibbs sampler (Caron, Teh & Murphy, Sec. 3.1).

Outputs are saved under ``results/gibbs_sampler/simulated/``.
"""

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")  # Works in terminals/WSL without graphical windows
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. SETTINGS AND SIMULATED DATA
# ============================================================
M = 10
m = 3
L = 1000
alpha_prior = 1.0
tau_prior = 1.0  # Gamma rate, NOT scale

n_chains = 4
n_iters = 3000
burn_in = 1000
n_predictive_replicates = 200

PROJECT_ROOT = Path(__file__).resolve().parents[2]
output_dir = PROJECT_ROOT / "results" / "gibbs_sampler" / "simulated"
output_dir.mkdir(parents=True, exist_ok=True)

rng_data = np.random.RandomState(42)  # Preserves the original simulated data
true_w = rng_data.gamma(alpha_prior / M, scale=1.0 / tau_prior, size=M)
true_p = true_w / true_w.sum()


def simulate_pl_data(w, n_rankings, top_m, rng):
    """Top-m rankings using independent exponential arrival times."""
    arrival_times = rng.exponential(scale=1.0 / w, size=(n_rankings, len(w)))
    return np.argsort(arrival_times, axis=1)[:, :top_m]


rankings = simulate_pl_data(true_w, L, m, rng_data)
print("True normalized weights:", np.round(true_p, 4))
print("Rankings shape:", rankings.shape)


# ============================================================
# 2. VECTORIZED FINITE GIBBS SAMPLER
# ============================================================
counts = np.bincount(rankings.ravel(), minlength=M)
posterior_shapes = alpha_prior / M + counts
samples = np.empty((n_chains, n_iters, M), dtype=float)


def gibbs_chain(rng):
    current_w = rng.lognormal(mean=-0.7, sigma=0.7, size=M)
    chain = np.empty((n_iters, M), dtype=float)

    for it in range(n_iters):
        # Z_li | w, rankings ~ Exp(sum of weights still available at step i)
        selected_w = current_w[rankings]
        previous_w = np.concatenate(
            [np.zeros((L, 1)), np.cumsum(selected_w[:, :-1], axis=1)], axis=1
        )
        total_w = current_w.sum()
        rates_Z = total_w - previous_w

        # Cancellation guard: recompute tiny remaining sums in log-space.
        fragile = rates_Z <= np.finfo(float).eps * total_w
        if np.any(fragile):
            for ell, i in np.argwhere(fragile):
                active = np.ones(M, dtype=bool)
                active[rankings[ell, :i]] = False
                rates_Z[ell, i] = np.exp(np.logaddexp.reduce(np.log(current_w[active])))
        if np.any(rates_Z <= 0) or not np.all(np.isfinite(rates_Z)):
            raise FloatingPointError("Invalid exponential rate in Z update")

        Z = rng.exponential(scale=1.0 / rates_Z)

        # Sum_{l,i} delta_lik * Z_li via weighted bincount.
        # After selecting item k at position i, it is absent at later stages.
        future_Z = np.cumsum(Z[:, ::-1], axis=1)[:, ::-1] - Z
        excluded_Z = np.bincount(
            rankings.ravel(), weights=future_Z.ravel(), minlength=M
        )
        exposure = np.maximum(Z.sum() - excluded_Z, 0.0)

        # w_k | Z, rankings ~ Gamma(alpha/M + n_k, tau + exposure_k)
        rates_w = tau_prior + exposure
        current_w = rng.gamma(posterior_shapes, scale=1.0 / rates_w)
        if np.any(current_w <= 0) or not np.all(np.isfinite(current_w)):
            raise FloatingPointError("Invalid sampled item weight")
        chain[it] = current_w / current_w.sum()  # Save only identifiable probabilities

    return chain


for c, seed in enumerate(np.random.SeedSequence(1234).spawn(n_chains)):
    samples[c] = gibbs_chain(np.random.default_rng(seed))
    print(f"Chain {c + 1}/{n_chains} completed")

post = samples[:, burn_in:, :]
pooled = post.reshape(-1, M)
p_mean = pooled.mean(axis=0)
p_low, p_high = np.quantile(pooled, [0.025, 0.975], axis=0)


# ============================================================
# 3. CONVERGENCE DIAGNOSTICS (SPLIT R-HAT AND APPROXIMATE ESS)
# ============================================================
def split_rhat(chains):
    """Classical split R-hat, calculated on post-burn-in normalized weights."""
    n = chains.shape[1] // 2
    halves = np.concatenate([chains[:, :n, :], chains[:, -n:, :]], axis=0)
    W = halves.var(axis=1, ddof=1).mean(axis=0)
    B = n * halves.mean(axis=1).var(axis=0, ddof=1)
    var_hat = ((n - 1) / n) * W + B / n
    return np.sqrt(var_hat / W)


def autocorr(x, max_lag):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x)
    acf = np.ones(max_lag + 1)
    if denom <= 0:
        acf[1:] = 0
        return acf
    for lag in range(1, max_lag + 1):
        acf[lag] = np.dot(x[:-lag], x[lag:]) / denom
    return acf


def approximate_ess(chains, max_lag=200):
    """Approximate effective sample size, using the initial-positive-pairs rule."""
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

print("\nItem | True p | Post. mean | 95% CI              | Split R-hat | ESS approx.")
for k in range(M):
    print(
        f"{k:4d} | {true_p[k]:.5f} | {p_mean[k]:.5f}    | "
        f"[{p_low[k]:.5f}, {p_high[k]:.5f}] | "
        f"{rhat[k]:.4f}      | {ess[k]:.0f}"
    )

mae = np.abs(true_p - p_mean).mean()
coverage = np.mean((true_p >= p_low) & (true_p <= p_high))
print(f"\nMAE normalized: {mae:.6f}")
print(f"True weights inside 95% CIs: {int(coverage * M)}/{M}")
print(f"Largest split R-hat: {rhat.max():.4f} (target < 1.01)")
print(f"Smallest approximate ESS: {ess.min():.0f} (higher is better)")

with (output_dir / "diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "item",
            "true_p",
            "posterior_mean",
            "ci_2_5",
            "ci_97_5",
            "split_rhat",
            "ess_approx",
        ]
    )
    for k in range(M):
        writer.writerow([k, true_p[k], p_mean[k], p_low[k], p_high[k], rhat[k], ess[k]])

np.savez_compressed(
    output_dir / "posterior_samples.npz",
    normalized_samples=samples,
    true_p=true_p,
    rankings=rankings,
    burn_in=burn_in,
)

# ============================================================
# 4. PLOTS: TRACE, POSTERIOR INTERVALS, ACF, POSTERIOR PREDICTIVE
# ============================================================
items = np.arange(M)
selected_items = np.argsort(p_mean)[-min(4, M) :][::-1]

fig, axes = plt.subplots(len(selected_items), 1, figsize=(11, 9), sharex=True)
axes = np.atleast_1d(axes)
for ax, k in zip(axes, selected_items):
    for c in range(n_chains):
        ax.plot(samples[c, :, k], alpha=0.65, linewidth=0.65, label=f"Chain {c+1}")
    ax.axvline(burn_in, color="black", linestyle="--", linewidth=1)
    ax.axhline(true_p[k], color="black", linestyle=":", linewidth=1)
    ax.set_ylabel(f"p[{k}]")
axes[0].legend(ncol=n_chains, fontsize=8)
axes[-1].set_xlabel("Iteration")
fig.suptitle("Trace plots: normalized weights (dashed = burn-in; dotted = true)")
fig.tight_layout()
fig.savefig(output_dir / "01_trace_plots.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.errorbar(
    items,
    p_mean,
    yerr=[p_mean - p_low, p_high - p_mean],
    fmt="o",
    capsize=4,
    label="Posterior mean + 95% credible interval",
)
ax.scatter(items, true_p, marker="x", s=75, label="True normalized weight", zorder=4)
ax.set_xticks(items)
ax.set_xlabel("Item")
ax.set_ylabel("Normalized weight")
ax.set_title("Posterior recovery of true item probabilities")
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "02_posterior_intervals.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
max_lag = min(100, post.shape[1] - 1)
for k in selected_items:
    rho = np.mean([autocorr(post[c, :, k], max_lag) for c in range(n_chains)], axis=0)
    ax.plot(np.arange(max_lag + 1), rho, label=f"Item {k}")
ax.axhline(0, color="black", linewidth=0.8)
ax.set(
    xlabel="Lag", ylabel="Mean autocorrelation", title="Post-burn-in autocorrelation"
)
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "03_autocorrelation.png", dpi=180)
plt.close(fig)

# Posterior predictive check for top-m inclusion frequency.
# Gumbel top-k is equivalent to sequential PL sampling and works in log-space.
rng_ppc = np.random.default_rng(2026)
indices = rng_ppc.choice(len(pooled), size=n_predictive_replicates, replace=True)
ppc_rates = np.empty((n_predictive_replicates, M))
for r, index in enumerate(indices):
    p = pooled[index]
    scores = np.log(p)[None, :] + rng_ppc.gumbel(size=(L, M))
    top_m = np.argpartition(scores, kth=M - m, axis=1)[:, -m:]
    ppc_rates[r] = np.bincount(top_m.ravel(), minlength=M) / L

observed_rate = np.bincount(rankings.ravel(), minlength=M) / L
ppc_mean = ppc_rates.mean(axis=0)
ppc_low, ppc_high = np.quantile(ppc_rates, [0.025, 0.975], axis=0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.errorbar(
    items,
    ppc_mean,
    yerr=[ppc_mean - ppc_low, ppc_high - ppc_mean],
    fmt="o",
    capsize=4,
    label="Posterior predictive mean + 95% interval",
)
ax.scatter(
    items, observed_rate, marker="x", s=75, label="Observed top-m frequency", zorder=4
)
ax.set_xticks(items)
ax.set(
    xlabel="Item",
    ylabel="Fraction of rankings containing item",
    title="Posterior predictive check: top-m frequencies",
)
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "04_posterior_predictive.png", dpi=180)
plt.close(fig)

print("\nResults saved in:", output_dir)
for file in sorted(output_dir.iterdir()):
    print("  -", file.name)

