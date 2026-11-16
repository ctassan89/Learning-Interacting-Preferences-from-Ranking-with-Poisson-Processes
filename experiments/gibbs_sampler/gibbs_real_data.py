"""Finite Bayesian Plackett-Luce Gibbs sampler on the REAL top-10 lists (Caron, Teh & Murphy, Sec. 3.1).

Same model and same updates as gibbs_simulated_data.py:
    w_k ~ Gamma(alpha/M, b)                      (b = rate; called tau_prior in the simulated script)
    Z_li | w ~ Exp(sum of the weights still in the race at step i of list l)
    w_k | Z ~ Gamma(alpha/M + n_k, b + exposure_k)

Data: data/processed/preference_lists_top10.csv (671 users, ordered top-10 lists) and
data/processed/catalogue.csv (the 9,025 movies with metadata = the universe of the model, M = 9,025).
15 listed movies are not in the catalogue (no metadata): they are removed from the lists, so 32 lists
have 8 or 9 movies. This is still exact: restricted to the catalogue, a PL top-10 list is a PL top-9 list
(in the race view, the catalogue movies of the list are the first catalogue movies to arrive).

Two tricks keep it fast on all 9,025 movies (no need to take a subset):
1. Never-ranked movies (n_k = 0, 77% of the catalogue) all have the same full conditional
   Gamma(alpha/M, b + sum(Z)), and only their total enters the likelihood. We sample their total
   w_star ~ Gamma(M0 * alpha/M, b + sum(Z)) (sum of independent Gammas with the same rate): exact.
2. Rescaling step: the likelihood does not see the total scale s = sum(w), so given the normalised
   weights s ~ Gamma(alpha, b) (its prior). Redrawing s every iteration is a valid Gibbs step and
   makes the chains mix faster. We save only normalised weights, which is what we compare anyway.

Posterior predictive checks (replicated lists from posterior draws):
  - per movie: number of lists containing it, observed vs predicted;
  - per pair of movies of the same saga: number of lists containing both. The base model gives every
    user the same popularities, so it cannot know that whoever loves one film of a saga also loves the next.

Outputs in results/gibbs_sampler/real/:
  diagnostics.csv           one row per ranked movie: n_k, posterior mean and 95% CI, R-hat, PPC interval
  ppc_pairs.csv             saga pairs: lists containing both, observed vs predicted
  posterior_samples.npz     normalised weights (every 10th iteration, float32)
  slide8_top_movies.pdf     top movies: posterior mean and 95% interval          (left of slide 8)
  slide8_ppc_pairs.pdf      saga pairs: lists containing both, observed vs predicted (right of slide 8)
"""

from pathlib import Path
import ast

import matplotlib

matplotlib.use("Agg")  # Works in terminals/WSL without graphical windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator


# ============================================================
# 1. SETTINGS AND REAL DATA
# ============================================================
alpha_prior = 2.0
b_prior = 1.0  # Gamma rate, NOT scale (tau_prior in gibbs_simulated_data.py)

n_chains = 4
n_iters = 3000
burn_in = 1000
thin_save = 10  # saved samples: every 10th iteration
n_predictive_replicates = 200
n_top = 10  # movies shown on slide 8
# pairs of movies of the same saga for the predictive check: (movieId, movieId, label)
saga_pairs = [
    (260, 1196, "Star Wars IV + V"),
    (1196, 1210, "Star Wars V + VI"),
    (4993, 5952, "Lord of the Rings 1 + 2"),
    (5952, 7153, "Lord of the Rings 2 + 3"),
    (858, 1221, "The Godfather I + II"),
    (1, 3114, "Toy Story 1 + 2"),
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
data_dir = PROJECT_ROOT / "data" / "processed"
output_dir = PROJECT_ROOT / "results" / "gibbs_sampler" / "real"
output_dir.mkdir(parents=True, exist_ok=True)

catalogue = pd.read_csv(data_dir / "catalogue.csv")
lists_df = pd.read_csv(data_dir / "preference_lists_top10.csv")
M = len(catalogue)
index_of = {mid: k for k, mid in enumerate(catalogue["movieId"])}
titles = catalogue["title"].to_numpy()

raw_lists = [ast.literal_eval(s) for s in lists_df["top_movies"]]
lists = [[index_of[mid] for mid in lst if mid in index_of] for lst in raw_lists]
n_removed = sum(len(a) - len(b) for a, b in zip(raw_lists, lists))
L = len(lists)
m = max(len(lst) for lst in lists)

# L x m array, -1 = empty slot (only in the few lists that lost a non-catalogue movie)
rankings = np.full((L, m), -1, dtype=int)
for ell, lst in enumerate(lists):
    rankings[ell, : len(lst)] = lst
valid = rankings >= 0
flat_items = rankings[valid]

counts = np.bincount(flat_items, minlength=M)  # n_k: number of lists containing movie k
ranked = np.flatnonzero(counts > 0)  # movies in at least one list
R = len(ranked)
M0 = M - R  # never-ranked movies, collapsed into w_star
pos_of = np.full(M, -1)
pos_of[ranked] = np.arange(R)
rk = np.where(valid, pos_of[np.maximum(rankings, 0)], -1)  # rankings in the 0..R-1 numbering
rk_flat = rk[valid]

print(f"Lists: {L} (top-{m}); {n_removed} listed movies not in the catalogue were removed")
print(f"Catalogue: M = {M} movies; in at least one list: {R}; never ranked: {M0} ({M0 / M:.1%})")


# ============================================================
# 2. GIBBS SAMPLER (never-ranked movies collapsed, rescaling step)
# ============================================================
shape_ranked = alpha_prior / M + counts[ranked]
shape_star = alpha_prior * M0 / M


def gibbs_chain(rng):
    w = rng.lognormal(mean=-0.7, sigma=0.7, size=R)  # ranked movies
    w_star = rng.gamma(shape_star, 1.0 / b_prior)  # total of the never-ranked movies
    chain = np.empty((n_iters, R + 1), dtype=float)  # normalised weights; last column = w_star share

    for it in range(n_iters):
        # Z_li | w ~ Exp(total weight minus the weights already chosen in list l)
        total = w.sum() + w_star
        selected_w = np.where(valid, w[np.maximum(rk, 0)], 0.0)
        previous_w = np.concatenate([np.zeros((L, 1)), np.cumsum(selected_w[:, :-1], axis=1)], axis=1)
        rates_Z = np.maximum(total - previous_w, np.finfo(float).tiny)
        Z = np.where(valid, rng.exponential(scale=1.0 / rates_Z), 0.0)

        # exposure_k = sum of Z over the steps where movie k was still in the race
        future_Z = np.cumsum(Z[:, ::-1], axis=1)[:, ::-1] - Z
        excluded_Z = np.bincount(rk_flat, weights=future_Z[valid], minlength=R)
        Z_total = Z.sum()
        exposure = np.maximum(Z_total - excluded_Z, 0.0)

        # w_k | Z ~ Gamma(alpha/M + n_k, b + exposure_k); never-ranked: exposure = sum(Z)
        w = rng.gamma(shape_ranked, scale=1.0 / (b_prior + exposure))
        w_star = rng.gamma(shape_star, scale=1.0 / (b_prior + Z_total))

        # rescaling: s | normalised weights ~ Gamma(alpha, b)
        s = w.sum() + w_star
        s_new = rng.gamma(alpha_prior, 1.0 / b_prior)
        w *= s_new / s
        w_star *= s_new / s
        if not (np.all(np.isfinite(w)) and np.all(w > 0) and np.isfinite(w_star)):
            raise FloatingPointError("Invalid sampled weight")
        chain[it, :R] = w / s_new
        chain[it, R] = w_star / s_new
    return chain


samples = np.empty((n_chains, n_iters, R + 1))
for c, seed in enumerate(np.random.SeedSequence(1234).spawn(n_chains)):
    samples[c] = gibbs_chain(np.random.default_rng(seed))
    print(f"Chain {c + 1}/{n_chains} completed")

post = samples[:, burn_in:, :]
pooled = post.reshape(-1, R + 1)
p_mean = pooled.mean(axis=0)
p_low, p_high = np.quantile(pooled, [0.025, 0.975], axis=0)


# ============================================================
# 3. CONVERGENCE DIAGNOSTICS (same as gibbs_simulated_data.py)
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
top = np.argsort(-p_mean[:R])[:n_top]  # top movies, in the 0..R-1 numbering
ess_top = approximate_ess(post[:, :, top])
print(f"\nLargest split R-hat over all {R} ranked movies: {rhat[:R].max():.4f} (target < 1.01)")
print(f"Smallest approximate ESS among the top {n_top}: {ess_top.min():.0f}")
print(f"Share of the {M0} never-ranked movies together: {p_mean[R]:.2e} "
      f"(each: {p_mean[R] / M0:.1e}, the same posterior for all of them)")


# ============================================================
# 4. POSTERIOR PREDICTIVE CHECK: number of lists containing each movie
# ============================================================
# Replicated lists from posterior draws, with the exponential race. The never-ranked block is one
# competitor with rate w_star (its share is about 1e-4, so splitting it into single movies changes nothing).
rng_ppc = np.random.default_rng(2026)
draws = pooled[rng_ppc.choice(len(pooled), size=n_predictive_replicates, replace=False)]
rep_counts = np.empty((n_predictive_replicates, R + 1))
list_len = valid.sum(axis=1)
pair_pos = np.array([[pos_of[index_of[a]], pos_of[index_of[c]]] for a, c, _ in saga_pairs])
rep_pairs = np.empty((n_predictive_replicates, len(saga_pairs)))
in_list = np.zeros((L, R), dtype=bool)
in_list[np.repeat(np.arange(L), valid.sum(axis=1)), rk_flat] = True
obs_pairs = np.array([(in_list[:, a] & in_list[:, c]).sum() for a, c in pair_pos])
for r, p in enumerate(draws):
    arrival = rng_ppc.exponential(size=(L, R + 1)) / p[None, :]
    first = np.argpartition(arrival, m - 1, axis=1)[:, :m]  # the m first arrivals, then in order
    order = np.take_along_axis(first, np.argsort(np.take_along_axis(arrival, first, 1), 1), 1)
    keep = np.arange(m)[None, :] < list_len[:, None]  # same list lengths as the data
    rep_counts[r] = np.bincount(order[keep], minlength=R + 1)
    has = lambda k: ((order == k) & keep).any(axis=1)  # noqa: E731
    rep_pairs[r] = [(has(a) & has(c)).sum() for a, c in pair_pos]
ppc_low, ppc_high = np.quantile(rep_counts, [0.025, 0.975], axis=0)
observed = counts[ranked]
inside_ppc = (ppc_low[:R] <= observed) & (observed <= ppc_high[:R])
print(f"Observed list counts inside the 95% predictive interval: {inside_ppc.mean():.1%} of the ranked movies")
pair_mean = rep_pairs.mean(axis=0)
pair_low, pair_high = np.quantile(rep_pairs, [0.025, 0.975], axis=0)
print("\nSaga pairs: lists containing both, observed vs predicted (mean, 95% interval)")
for (a, c, lab), o, mu, lo_, hi_ in zip(saga_pairs, obs_pairs, pair_mean, pair_low, pair_high):
    print(f"  {lab:26s} observed {o:3d} | predicted {mu:4.1f} [{lo_:.0f}, {hi_:.0f}] | x{o / max(mu, 1e-9):.1f}")
pd.DataFrame({
    "pair": [lab for *_, lab in saga_pairs],
    "movieId_a": [a for a, _, _ in saga_pairs], "movieId_b": [c for _, c, _ in saga_pairs],
    "observed": obs_pairs, "predicted_mean": pair_mean, "ppc_2_5": pair_low, "ppc_97_5": pair_high,
}).to_csv(output_dir / "ppc_pairs.csv", index=False)

diag = pd.DataFrame({
    "movieId": catalogue["movieId"].to_numpy()[ranked],
    "title": titles[ranked],
    "n_lists": observed,
    "posterior_mean": p_mean[:R],
    "ci_2_5": p_low[:R],
    "ci_97_5": p_high[:R],
    "split_rhat": rhat[:R],
    "ppc_2_5": ppc_low[:R],
    "ppc_97_5": ppc_high[:R],
}).sort_values("posterior_mean", ascending=False)
diag.to_csv(output_dir / "diagnostics.csv", index=False)
np.savez_compressed(
    output_dir / "posterior_samples.npz",
    normalized_samples=samples[:, ::thin_save, :].astype(np.float32),  # last column: never-ranked total
    ranked_movieId=catalogue["movieId"].to_numpy()[ranked],
    thin=thin_save, burn_in=burn_in, alpha=alpha_prior, b=b_prior,
)

print("\nTop movies: n_lists | posterior mean | 95% CI | R-hat")  # noqa: E501
for k in top:
    print(f"  {titles[ranked[k]][:40]:40s} {observed[k]:3d} | {p_mean[k]:.5f} | "
          f"[{p_low[k]:.5f}, {p_high[k]:.5f}] | {rhat[k]:.4f}")


# ============================================================
# 5. FIGURES FOR SLIDE 8 (sized for the 16:9 deck, same style as slide 7)
# ============================================================
BRAND, ACCENT, INK2, MUTED, HAIR = "#184F95", "#EB6834", "#52514E", "#898781", "#E1E0D9"
plt.rcParams.update({
    "font.family": ["LM Sans 10", "Latin Modern Sans", "DejaVu Sans"],
    "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})
CM = 1 / 2.54
HEIGHT, BOTTOM, TOP = 4.6 * CM, 0.2, 0.86  # same vertical layout for both figures


def short(t, n=26):
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def style(ax):
    ax.grid(True, axis="x", color=HAIR, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(MaxNLocator(4))


# left: posterior popularity of the top movies
fig = plt.figure(figsize=(7.6 * CM, HEIGHT))
ax = fig.add_axes([0.45, BOTTOM, 0.52, TOP - BOTTOM])
y = np.arange(n_top)[::-1]
ax.hlines(y, 100 * p_low[top], 100 * p_high[top], color=BRAND, lw=2.6, alpha=0.35)
ax.scatter(100 * p_mean[top], y, s=16, color=BRAND, zorder=3, linewidths=0)
ax.set_yticks(y)
ax.set_yticklabels([short(titles[ranked[k]]) for k in top])
ax.set_ylim(-0.6, n_top - 0.4)
ax.set_xlabel("normalised weight (%)")
ax.set_xlim(0, None)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
style(ax)
ax.set_title("posterior mean and 95% interval", fontsize=7.5, color=INK2, pad=4)
fig.savefig(output_dir / "slide8_top_movies.pdf")
fig.savefig(output_dir / "slide8_top_movies_preview.png", dpi=250)
plt.close(fig)

# right: predictive check on saga pairs
n_pairs = len(saga_pairs)
fig = plt.figure(figsize=(7.0 * CM, HEIGHT))
ax = fig.add_axes([0.44, BOTTOM, 0.53, TOP - BOTTOM])
yp = np.arange(n_pairs)[::-1]
ax.hlines(yp, pair_low, pair_high, color=BRAND, lw=2.6, alpha=0.35)
ax.scatter(pair_mean, yp, s=16, color=BRAND, zorder=3, linewidths=0, label="predicted, 95% interval")
ax.scatter(obs_pairs, yp, s=30, marker="x", color=ACCENT, linewidths=1.3, zorder=4, label="observed")
ax.set_yticks(yp)
ax.set_yticklabels([lab for *_, lab in saga_pairs])
ax.set_ylim(-0.6, n_pairs - 0.4)
ax.set_xlabel(f"lists containing both (of {L})")
ax.set_xlim(0, 1.12 * max(obs_pairs.max(), pair_high.max()))
style(ax)
ax.legend(loc="lower center", bbox_to_anchor=(0.32, 0.99), ncol=2, frameon=False, fontsize=7,
          handletextpad=0.3, columnspacing=1.0, handlelength=1.2)
fig.savefig(output_dir / "slide8_ppc_pairs.pdf")
fig.savefig(output_dir / "slide8_ppc_pairs_preview.png", dpi=250)
plt.close(fig)

print("\nResults saved in:", output_dir)
for file in sorted(output_dir.iterdir()):
    print("  -", file.name)
