"""Compare all text-embedding runs (and two reference embeddings).

Merges the shared ``results/metrics.csv`` (rows of ``text_*`` and ``ref_*``
methods) with ``results/TextEmbeddings/extra_metrics.csv`` into
``results/TextEmbeddings/summary.csv`` and draws ``summary.png`` (one panel per
key metric, one bar per run).

With ``--with-baselines`` it first evaluates two references with the same
pipeline, so text encoders are compared with what the team already has:
    ref_tfidf_text      the text block of the shared matrix (TF-IDF + SVD, cols 0-99)
    ref_feature_matrix  the whole shared feature matrix v2 (902 columns)

Usage: python -m experiments.TextEmbeddings.compare [--with-baselines]
"""

import argparse

import numpy as np
import pandas as pd

from experiments.TextEmbeddings.common import (
    EXTRA_METRICS_PATH,
    PROJECT_ROOT,
    RESULTS_DIR,
    TEXT_RESULTS_DIR,
    evaluate_run,
    load_movies,
)

# (space, column, label, higher is better)
KEY_METRICS = (
    ("umap2d", "k10_silhouette", "Silhouette, UMAP 2D (k=10)", True),
    ("umap2d", "k10_davies_bouldin", "Davies-Bouldin, UMAP 2D (k=10)", False),
    ("native", "k10_genre_purity", "Genre purity, native (k=10)", True),
    ("native", "genre_knn_heldout", "kNN genre Jaccard, held-out", True),
    ("native", "genre_f1_micro_heldout", "kNN genre F1, held-out", True),
    ("native", "placement_gap", "Held-out placement gap (~1 best)", None),
    ("umap2d", "trustworthiness", "Trustworthiness of UMAP 2D", True),
    ("native", "taste_auc_time", "Taste AUC, time-matched", True),
)
FAMILY_COLORS = {  # fixed order: reference palette slots 1-2, neutral for references
    "pretrained": "#2a78d6",
    "classical (LSA)": "#eb6834",
    "reference": "#8c8b86",
}


def evaluate_baselines() -> None:
    movies = load_movies()
    z = np.load(PROJECT_ROOT / "data" / "processed" / "features.npz")
    X = pd.DataFrame(z["X"]).set_index(pd.Index(z["movieId"]))
    X = X.loc[movies["movieId"]].to_numpy(dtype=np.float32)  # catalogue order
    for name, Z in (("ref_tfidf_text", X[:, 0:100]), ("ref_feature_matrix", X)):
        (TEXT_RESULTS_DIR / name).mkdir(parents=True, exist_ok=True)
        evaluate_run(name, Z, movies)


def family(method: str) -> str:
    if method.startswith("ref_"):
        return "reference"
    if method.startswith("text_lsa"):
        return "classical (LSA)"
    return "pretrained"


def summary_table() -> pd.DataFrame:
    shared = pd.read_csv(RESULTS_DIR / "metrics.csv")
    shared = shared[shared["method"].str.match(r"^(text|ref)_")]
    extra = pd.read_csv(EXTRA_METRICS_PATH)
    table = shared.merge(
        extra, on=["method", "space"], how="outer", suffixes=("", "_extra")
    )
    return table.drop(columns=[c for c in table if c.endswith("_extra")])


def plot_summary(table: pd.DataFrame, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    methods = sorted(table["method"].unique(), key=lambda m: (family(m), m))
    n_cols = 4
    n_rows = int(np.ceil(len(KEY_METRICS) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 0.28 * len(methods) * n_rows + 1.5),
        sharey=True,
    )
    for ax, (space, column, label, higher) in zip(axes.flat, KEY_METRICS):
        rows = table[table["space"] == space].set_index("method")
        values = [
            rows[column].get(m, np.nan) if column in rows else np.nan for m in methods
        ]
        y = np.arange(len(methods))
        ax.barh(
            y,
            values,
            height=0.7,
            color=[FAMILY_COLORS[family(m)] for m in methods],
            edgecolor="white",
            linewidth=1,
        )
        if column == "placement_gap":
            ax.axvline(1.0, color="#52514e", linewidth=1, linestyle="--")
        hint = (
            ""
            if higher is None
            else (" (higher = better)" if higher else " (lower = better)")
        )
        ax.set_title(label + hint, fontsize=9, color="#0b0b0b", loc="left")
        ax.set_yticks(y, methods, fontsize=7, color="#52514e")
        ax.tick_params(axis="x", labelsize=7, colors="#52514e")
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.invert_yaxis()
    for ax in list(axes.flat)[len(KEY_METRICS) :]:
        ax.set_visible(False)
    fig.legend(
        handles=[Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()],
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-baselines",
        action="store_true",
        help="evaluate the TF-IDF text block and the full shared matrix first",
    )
    args = parser.parse_args()
    if args.with_baselines:
        evaluate_baselines()
    table = summary_table()
    out = TEXT_RESULTS_DIR / "summary.csv"
    table.to_csv(out, index=False, float_format="%.4f")
    plot_summary(table, TEXT_RESULTS_DIR / "summary.png")
    cols = ["method", "space"] + [c for _, c, _, _ in KEY_METRICS if c in table]
    print(table[cols].sort_values(["space", "method"]).to_string(index=False))
    print(f"\nsaved: {out.relative_to(PROJECT_ROOT)}, summary.png")


if __name__ == "__main__":
    main()
