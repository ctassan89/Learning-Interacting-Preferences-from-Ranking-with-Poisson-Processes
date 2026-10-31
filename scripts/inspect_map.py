"""What are the islands of a 2D map? (R4 project)

Draws the 2D map of an embedding with four colourings and describes every separate group ("island").
For a high-dimensional input the map is UMAP fitted on train movies with the same settings as
evaluate_embedding.py --umap, so it is the same map you already saw.
    <name>_map_genre.png      colour = first genre (as in evaluate_embedding.py)
    <name>_map_language.png   colour = original language (8 most common)
    <name>_map_saga.png       colour = the 10 largest sagas; other films grey
    <name>_map_year.png       colour = release year
    <name>_map_budget.png     colour = budget known / unknown in TMDb
    <name>_islands.txt        one paragraph per island: size, main language / first genre / saga /
                              director with their share (and the share in the whole catalogue),
                              median year, best-known titles (most rated)
    embeddings/<name>_umap2d.npz   the 2D coordinates (only if UMAP was run)
Islands = groups of points separated by empty space: DBSCAN on the 2D coordinates with a radius of
--gap times the size of the map (default 1.5%); groups smaller than --min-island films are ignored.

Usage (repository root):
    python scripts/inspect_map.py                                    # data/processed/features.npz
    python scripts/inspect_map.py --embedding results/embeddings/pumap_2d_nn15_md0.1.npz --name pumap_2d
An embedding that is already 2D is used as it is.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from evaluate_embedding import SEED, align, load_embedding, plot_map, save_embedding  # noqa: E402


def to_2d(Z, train):
    if Z.shape[1] == 2:
        return Z, False
    import umap
    return umap.UMAP(n_components=2, random_state=SEED).fit(Z[train]).transform(Z), True


def plot_categories(xy, labels, top, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = pd.Series(labels).fillna("").to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    other = ~np.isin(labels, top)
    ax.scatter(xy[other, 0], xy[other, 1], s=3, c="lightgrey", label="other", linewidths=0)
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(top):
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=4 if len(top) > 8 else 3, color=cmap(i % 10), label=str(c)[:30],
                   linewidths=0)
    ax.set_title(title)
    ax.legend(markerscale=4, fontsize=7, loc="best", frameon=False)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_year(xy, year, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 6))
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=3, c=year, cmap="viridis", linewidths=0)
    fig.colorbar(sc, ax=ax, label="year")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def share(values, overall=None):
    """'most common value (its share)' of a group, optionally with its share in the whole catalogue."""
    v = pd.Series(values)
    vc = v.dropna().value_counts()
    if vc.empty:
        return "-"
    top, frac = vc.index[0], vc.iloc[0] / len(v)
    if frac < 0.05:
        return f"none dominant (most common: {top} {frac:.1%})"
    text = f"{top} {frac:.0%}"
    if overall is not None:
        text += f" (catalogue {(pd.Series(overall) == top).mean():.0%})"
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default=REPO / "data" / "processed" / "features.npz")
    p.add_argument("--embedding", default=None, help="an embedding file (Z, movieId) instead of the features")
    p.add_argument("--name", default=None, help="used in the file names (default: raw_features or the "
                                                 "embedding file name)")
    p.add_argument("--catalogue", default=REPO / "data" / "processed" / "catalogue.csv")
    p.add_argument("--movies", default=REPO / "data" / "processed" / "movies_clean.csv")
    p.add_argument("--ratings", default=REPO / "data" / "raw" / "ratings_small.csv")
    p.add_argument("--out", default=REPO / "results")
    p.add_argument("--gap", type=float, default=0.015, help="island radius, as a fraction of the map size")
    p.add_argument("--min-island", type=int, default=15)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    catalogue = pd.read_csv(a.catalogue)
    if a.embedding:
        Z, ids = load_embedding(a.embedding)
        name = a.name or Path(a.embedding).stem
    else:
        z = np.load(a.features)
        Z, ids = z["X"].astype(float), z["movieId"]
        name = a.name or "raw_features"
    Z = align(Z, ids, catalogue)
    movies = pd.read_csv(a.movies).set_index("movieId").loc[catalogue.movieId].reset_index()
    n_ratings = pd.read_csv(a.ratings).movieId.value_counts().reindex(catalogue.movieId).fillna(0).to_numpy()
    train = (catalogue.split == "train").to_numpy()

    print(f"{name}: {Z.shape[0]} movies, {Z.shape[1]} dimensions" +
          (" -> UMAP 2D (fitted on train movies) ..." if Z.shape[1] > 2 else ""), flush=True)
    xy, ran_umap = to_2d(Z, train)
    if ran_umap:
        save_embedding(out / "embeddings" / f"{name}_umap2d.npz", xy, catalogue.movieId)

    first_genre = catalogue.genres.fillna("").str.split("|").str[0].replace("", "none").to_numpy()
    language = movies.language.to_numpy()
    saga = movies.collection.to_numpy()
    director = movies.directors.fillna("").str.split("|").str[0].replace("", np.nan).to_numpy()
    year = catalogue.year.to_numpy()
    budget = np.where(movies.budget.notna(), "budget known", "budget unknown")

    plot_map(xy, catalogue, out / f"{name}_map_genre.png", f"{name}: first genre")
    plot_categories(xy, language, list(pd.Series(language).value_counts().index[:8]),
                    out / f"{name}_map_language.png", f"{name}: original language")
    plot_categories(xy, saga, list(pd.Series(saga).value_counts().index[:10]),
                    out / f"{name}_map_saga.png", f"{name}: the 10 largest sagas")
    plot_year(xy, year, out / f"{name}_map_year.png", f"{name}: release year")
    plot_categories(xy, budget, ["budget known", "budget unknown"], out / f"{name}_map_budget.png",
                    f"{name}: budget in TMDb")

    from sklearn.cluster import DBSCAN
    size = np.linalg.norm(xy.max(axis=0) - xy.min(axis=0))
    labels = DBSCAN(eps=a.gap * size, min_samples=5).fit_predict(xy)
    groups = pd.Series(labels[labels >= 0]).value_counts()
    groups = groups[groups >= a.min_island]
    lines = [f"{name}: {len(groups)} groups of >= {a.min_island} films separated by empty space "
             f"(radius {a.gap:.1%} of the map); {np.sum(labels < 0)} scattered films belong to none.",
             "Shares: most common value in the group (and in the whole catalogue).", ""]
    for rank, (g, n) in enumerate(groups.items()):
        m = labels == g
        best = np.argsort(-n_ratings[m])[:5]
        titles = catalogue.title.to_numpy()[m][best]
        years = year[m][best]
        examples = "; ".join(f"{t} ({y:.0f})" if not np.isnan(y) else t for t, y in zip(titles, years))
        head = "MAIN CONTINENT" if rank == 0 else f"island {rank}"
        lines += [f"{head}: {n} films ({n / len(xy):.1%})",
                  f"  language  {share(language[m], language)}",
                  f"  genre     {share(first_genre[m], first_genre)}",
                  f"  saga      {share(saga[m])}   (films in any saga: {pd.notna(saga[m]).mean():.0%})",
                  f"  director  {share(director[m])}",
                  f"  budget known {np.mean(budget[m] == 'budget known'):.0%} "
                  f"(catalogue {np.mean(budget == 'budget known'):.0%})",
                  f"  median year {np.nanmedian(year[m]):.0f}",
                  f"  e.g. {examples}", ""]
    text = "\n".join(lines)
    (out / f"{name}_islands.txt").write_text(text)
    print(text)
    print(f"saved: {out}/{name}_map_genre.png, _map_language.png, _map_saga.png, _map_year.png, "
          f"_map_budget.png, _islands.txt")


if __name__ == "__main__":
    main()
