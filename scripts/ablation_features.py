"""Block-by-block test of the feature matrix (R4 project).

Questions:
  1. which blocks of features (text, genres, cast, directors, ...) carry information about users' tastes?
  2. which encoding of the list blocks works better: v1 (SVD / 0-1 columns) or v2 (random codes)?
     (see ENCODING in build_features.py)

Every variant of the matrix is tested directly in its full dimension (no reduction, no UMAP) with the
taste tests of evaluate_embedding.py: movies in the same user's top-10 list should be closer than
  random   random pairs from all users' lists (same popularity)
  time     random pairs from the lists of the ~20 users active at about the same time
           (removes the advantage of the year: a user active in 1997 cannot have 2010 films in the list)
  user     a top-10 movie and a movie the same user disliked (rating <= 2.5)
For each test: AUC (0.5 = no signal) and the difference from v2; "*" = the 95% paired bootstrap interval
over users does not contain 0, so the difference is not just noise from the choice of users.
Also the genre agreement of the 10 nearest neighbours (train movies). Full intervals and the AUC on pairs
with a held-out movie are in results/ablation.csv.

Variants:
  complete            v2 (every list block as random codes, tuned weights), v1, the first draft of v2
                      (30 Oct), v2 with the weights of v1, saga weight halved
  one block alone     each block by itself (list blocks in both encodings); the numeric block also split
                      into year / runtime / budget (filled by regression, or median + 0/1 flag as before)
  v2 minus one block  v2 without each block in turn (negative difference = the block helps)
Note: in every variant the writers are defined as in v2 (whole Writing department), so "v1" here
differs slightly from the first data/processed files.

Usage (repository root, after build_features.py has written data/processed):
    python scripts/ablation_features.py
Output: results/ablation.csv, results/ablation_neighbors.txt. Runs in a few minutes.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import build_features as bf                                                           # noqa: E402
from evaluate_embedding import (N_BOOT, SEED, genre_agreement, genre_sets,               # noqa: E402
                                neighbours_text, taste_tests, user_lists)

SOURCES = {"v1": dict(encoding=bf.ENCODING_V1, budget="flag"),         # how each source is built
           "v2a": dict(encoding=bf.ENCODING_V2A, budget="flag"),
           "v2": dict(encoding=bf.ENCODING),                          # budget filled by regression
           "v2flag": dict(encoding=bf.ENCODING, budget="flag")}       # budget: median + 0/1 flag
TESTS = ["random", "time", "user"]


def label(enc):
    kind, _, k = enc
    return f"svd {k}" if kind == "svd" else kind


def q(x, c):
    return float(np.percentile(x, c))


# ----------------------------------------------------------------------------- data
def load_setup(raw, catalogue_path, sources):
    """Tidy data, catalogue, train mask, users (top / disliked movies, time), and every block of every
    source (FeatureBuilder settings) scaled to total variance 1 (no weights): lib[(source, block)]. The
    numeric block is also split into 'year', 'runtime', 'budget' (log budget, plus the 0/1 flag if any) and
    'runtime_budget', with the same scale as inside the numeric block."""
    d, _, _ = bf.prepare(raw)
    catalogue = pd.read_csv(catalogue_path)
    if len(catalogue) != len(d) or (catalogue.movieId.to_numpy() != d.movieId.to_numpy()).any() \
            or (catalogue.split.to_numpy() != d.split.to_numpy()).any():
        sys.exit("catalogue.csv does not match the raw data (different movies or split): regenerate it "
                 "with build_features.py.")
    train = (d.split == "train").to_numpy()
    lib = {}
    for source, kw in sources.items():
        blocks, _ = bf.FeatureBuilder(**kw).fit(d[train]).transform_blocks(d)
        lib.update({(source, b): B.astype(np.float32) for b, B in blocks.items()})
        num = lib[(source, "numeric")]                    # columns: year, log_runtime, log_budget [, flag]
        lib[(source, "year")], lib[(source, "runtime")] = num[:, :1], num[:, 1:2]
        lib[(source, "budget")], lib[(source, "runtime_budget")] = num[:, 2:], num[:, 1:]
    users = user_lists(pd.read_csv(Path(raw) / "ratings_small.csv"), catalogue)
    return d, catalogue, train, users, lib


def assemble(lib, spec):
    """spec = {block: (source, weight)} -> the weighted matrix."""
    return np.hstack([lib[(src, b)] * w for b, (src, w) in spec.items()])


def describe_tests(tests):
    t = tests["random"]
    return (f"{t.n_users} users, {len(t.pos)} same-list pairs; time groups of about 20 users "
            f"(median span {tests['time'].median_span_years:.1f} years); "
            f"{tests['user'].n_users_with_low} users with disliked movies (rating <= 2.5)")


# ----------------------------------------------------------------------------- variants
def make_variants():
    """(group, name, {block: (source, weight)}); source = which encoding the block comes from."""
    W, W1, order = bf.WEIGHTS, bf.WEIGHTS_V1, bf.BLOCK_ORDER
    v1 = {b: ("v1", W1[b]) for b in order}
    v2 = {b: ("v2", W[b]) for b in order}
    out = [("complete", "v2", v2), ("complete", "v1", v1),
           ("complete", "v2 draft 30 Oct", {b: ("v2a", W1[b]) for b in order}),
           ("complete", "v2 with the weights of v1", {b: ("v2", W1[b]) for b in order}),
           ("complete", "v2 with budget flag (before)", {**v2, "numeric": ("v2flag", W["numeric"])}),
           ("complete", "v2, saga weight 0.5", {**v2, "collection": ("v2", 0.5 * W["collection"])})]
    for b in order:
        if b in bf.SET_BLOCKS:
            out.append(("one block alone", f"only {b} ({label(bf.ENCODING_V1[b])})", {b: ("v1", 1.0)}))
            out.append(("one block alone", f"only {b} (code)", {b: ("v2", 1.0)}))
        elif b == "numeric":
            out += [("one block alone", "only numeric", {b: ("v2", 1.0)}),
                    ("one block alone", "only year", {"year": ("v2", 1.0)}),
                    ("one block alone", "only runtime", {"runtime": ("v2", 1.0)}),
                    ("one block alone", "only budget (regression)", {"budget": ("v2", 1.0)}),
                    ("one block alone", "only budget + flag (before)", {"budget": ("v2flag", 1.0)})]
        else:
            out.append(("one block alone", f"only {b}", {b: ("v2", 1.0)}))
    for b in order:
        out.append(("v2 minus one block", f"v2 without {b}", {k: v for k, v in v2.items() if k != b}))
    no_num = {k: v for k, v in v2.items() if k != "numeric"}
    out.append(("v2 minus one block", "v2 without year", {**no_num, "runtime_budget": ("v2", W["numeric"])}))
    out.append(("v2 minus one block", "v2 without runtime + budget", {**no_num, "year": ("v2", W["numeric"])}))
    out.append(("v2 minus one block", "v2 without budget",
                {**no_num, "year": ("v2", W["numeric"]), "runtime": ("v2", W["numeric"])}))
    return out


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default=REPO / "data" / "raw")
    p.add_argument("--catalogue", default=REPO / "data" / "processed" / "catalogue.csv")
    p.add_argument("--out", default=REPO / "results")
    p.add_argument("--boot", type=int, default=N_BOOT, help="bootstrap resamples of the users")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("reading the data and fitting the encodings on train movies (v1, v2 draft, v2) ...", flush=True)
    d, catalogue, train, users, lib = load_setup(a.raw, a.catalogue, SOURCES)
    tests = taste_tests(users, ~train, a.boot)
    genres = genre_sets(catalogue)
    print(f"  done in {time.time() - t0:.0f} s\n{describe_tests(tests)}", flush=True)
    header = (f"{'variant':<32}{'dim':>5}  " + "".join(f"{'AUC ' + t:<12}{'vs v2':<10}" for t in TESTS)
              + "genre kNN")

    rows, boots, neighbours, group = [], {}, [], None
    for g_name, name, spec in make_variants():
        if g_name != group:
            group = g_name
            print(f"\n--- {group}\n{header}")
        Z = assemble(lib, spec)
        res = {t: tests[t].run(Z) for t in TESTS}
        boots[name] = res
        g = genre_agreement(Z, train, ~train, genres, np.random.default_rng(SEED))
        row = {"group": group, "variant": name, "dim": Z.shape[1]}
        line = f"{name:<32}{Z.shape[1]:>5}  "
        for t in TESTS:
            (auc, bt), (aho, _) = res[t]["all"], res[t]["heldout"]
            ref, bref = boots["v2"][t]["all"]
            dlo, dhi = q(bt - bref, 2.5), q(bt - bref, 97.5)
            row.update({f"{t}_auc": auc, f"{t}_lo": q(bt, 2.5), f"{t}_hi": q(bt, 97.5),
                        f"{t}_diff_vs_v2": auc - ref, f"{t}_diff_lo": dlo, f"{t}_diff_hi": dhi,
                        f"{t}_auc_heldout": aho})
            star = "*" if (dlo > 0 or dhi < 0) else " "
            diff = "" if name == "v2" else f"{auc - ref:+.4f}{star}"
            line += f"{auc:<12.4f}{diff:<10}"
        row.update({"genre_knn_train": g["genre_knn_train"], "genre_knn_heldout": g["genre_knn_heldout"]})
        rows.append(row)
        print(line + f"{g['genre_knn_train']:.3f}", flush=True)
        if name in ("v1", "v2"):
            neighbours += [f"=== {name}"] + neighbours_text(Z, catalogue) + [""]

    pd.DataFrame(rows).to_csv(out / "ablation.csv", index=False, float_format="%.4f")
    (out / "ablation_neighbors.txt").write_text("\n".join(neighbours))
    print(f"\ngenre agreement of random pairs: {g['genre_random_pairs']:.3f}; AUC with no signal: 0.5")
    print("Reading: '*' = the 95% interval of the difference from v2 does not contain 0 (a real difference).\n"
          "'v2 without X' below 0 = block X helps. If the year helps in 'random' but not in 'time', its\n"
          "signal comes from when users were active, not from taste. Genre kNN is high whenever the genres\n"
          "block is in the matrix.")
    print(f"saved: {out / 'ablation.csv'}, {out / 'ablation_neighbors.txt'}  ({time.time() - t0:.0f} s in total)")


if __name__ == "__main__":
    main()
