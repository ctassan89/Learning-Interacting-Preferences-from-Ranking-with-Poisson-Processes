"""Choose the block weights of the feature matrix (R4 project) without fooling ourselves.

The weights decide how much each block counts in the distance between two films. We choose them with a
taste test (see evaluate_embedding.py), using HALF of the users only (half A); the chosen weights are then
checked on the OTHER half (half B), which played no role in the choice: a gain that does not survive on
half B was luck, not a better matrix.

The choice uses one test (--criterion, default "time": random pairs from users active at about the same
time, so the year gets no credit just because a user active in 1997 cannot have 2010 films in the list).
On half B all three tests are reported (random, time, user = top-10 vs the same user's disliked movies).

Grid (absolute weights; the list blocks share one multiplier "names"):
    text       0.5, 1
    genres     1, 1.5, 2
    language   0.25, 0.5, 1
    numeric    1, 1.5, 2        (year, runtime, budget)
    names      0.5, 1           (keywords, cast, directors, writers, companies, sagas)
For the 5 best combinations on half A and for the current weights of build_features.py we report, on
half B, each test's AUC and its difference from the current weights ("*" = 95% paired bootstrap interval
without 0), and the genre agreement of the 10 nearest neighbours.

Usage (repository root):  python scripts/tune_weights.py [--criterion time|user|random]
Output: results/weights_grid.csv (every combination, half A), results/weights_best.csv (half B).
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import build_features as bf                                                     # noqa: E402
from ablation_features import TESTS, assemble, describe_tests, load_setup, q       # noqa: E402
from evaluate_embedding import N_BOOT, SEED, genre_agreement, genre_sets, taste_tests  # noqa: E402

GRID = {"text": [0.5, 1.0], "genres": [1.0, 1.5, 2.0], "language": [0.25, 0.5, 1.0],
        "numeric": [1.0, 1.5, 2.0], "names": [0.5, 1.0]}
N_TOP = 5


def weights_of(combo):
    w = dict(bf.WEIGHTS)
    for k in ("text", "genres", "language", "numeric"):
        w[k] = combo[k]
    for b in bf.SET_BLOCKS:
        w[b] = bf.WEIGHTS[b] * combo["names"]
    return w


def matrix(lib, combo):
    return assemble(lib, {b: ("v2", w) for b, w in weights_of(combo).items()})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--criterion", choices=TESTS, default="time", help="taste test used for the choice")
    p.add_argument("--raw", default=REPO / "data" / "raw")
    p.add_argument("--catalogue", default=REPO / "data" / "processed" / "catalogue.csv")
    p.add_argument("--out", default=REPO / "results")
    p.add_argument("--boot", type=int, default=N_BOOT, help="bootstrap resamples of the users")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("reading the data and fitting the v2 encoding on train movies ...", flush=True)
    d, catalogue, train, users, lib = load_setup(a.raw, a.catalogue, {"v2": dict(encoding=bf.ENCODING)})
    order = np.random.default_rng(SEED).permutation(len(users))
    half = len(users) // 2
    tests_a = taste_tests([users[i] for i in order[:half]], ~train, a.boot, seed=SEED)
    tests_b = taste_tests([users[i] for i in order[half:]], ~train, a.boot, seed=SEED + 1)
    print(f"  done in {time.time() - t0:.0f} s\nhalf A (choice): {describe_tests(tests_a)}\n"
          f"half B (check):  {describe_tests(tests_b)}", flush=True)

    current = {"text": bf.WEIGHTS["text"], "genres": bf.WEIGHTS["genres"], "language": bf.WEIGHTS["language"],
               "numeric": bf.WEIGHTS["numeric"], "names": 1.0}
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    if current not in combos:
        combos.append(current)
    print(f"\nhalf A: {len(combos)} weight combinations, chosen by the '{a.criterion}' test ...", flush=True)
    rows = []
    for c in combos:
        Z = matrix(lib, c)
        rows.append({**c, **{f"{t}_auc_A": tests_a[t].run(Z, heldout=False)["all"][0] for t in TESTS},
                     "current": c == current})
    grid = pd.DataFrame(rows).sort_values(f"{a.criterion}_auc_A", ascending=False).reset_index(drop=True)
    grid.insert(0, "rank_A", np.arange(1, len(grid) + 1))
    grid.to_csv(out / "weights_grid.csv", index=False, float_format="%.4f")

    chosen = grid.head(N_TOP)
    if not chosen.current.any():
        chosen = pd.concat([chosen, grid[grid.current]])
    genres = genre_sets(catalogue)
    keys = list(GRID)
    Zc = matrix(lib, current)
    ref = {t: tests_b[t].run(Zc, heldout=False)["all"] for t in TESTS}
    print(f"\nhalf B (users not used for the choice)\n{'rank A':>6}  {'text genres lang numeric names':<34}"
          + "".join(f"{'AUC ' + t:<12}{'vs current':<11}" for t in TESTS) + "genre kNN")
    best = []
    for _, r in chosen.iterrows():
        c = {k: r[k] for k in keys}
        Z = matrix(lib, c)
        g = genre_agreement(Z, train, ~train, genres, np.random.default_rng(SEED))
        row = {"rank_A": int(r.rank_A), **c, "current": bool(r.current)}
        w = " ".join(f"{c[k]:g}" for k in keys) + ("  (current)" if r.current else "")
        line = f"{int(r.rank_A):>6}  {w:<34}"
        for t in TESTS:
            auc, bt = tests_b[t].run(Z, heldout=False)["all"]
            dlo, dhi = q(bt - ref[t][1], 2.5), q(bt - ref[t][1], 97.5)
            row.update({f"{t}_auc_A": r[f"{t}_auc_A"], f"{t}_auc_B": auc, f"{t}_B_lo": q(bt, 2.5),
                        f"{t}_B_hi": q(bt, 97.5), f"{t}_diff_vs_current_B": auc - ref[t][0],
                        f"{t}_diff_lo": dlo, f"{t}_diff_hi": dhi})
            star = "*" if (dlo > 0 or dhi < 0) else " "
            diff = "" if r.current else f"{auc - ref[t][0]:+.4f}{star}"
            line += f"{auc:<12.4f}{diff:<11}"
        row.update({"genre_knn_train": g["genre_knn_train"], "genre_knn_heldout": g["genre_knn_heldout"]})
        best.append(row)
        print(line + f"{g['genre_knn_train']:.3f}", flush=True)
    pd.DataFrame(best).to_csv(out / "weights_best.csv", index=False, float_format="%.4f")

    top, crit = best[0], a.criterion
    if top["current"]:
        verdict = "the current weights are already the best on half A: keep them."
    elif top[f"{crit}_diff_lo"] > 0:
        verdict = (f"the best weights on half A also win on half B in the '{crit}' test "
                   f"(+{top[f'{crit}_diff_vs_current_B']:.4f}, interval above 0): worth adopting, after a "
                   "look at the neighbours and at the other two tests.")
    else:
        verdict = (f"the gain of the best weights on half A does not survive on half B in the '{crit}' test: "
                   "keep the current weights.")
    print(f"\nVerdict: {verdict}")
    print("'*' = the 95% interval of the difference from the current weights does not contain 0.")
    print(f"saved: {out / 'weights_grid.csv'}, {out / 'weights_best.csv'}  ({time.time() - t0:.0f} s in total)")


if __name__ == "__main__":
    main()
