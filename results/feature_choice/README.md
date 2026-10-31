# How the feature matrix was chosen (31 October 2026)

Evidence for the choices in `data/processed/README.md`. "v2" in these files is the committed matrix.

- `ablation.csv`: every variant of the matrix (v1, v2, each block alone, v2 without each block) with the three taste tests of `scripts/evaluate_embedding.py` (random, time-matched, top-10 vs disliked), 95% bootstrap intervals over users, and genre agreement of the nearest neighbours. From `scripts/ablation_features.py`.
- `weights_grid.csv`: 108 weight combinations, taste AUC on half A of the users. `weights_best.csv`: the 5 best on half A and the chosen weights, checked on half B. From `scripts/tune_weights.py`.
- `map_v1_genre.png`, `neighbors_v1.txt`: UMAP 2D map and nearest neighbours of version 1 (SVD encoding).
- `map_v2_genre.png`, `map_v2_budget.png`, `islands_v2.txt`, `neighbors_v2.txt`: the same for version 2.
