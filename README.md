# Learning-Interacting-Preferences-from-Ranking-with-Poisson-Processes

## Repository layout
- `data/raw/`: Kaggle "The Movies Dataset" files (`movies_metadata.csv`, `credits.csv`, `keywords.csv`, `links_small.csv`, `ratings_small.csv`); not committed.
- `data/processed/`: shared catalogue, feature matrix and preference lists (see its README).
- `preference_dataset/`: notebook that builds the users' top-10 preference lists.
- `scripts/build_features.py`: builds `data/processed/` from `data/raw/`.
- `scripts/evaluate_embedding.py`: shared evaluation of any movie embedding (clusters, genres, taste tests).
- `scripts/inspect_map.py`: 2D maps of an embedding coloured by genre, language, saga, year, budget, and a description of its islands.
- `scripts/embed_parametric_umap.py`: parametric UMAP embedding.
- `scripts/ablation_features.py`, `scripts/tune_weights.py`: the analyses used to choose the encoding and the weights of the feature matrix (results in `results/feature_choice/`).
- `results/`: outputs of the scripts; not committed, except `results/feature_choice/`.

## Setup
```
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_features.py        # after putting the Kaggle files in data/raw/
```
