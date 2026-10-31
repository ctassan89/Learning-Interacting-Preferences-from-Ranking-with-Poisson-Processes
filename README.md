# Learning-Interacting-Preferences-from-Ranking-with-Poisson-Processes

## Repository layout
- `data/raw/`: Kaggle "The Movies Dataset" files (`movies_metadata.csv`, `credits.csv`, `keywords.csv`, `links_small.csv`, `ratings_small.csv`); not committed.
- `data/processed/`: shared catalogue, feature matrix and preference lists (see its README).
- `preference_dataset/`: notebook that builds the users' top-10 preference lists.
- `scripts/build_features.py`: builds `data/processed/` from `data/raw/`.
- `scripts/evaluate_embedding.py`: shared evaluation of any movie embedding (clusters, genres, taste tests).
- `results/`: outputs of the scripts; not committed.

## Setup (Python 3.13)
```
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
