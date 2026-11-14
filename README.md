# Learning-Interacting-Preferences-from-Ranking-with-Poisson-Processes

## Repository layout
- `data/raw/`: Kaggle "The Movies Dataset" files (`movies_metadata.csv`, `credits.csv`, `keywords.csv`, `links_small.csv`, `ratings_small.csv`); not committed.
- `data/processed/`: shared catalogue, feature matrix and preference lists (see its README).
- `preference_dataset/`: notebook that builds the users' top-10 preference lists.
- `scripts/build_features.py`: builds `data/processed/` from `data/raw/`.
- `scripts/evaluate_embedding.py`: shared evaluation of any movie embedding (clusters, genres, taste tests).
- `experiments/dimensionality_reduction/`: AE, VAE, and BetaVAE training
  entry points plus shared training, post-processing, PCA, UMAP, low-dimensional,
  and architecture-ablation utilities.
- `experiments/gibbs_sampler/`: Plackett--Luce Gibbs sampler experiments.
- `results/dimensionality_reduction/`: dimensionality-reduction outputs; not
  committed except for the curated PCA artifacts already in version control.
- `results/gibbs_sampler/`: Gibbs diagnostics and posterior samples; not committed.

## Experiments

Run experiments from the repository root with module syntax.  Dimensionality
reduction training entry points follow the model/configuration naming scheme:

```bash
python -m experiments.dimensionality_reduction.AE.train_ae_A
python -m experiments.dimensionality_reduction.VAE.train_vae_A
python -m experiments.dimensionality_reduction.BetaVAE.train_beta_vae_A --beta 0.1
python -m experiments.dimensionality_reduction.visualize_pca
```

The simulated Plackett--Luce sampler is available as:

```bash
python -m experiments.gibbs_sampler.gibbs_simulated_data
```

It writes trace plots, diagnostics, posterior samples, and posterior predictive
checks to `results/gibbs_sampler/simulated/`.  A real-data Gibbs entry point is
not included in the current working tree; `results/gibbs_sampler/real_data/` is
reserved for it when supplied.

## Setup (Python 3.13)
```
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
