# Text embeddings of the movies (Sara)

Step 1 of the project asks for a movie embedding that
1. uses **only the movies' own characteristics** (never user ratings),
2. has **well separated clusters**, also after projection to 2D/3D,
3. **generalizes to new movies**: an unseen movie must be placeable in the same space.

This folder tries the *text* route: describe every movie with a short piece of text and
turn it into a vector with a text encoder. Other members of the team try linear methods,
Kernel PCA, autoencoders and cluster-focused models on the shared feature matrix; all
methods use the same catalogue, split and evaluation, so they can be compared directly.

## Approach in one picture

```
movies_clean.csv ──► input text (3 variants) ──► encoder (5) ──► L2-normalized vectors
                                                                   │
                              results/TextEmbeddings/<run>/embedding.npz (9,025 movies)
                                                                   │
           shared evaluation (scripts/evaluate_embedding.py) + extra metrics, on
           the native space and on UMAP 2D/3D fitted on training movies only
```

One **run** = one encoder × one input variant, named `text_<encoder>_<variant>`
(e.g. `text_mpnet_meta_core`): 5 encoders × 3 inputs = **15 runs**.

## Data and split (shared with the team)

- Movies: the 9,025 films of `data/processed/catalogue.csv`, with their metadata from
  `data/processed/movies_clean.csv` (built by `scripts/build_features.py`).
- Split: the shared **8,171 train / 854 held-out** split (seed 2026). Held-out movies play
  the role of "new movies".
- Anything that is fitted (LSA vocabulary and SVD, the choice of hyper-parameters, UMAP,
  KMeans/GMM) uses **training movies only**. Hyper-parameters are chosen on a validation
  subset (15%) of the training movies, never on held-out movies.
- Ratings (`data/raw/ratings_small.csv`) are used **only to evaluate** (taste tests).

## Input texts (`text_inputs.py`)

| variant | content |
|---|---|
| `overview` | overview + tagline (title if a movie has no overview: 13 movies) |
| `meta_core` | `Title: Toy Story (1995). Genres: Animation, Comedy, Family. Director: John Lasseter. Cast: Tom Hanks, Tim Allen, Don Rickles. Keywords: jealousy, toy, boy, friendship, friends. Plot: …` |
| `meta_rich` | `meta_core` + language, writers (top 3), production companies (top 3), collection/saga |

Design choices:
- Long lists are cut (3 actors in billing order, 5 keywords, 2 directors, 3 writers,
  3 companies) so that the text stays short and the plot is not drowned.
- **The plot comes last**: if a text is longer than the encoder's limit (256–512 tokens),
  only the end of the plot is cut, never the metadata. The share of truncated texts is
  saved in each run's `config.json` (e.g. 0.8% for MiniLM on `meta_core`).
- Empty fields are **omitted**, not written as "unknown", so movies with missing metadata
  do not share a token that would pull them together.

## Encoders (`encoders.py`)

| key | model | dim | fitted on our data? |
|---|---|---|---|
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | no (pretrained) |
| `mpnet` | `sentence-transformers/all-mpnet-base-v2` | 768 | no (pretrained) |
| `e5` | `intfloat/e5-base-v2` (prefix `"query: "`) | 768 | no (pretrained) |
| `bge` | `BAAI/bge-base-en-v1.5` | 768 | no (pretrained) |
| `lsa` | TF-IDF (1–2-grams, English stop words, sublinear tf) + truncated SVD | 100–300 | **yes, train only** |

- **Pretrained models** (Sentence-BERT family, e5, bge) are used as they are: nothing is
  learned from our movies, so there is no possible leakage.
- **LSA** is the classical baseline. The TF-IDF vocabulary, IDF weights and SVD are fitted
  on training texts only; the number of components (100 / 200 / 300) is chosen on the
  validation subset of the training movies. It replaces **Doc2Vec**, which was planned but
  dropped because its library (gensim) does not install on Python 3.14.
- Every vector is **L2-normalized**, so the Euclidean distance used by the shared evaluator
  is equivalent to cosine similarity (`|a − b|² = 2 − 2·cos(a, b)`).

**New movies** (requirement 3): a new movie only needs its metadata. Build its text with the
same function and embed it; pretrained models need nothing else, LSA uses the encoder fitted
on training movies (saved as `encoder.joblib`):

```python
from experiments.TextEmbeddings.common import embed_new_movies
movie = {"title": "...", "year": 2024, "genres": ["Drama"], "directors": ["..."],
         "cast": ["...", "..."], "keywords": ["..."], "text": "Plot of the movie..."}
z = embed_new_movies([movie], "mpnet", "meta_core")   # 1 x 768, same space
```

## Main functions

| file | function | what it does |
|---|---|---|
| `data.py` | `load_movies()` | loads `movies_clean.csv` in catalogue order, checks ids and split, parses the `|` lists |
| | `validation_split()` | train → sub-train / validation positions (hyper-parameter choice only) |
| `text_inputs.py` | `build_text()`, `build_texts()` | input text of one movie / of all movies for a variant |
| `encoders.py` | `make_encoder(key)` | returns an encoder with `.embed(texts)` (and `.fit()` for LSA) |
| `common.py` | `run_text_embedding(key, variant)` | builds texts, embeds, saves, evaluates one run |
| | `embed_catalogue()` | embeds the 9,025 movies, or reuses the cached `embedding.npz` |
| | `select_params()` | chooses the LSA configuration on the validation subset |
| | `evaluate_run()` | shared evaluation + extra metrics on native, UMAP 2D and UMAP 3D |
| | `embed_new_movies()` | places new movies in the space of a run |
| `extra_metrics.py` | `extra_metrics()` | all extra metrics for one space (see below) |
| `compare.py` | `main()` | builds `summary.csv` and `summary.png`, optionally with the baselines |

## Evaluation

### 1. Shared protocol (`scripts/evaluate_embedding.py`, same for every team member)
Computed on the native space and on **UMAP 2D/3D fitted on training movies** (seed 2026):

| metric | meaning | better |
|---|---|---|
| `k10/k20_silhouette` | KMeans (10 / 20 clusters, fitted on train): cohesion vs separation | higher |
| `k10/k20_davies_bouldin` | average similarity between each cluster and its closest one | lower |
| `k10/k20_calinski_harabasz` | between-cluster vs within-cluster dispersion | higher |
| `k10/k20_genre_purity` | share of movies whose first genre is the majority one of their cluster | higher |
| `genre_knn_train/heldout` | mean genre Jaccard between a movie and its 10 nearest neighbours (vs `genre_random_pairs`) | higher |
| `taste_auc`, `_time`, `_user` | do movies in the same user's top-10 lie closer than random pairs / time-matched pairs / that user's disliked movies? 0.5 = no signal, with 95% bootstrap interval | higher |
| `*_heldout` | the same metrics for held-out movies | close to train |

It also writes the nearest neighbours of 8 reference movies (`<run>_neighbors.txt`) and the
2D map coloured by genre (`<run>_umap2d_map.png`).

### 2. Extra metrics (`extra_metrics.py` → `results/TextEmbeddings/extra_metrics.csv`)
Reused from Carlo's `metrics/metrics.py` (everything that applies to an embedding without
a decoder; reconstruction errors do not apply to text encoders):

| metric | meaning |
|---|---|
| `gmm_*` | Gaussian mixture with 10 components (diagonal covariances in high dimension, full in 2D/3D): silhouette, DB, CH, multi-label purity, held-out assignment |
| `hdbscan_*` | density-based clusters in 2D/3D: number of clusters, noise share, scores without noise |
| `kmeans_purity_ml(_heldout)` | multi-label genre purity (a movie counts for every genre it has) |
| `heldout_genre_jaccard_cosine` | held-out → train neighbour genre similarity with the cosine metric |

New, chosen for requirements 2 and 3:

| metric | meaning | ideal |
|---|---|---|
| `trustworthiness` | how well UMAP 2D/3D keeps the 10-nearest-neighbour structure of the native space (are the clusters we see in 2D real?) | 1 |
| `placement_gap` | median distance from a held-out movie to its nearest train movie ÷ the same for train movies (leave-one-out) | ≈ 1 (> 1: new movies land in empty regions) |
| `genre_f1_micro/macro_heldout` | genres of a held-out movie predicted by the vote of its 10 nearest train movies (vs `_train`) | high, close to train |
| `same_collection_nn`, `same_director_nn` | share of movies whose nearest neighbour is from the same saga / director | high; trivially high for `meta_*` inputs that contain the names, so mainly informative for `overview` |
| `anisotropy` | mean cosine of random pairs: high means vectors fill a narrow cone, which hurts kNN and clustering | low |

### 3. Reference rows (`compare.py --with-baselines`)
- `ref_tfidf_text`: the text block of the shared feature matrix (TF-IDF + SVD of
  overview + tagline, columns 0–99);
- `ref_feature_matrix`: the whole shared feature matrix v2 (902 columns).

## How to run

From the repository root, Python 3.14 venv:

```bash
pip install -r requirements.txt -r experiments/TextEmbeddings/requirements.txt
python -m experiments.TextEmbeddings.embed_mpnet --input all       # or overview / meta_core / meta_rich
python -m experiments.TextEmbeddings.embed_lsa --input all
python -m experiments.TextEmbeddings.compare --with-baselines       # summary table and chart
python -m unittest tests.test_text_embeddings
```

Options of the `embed_*` scripts: `--force` (re-embed even if cached), `--no-umap`,
`--skip-eval` (only compute and save the embeddings). Pretrained models are downloaded
from Hugging Face the first time (~90–450 MB each). A full run of the grid (15 runs + baselines)
took about 1.5 hours on a laptop, mostly UMAP.

## Outputs (`results/`, not committed)

```
results/
├── metrics.csv                         shared metrics, one row per (method, space)
├── <run>_neighbors.txt                 neighbours of the reference movies
├── <run>_umap2d_map.png                2D map coloured by genre
└── TextEmbeddings/
    ├── extra_metrics.csv               extra metrics, one row per (method, space)
    ├── summary.csv, summary.png        comparison of all runs (compare.py)
    └── <run>/
        ├── embedding.npz               Z (9,025 × d) + movieId, shared format
        ├── umap2d.npz, umap3d.npz      the projections used for the metrics
        ├── config.json                 model, parameters, timing, truncated share
        ├── sample_texts.txt            the text the encoder actually read (4 movies)
        └── encoder.joblib              LSA only: fitted encoder for new movies
```

## Results (full grid, 3 November 2026)

All numbers from `results/TextEmbeddings/summary.csv` (regenerate with the commands above).

**A warning before reading the genre metrics.** The `meta_*` texts contain the genre names
("Genres: Animation, Comedy"), and the shared feature matrix has a genre block with weight
1.5. Genre purity, kNN genre agreement and genre F1 are therefore partly circular for these
rows: they reward having the genres in the input. The **taste tests** use user rankings that
no method has seen, so they are the fairest comparison.

### Taste tests (native space, 95% bootstrap interval)

| run | random pairs | time-matched | same user's dislikes |
|---|---|---|---|
| `text_e5_meta_core` | **0.592** [0.584, 0.600] | 0.571 [0.564, 0.579] | 0.578 [0.569, 0.586] |
| `text_mpnet_meta_rich` | 0.590 [0.582, 0.598] | 0.569 [0.561, 0.577] | **0.579** [0.571, 0.587] |
| `ref_feature_matrix` | 0.585 [0.577, 0.593] | **0.572** [0.563, 0.580] | 0.561 [0.551, 0.570] |
| `text_lsa_meta_core` | 0.580 [0.573, 0.586] | 0.563 [0.556, 0.569] | 0.539 [0.532, 0.547] |
| `text_bge_meta_core` | 0.564 [0.556, 0.571] | 0.553 [0.545, 0.561] | 0.544 [0.534, 0.553] |
| `text_minilm_overview` | 0.542 | 0.538 | 0.550 |
| `ref_tfidf_text` | 0.511 | 0.510 | 0.503 |

The best text embeddings (e5 and mpnet on metadata text) are **as good as the shared feature
matrix** on random and time-matched pairs (overlapping intervals), and **better at separating
a user's liked from disliked movies** (0.578–0.579 vs 0.561).

### Main findings

1. **Input text matters more than the model.** For every pretrained model, `meta_core` beats
   `overview` on all metrics, e.g. e5 taste AUC 0.552 → 0.592, held-out genre F1 0.60 → 0.72.
   `meta_rich` is about the same as `meta_core`: writers, companies and saga add little.
2. **Models.** e5 is the best overall; mpnet is very close on taste; bge has the best
   neighbours by eye but lower taste AUC; MiniLM is the weakest but about 6× faster to encode (47 s vs ~5 min for the
   catalogue on a laptop).
3. **New movies are placed well** (requirement 3): `placement_gap` is 0.97–1.01 for every text
   run, and held-out metrics match training ones (e.g. e5 `meta_core` kNN genre Jaccard 0.446
   held-out). Pretrained encoders have nothing fitted, so there is nothing to overfit.
4. **Cluster separation** (requirement 2) after UMAP 2D: silhouette (k=10) 0.35–0.38 for the
   pretrained runs, 0.41 for LSA on metadata, 0.47 for the shared matrix. The maps show one
   continent organised by genre (comedy, drama, horror, action, documentaries at the edges);
   HDBSCAN finds only 2–7 dense groups for e5/bge, i.e. clusters are **regions, not islands**.
   Trustworthiness of the 2D map is 0.76–0.83 (0.93 for LSA on metadata), so the 2D picture
   reflects the real neighbourhoods reasonably well.
5. **LSA (TF-IDF + SVD) on metadata is a strong classical baseline**: highest genre scores
   among text runs (partly circular, see warning) and good taste AUC, but it relies on exact
   word overlap. On `overview` alone it is the weakest method.
6. **Do not trust `text_lsa_overview`'s silhouette (0.70).** Its trustworthiness is the
   lowest (0.58) and its map shows a single dense blob of movies whose overview shares
   almost no vocabulary with the rest: KMeans isolates the blob, which inflates the
   silhouette without any meaningful separation.
7. **Neighbours** (`results/<run>_neighbors.txt`), e.g. `text_bge_meta_core`:
   Pulp Fiction → Jackie Brown, Reservoir Dogs, Heat; The Matrix → its two sequels, The
   Animatrix; Toy Story → its sequels, Toy Story of Terror!. With `overview` only the plot
   drives neighbours (Toy Story → Child's Play 3, a film about a living toy).

**Recommended text embedding: `text_e5_meta_core`** (768 dimensions): best taste AUC, good
held-out placement, sensible neighbours; `text_mpnet_meta_rich` is an equivalent alternative.

## Limitations

- Silhouette values in the native space (hundreds of dimensions) are low for every text
  encoder (≈ 0.02); cluster separation should be compared on the UMAP 2D/3D projections.
- Pretrained encoders were trained on web text that may include descriptions of these
  movies; this does not use our ratings, but it is outside knowledge.
- For `meta_*` inputs, metrics about saga/director neighbours are partly circular, because
  the names are in the text.
- Most overviews are in English; for the few non-English ones the English models are weaker.