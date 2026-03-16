# Hitmakers — Predicting Billboard Success

> **Can we predict whether a newly charting artist will become a hitmaker or a one-hit wonder?**

This project builds a machine-learning pipeline that predicts whether an artist who scores their first Billboard Hot 100 top-20 hit will go on to chart again (`top_20_hitmaker = 1`) or remain a one-hit wonder (`top_20_hitmaker = 0`).

| | |
|---|---|
| **Dataset** | 759 artists × 26 features (2000–2019 debut window) |
| **Target** | `top_20_hitmaker` — binary (1 = multiple top-20 hits, 0 = exactly one) |
| **Class balance** | ~57 % one-hit wonders · ~43 % hitmakers |
| **Best model** | XGBoost (Test AUC ≈ 0.78, tuned with Optuna + forward selection) |

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Sources](#data-sources)
3. [Data Pipeline](#data-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Model Comparison Pipeline](#model-comparison-pipeline)
6. [Models Evaluated](#models-evaluated)
7. [Key Results](#key-results)
8. [Repository Structure](#repository-structure)
9. [Getting Started](#getting-started)

---

## Project Overview

Artists break onto the Billboard Hot 100 every year, but only a fraction sustain chart success. This project combines Billboard chart history (1958–2026), MusicBrainz metadata, collaboration-network analysis, and genre tagging to predict — at the moment of an artist's first top-20 hit — whether they will chart again.

The modelling pipeline is fully automated: hyperparameter tuning (Optuna), feature selection (forward selection + SHAP), genre consolidation, centrality ablation, and leakage-safe threshold tuning all run end-to-end in a single notebook.

---

## Data Sources

| File / Source | Description |
|---------------|-------------|
| `billboard_hot100_1958_2026.csv` | Week-by-week Billboard Hot 100 entries (1958–2026), ~350 K rows |
| `billboard_hot100_songs_final.csv` | Deduplicated Hot 100 songs (one row per unique song, ~30 K rows) |
| `billboard_200_albums_final.csv` | Deduplicated Billboard 200 albums (~19 K rows) |
| `kang_data_w_spotify.csv` | Kang/Kwon academic dataset with verified Spotify IDs |
| `gabminamedez_spotify_data.csv` | Spotify audio features ([GitHub](https://github.com/gabminamedez/spotify-data)) |
| `google_trends_top3000.csv` | Monthly Google Trends interest for top artists (via `pytrends`) |
| MusicBrainz PostgreSQL DB (Docker) | Artist IDs, genre tags, collaboration edges |

---

## Data Pipeline

The raw data goes through **8 feature-engineering stages** before reaching the model-ready dataset. Full details are in `data_preparation.ipynb` and the notebooks under `pipeline_supplement/`.

```
External Sources → Stage 1–8 → df_artists.csv (13,655 artists × 44 cols)
                                       │
                               data_preparation.ipynb
                                       │
                               df_artists_final.csv (759 artists × 26 cols + target)
```

### Stage Summary

| Stage | Name | Key Transformations |
|:-----:|------|---------------------|
| 1 | **Billboard Cleaning & Artist Verification** | Split collaborations ("Drake Feat. Rihanna" → separate rows); MusicBrainz fuzzy matching; name normalization |
| 2 | **Artist / Song Aggregation & Target** | Per-artist chart stats (total songs, #1 hits, top-10/20/50 counts); milestone-year columns; `top_10_hitmaker_songs` flag |
| 3 | **Artist Name Deduplication** | Removed collab artifacts (`feat.`/`ft.`); deduplicated `the`/non-`the` variants; manual keep-lists |
| 4 | **Label & Genre Tagging** | MusicBrainz genre queries; 546-genre → 18-major-category mapping; one-hot genre columns |
| 5 | **MusicBrainz ID Corrections** | Fixed 564 wrong MBIDs; filled missing IDs from Kang dataset |
| 6 | **Song-Level Features** | Created `df_songs` with recording MBIDs, genre tags, Spotify audio features |
| 7 | **Network Metrics** | Built collaboration graph from MusicBrainz; computed degree, closeness, betweenness, eigenvector centrality (rolling 5-year windows) |
| 8 | **Final Assembly** | Dropped redundant columns; merged Spotify features for first top-20 hit; created `top_20_hitmaker` target |

### `data_preparation.ipynb` (df_artists → df_artists_final)

| Step | Operation |
|:----:|-----------|
| 1 | Filter to artists with first top-20 hit in **2000–2019** |
| 2 | Drop identifier / non-feature columns (MBIDs, Spotify IDs, raw genre strings) |
| 3 | One-hot encode 18 genre categories + `artist_genre_unknown` flag + genre count |
| 4 | Drop null targets and duplicate rows |
| 5 | Drop Spotify audio features (~40 % missing) |
| 6 | Drop collinear network metrics (`degree_centrality`, `power_of_connected_artists`) |
| 7 | Fill remaining network metric nulls with 0 |

---

## Feature Engineering

### Feature Categories (26 total)

| Category | Features | Examples |
|----------|:--------:|---------|
| **Chart statistics** | ~5 | `total_charting_songs`, `#1_hit_count`, `highest_charting_position`, `wks_on_chart` |
| **Genre (one-hot)** | ~18 + count | `artist_genre_Pop`, `artist_genre_Hip Hop/Rap`, `#_of_genres_artist` |
| **Network metrics** | 3 | `harmonic_closeness_centrality`, `betweenness_centrality`, `eigenvector_centrality` (all rolling 5-year, at year of first top-20 hit) |

### 18 Major Genre Categories

> Blues · Classical · Country/Americana · Easy Listening/Vocal · Electronic/Dance · Experimental/Avant-Garde · Folk · Gospel/Christian/Religious · Hip Hop/Rap · Jazz · Latin · Metal · Pop · Punk/Hardcore · R&B/Soul/Funk · Reggae/Caribbean · Rock · World Music

### Network Metrics

| Metric | Meaning |
|--------|---------|
| `harmonic_closeness_centrality` | Average distance to all other artists (handles disconnected components) |
| `betweenness_centrality` | How often an artist is a "bridge" between communities |
| `eigenvector_centrality` | Connected to other well-connected artists |

---

## Model Comparison Pipeline

The model comparison notebook (`Hitmakers_model_compare.ipynb`) runs an **8-step automated pipeline** per model:

| Step | Name | Purpose |
|:----:|------|---------|
| 1 | **Full-Feature Optuna Tuning** | Tune hyperparameters on all 26 features. Objective: $\text{AUC} - \lambda \times \text{gap}$ ($\lambda = 0.3$) — rewards high CV AUC while penalising train/val overfit gap |
| 2 | **CV Feature Importance** | SHAP `TreeExplainer` for tree models; permutation importance for LR / AdaBoost. Computed on validation folds only |
| 3 | **Genre Consolidation** | Keep high-signal genres (importance > mean for SHAP models, > 0 for permutation); merge remainder → `artist_genre_other` |
| 4 | **Forward Selection** | Greedy feature addition ordered by Step 2 importance; track CV AUC and overfit gap at each $n$ |
| 5 | **Optuna Re-Tune + Winner** | Re-tune on $n_{\text{peak}}$ and $n_{\text{gap}}$ candidates; select winner by penalised score (min 5 features) |
| 6 | **Centrality Ablation** | Test all $2^k$ subsets of centrality features; drop any subset that improves raw CV AUC |
| 7 | **Final Evaluation** | Fit on full training set; evaluate on held-out test set (single touch) |
| 8 | **OOF Threshold Tuning** | Leakage-safe threshold from out-of-fold training predictions; precision ≥ 0.60 fallback |

### Validation Strategy

- **80 / 20 stratified train-test split** (`random_state=42`)
- **5-fold stratified cross-validation** on training set
- Test set touched **once** for final reporting

---

## Models Evaluated

| Model | Type | Key Properties |
|-------|------|----------------|
| Stratified Baseline | Baseline | Predicts class ratio (~43 % hitmaker) |
| Logistic Regression | Linear | L2-regularised, `StandardScaler` applied |
| Random Forest | Ensemble (bagging) | `class_weight='balanced'` for imbalance |
| XGBoost | Gradient boosting | L1/L2 regularisation, column sampling |
| CatBoost | Gradient boosting | Ordered boosting, built-in regularisation |
| AdaBoost | Adaptive boosting | Decision-tree stumps (Freund & Schapire 1997) |

---

## Key Results

All models substantially outperform the stratified baseline (~0.50 AUC). The pipeline produces per-model diagnostics including ROC curves, confusion matrices, calibration curves, precision-recall curves, cumulative gain (lift) charts, and both unsigned and signed SHAP heatmaps.

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **AUC-ROC** | Primary ranking metric (discrimination ability) |
| **Log Loss** | Penalises confident wrong predictions |
| **Brier Score** | Calibration — closeness of predicted probabilities to true outcomes |
| **Precision / Recall / F1** | Classification performance at tuned threshold |
| **Lift** | Cumulative gain at 10 %, 20 %, 30 %, 50 % of artists screened |

---

## Repository Structure

```
├── README.md                          # This file
├── DATA_PIPELINE.md                   # Detailed data pipeline documentation
├── data_preparation.ipynb             # df_artists.csv → df_artists_final.csv
├── Hitmakers_model_compare.ipynb      # Full 8-step model comparison pipeline
├── Hitmakers_temporal_split.ipynb     # Temporal train/test split variant
│
├── df_artists_final.csv               # Model-ready dataset (759 × 27)
├── df_artists.csv                     # Pre-filtered artist dataset (13,655 × 44)
├── df_songs.csv                       # Song-level features (~30 K rows)
├── df_albums.csv                      # Album-level data
├── billboard_hot100_1958_2026.csv     # Raw Billboard Hot 100 weekly data
├── billboard_hot100_songs_final.csv   # Deduplicated songs
├── billboard_200_albums_final.csv     # Deduplicated albums
│
├── pipeline_supplement/               # Upstream pipeline notebooks (Stages 1–8)
├── McNally_Jupyter_Notebooks/         # EDA, data cleaning, network analysis notebooks
├── McNally_Network_Analysis_Data/     # Network graph data files
├── GS/                                # Google Trends experiment notebooks
├── Old_CSVs/                          # Archived intermediate CSVs
├── UPDATE_YUNDI/                      # Supplementary update notebooks
│
├── ml_sandbox_*.ipynb                 # Iterative modelling experiments
├── Model_Comparison.ipynb             # Finalized model comparison draft
```

---

## Getting Started

### Requirements

```
python >= 3.9
pandas
numpy
scikit-learn
xgboost
catboost
optuna
shap
matplotlib
seaborn
```

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd spring-2026-hitmakers
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn xgboost catboost optuna shap matplotlib seaborn
   ```

3. **Run data preparation** (optional — `df_artists_final.csv` is already provided)
   ```
   Open data_preparation.ipynb and run all cells
   ```

4. **Run the model comparison pipeline**
   ```
   Open Hitmakers_model_compare.ipynb and run all cells
   ```
   This will execute the full 8-step pipeline for all 6 models, produce diagnostic plots, and output the cross-model comparison table.

---

*Spring 2026 — Hitmakers Team Project*
