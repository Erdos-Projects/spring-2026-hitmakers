# Hitmakers (FEEL FREE TO MAKE CHANGE! PUTTING THE FORMAT HERE FOR NOW!)

> **Can we predict whether a newly charting artist will become a hitmaker or a one-hit wonder?**

This project builds a machine-learning pipeline that predicts whether an artist who scores their first Billboard Hot 100 top-20 hit will go on to chart again (`top_20_hitmaker = 1`) or remain a one-hit wonder (`top_20_hitmaker = 0`).

| | |
|---|---|
| **Dataset** | 759 artists × 26 features (2000–2019 debut window) |
| **Target** | `top_20_hitmaker` — binary (1 = multiple top-20 hits, 0 = exactly one) |
| **Class balance** | ~57 % one-hit wonders · ~43 % hitmakers |
| **Best model(Edit later)** | XGBoost (Test AUC ≈ 0.78, tuned with Optuna + forward selection) |

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Sources](#data-sources)
3. [Data Pipeline](#data-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Model Comparison Pipeline](#model-comparison-pipeline)
6. [Models Evaluated](#models-evaluated)
7. [Model Selection & Stability](#model-selection--stability)
8. [Robustness Check: Naked Models](#robustness-check-naked-models)
9. [Final Model: Random Forest](#final-model-random-forest)
10. [Repository Structure](#repository-structure)
11. [Getting Started](#getting-started)

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

The model comparison notebook (`Model_Comparison_Final.ipynb`) runs an **8-step automated pipeline** per model:

| Step | Name | Purpose |
|:----:|------|---------|
| 1 | **Full-Feature Optuna Tuning** | Tune hyperparameters on all 26 features. Objective: $\text{AUC} - \lambda \times \text{gap}$ ($\lambda = 0.3$) — rewards high CV AUC while penalising train/val overfit gap |
| 2 | **CV Feature Importance** | SHAP `TreeExplainer` for tree models; permutation importance for LR / AdaBoost. Computed on validation folds only |
| 3 | **Genre Consolidation** | Keep high-signal genres (importance > mean for SHAP models, > 0 for permutation); merge remainder → `artist_genre_other` |
| 4 | **Forward Selection** | Greedy feature addition ordered by Step 2 importance; track CV AUC and overfit gap at each $n$ |
| 5 | **Optuna Re-Tune + Winner** | Re-tune on $n_{\text{peak}}$ and $n_{\text{gap}}$ candidates; select winner by penalised score (min 5 features) |
| 6 | **Centrality Ablation** | Test all $2^3 = 8$ subsets of the 3 centrality features; keep the subset that maximises the penalised score ($\text{CV AUC} - \lambda \times \text{gap}$) |
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
| Logistic Regression | Linear | L2-regularised, `StandardScaler` applied — preliminary study only |
| Random Forest | Ensemble (bagging) | `class_weight='balanced'` for imbalance |
| XGBoost | Gradient boosting | L1/L2 regularisation, column sampling |
| LightGBM | Gradient boosting | Histogram-based, leaf-wise growth |
| CatBoost | Gradient boosting | Ordered boosting, built-in regularisation |
| AdaBoost (Linear) | Adaptive boosting | Logistic regression base learner |
| AdaBoost (Tree) | Adaptive boosting | Decision-tree stump base learner |

---

## Model Selection & Stability

With only 759 artists (607 in training) and 5-fold CV (~121 artists per fold), cross-validation AUC estimates are noisy. Optuna can exploit this noise, producing hyperparameters that look good on CV but don't generalise — a form of hyperparameter overfitting.

To check this, we ran a **bootstrap validation** (`Bootstrap_validation.ipynb`, B=25): resample the training set with replacement, run the full tuning pipeline each iteration, evaluate on the fixed test set.

![Bootstrap Validation](Complementary%20Study/Bootstrap_validation.png)

| Model | Single-run AUC | Bootstrap mean | Bootstrap std | 90% CI | Δ vs Baseline | Δ vs Single-run |
|-------|:--------------:|:--------------:|:-------------:|--------|:-------------:|:---------------:|
| XGBoost | 0.774 | 0.739 | 0.023 | [0.697, 0.775] | +0.233 | −0.035 |
| Random Forest | 0.767 | 0.745 | 0.021 | [0.715, 0.774] | +0.239 | −0.022 |
| CatBoost | 0.753 | 0.719 | 0.032 | [0.671, 0.750] | +0.213 | −0.034 |

All three models consistently outperform the stratified baseline (~0.506 AUC) by over 0.21 AUC points across all bootstrap resamples — confirming the signal in the data is real, not an artifact of a single lucky split.

**Random Forest shows the smallest standard deviation, tightest confidence interval, and smallest drop from the single-run AUC (−0.022)**, making it the most stable and reproducible choice on a dataset of this size. RF is selected as the final model.

---

## Robustness Check: Naked Models

To test whether performance depends on tuning or on the signal in the data, we ran a bootstrap (`Naked_Model_Bootstrap_Threshold.ipynb`, B=100) using Random Forest with fixed, untuned hyperparameters — no Optuna, no feature selection, all 26 features.

![Naked RF Bootstrap](Complementary%20Study/Naked_RF_Bootstrap.png)

| Metric | Tuned RF (single run) | Naked RF mean | Naked RF std | 90% CI |
|--------|:---------------------:|:-------------:|:------------:|--------|
| AUC | 0.767 | 0.760 | 0.012 | [0.738, 0.779] |
| Precision | 0.617 | 0.580 | 0.044 | [0.523, 0.672] |
| Recall | 0.758 | 0.749 | 0.071 | [0.636, 0.848] |
| F1 | 0.680 | 0.650 | 0.020 | [0.620, 0.681] |

The tuned single-run AUC (0.767) sits well within the naked bootstrap distribution — the two are statistically indistinguishable. **The signal in the data is the main driver of performance, not the tuning.**

However, recall tells a different story. With a std of 0.071 and a 90% CI spanning 0.21, a naked RF can produce recall anywhere from 0.64 to 0.85 depending on the training sample. For a model intended to reliably identify hitmakers, this level of variability is unacceptable.

---

## Final Model: Random Forest

The naked bootstrap established that recall is too variable without any tuning. Rather than running the full heavy pipeline, `Final_Model_RandomForest.ipynb` uses a deliberately **light and conservative** tuning approach: 8 Optuna trials (vs 30+ elsewhere), a stronger gap penalty (λ=0.5 vs 0.3), tighter regularization bounds, SHAP-based genre consolidation, top 12 features by importance, centrality ablation, and a precision floor of 0.60 on threshold tuning.

**Best parameters found:** `n_estimators=181`, `max_depth=2`, `min_samples_leaf=14`, `max_features=log2` — shallow trees, strongly regularized.

| Metric | Value |
|--------|-------|
| Test AUC | 0.773 |
| Train–Test Gap | 0.008 |
| Log Loss | 0.597 |
| Precision | 0.627 |
| Recall | 0.712 |
| F1 | 0.667 |
| Threshold | 0.50 |
| Total leaves | 717 (avg 4.0 per tree) |

**Bootstrap stability** (`Final_Model_RandomForest.ipynb`, B=100, full pipeline):

| Metric | Single run | Mean | Std | 95% CI |
|--------|:----------:|:----:|:---:|--------|
| AUC | 0.773 | 0.767 | 0.014 | [0.739, 0.791] |
| Precision | 0.627 | 0.584 | 0.054 | [0.494, 0.690] |
| Recall | 0.712 | 0.762 | 0.067 | [0.636, 0.909] |
| F1 | 0.667 | 0.656 | 0.020 | [0.618, 0.693] |

Each bootstrap iteration re-runs the full pipeline — feature selection, 8 Optuna trials, and threshold tuning — making this the most honest stability check. All single-run metrics fall within their bootstrap confidence intervals, confirming the result is reproducible and not a lucky draw. Recall carries the most variance (std=0.067), reflecting the fundamental limit of a dataset of this size rather than a flaw in the model.

---

## Repository Structure

```
├── README.md                          # This file
├── DATA_PIPELINE.md                   # Detailed data pipeline documentation
├── data_preparation.ipynb             # Main data preparation notebook
├── Hitmakers_model_compare.ipynb      # Full 8-step model comparison pipeline
├── Hitmakers_temporal_split.ipynb     # Temporal train/test split variant
│
├── df_artists_final.csv               # Model-ready dataset (759 × 27)
├── catboost_info/                     # CatBoost training outputs and logs
│   ├── catboost_training.json
│   ├── learn_error.tsv
│   ├── time_left.tsv
│   ├── learn/
│   └── tmp/
├── Datasets/
│   ├── billboard_200_albums_final.csv
│   ├── Main_Data/
│   │   ├── billboard_hot100_1958_2026.csv
│   │   ├── billboard_hot100_songs_final.csv
│   │   ├── df_albums.csv
│   │   ├── df_artists.csv
│   │   ├── df_artists_network_metrics.csv
│   │   ├── df_artists_with_network_metrics.csv
│   │   └── df_songs.csv
│   └── Pipeline_supplement/
│       ├── billboard_data_cleaning_pt_1_McNally.ipynb
│       ├── billboard_data_cleaning_pt_2_McNally.ipynb
│       ├── billboard_data_cleaning_pt_4.ipynb
│       ├── billboard_data_cleaning_pt_5_genre.ipynb
│       ├── billboard_data_cleaning_pt_6_condensing.ipynb
│       ├── billboard_data_cleaning_pt_7_filling_in_missing_artist_ids.ipynb
│       ├── billboard_data_cleaning_pt_8_new_network_metrics.ipynb
│       ├── billboard_network_data_merge.ipynb
│       ├── df_artists_clean.ipynb
│       ├── df_songs_create.ipynb
│       ├── EDA_1_+billboard_data_cleaning_pt_3.ipynb
│       ├── google_trends_engineered.ipynb
│       └── Googletrend_dataset.ipynb
├── ml_sandbox/
│   ├── ml_sandbox_13_model_compare.ipynb
│   ├── ml_sandbox_14_first_charting_song.ipynb
│   ├── ml_sandbox_15_df_artists_final_clean.ipynb
│   ├── ml_sandbox_16_final_xgboost_tuning.ipynb
│   ├── ml_sandbox_17_model_compare_final_xgboost_tuning.ipynb
│   ├── ml_sandbox_18_catboost.ipynb
│   ├── ml_sandbox_18_model_compare_final_xgboost_tuning_with_threshold_tuning.ipynb
│   ├── ml_sandbox_19_adaboost_linear.ipynb
│   ├── ml_sandbox_20_adaboost_tree.ipynb
│   ├── ml_sandbox_21_pipeline_compare.ipynb
│   ├── ml_sandbox_22_explainability.ipynb
│   ├── ml_sandbox_23_model_selection.ipynb
│   ├── ml_sandbox_24_executive_summary.ipynb
│   └── ml_sandbox_25_google_trends_comparison.ipynb
├── Preliminary Study/
│   ├── GS/
│   │   ├── df_songs_google_decay.csv
│   │   ├── extra_model_compare.ipynb
│   │   ├── google_trends_engineered.ipynb
│   │   ├── Googletrend_dataset.ipynb
│   │   ├── more_boost.ipynb
│   │   └── catboost_info/
│   ├── McNally_Jupyter_Notebooks/
│   │   ├── billboard_data_cleaning_pt_1_McNally.ipynb
│   │   ├── billboard_data_cleaning_pt_2_McNally.ipynb
│   │   ├── ...
│   ├── McNally_Network_Analysis_Data/
│   │   ├── master_edge_list_all_artists.parquet
│   │   ├── master_edge_list_top_10_artists_only.parquet
│   │   ├── networks_all_artists/
│   │   └── networks_top_10_artists_only/
│   ├── Old_CSVs/
│   │   ├── billboard_24years_lyrics_spotify.csv
│   │   ├── df_albums_old.csv
│   │   ├── ...
│   └── UPDATE_YUNDI/
│       ├── Build_Comprehensive_Dataset.ipynb
│       ├── COMPREHENSIVE_DATASET_README.md
│       ├── DATA_PIPELINE_DIAGRAMS.md
│       ├── EDA_comp.ipynb
│       ├── Modeling_Preliminary.ipynb
│       ├── MusicBrainz_Data Understanding.ipynb
│       ├── MusicBrainz_DataExtract.ipynb
│       ├── artist_child_csv/
│       ├── df_comprehensive_hitmaker_prediction.csv
│       ├── df_comprehensive_hitmaker_prediction_DICTIONARY.csv
│       ├── ml_sandbox_0306.ipynb
│       └── ml_sandbox_update.ipynb
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
