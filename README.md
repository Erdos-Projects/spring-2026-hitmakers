# Hitmakers vs. One-Hit Wonders
### Predicting Sustained Success in the Music Industry

**Authors:** [Vishal Gupta](https://github.com/vishalguptaud) · [Yundi Kong](https://github.com/Ykong21) · [James McNally](https://github.com/jamesemcnally) · [Guillermo Sanmarco](https://github.com/sanmarcog)

📄 [Executive Summary](Hitmakers%20Executive%20Summary.pdf)

> **Is it possible to predict whether a newly charting musician will become a hitmaker or a one-hit wonder?**

This project builds a machine-learning pipeline that predicts whether an artist who scores their first Billboard Hot 100 top-20 hit will produce more top 20 hits in the future (`top_20_hitmaker = 1`) or remain a one-hit wonder (`top_20_hitmaker = 0`).

| | |
|---|---|
| **Dataset** | 759 artists × 26 features (2000–2019 debut window) |
| **Target** | `top_20_hitmaker` — binary (1 = multiple top-20 hits, 0 = exactly one) |
| **Class balance** | ~57 % one-hit wonders · ~43 % hitmakers |
| **Final model** | Random Forest (Test AUC = 0.773, Recall = 0.712, light Optuna tuning) |

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
10. [Complement: Reporting Predicted Probabilities](#complement-reporting-predicted-probabilities)
11. [Repository Structure](#repository-structure)
12. [Getting Started](#getting-started)

---

## Project Overview

Artists break onto the Billboard Hot 100 every year, but only a fraction sustain chart success. This project combines Billboard chart history (1958–2026), MusicBrainz metadata, collaboration-network analysis, and genre tagging to predict — at the moment of an artist's first top-20 hit — whether they will produce additional hit songs in the future.

<!-- The modelling pipeline combines hyperparameter tuning (Optuna), feature selection (forward selection + SHAP), genre consolidation, centrality ablation, and leakage-safe threshold tuning. A lighter version of this pipeline is used for the final model, prioritising stability and interpretability over exhaustive search. -->

---

## Data Sources

| File / Source | Description |
|---------------|-------------|
| `billboard_hot100_1958_2026.csv` | Week-by-week Billboard Hot 100 entries (1958–2026), ~350 K rows |
| `billboard_hot100_songs_final.csv` | Deduplicated Hot 100 songs (one row per unique song, ~30 K rows, derived from weekly data) |
| `billboard_200_albums_final.csv` | Deduplicated Billboard 200 albums (~19 K rows, derived from weekly data) |
| `kang_data_w_spotify.csv` | Kang academic dataset with verified Spotify IDs ([GitHub](https://github.com/inwonakng/predicting-musician-success.git))|
| `google_trends_top3000.csv` | Monthly Google Trends interest for top artists (via `pytrends`) |
| MusicBrainz PostgreSQL DB (Docker) | Artist IDs, genre tags, collaboration edges |

---

## Data Pipeline

The raw data goes through **8 feature-engineering stages** before reaching the model-ready dataset. Full details are in `data_preparation.ipynb` and the notebooks under `Pipeline_supplement/`.

![Data Pipeline](Complementary%20Study/Data%20pipeline.png)

### Stage Summary

| Stage | Name | Key Transformations |
|:-----:|------|---------------------|
| 1 | **Billboard Cleaning & Artist Verification** | Split collaborations ("Drake Feat. Rihanna" → separate rows); MusicBrainz fuzzy matching; name normalization |
| 2 | **Artist / Song Aggregation & Target** | Per-artist chart stats (total songs, #1 hits, top-10/20/50 counts); milestone-year columns |
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
| `harmonic_closeness_centrality` | Average inverse distance to all other artists (handles disconnected components) |
| `betweenness_centrality` | How often an artist is a "bridge" between communities |
| `eigenvector_centrality` | How connected an artist is to other well-connected artists |

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
| SVM | Kernel-based | RBF kernel, feature scaling — preliminary study only |
| Random Forest | Ensemble (bagging) | `class_weight='balanced'` for imbalance |
| XGBoost | Gradient boosting | L1/L2 regularisation, column sampling |
| LightGBM | Gradient boosting | Histogram-based, leaf-wise growth |
| CatBoost | Gradient boosting | Ordered boosting, built-in regularisation |
| AdaBoost (Linear) | Adaptive boosting | Logistic regression base learner |
| AdaBoost (Tree) | Adaptive boosting | Decision-tree stump base learner |

---

## Model Selection & Stability

With only 759 artists (607 in training) and 5-fold CV (~121 artists per fold), cross-validation AUC estimates are noisy. Optuna can exploit this noise, producing hyperparameters that look good on CV but don't generalise — a form of hyperparameter overfitting.

To check this, we ran a **bootstrap validation** (`Bootstrap_validation.ipynb`, B=25): resample the training set with replacement, run a simplified tuning pipeline (10 Optuna trials) each iteration, evaluate on the fixed test set.

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

The naked bootstrap established that recall is too variable without any tuning. Rather than running the full heavy pipeline, `Final_Model_RandomForest.ipynb` uses a deliberately **light and conservative** tuning approach: 8 Optuna trials (vs 30+ elsewhere), a stronger gap penalty (λ=0.5 vs 0.3), tighter regularization bounds, SHAP-based genre consolidation, top 12 features by importance, centrality ablation, and maximized F1 score constrained to a precision floor of 0.60 on threshold tuning.

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

### Bootstrap Stability

To validate the final model, we re-ran the complete pipeline — feature selection, 8 Optuna trials, and threshold tuning — on 100 bootstrap resamples of the training set, evaluating each on the fixed test set. This is the most honest stability check: it captures not just sampling variability but also tuning variability.

![Final RF Bootstrap](Complementary%20Study/Final_RF_Bootstrap.png)

| Metric | Single run | Mean | Std | 95% CI |
|--------|:----------:|:----:|:---:|--------|
| AUC | 0.773 | 0.767 | 0.014 | [0.739, 0.791] |
| Precision | 0.627 | 0.584 | 0.054 | [0.494, 0.690] |
| Recall | 0.712 | 0.762 | 0.067 | [0.636, 0.909] |
| F1 | 0.667 | 0.656 | 0.020 | [0.618, 0.693] |

All single-run metrics fall within their bootstrap confidence intervals, confirming the result is reproducible and not a lucky draw. Recall carries the most variance (std=0.067), which is expected on a dataset of this size — but it does not undermine the model.

Two things support this. First, the pipeline consistently selects 10 features across all 100 bootstrap iterations, converging every time on the same three drivers: **charting history, network position, and genre**. That consistency is itself a finding — the model is not fishing for features, it is rediscovering the same signal. Second, the precision advantage over a fully naked model (0.627 vs 0.580) is a direct result of those deliberate feature choices, not tuning luck. The pipeline's contribution is not that it squeezes extra AUC — it is that it builds a model we can explain and defend.

---

## Complement: Reporting Predicted Probabilities

The sections above treat the model as a binary classifier. As a complement, we investigate whether the model's raw predicted probabilities are trustworthy enough to report directly (e.g. "this artist has a 72% chance of becoming a hitmaker").

### Calibration across all models

As a first step, we computed calibration curves for all six models in the comparison pipeline (`Model_Comparison_Final.ipynb`). 

![Calibration curves and predicted probability distributions for all 6 models](Complementary%20Study/calibration_curves_all_models.png)

**Key findings:**

- **Random Forest and XGBoost** are the best-calibrated models — their curves track the diagonal reasonably well in the mid-range (0.3–0.7), where most predictions fall. Crucially, neither model ever produces extreme probabilities: RF tops out at 0.74, XGBoost at 0.84.
- **LightGBM and CatBoost** are overconfident: LightGBM reaches 0.995 and assigns probabilities above 90% to 12 test artists; CatBoost reaches 0.976.
- **AdaBoost Linear** is essentially broken as a probability estimator — it collapses nearly all predictions to a single value ~0.47.

This analysis further supports the selection of Random Forest as the final model — it is the only model that is both well-discriminating (AUC 0.767) and produces probabilities that are plausibly meaningful.

---

### Calibrating the Final Random Forest

For RF, the raw probabilities are compressed (range 0.26–0.74) and deviate from the diagonal at the extremes. To make the probabilities genuinely reportable, we applied post-hoc calibration using two standard methods on top of the final tuned RF. Both calibrators are trained on the OOF predictions.

![Calibration comparison for the final RF: uncalibrated vs sigmoid vs isotonic](Complementary%20Study/calibration_rf_sigmoid_isotonic.png)

| Method | Threshold | AUC | Brier Score | Log Loss | Precision | Recall | F1 |
|--------|:---------:|:---:|:-----------:|:--------:|:---------:|:------:|:--:|
| Uncalibrated | 0.50 | 0.773 | 0.205 | 0.597 | 0.627 | 0.712 | 0.667 |
| Sigmoid | 0.43 | **0.773** | 0.198 | 0.581 | 0.603 | 0.712 | 0.653 |
| Isotonic | 0.42 | 0.768 | **0.191** | **0.559** | 0.603 | 0.712 | 0.653 |

**Why Sigmoid wins:**

- Sigmoid and isotonic produce **identical classification metrics** (same Precision, Recall, F1) at their respective thresholds.
- Sigmoid **preserves AUC exactly** (0.773 vs 0.768 for isotonic), meaning the ranking of artists is unaffected.
- Isotonic regression is known to overfit on small datasets (sklearn recommends n > 1000); our training set has 607 artists, making sigmoid the safer choice.
- Both calibrated models improve Brier score and log loss over the uncalibrated baseline, confirming that probabilities are genuinely more trustworthy after calibration.

The cost of calibration is a modest drop in F1 (0.667 → 0.653) and precision (0.627 → 0.603), shared equally by both methods. This is the honest price of making probabilities meaningful rather than just optimising for the binary decision.

---

## Repository Structure

```
├── README.md
├── df_artists_final.csv               # Model-ready dataset (759 × 27)
├── data_preparation.ipynb             # df_artists → df_artists_final
├── EDA_on_df_artists_final.ipynb      # Exploratory analysis
├── Model_Comparison_Final.ipynb       # 8-step pipeline, 6 models
├── Bootstrap_validation.ipynb         # Bootstrap validation of Model_Comparison_Final models (B=25)
├── bootstrap_results.pkl
├── Final_Model_RandomForest.ipynb     # Final model + full pipeline bootstrap (B=100)
│
├── Complementary Study/               # Additional analyses
│   ├── Final_Model_RandomForest_Complement.ipynb  # SHAP + extra plots for final model
│   ├── Model_Comparison_Final_SHAP.ipynb          # SHAP waterfall for all 6 models
│   ├── best3-disagreement.ipynb                   # Disagreement analysis, top 3 models
│   ├── Naked_Model_Bootstrap_Threshold.ipynb      # Naked RF bootstrap (B=100)
│   ├── RF_light_grid.ipynb                        # Light RF with ad-hoc genre consolidation
│   ├── RF_light_grid_bootstrap.ipynb              # Bootstrap validation of Light RF (B=25)
│   └── google_trends_signal_check.ipynb           # Google Trends ablation
│
├── Datasets/
│   ├── Main_Data/                     # Raw and intermediate CSVs
│   └── Pipeline_supplement/           # Data cleaning notebooks (stages 1–8)
│
└── Preliminary Study/                 # Early exploration (reference only)
```

---

## Getting Started


### Requirements

Python >= 3.9. All dependencies are listed in [`requirements.txt`](requirements.txt).

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd spring-2026-hitmakers
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run data preparation** (optional — `df_artists_final.csv` is already provided)
   ```
   Open data_preparation.ipynb and run all cells
   ```

4. **Run the model comparison pipeline**
   ```
   Open Model_Comparison_Final.ipynb and run all cells
   ```
   This will execute the full 8-step pipeline for all 6 models, produce diagnostic plots, and output the cross-model comparison table.

---

*Spring 2026 — Hitmakers Team Project*
