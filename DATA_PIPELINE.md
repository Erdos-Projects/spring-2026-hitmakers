# Data Pipeline: Billboard Hitmaker Prediction

## Project Goal

Predict whether an artist who achieves their first Billboard Hot 100 top-20 hit will go on to have additional top-20 hits (`top_20_hitmaker = 1`) or remain a one-hit wonder (`top_20_hitmaker = 0`).

**Final model input:** `df_artists_final.csv` (759 artists × 26 features + 1 target)
**Final model:** XGBoost classifier (tuned in `ml_sandbox_16_final_xgboost_tuning.ipynb`)

---

## Pipeline Overview

```
EXTERNAL DATA SOURCES
├─ billboard_hot100_1958_2026.csv    
├─ kang_data_w_spotify.csv           (Kang/Kwon academic dataset)
├─ gabminamedez_spotify_data.csv     (GitHub: gabminamedez/spotify-data)
├─ google_trends (via pytrends API)
└─ MusicBrainz database (Docker)
        │
        ▼  
┌─────────────────────────────────────────────────────────┐
│  billboard_hot100_songs_final.csv  (unique songs)       │
│  billboard_200_albums_final.csv    (unique albums)      │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Billboard Cleaning & Artist Verification      │
│  (billboard_data_cleaning_pt_1)                         │
│  • MusicBrainz API artist verification                  │
│  • Collaboration splitting (feat./x/&)                  │
│  • Performer name normalization                         │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: Aggregation & Target Construction             │
│  (billboard_data_cleaning_pt_2, EDA_1_+pt_3)           │
│  • Per-artist career stats                              │
│  • Milestone-year columns                               │
│  • Hitmaker flag                                        │
│  → df_artists_basic.csv                                 │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3: Artist Name Deduplication                     │
│  (df_artists_clean)                                     │
│  • Remove collab artifacts (feat., x, /)                │
│  • Deduplicate the/non-the variants                     │
│  • Manual keep-lists for legitimate names               │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────┴──────────────────────────────────┐
│          │            │            │                     │
▼          ▼            ▼            ▼                     ▼
Stage 4  Stage 5     Stage 6     Stage 7              Stage 8
Labels   MBID Fix    Songs +     Network              Cleanup
+ Genre  (pt_7)      Spotify     Metrics              + Final
(pt_4,5)             (df_songs,  (pt_8,               Assembly
                      spotify_   network_              (pt_6,
                      merge)     merge)                df_artists
                                                       _clean)
│          │            │            │                     │
└──────────┴────────────┴────────────┴─────────────────────┘
                       │
                       ▼
              df_artists.csv
           (13,655 artists × 44 cols)
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  data_preparation.ipynb                                 │
│  • Filter 2000–2019 artists                             │
│  • One-hot encode genres                                │
│  • Drop Spotify audio / collinear features              │
│  • Fill network nulls with 0                            │
│  → df_artists_final.csv (759 × 27)                     │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  ml_sandbox_16_final_xgboost_tuning.ipynb               │
│  • XGBoost with tuned hyperparameters                   │
│  • 80/20 stratified train/test split                    │
│  • Cross-validated evaluation                           │
└─────────────────────────────────────────────────────────┘
```

---

## Stage-by-Stage Detail

### Stage 0: Raw Data Acquisition

| File | Rows | Source | Notes |
|------|------|--------|-------|
| `billboard_hot100_1958_2026.csv` | ~350K | Scraped/downloaded | Week-by-week Hot 100 entries. **How it was obtained is not documented in the repo.** |
| `billboard_hot100_songs_final.csv` | ~30K | Derived from weekly data | One row per unique song. **The deduplication step is not in the repo** — loaded as a pre-existing file. |
| `billboard_200_albums_final.csv` | ~19K | Derived from weekly data | One row per unique album. **Same — dedup step not in repo.** |
| `kang_data_w_spotify.csv` | ~9K | Kang/Kwon academic dataset | Verified artist IDs + Spotify data. Used for ID corrections in Stage 5. |
| `gabminamedez_spotify_data.csv` | ~170K | [GitHub](https://github.com/gabminamedez/spotify-data) | Spotify audio features. Covers songs up to ~2020. |
| MusicBrainz DB | ~2M artists | Local Docker dump | Used for artist verification, genre tags, collaboration edges, UUIDs. |

### Stage 1: Billboard Cleaning & Artist Verification

**Notebook:** `McNally_Jupyter_Notebooks/billboard_data_cleaning_pt_1_McNally.ipynb`

- Loaded deduplicated songs (`billboard_hot100_songs_unique.csv`) and albums (`billboard_200_albums_unique.csv`)
- Verified each artist name against the MusicBrainz API:
  - Exact name match + alias matching
  - Used to distinguish real band names (e.g. "Simon & Garfunkel") from collaboration credits (e.g. "Drake Featuring Rihanna")
- Split collaboration performers into separate rows, preserving `original_performer`
- Normalized performer names → `performer_normalized`
- **Output:** `billboard_hot100_songs_cleaned.csv`, `billboard_200_albums_cleaned.csv`, `billboard_songs_and_albums_combined_all.csv`

### Stage 2: Artist/Song Aggregation & Target Construction

**Notebooks:** `billboard_data_cleaning_pt_2_McNally.ipynb`, `EDA_1_+billboard_data_cleaning_pt_3.ipynb`

- Aggregated per-artist statistics from song/album data:
  - `first_song_year`, `last_song_year`, `years_active_on_charts`
  - `total_charting_songs`, `total_charting_albums`
  - `#1_hit_song_count`, `#1_hit_album_count`
  - `top_10_song_count`, `top_10_album_count`, `top_20_song_count`, etc.
  - `highest_charting_song_position`, `highest_charting_album_position`
- Computed milestone-year columns: `first_year_top_10_songs`, `first_year_top_20_songs`, `first_year_num1_songs`, etc.
- Created `top_10_hitmaker_songs` binary flag (>1 top-10 song)
- **Output:** `df_artists_basic.csv`

### Stage 3: Artist Name Deduplication

**Notebook:** `McNally_Jupyter_Notebooks/df_artists_clean.ipynb`

Extensive cleanup of artist names to ensure one row per real artist:

| Pattern | Action | Examples |
|---------|--------|----------|
| `feat.`/`ft.`/`featuring` | Drop | `"Drake Featuring Rihanna"` |
| `x` collabs | Drop | `"XXXTENTACION x Lil Pump"` |
| Slash collabs | Drop (with keep-list) | Keep: `ac/dc`, `gza/genius`; Drop: `"Jay-Z / Kanye West"` |
| `presents` | Drop | `"DJ Khaled Presents..."` |
| Comma collabs | Drop (with keep-list) | Keep: `tyler, the creator`, `earth, wind & fire`; Drop: `"Drake, Future"` |
| `or` collabs | Drop (with keep-list) | Keep: `dead or alive`; Drop: `"this or that"` |
| Parenthetical credits | Drop (with keep-list) | Keep: `(hed) p.e.`; Drop: `"artist (with other artist)"` |
| `the`/non-`the` variants | Keep higher-charting variant | `the rolling stones` > `rolling stones` |
| Orchestra/ensemble artifacts | Drop | `"his orch."`, `"orchestra"` |
| Typo variants | Drop | `"30 secnds to mars"` |

**Result:** ~13,600 → ~13,400 unique artists

### Stage 4: Label Aggregation & Genre Tagging

**Notebooks:** `billboard_data_cleaning_pt_4.ipynb`, `billboard_data_cleaning_pt_5_genre.ipynb`

**Label aggregation (pt_4):**
- Aggregated record labels from songs and albums → `aggregate_labels` per artist
- Created `first_top_10_hit_song_genre_tags`, `first_top_10_hit_album_genre_tags`
- Created `genre_tags_through_first_top_10_hit` (all genres up to first top-10 year)

**Genre mapping (pt_5):**
- Queried MusicBrainz for genre tags per artist, song, and album
- Built comprehensive **546-genre → 18-major-category mapping**:

| # | Major Category | Example MusicBrainz Genres |
|---|---------------|---------------------------|
| 1 | Rock | rock, classic rock, hard rock, psychedelic rock, grunge |
| 2 | Pop | pop, synth-pop, dance-pop, electropop, teen pop |
| 3 | Hip Hop/Rap | hip hop, rap, trap, gangsta rap, conscious hip hop |
| 4 | R&B/Soul/Funk | r&b, soul, funk, neo-soul, quiet storm |
| 5 | Electronic/Dance | electronic, house, techno, edm, dubstep |
| 6 | Country/Americana | country, americana, country rock, outlaw country |
| 7 | Jazz | jazz, smooth jazz, bebop, jazz fusion |
| 8 | Blues | blues, electric blues, delta blues |
| 9 | Metal | metal, heavy metal, death metal, thrash metal |
| 10 | Punk/Hardcore | punk, punk rock, hardcore punk, pop punk |
| 11 | Folk | folk, folk rock, singer-songwriter |
| 12 | Latin | latin, reggaeton, salsa, bachata, latin pop |
| 13 | Gospel/Christian/Religious | gospel, christian rock, worship, ccm |
| 14 | Classical | classical, opera, orchestral, chamber music |
| 15 | Reggae/Caribbean | reggae, ska, dancehall, dub |
| 16 | World Music | afrobeat, bhangra, fado, bossa nova |
| 17 | Experimental/Avant-Garde | experimental, noise, avant-garde, industrial |
| 18 | Easy Listening/Vocal | easy listening, lounge, adult contemporary |

- Created `musicbrainz_major_genre_categories`, `spotify_major_genre_categories`, `combined_major_genre_categories` (union of MB + Spotify)

### Stage 5: MusicBrainz ID Corrections

**Notebooks:** `billboard_data_cleaning_pt_7_filling_in_missing_artist_ids.ipynb`, `not_for_final_github_add_MBID_to_dfs.ipynb`

- Assigned `musicbrainz_mbid` (UUID) via case-insensitive name matching against local MusicBrainz DB
- Cross-referenced MusicBrainz integer IDs with Kang et al. verified dataset
- **Fixed 564 wrong MusicBrainz artist IDs** (e.g. "The Beatles" had been mapped to a Philly doo-wop group named "The Beatles")
- Filled previously-null IDs from Kang dataset
- Synced corrected IDs into `df_songs` and `df_albums`
- Re-synced `spotify_id` and `spotify_genres` from Kang dataset
- Renamed `performer_normalized` → `name` in `df_artists`

### Stage 6: Song-Level Feature Engineering

**Notebooks:** `df_songs_create.ipynb`, `spotify_data_merge.ipynb`

- Created `df_songs` from `billboard_hot100_songs_final.csv`:
  - Renamed columns: `performer_normalized` → `name`, `performer` → `performer_pre_normalized`
  - Queried MusicBrainz for `musicbrainz_recording_mbid` per song
  - Added song-level and album-level genre tags from MusicBrainz
  - Created major genre categorizations using the 546→18 mapping
  - Added `#_of_genres_song`, `#_of_genres_aggregate`
  - Added record label data from MusicBrainz
- Merged Spotify audio features from `gabminamedez_spotify_data.csv`:
  - Matched on (title + artist name), exploding multi-artist Spotify entries
  - Features: acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence, mode, key, popularity, explicit, duration_ms
  - Coverage: ~30-40% of songs matched (Spotify data covers up to ~2020)
- **Output:** `df_songs.csv` (~30K rows × ~37 columns)

### Stage 7: Network Metrics

**Notebooks:** `billboard_data_cleaning_pt_8_new_network_metrics.ipynb`, `billboard_network_data_merge.ipynb`

**Phase 1 — Edge list construction:**
- Queried MusicBrainz for all collaboration edges (artists sharing `artist_credit` on a `release_group`)
- Also added track-level featuring credits
- **Output:** `master_edge_list_all_artists.parquet`

**Phase 2 — Network snapshots:**
- Built yearly + rolling 5-year window edge lists for each year 1958–2024
- Stored in `McNally_Network_Analysis_Data/networks_all_artists/`

**Phase 3 — Centrality computation:**
- For each artist, computed graph metrics using NetworkX at two time anchors:
  - `top10` = year of artist's first top-10 hit
  - `firstsong` = year of artist's first charting song
- Metrics computed:
  - `degree_centrality` — fraction of artists in network connected to this artist
  - `closeness_centrality` — inverse average distance to all other artists
  - `harmonic_closeness_centrality` — handles disconnected components gracefully
  - `betweenness_centrality` — how often artist lies on shortest paths (bridge role)
  - `eigenvector_centrality` — connected to other well-connected artists
  - `power_of_connected_artists` — sum of collaborators' chart power:
    - Song power = `101 - peak_position` (scale 1–100)
    - Album power = `(201 - peak_position) / 2` (normalized to same scale)
    - Per collaborator: max(song power, album power)
    - Total: sum across all collaborators in the window
- Merged network metrics into `df_artists` via `performer_normalized`
- **Output:** `df_artists_network_metrics.csv` (26 columns)

### Stage 8: Column Cleanup & Final Assembly

**Notebooks:** `billboard_data_cleaning_pt_6_condensing.ipynb`, `df_artists_clean.ipynb` (final cells)

- Dropped redundant/unused columns: `last_song_year`, `last_album_year`, `tags`, `top_20_song_count`, `top_50_song_count`, etc.
- Converted year columns to nullable `Int64` type
- Merged Spotify audio features for each artist's first top-20 hit song (from `df_songs`)
- Added `first_top_20_song_major_genres` mapped from first top-20 hit in `df_songs`
- Added `top_20_hit_song_#_wks_on_chart_any_position` (weeks first top-20 hit stayed on chart)
- Created **`top_20_hitmaker`** target variable:
  - `1` = artist had ≥2 top-20 songs
  - `0` = artist had exactly 1 top-20 song
  - `NaN` = artist had no top-20 songs
- **Output:** `df_artists.csv` (13,655 × 44)

---

## Supplementary Pipelines (not used in final model)

### Google Trends

**Notebooks:** `GS/Googletrend_dataset.ipynb`, `GS/google_trends_engineered.ipynb`

- Pulled monthly Google Trends interest (2004–2021) for top 500 artists via `pytrends`
  - YouTube search + web search, US only, music category (cat=35)
  - Checkpointed every 50 artists
- Engineered `google_trend_decay` feature (decline in search interest after first hit)
- **Output:** `google_trends_top500.csv`, `df_songs_google_decay.csv`
- **Status:** Used in experimental model sandboxes (6, 7, 8) but **not in the final XGBoost model**

### Yundi's Comprehensive Dataset

**Notebooks:** `UPDATE_YUNDI/Build_Comprehensive_Dataset.ipynb`, `UPDATE_YUNDI/MusicBrainz_DataExtract.ipynb`

- Alternative/parallel dataset construction: 2,420 artists × 50 columns
- Different target definition (`is_1hit_wonder` based on top-10 threshold)
- Aggregated Google Trends to artist-level averages
- Genre variety from `df_songs` + MusicBrainz
- Spotify audio mean + stddev per artist
- Wikipedia pageviews (2015–2024) via Wikimedia API
- **Output:** `df_comprehensive_hitmaker_prediction.csv`
- **Status:** Separate modeling track with its own experiments (`UPDATE_YUNDI/ml_sandbox_*.ipynb`)

### Wikipedia & YouTube Data

**Notebook:** `McNally_Jupyter_Notebooks/youtube_data_pull.ipynb`

- Loaded 4 additional Kaggle Spotify datasets for cross-referencing
- Pulled Wikipedia annual pageviews (2015–2024) via Wikimedia API (using MusicBrainz Wikipedia URLs)
- Collected YouTube links for artists
- **Status:** Exploratory — **not integrated into any model**

---

## Data Files Reference

### Raw / External Data
| File | Rows | Description |
|------|------|-------------|
| `billboard_hot100_1958_2026.csv` | ~350K | Week-by-week Billboard Hot 100 entries |
| `billboard_hot100_songs_final.csv` | ~30K | Deduplicated unique songs |
| `billboard_200_albums_final.csv` | ~19K | Deduplicated unique albums |

### Intermediate Data
| File | Rows × Cols | Description |
|------|-------------|-------------|
| `df_songs.csv` | ~30K × 37 | Song-level features: chart stats, MusicBrainz IDs, genre tags, Spotify audio, labels |
| `df_albums.csv` | ~19K × 34 | Album-level features: chart stats, genre tags, labels |
| `df_artists.csv` | 13,655 × 44 | Artist-level features: career stats, genre, network metrics, Spotify audio, target |
| `df_artists_network_metrics.csv` | 13,655 × 26 | Network centrality metrics (standalone) |
| `df_artists_with_network_metrics.csv` | ~13K × 60+ | Full artist profile before cleaning |

### Final Data
| File | Rows × Cols | Description |
|------|-------------|-------------|
| `df_artists_final.csv` | 759 × 27 | Model-ready dataset (2000–2019 artists, feature-selected) |

---

## Feature Dictionary: `df_artists_final.csv`

### Career Features (3)
| Feature | Type | Description |
|---------|------|-------------|
| `years_through_first_top_20_hit` | float | Years from first charting song to first top-20 hit |
| `#_of_charting_songs_through_first_top_20_hit` | float | Number of songs that charted before/including the first top-20 hit |
| `top_20_hit_song_#_wks_on_chart_any_position` | float | Weeks the artist's first top-20 hit spent on chart at any position (87 nulls — artists whose top-20 hit wasn't matched in df_songs) |

### Genre Features (20)
| Feature | Type | Description |
|---------|------|-------------|
| `artist_genre_Blues` … `artist_genre_World Music` | bool→int | One-hot encoded from `combined_major_genre_categories` (18 genres). Source: MusicBrainz + Spotify. Multi-genre artists have multiple 1s. |
| `artist_genre_unknown` | int | 1 if no genre data available, 0 otherwise |
| `#_of_genres_artist` | int | Count of distinct genre categories assigned to the artist |

### Network Features (3)
| Feature | Type | Description |
|---------|------|-------------|
| `harmonic_closeness_centrality_top20_rolling5` | float | Harmonic closeness in the 5-year collaboration network ending at the year of the artist's first top-20 hit. Higher = closer to more artists. Handles disconnected components. |
| `betweenness_centrality_top20_rolling5` | float | How often the artist lies on shortest paths between other artists. Higher = more of a "bridge" between communities. |
| `eigenvector_centrality_top20_rolling5` | float | Connected to other well-connected artists. Higher = more central in the collaboration network. |

### Target (1)
| Feature | Type | Description |
|---------|------|-------------|
| `top_20_hitmaker` | float | **1.0** = artist had ≥2 top-20 songs, **0.0** = exactly 1 top-20 song |

**Class balance:** 431 non-hitmakers (56.8%) vs 328 hitmakers (43.2%)

---

## Key Decisions & Rationale

### Why filter to 2000–2019?
- Network metrics require a rolling 5-year window, so pre-2000 data has sparse collaboration signals
- Post-2019 artists may not have had enough time to accumulate additional hits
- This window gives a balanced dataset with sufficient feature coverage

### Why drop Spotify audio features?
- ~40% missing values (Spotify data from `gabminamedez` only covers through ~2020, and not all songs match)
- Imputation at this scale would introduce bias; dropping preserves data integrity

### Why drop `degree_centrality` and `power_of_connected_artists`?
- High multicollinearity with `eigenvector_centrality` (VIF analysis in `ml_sandbox_4`)
- Keeping only `harmonic_closeness`, `betweenness`, and `eigenvector` centrality reduces redundancy

### Why use `top20_rolling5` network metrics?
- Rolling 5-year window captures recent collaboration activity without being as noisy as single-year
- Anchored at year of first top-20 hit (the prediction point)

### Why fill network nulls with 0?
- Null = artist had no MusicBrainz collaboration data in that window
- 0 is the correct semantic value: "no measured centrality in the collaboration network"

### Why fix 564 MusicBrainz IDs?
- Initial name-matching against MusicBrainz produced false matches (common names mapped to wrong artists)
- Cross-referencing with the Kang et al. verified dataset corrected these, ensuring genre tags and network edges are accurate

---

## Notebook Inventory

### Main Pipeline (in execution order)
| # | Notebook | Stage | Role |
|---|----------|-------|------|
| 1 | `McNally_Jupyter_Notebooks/billboard_data_cleaning_pt_1_McNally.ipynb` | 1 | Billboard cleaning, artist verification, collab splitting |
| 2 | `McNally_Jupyter_Notebooks/billboard_data_cleaning_pt_2_McNally.ipynb` | 2 | Per-artist aggregation, initial df_artists |
| 3 | `McNally_Jupyter_Notebooks/EDA_1_+billboard_data_cleaning_pt_3.ipynb` | 2 | EDA + milestone-year columns |
| 4 | `McNally_Jupyter_Notebooks/not_for_final_github_add_MBID_to_dfs.ipynb` | 5 | Initial MBID (UUID) assignment |
| 5 | `McNally_Jupyter_Notebooks/billboard_data_cleaning_pt_4.ipynb` | 4 | Label aggregation, genre tag aggregation |
| 6 | `McNally_Jupyter_Notebooks/billboard_data_cleaning_pt_5_genre.ipynb` | 4 | 546→18 genre mapping engine |
| 7 | `McNally_Jupyter_Notebooks/billboard_data_cleaning_pt_6_condensing.ipynb` | 8 | Column cleanup, type conversions |
| 8 | `McNally_Jupyter_Notebooks/billboard_data_cleaning_pt_7_filling_in_missing_artist_ids.ipynb` | 5 | Kang cross-ref, fix 564 wrong IDs |
| 9 | `McNally_Jupyter_Notebooks/df_songs_create.ipynb` | 6 | Create df_songs with MusicBrainz enrichment |
| 10 | `McNally_Jupyter_Notebooks/spotify_data_merge.ipynb` | 6 | Merge Spotify audio features into df_songs |
| 11 | `McNally_Jupyter_Notebooks/billboard_data_cleaning_pt_8_new_network_metrics.ipynb` | 7 | Build collaboration network, compute centrality |
| 12 | `McNally_Jupyter_Notebooks/billboard_network_data_merge.ipynb` | 7 | Merge network metrics into df_artists |
| 13 | `McNally_Jupyter_Notebooks/df_artists_clean.ipynb` | 3, 8 | Artist dedup + final assembly + target variable |
| 14 | `data_preparation.ipynb` | 9 | df_artists.csv → df_artists_final.csv |
| 15 | `ml_sandbox_16_final_xgboost_tuning.ipynb` | 10 | Final XGBoost model |

### Supplementary / Exploratory
| Notebook | Purpose |
|----------|---------|
| `GS/Googletrend_dataset.ipynb` | Pull Google Trends data via pytrends |
| `GS/google_trends_engineered.ipynb` | Engineer google_trend_decay feature |
| `McNally_Jupyter_Notebooks/youtube_data_pull.ipynb` | Wikipedia pageviews, Kaggle Spotify datasets |
| `UPDATE_YUNDI/MusicBrainz_DataExtract.ipynb` | Flatten MusicBrainz JSON dump |
| `UPDATE_YUNDI/Build_Comprehensive_Dataset.ipynb` | Yundi's alternative 2,420-row dataset |
| `McNally_Jupyter_Notebooks/ml_sandbox_*.ipynb` | Model experiments (1–14) |
| `McNally_Jupyter_Notebooks/EDA_*.ipynb` | Exploratory data analysis |

---

## External Dependencies

| Dependency | Purpose | Required For |
|------------|---------|--------------|
| MusicBrainz PostgreSQL DB (Docker) | Artist verification, genre tags, collaboration edges, UUIDs | Stages 1, 4, 5, 6, 7 |
| MusicBrainz API (remote) | Artist name verification | Stage 1 |
| Spotify API / datasets | Audio features, genre enrichment | Stage 6 |
| Google Trends (pytrends) | Search interest time series | Google Trends pipeline |
| Wikimedia API | Wikipedia pageviews | Exploratory (unused) |
| NetworkX | Graph centrality computation | Stage 7 |

> **Note:** To reproduce `df_artists_final.csv` from `df_artists.csv`, you only need **pandas**. The external dependencies are only needed to rebuild the upstream CSVs from raw data.

---

## Reproducibility

```bash
# To regenerate df_artists_final.csv from df_artists.csv:
# Run all cells in data_preparation.ipynb

# To run the final model:
# Run all cells in ml_sandbox_16_final_xgboost_tuning.ipynb
```

### Python Environment
- Python 3.12+
- Key packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `shap`
- For upstream rebuild: `psycopg2`, `networkx`, `requests`, `tqdm`, `pytrends`
