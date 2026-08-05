# Reel Match — Hybrid Movie Recommender

Given a movie you liked, **Reel Match** recommends similar titles by blending three independent signals:

- **Collaborative filtering** — behavioral similarity from co-rating patterns
- **Content-based similarity** — TF-IDF over genres and tags
- **Cast/director overlap** — shared top-billed actors and directors

These are combined through a **confidence-weighted hybrid model** that dynamically adjusts trust in each signal based on how much rating data a movie actually has.

Built **end-to-end from scratch** (no recommender libraries) on the **MovieLens** dataset, enriched with **TMDB** cast/crew metadata, and validated against a **held-out evaluation harness** (Precision@K / Recall@K).

## Problem Framing

**Goal:** given a movie a user likes, recommend similar movies they're likely to enjoy — ranked by predicted relevance.

**Primary interaction mode:** item-to-item similarity. User selects one movie → system returns top-N similar movies (e.g. *Infinity War* → *Endgame*, *Justice League*).

**Why hybrid, not a single method:**
- Collaborative filtering alone fails on sparse movies — **62.5%** of the catalog has fewer than 5 ratings (see [Dataset](#dataset)).
- Content and cast/director signals don't need rating history, making them natural fallbacks for cold-start movies.

**Success metric:** offline proxy via **Precision@K / Recall@K** — using held-out real ratings to check whether the system would have surfaced movies a user actually liked, since no live users were available to test against.

**Out of scope:** friend/social-based recommendations, live feedback loop, real-time serving, multi-movie profile-based recommendations.

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `aiohttp` · `Streamlit` · MovieLens dataset · TMDB API

## How to Run

**1. Install dependencies**
```bash
pip install pandas numpy scikit-learn aiohttp streamlit
```

**2. Get the data**
- Download `ml-latest-small` from [MovieLens](https://grouplens.org/datasets/movielens/) → place in `data/raw/`
- Get a free TMDB Read Access Token from [TMDB API settings](https://www.themoviedb.org/settings/api)
- Set it as an environment variable:
```bash
export TMDB_API_KEY=your_read_access_token_here
```

**3. Run the data pipeline**
```bash
python tmdb_enrich.py       # enrich movies with cast/crew data
# then run EDA.ipynb to clean and export movies_clean.csv to data/processed/
```

**4. Launch the app**
```bash
streamlit run app.py
```

## Dataset

**MovieLens `ml-latest-small`** — 100,836 ratings across 610 users and 9,742 movies (ratings, genres, tags).

**TMDB API** — cast and director metadata merged in via MovieLens's `links.csv` (movieId → tmdbId mapping).

**Coverage after merge:**

| Metric | Value |
|---|---|
| Movies with cast data | 8,997 / 9,742 (92.4%) |
| Movies with crew (director) data | 9,021 / 9,742 (92.6%) |
| Movies with ≥1 tag | 1,572 / 9,742 (16.1%) |

**Ratings sparsity (the core challenge this project addresses):**

| Metric | Value |
|---|---|
| Median ratings per movie | 3 |
| Mean ratings per movie | 10.4 |
| Movies with <5 ratings | 62.5% |
| Movies with <10 ratings | 76.7% |

This long-tail sparsity — most movies have very few ratings, a handful have hundreds — is the direct justification for a hybrid approach: collaborative filtering alone cannot reliably serve the majority of the catalog.

If `w_cf + w_cast` would exceed 1, both are scaled down proportionally so all three weights always sum to exactly 1 (see [Design Decisions](#key-design-decisions)).

**Tuned parameters** (via grid search, see [Evaluation](#evaluation-methodology)): `α_max = 0.85`, `k = 3`, `cast_cap = 0.28`

## Key Design Decisions

**Soft weighting over a hard CF cutoff.** Rather than dropping movies below a ratings threshold (losing 62.5% of the catalog from CF), CF's influence scales continuously with rating count via a Bayesian shrinkage function — the same principle used in the popularity baseline and IMDB's weighted rating formula. No movie is fully excluded; low-data movies just lean harder on content/cast.

**Top-5 billed cast only.** Full cast lists dilute similarity with background/minor roles. Capping to the top 5 by billing order keeps the signal focused on actors that actually drive audience interest.

**TF-IDF `min_df` tuning.** An initial `min_df=2` filtered out enough rare tag words that hundreds of movies collapsed to identical genre-only vectors, creating large similarity ties (e.g. 725 movies tied at 1.0 with *Toy Story*). Lowering to `min_df=1` restored genuine differentiation.

**Proportional weight clamping.** Since `w_cf` and `w_cast` are computed independently, their sum could exceed 1 for well-rated movies with cast data. Both are scaled down proportionally in that case, guaranteeing all three weights always sum to exactly 1 rather than silently producing an invalid (negative) content weight.

## Evaluation Methodology

**Held-out design (adapted for item-to-item):** standard recommender evaluation holds out ratings per user, but this system's primary mode is item-to-item, not profile-based. So for each user, one liked movie is hidden (the "held-out" movie) while another is fed to the recommender as the "known" input — testing whether the system surfaces the held-out movie given only the known one.

**Relevance threshold:** a rating ≥ 4.0 counts as "liked."

**Metrics:** Precision@10 and Recall@10 — with a single held-out movie per user, recall simplifies to hit rate.

**Sanity-checked before trusting real results:** the harness was first run against a random recommender, which correctly scored ~0.0 — confirming the evaluation logic itself was sound before any real model was tested.

**Hyperparameter tuning:** a grid search over `α_max`, `cast_cap`, and `k` (shrinkage constant) was run against this same harness, precomputing per-movie component scores once to make ~3,800 combinations tractable, rather than hand-picking weights.

## Results

| Model | Precision@10 | Recall@10 | vs Baseline |
|---|---|---|---|
| Popularity baseline (confidence-weighted) | 0.00148 | 0.0148 | 1x |
| Item-based CF alone | 0.00700 | 0.0700 | 4.7x |
| Content-based alone | ~0.0 at K=10 | ~0.0 | below baseline |
| Cast/director alone | 0.00049 | 0.0049 | below baseline |
| **Tuned hybrid** | **0.00740** | **0.0740** | **5x** |

**Content-based, isolated:** while it scored ~0 at K=10 in aggregate, rank analysis on held-out movies showed a median rank of ~4,031 out of 9,741 — weak but non-random signal, occasionally ranking a correct match as high as position 50.

**Cast/director, isolated:** scored below the popularity baseline alone. With only top-5 billed cast considered, most movie pairs share zero cast members — the signal is sparse but precise on the rare pairs where it fires (e.g. franchise sequels).

**Hybrid vs CF-alone:** the tuned hybrid provides a modest but real improvement over CF alone (+5.7%) in aggregate — see [Key Finding](#key-finding-segmented-testing) for how this breaks down by movie popularity.

## Key Finding: Segmented Testing

The aggregate hybrid result (+5.7% over CF alone) hides a more important pattern. Splitting evaluation pairs by the input movie's rating count tells a different story:

| Segment | N pairs | CF Precision@10 | Hybrid Precision@10 | Lift |
|---|---|---|---|---|
| Sparse (<10 ratings) | 61 | 0.00492 | 0.00328 | **−33.3%** |
| Mid (10–50 ratings) | 180 | 0.00667 | 0.00667 | +0.0% |
| Well-rated (50+ ratings) | 367 | 0.00763 | 0.00845 | +10.7% |

**This contradicts the project's original design hypothesis.** The hybrid was intended to rescue sparse, cold-start movies by falling back on content/cast signal — instead, it underperforms CF-alone specifically in that segment, while helping most where CF already worked well.

**Root cause, isolated:** within the sparse segment specifically, both content-alone and cast-alone scored **0.0 precision** (not just weak — zero hits). Any weight redistributed away from CF toward these fallback signals diluted the only signal that was actually producing hits, rather than adding value.

**Caveat:** the sparse segment contains only 61 pairs — a difference of roughly one hit shifts the precision noticeably. This finding is directional, not statistically conclusive, and would benefit from a larger held-out sample before being treated as final.

## Known Limitations

- **Cold-start fallback signals need strengthening.** Content and cast/director similarity are currently too weak on their own to reliably rescue sparse movies — the project's core hybrid rationale — as shown in [segmented testing](#key-finding-segmented-testing). Richer content features (e.g. plot summaries, keywords beyond MovieLens tags) would likely help.
- **Small evaluation sample for segmented analysis.** 608 total held-out pairs (61 in the sparse segment) is enough for aggregate comparisons but limits confidence in segment-level findings. A multi-pair-per-user evaluation design would increase statistical power.
- **Cast similarity is O(n) per query**, computed on demand rather than precomputed into a matrix like CF and content similarity — a deliberate tradeoff given dataset size, but would need optimizing for production-scale use.
- **No live user feedback loop.** Success is measured entirely via offline proxy metrics (Precision@K/Recall@K) against historical ratings, not real user interactions.
- **Catalog scope is fixed** to MovieLens's ~9,700 movies — no mechanism for adding new movies without rerunning the data pipeline.