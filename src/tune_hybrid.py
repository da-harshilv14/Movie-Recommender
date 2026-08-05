"""
tune_hybrid.py — Fast grid search by precomputing component scores once,
then only varying the weighted combination per grid point.
"""

import numpy as np
import pandas as pd
import itertools

from evalution import get_liked_movies, make_eval_pairs
import hybrid as hybrid_module

RATINGS_PATH = "../data/ratings.csv"

# Full-precision grid, but kept to a sane RANGE (not 0-1) to stay tractable.
# alpha near 0 or 1 is rarely useful anyway — search where it matters.
ALPHA_MAX_OPTIONS = np.round(np.arange(0.50, 0.91, 0.01), 2)   # 41 values
CAST_CAP_OPTIONS = np.round(np.arange(0.00, 0.31, 0.01), 2)    # 31 values
K_SHRINKAGE_OPTIONS = [3, 5, 10]


def precompute_component_scores(pairs):
    """
    For each UNIQUE 'known' movie across all eval pairs, precompute:
      - cf_scores, content_scores, cast_scores (aligned arrays)
      - n_ratings, has_cf, has_cast

    Returns: dict {movie_id: {"cf": arr, "content": arr, "cast": arr,
                               "n_ratings": int, "has_cf": bool, "has_cast": bool}}
    """
    unique_known = {known for known, _, _ in pairs}
    cache = {}

    for movie_id in unique_known:
        cf_scores = hybrid_module._get_cf_scores_aligned(movie_id)
        content_scores = hybrid_module._get_content_scores_aligned(movie_id)
        cast_scores = hybrid_module._get_cast_scores_aligned(movie_id)

        has_cf = movie_id in hybrid_module.cf_module._movie_id_to_idx
        n_ratings = int(hybrid_module._movies_with_idx.loc[movie_id, "n_ratings"])
        has_cast = len(hybrid_module.cast_module._movie_people.get(movie_id, set())) > 0

        cache[movie_id] = {
            "cf": cf_scores, "content": content_scores, "cast": cast_scores,
            "n_ratings": n_ratings, "has_cf": has_cf, "has_cast": has_cast,
        }

    return cache


def evaluate_combo(alpha_max, cast_cap, k_shrink, pairs, cache, k=10):
    total_hits = 0

    for known, heldout, user in pairs:
        c = cache[known]

        w_cf = alpha_max * (c["n_ratings"] / (c["n_ratings"] + k_shrink)) if c["has_cf"] else 0.0
        w_cast = cast_cap if c["has_cast"] else 0.0
        w_cf, w_content, w_cast = hybrid_module.compute_weights(w_cf, w_cast)

        final_scores = w_cf * c["cf"] + w_content * c["content"] + w_cast * c["cast"]

        ranked_positions = np.argsort(final_scores)[::-1]
        top_k_ids = []
        for pos in ranked_positions:
            candidate_id = hybrid_module._all_movie_ids[pos]
            if candidate_id == known:
                continue
            top_k_ids.append(candidate_id)
            if len(top_k_ids) == k:
                break

        if heldout in top_k_ids:
            total_hits += 1

    precision = total_hits / (len(pairs) * k)
    recall = total_hits / len(pairs)
    return precision, recall


def run_grid_search(pairs):
    cache = precompute_component_scores(pairs)
    results = []

    combos = list(itertools.product(ALPHA_MAX_OPTIONS, CAST_CAP_OPTIONS, K_SHRINKAGE_OPTIONS))
    print(f"Testing {len(combos)} combinations...")

    for i, (alpha_max, cast_cap, k_shrink) in enumerate(combos):
        precision, recall = evaluate_combo(alpha_max, cast_cap, k_shrink, pairs, cache)
        results.append({
            "alpha_max": alpha_max, "cast_cap": cast_cap, "k_shrink": k_shrink,
            "precision": precision, "recall": recall,
        })
        if i % 500 == 0:
            print(f"  {i}/{len(combos)} done")

    return results


if __name__ == "__main__":
    ratings = pd.read_csv(RATINGS_PATH)
    liked = get_liked_movies(ratings)
    pairs = make_eval_pairs(liked)

    results = run_grid_search(pairs)
    results_df = pd.DataFrame(results).sort_values("precision", ascending=False)
    print(results_df.head(20).to_string(index=False))
    results_df.to_csv("../data/processed/grid_search_results.csv", index=False)