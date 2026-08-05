"""
diagnose_sparse.py — Isolate why the hybrid underperforms CF-alone
specifically in the sparse (<10 ratings) bucket.

Tests three things:
  1. Average weight breakdown for sparse-bucket movies
  2. Content-alone and cast-alone precision, isolated to the sparse bucket
  3. A variant hybrid with cast_cap forced to 0, to isolate cast's effect
"""

import numpy as np
import pandas as pd

from evalution import evaluate, get_liked_movies, make_eval_pairs
from segmented_test import bucket_pairs
import hybrid as hybrid_module
import content_based as content_module
import cast_similarity as cast_module
import item as cf_module

RATINGS_PATH = "../data/ratings.csv"
K = 10


def average_weights_for_bucket(pairs: list) -> dict:
    """Average w_cf, w_content, w_cast across all 'known' movies in this bucket."""
    known_movies = {known for known, _, _ in pairs}
    all_weights = [hybrid_module.get_weights(m) for m in known_movies]

    avg = {
        "cf": np.mean([w["cf"] for w in all_weights]),
        "content": np.mean([w["content"] for w in all_weights]),
        "cast": np.mean([w["cast"] for w in all_weights]),
    }
    return avg


def hybrid_no_cast_recommend_fn(movie_id, k: int = K) -> list:
    """
    Same as hybrid_recommend_fn, but forces w_cast = 0 for everyone,
    redistributing that weight to content instead. Isolates whether
    cast specifically is responsible for the sparse-bucket drop.
    """
    has_cf = movie_id in cf_module._movie_id_to_idx
    if has_cf:
        n_ratings = int(hybrid_module._movies_with_idx.loc[movie_id, "n_ratings"])
        w_cf = hybrid_module.ALPHA_MAX * (n_ratings / (n_ratings + hybrid_module.K_SHRINKAGE))
    else:
        w_cf = 0.0

    w_cf = min(w_cf, 1.0)
    w_content = 1.0 - w_cf
    w_cast = 0.0

    cf_scores = hybrid_module._get_cf_scores_aligned(movie_id)
    content_scores = hybrid_module._get_content_scores_aligned(movie_id)

    final_scores = w_cf * cf_scores + w_content * content_scores

    ranked_positions = np.argsort(final_scores)[::-1]
    recommendations = []
    for pos in ranked_positions:
        candidate_id = hybrid_module._all_movie_ids[pos]
        if candidate_id == movie_id:
            continue
        recommendations.append(candidate_id)
        if len(recommendations) == k:
            break
    return recommendations


def run_diagnosis(pairs: list):
    buckets = bucket_pairs(pairs)
    sparse_pairs = buckets["sparse (<10 ratings)"]

    print(f"Sparse bucket: {len(sparse_pairs)} pairs\n")

    # 1. Weight breakdown
    avg_weights = average_weights_for_bucket(sparse_pairs)
    print("--- Average weights for sparse-bucket movies ---")
    print(f"  w_cf:      {avg_weights['cf']:.4f}")
    print(f"  w_content: {avg_weights['content']:.4f}")
    print(f"  w_cast:    {avg_weights['cast']:.4f}\n")

    # 2. Isolated content-alone and cast-alone precision, sparse bucket only
    print("--- Isolated precision, SPARSE BUCKET ONLY ---")
    content_result = evaluate(content_module.content_based_recommendation_func, sparse_pairs, k=K)
    cast_result = evaluate(cast_module.cast_based_recommend_fn, sparse_pairs, k=K)
    cf_result = evaluate(cf_module.item_cf_recommend_fn, sparse_pairs, k=K)
    hybrid_result = evaluate(hybrid_module.hybrid_recommend_fn, sparse_pairs, k=K)

    print(f"  CF alone:       {cf_result['precision_at_k']:.5f}")
    print(f"  Content alone:  {content_result['precision_at_k']:.5f}")
    print(f"  Cast alone:     {cast_result['precision_at_k']:.5f}")
    print(f"  Hybrid:         {hybrid_result['precision_at_k']:.5f}\n")

    # 3. Variant: hybrid with cast forced to 0
    print("--- Variant: hybrid with cast_cap forced to 0 ---")
    no_cast_result = evaluate(hybrid_no_cast_recommend_fn, sparse_pairs, k=K)
    print(f"  Hybrid (no cast): {no_cast_result['precision_at_k']:.5f}")
    print(f"  Hybrid (with cast): {hybrid_result['precision_at_k']:.5f}")
    print(f"  CF alone (reference): {cf_result['precision_at_k']:.5f}")


if __name__ == "__main__":
    ratings = pd.read_csv(RATINGS_PATH)
    liked = get_liked_movies(ratings)
    pairs = make_eval_pairs(liked)
    run_diagnosis(pairs)