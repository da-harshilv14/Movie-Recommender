"""
segmented_test.py — Compare hybrid vs CF-alone precision, split by
n_ratings buckets. Tests whether the hybrid actually helps where it's
supposed to (sparse movies), not just in aggregate.
"""

import pandas as pd

from evalution import evaluate, get_liked_movies, make_eval_pairs
from hybrid import hybrid_recommend_fn, _movies_with_idx
import item as cf_module

RATINGS_PATH = "../data/ratings.csv"
K = 10

# Bucket boundaries — informed by your EDA: median was 3, 75th percentile was 9
BUCKETS = {
    "sparse (<10 ratings)": (0, 10),
    "mid (10-50 ratings)": (10, 50),
    "well-rated (50+ ratings)": (50, float("inf")),
}


def bucket_pairs(pairs: list) -> dict:
    """Split eval pairs into buckets based on the KNOWN movie's n_ratings."""
    buckets = {name: [] for name in BUCKETS}

    for known, heldout, user in pairs:
        if known not in _movies_with_idx.index:
            continue  # known movie not in catalog at all — skip

        n_ratings = int(_movies_with_idx.loc[known, "n_ratings"])

        for bucket_name, (low, high) in BUCKETS.items():
            if low <= n_ratings < high:
                buckets[bucket_name].append((known, heldout, user))
                break

    return buckets


def evaluate_on_pairs(recommend_fn, pairs: list, k: int = K) -> dict:
    """Thin wrapper — reuses the existing evaluate() from evaluation.py."""
    return evaluate(recommend_fn, pairs, k=k)


def run_segmented_comparison(pairs: list):
    buckets = bucket_pairs(pairs)

    print(f"{'Bucket':<25} {'N pairs':<10} {'CF Precision':<15} {'Hybrid Precision':<18} {'Lift'}")
    print("-" * 90)

    for bucket_name, bucket_pairs_list in buckets.items():
        if not bucket_pairs_list:
            print(f"{bucket_name:<25} {'0 pairs — skipped':<10}")
            continue

        cf_result = evaluate_on_pairs(cf_module.item_cf_recommend_fn, bucket_pairs_list, k=K)
        hybrid_result = evaluate_on_pairs(hybrid_recommend_fn, bucket_pairs_list, k=K)

        cf_p = cf_result["precision_at_k"]
        hybrid_p = hybrid_result["precision_at_k"]

        if cf_p > 0:
            lift = (hybrid_p - cf_p) / cf_p
            lift_str = f"{lift:+.1%}"
        elif hybrid_p > 0:
            lift_str = "CF=0, hybrid>0"
        else:
            lift_str = "both 0"

        print(f"{bucket_name:<25} {len(bucket_pairs_list):<10} {cf_p:<15.5f} {hybrid_p:<18.5f} {lift_str}")

    print("-" * 90)
    print("Note: buckets with few pairs (<50 or so) are noisy — treat their")
    print("precision numbers as directional, not precise, until validated on")
    print("a larger evaluation set.")


if __name__ == "__main__":
    ratings = pd.read_csv(RATINGS_PATH)
    liked = get_liked_movies(ratings)
    pairs = make_eval_pairs(liked)
    run_segmented_comparison(pairs)