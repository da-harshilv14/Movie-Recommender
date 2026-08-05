import pandas as pd
import numpy as np

import item as cf_module
import cast_similarity as cast_module
import content_based as content_module

MOVIES_PATH = "../data/processed/movies_clean.csv"
K = 10

_movies = pd.read_csv(MOVIES_PATH)
_movies_with_idx = _movies.set_index("movieId")

_all_movie_ids = _movies["movieId"].tolist()
_all_movie_id_to_pos = {mid : i for i, mid in enumerate(_all_movie_ids)}
N = len(_all_movie_ids)

def compute_weights(w_cf: float, w_cast: float) -> tuple:
    """
    Returns (w_cf, w_content, w_cast), guaranteed to sum to 1.0 and all
    non-negative — even when w_cf + w_cast would otherwise exceed 1.
    """
    total_priority = w_cf + w_cast
    if total_priority > 1.0:
        # scale both down proportionally so they still sum to 1
        w_cf = w_cf / total_priority
        w_cast = w_cast / total_priority
        w_content = 0.0
    else:
        w_content = 1.0 - total_priority

    return w_cf, w_content, w_cast

ALPHA_MAX = 0.85
CAST_WEIGHT_CAP = 0.28
K_SHRINKAGE = 3

def get_weights(movie_id) -> dict:
    has_cf = movie_id in cf_module._movie_id_to_idx
    if has_cf:
        n_ratings = int(_movies_with_idx.loc[movie_id, "n_ratings"])
        w_cf_raw = ALPHA_MAX * (n_ratings / (n_ratings + K_SHRINKAGE))
    else:
        w_cf_raw = 0.0

    people = cast_module._movie_people.get(movie_id, set())
    w_cast_raw = CAST_WEIGHT_CAP if len(people) > 0 else 0.0

    w_cf, w_content, w_cast = compute_weights(w_cf_raw, w_cast_raw)

    return {"cf": w_cf, "content": w_content, "cast": w_cast}

def _get_cf_scores_aligned(movie_id) -> np.ndarray:
    """CF similarity of movie_id against every movie in _all_movie_ids, 0 where unavailable."""
    scores = np.zeros(N)
    if movie_id not in cf_module._movie_id_to_idx:
        return scores  # CF has no data for this movie at all

    row_idx = cf_module._movie_id_to_idx[movie_id]
    cf_row = cf_module._similarity_matrix[row_idx]

    for i, other_id in enumerate(cf_module._movie_ids):
        pos = _all_movie_id_to_pos.get(other_id)
        if pos is not None:
            scores[pos] = cf_row[i]
    return scores

def _get_content_scores_aligned(movie_id) -> np.ndarray:
    """Content similarity of movie_id against every movie in _all_movie_ids."""
    scores = np.zeros(N)
    if movie_id not in content_module._movie_id_to_idx:
        return scores

    row_idx = content_module._movie_id_to_idx[movie_id]
    content_row = content_module._content_similariy[row_idx]

    for i, other_id in enumerate(content_module._movies_ids):
        pos = _all_movie_id_to_pos.get(other_id)
        if pos is not None:
            scores[pos] = content_row[i]
    return scores

def _get_cast_scores_aligned(movie_id) -> np.ndarray:
    """Cast/director Jaccard similarity of movie_id against every movie."""
    scores = np.zeros(N)
    movie_set = cast_module._movie_people.get(movie_id, set())
    if not movie_set:
        return scores  # nothing to compare — leave all zeros

    for other_id, other_set in cast_module._movie_people.items():
        if other_id == movie_id:
            continue
        pos = _all_movie_id_to_pos.get(other_id)
        if pos is not None:
            scores[pos] = cast_module.jaccard_similarity(movie_set, other_set)
    return scores


def hybrid_recommend_fn(movie_id, k: int = K) -> list:
    weights = get_weights(movie_id)

    cf_scores = _get_cf_scores_aligned(movie_id)
    content_scores = _get_content_scores_aligned(movie_id)
    cast_scores = _get_cast_scores_aligned(movie_id)

    final_scores = (
        weights["cf"] * cf_scores
        + weights["content"] * content_scores
        + weights["cast"] * cast_scores
    )

    ranked_positions = np.argsort(final_scores)[::-1]

    recommendations = []
    for pos in ranked_positions:
        candidate_id = _all_movie_ids[pos]
        if candidate_id == movie_id:
            continue
        recommendations.append(candidate_id)
        if len(recommendations) == k:
            break

    return recommendations

if __name__ == "__main__":
    print(get_weights(1))
    print(hybrid_recommend_fn(1))
