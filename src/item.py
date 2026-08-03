import pandas as pd
import numpy as np 

RATINGS_PATH = "../data/ratings.csv"
K = 10

"""
    IMPORTANT DESIGN NOTE: filling missing ratings with 0 is a real
    simplification worth understanding, not glossing over. A 0 here
    doesn't mean "user rated this 0 stars" — it means "no rating at all."
    Since real ratings are 0.5-5.0, 0 never collides with a real value,
    but it DOES affect the math: cosine similarity computed this way is
    implicitly treating "unrated" as "very different from any rated
    value," which is a known simplification in basic item-based CF.
    (Worth a sentence in your documentation later — this is exactly the
    kind of tradeoff an interviewer would want you to be aware of.)


    Right now this won't crash your evaluation (your eval pairs only ever come from rated/liked movies), but it will crash later the moment you call item_cf_recommend_fn on a movie with 0 or very few ratings — exactly the cold-start case your hybrid design is supposed to handle gracefully.
    Not something to fix in this file necessarily — but flag it as a known gap: item-based CF alone cannot serve unrated/sparse movies, which is precisely why content + cast similarity exists as fallback in your hybrid. Worth a one-line comment in the code noting this is intentional, not forgotten.
"""
def build_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    user_item_matrix = ratings_df.pivot_table(index="movieId", columns="userId", values="rating", fill_value=0)
    return user_item_matrix

def compute_item_similarity(matrix: pd.DataFrame) -> np.ndarray:
    np_array = matrix.to_numpy()
    norms = np.linalg.norm(np_array, axis=1)
    safe_norms = np.where(norms == 0, 1, norms)
    M_normalized = np_array / safe_norms[:, np.newaxis]
    similarity_matrix = M_normalized @ M_normalized.T
    return similarity_matrix

_ratings = pd.read_csv(RATINGS_PATH)
_matrix = build_user_item_matrix(_ratings)
_similarity_matrix = compute_item_similarity(_matrix)
_movie_ids = _matrix.index.tolist()
_movie_id_to_idx = {mid: i for i, mid in enumerate(_movie_ids)}

def item_cf_recommend_fn(movie_id, k:int = K) -> list:
    row_idx = _movie_id_to_idx[movie_id]
    movie_row = _similarity_matrix[row_idx]
    movie_sorted_idx = np.argsort(movie_row)[::-1]
    movie_idx_top_k = [idx for idx in movie_sorted_idx if _movie_ids[idx] != movie_id][:k]
    top_k_idx = [_movie_ids[idx] for idx in movie_idx_top_k]
    return top_k_idx

if __name__ == "__main__":
    print(item_cf_recommend_fn(1))