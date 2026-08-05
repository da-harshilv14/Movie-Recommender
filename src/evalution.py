import pandas as pd
from collections import defaultdict
from pprint import pprint
import random

RELEVANCE_THRESHOLD = 4.0
K = 10
random.seed(42)
moviesDF = pd.read_csv("../data/processed/movies_clean.csv")


def get_liked_movies(rating_df: pd.DataFrame) -> dict:
    highratingsDF = rating_df[rating_df['rating'] >= RELEVANCE_THRESHOLD]
    user_counts = highratingsDF.groupby('userId')['userId'].transform('count')

    result = highratingsDF[user_counts >= 2][['userId', 'movieId', 'rating']]

    liked_movies = result.groupby('userId')['movieId'].apply(list).to_dict()
    return liked_movies

def make_eval_pairs(liked_movies: dict, seet: int = 42) -> list[tuple]:
    # For now only one (known, held, userid) tuple for each userid
    ans = []
    random.seed(seet)
    for userid, movies in liked_movies.items():
        movie_to_hold = random.choice(movies)
        options = [c for c in movies if c != movie_to_hold]
        movie_known = random.choice(options)
        ans.append((movie_known, movie_to_hold, userid))
    return ans

def evaluate(recommend_fn, eval_pairs: list[tuple], k: int = K) -> dict:
    total_hits = 0
    for known, heldout, user in eval_pairs:
        k_recommendations = recommend_fn(known)
        if heldout in k_recommendations:
            total_hits+=1
    
    precision_at_k = (total_hits)/ (len(eval_pairs) * k)
    
    # for now only ONE-held-out-per-user
    # can change if multiple movies per user 
    # change denominator to (total relevant movies, not total pairs)
    recall_at_k = (total_hits)/(len(eval_pairs))

    return {"precision_at_k":precision_at_k, "recall_at_k": recall_at_k, "n_pairs": len(eval_pairs)}

def dummy_recommend_fn(movie_id, k=K) -> list[int]:
    movies_with_IDs = moviesDF["movieId"].to_list()
    movieList = random.choices(movies_with_IDs, k=k)
    return movieList

from baseline import baseline_recommend_fn
from item import item_cf_recommend_fn
from content_based import content_based_recommendation_func
from cast_similarity import cast_based_recommend_fn
from hybrid import hybrid_recommend_fn

if __name__ == "__main__":
    ratings = pd.read_csv("../data/ratings.csv")  # adjust path
    liked = get_liked_movies(ratings)
    pairs = make_eval_pairs(liked)
    # results = evaluate(baseline_recommend_fn, pairs)
    results = evaluate(hybrid_recommend_fn, pairs)
    print(results)