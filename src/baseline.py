import pandas as pd

MOVIE_PATH = "../data/processed/movies_clean.csv"
RATING_PATH = "../data/ratings.csv"

K = 10

def compute_popularity_rating(ratings_df: pd.DataFrame) -> list:
    
    # 1. Group by movie and count rows
    movie_ratings_count = ratings_df.groupby("movieId").size()
    movie_ratings_sum = ratings_df.groupby("movieId")["rating"].mean()

    # 2. Reset the index to turn it back into a clean DataFrame
    movie_ratings_count = movie_ratings_count.reset_index(name="num_ratings")
    movie_ratings_sum = movie_ratings_sum.reset_index(name="average_ratings")

    m = 2

    weighted_ratings = movie_ratings_count.merge(movie_ratings_sum, on="movieId")
    weighted_ratings["weighted_rating"] = ((weighted_ratings["num_ratings"] / (weighted_ratings["num_ratings"] + m) ) * weighted_ratings["average_ratings"]) + (m / (m + weighted_ratings["num_ratings"])) * (weighted_ratings["average_ratings"].mean())
    ans = weighted_ratings.filter(items=["movieId", "weighted_rating", "rating"]).sort_values(by="weighted_rating", ascending=False).head(K)
    lisst = ans["movieId"].to_list()


    return lisst

_ratings = pd.read_csv(RATING_PATH)
POPULARITY_RANKING = compute_popularity_rating(ratings_df=_ratings)

def baseline_recommend_fn(movie_id, k: int=K) -> list:
    subset_movies = POPULARITY_RANKING[:k+1]
    top_k_list = [m_id for m_id in subset_movies if m_id != movie_id][:k]
    return top_k_list

if __name__ == "__main__":
    # quick manual sanity check before plugging into evaluate()
    print(baseline_recommend_fn(77))  # try any real movieId from your data