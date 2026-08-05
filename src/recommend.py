import pandas as pd
import numpy as np
from hybrid import hybrid_recommend_fn, _movies_with_idx

MOVIES_PATH = "../data/processed/movies_clean.csv"
_movies = pd.read_csv(MOVIES_PATH)
_movieid_to_title = _movies.set_index("movieId")["title"].to_dict()


def find_movie_by_title(query : str) -> pd.DataFrame :
    movies_with_title = _movies[_movies["title"].str.contains(query, na=False, case=False)]
    return movies_with_title

def recommend_by_title(query: str, k: int = 10) -> list:
    movie_with_title = find_movie_by_title(query)
    if movie_with_title.shape[0] == 0:
        return "No movie with the title exists."

    # For now just take the first recommended
    movie_id = movie_with_title.iloc[0]["movieId"]
    movie_recommendations = hybrid_recommend_fn(movie_id=movie_id, k=6)
    movie_rec_titles = [_movieid_to_title[movie_id] for movie_id in movie_recommendations]
    return movie_rec_titles

if __name__ == "__main__":
    results = recommend_by_title("Weekend at bernie")
    print(results)



