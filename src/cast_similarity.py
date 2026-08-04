import pandas as pd
import numpy as np

CAST_PATH = "../data/processed/cast.csv"
CREW_PATH = "../data/processed/crew.csv"
MOVIE_PATH = "../data/processed/movies_clean.csv"
K = 5


def build_movie_people_stats(cast_df: pd.DataFrame, crew_df: pd.DataFrame) -> dict:
    cast_filter = cast_df[cast_df['order'] < 5]
    movie_to_people = cast_filter.groupby(by="movie_id")["person_id"].apply(set)
    movie_to_crew = crew_df.groupby(by="movie_id")['person_id'].apply(set)
    ans  = movie_to_people.combine(
        movie_to_crew, 
        lambda set1, set2: (set1 if pd.notna(set1) else set()) | (set2 if pd.notna(set2) else set())
    )
    return ans.to_dict()

def jaccard_similarity(set_a:set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    else:
        return len(set_a & set_b) / len(set_a | set_b)

_cast = pd.read_csv(CAST_PATH)
_crew = pd.read_csv(CREW_PATH)
_movies = pd.read_csv(MOVIE_PATH)
_movie_people = build_movie_people_stats(_cast, _crew)
_movie_ids = _movies["movieId"].tolist()


def cast_based_recommend_fn(movie_id, k:int=K) -> list:
    movie_set = _movie_people.get(movie_id, set())

    if not movie_set:
        return []
    
    movie_ratings = [[movie, jaccard_similarity(movie_set, _movie_people[movie])] for movie in _movie_people if movie_id != movie]
    movie_ratings.sort(key=lambda x:x[1], reverse=True)
    ans = [movie[0] for movie in movie_ratings][:k]
    return ans

if __name__ == "__main__":
    cast_based_recommend_fn(1)
