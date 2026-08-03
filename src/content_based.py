import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from item import compute_item_similaritty #own cosine similarity code (take dataframe as params)

MOVIES_PATH = "../data/processed/movies_clean.csv"
K = 10

def build_tfidf_matrix(movies_df: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
    movies_df["content_text"] = movies_df["content_text"].fillna("")
    X = vectorizer.fit_transform(movies_df["content_text"])
    return vectorizer, X

def compute_content_similarity(tfidf_matrix):
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

_movies = pd.read_csv(MOVIES_PATH)
_vectorizer, _tfidf_matrix = build_tfidf_matrix(_movies)
_content_similariy = compute_content_similarity(_tfidf_matrix)
_movies_ids = _movies["movieId"].tolist()
_movie_id_to_idx = {mid: i for i, mid in enumerate(_movies_ids)}

def content_based_recommendation_func(movie_id, k:int=K) -> list:
    movie_idx = _movie_id_to_idx[movie_id]
    movie_row = np.argsort(_content_similariy[movie_idx])[::-1]
    movie_row = [movie for movie in movie_row if _movies_ids[movie] != movie_id][:k]
    top_k_idx = [_movies_ids[idx] for idx in movie_row]
    return top_k_idx

if __name__ == "__main__":
    print(content_based_recommendation_func(1))
