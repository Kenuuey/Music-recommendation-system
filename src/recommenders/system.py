from typing import Optional, List
import pandas as pd
from src.recommenders.non_personalized import NonPersonalizedRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.collaborative import UserBasedRecommender, ItemBasedRecommender

class MusicRecommenderSystem:
    """Unified system combining multiple types of recommenders."""

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        tracks_df: pd.DataFrame,
        genres_df: Optional[pd.DataFrame] = None,
        lyrics_df: Optional[pd.DataFrame] = None,
        top_k_neighbors: int = 50
    ):
        self.interactions_df = interactions_df
        self.tracks_df = tracks_df
        self.genres_df = genres_df
        self.lyrics_df = lyrics_df
        self.top_k_neighbors = top_k_neighbors

        # Initialize individual recommenders
        self.non_personalized = NonPersonalizedRecommender(interactions_df, tracks_df, genres_df) \
            if genres_df is not None else None
        self.content_based = ContentBasedRecommender(interactions_df, tracks_df, lyrics_df) \
            if lyrics_df is not None else None
        self.user_based = UserBasedRecommender(interactions_df, tracks_df, top_k_neighbors)
        self.item_based = ItemBasedRecommender(interactions_df, tracks_df, top_k_neighbors)

    def train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        """Split interactions into train/test for collaborative recommenders."""
        self.user_based.train_test_split(test_size, random_state)
        self.item_based.train_test_split(test_size, random_state)

    def fit(self):
        """Fit all recommenders."""
        if self.non_personalized:
            self.non_personalized.fit()
        if self.content_based:
            pass  # No fitting required unless classifier approach is used
        self.user_based.fit()
        self.item_based.fit()

    # ---------------- Recommendation API ----------------
    def recommend_user(self, user_id: int, k: int = 10, method: str = 'user') -> pd.DataFrame:
        """Recommend tracks for a user."""
        if method == 'user':
            return self.user_based.recommend_for_user(user_id, k)
        elif method == 'item':
            return self.item_based.recommend_for_track(user_id, k)  # treating user_id as track_id?
        else:
            raise ValueError("method must be 'user' or 'item'")

    def recommend_track(self, track_id: int, k: int = 10) -> pd.DataFrame:
        """Recommend tracks similar to a given track."""
        return self.item_based.recommend_for_track(track_id, k)

    def recommend_keyword(self, keyword: str, k: int = 50, method: str = 'baseline', **kwargs) -> pd.DataFrame:
        """Recommend tracks by keyword using content-based approaches."""
        if self.content_based is None:
            raise ValueError("Content-based recommender not initialized (lyrics_df missing).")

        if method == 'baseline':
            return self.content_based.baseline(keyword, k=k)
        elif method == 'word2vec':
            model = kwargs.get('model')
            topn = kwargs.get('topn', 10)
            threshold = kwargs.get('threshold', 5)
            return self.content_based.word2vec(keyword, model, topn=topn, threshold=threshold, k=k)
        elif method == 'classifier':
            label_by_genre = kwargs.get('label_by_genre', False)
            return self.content_based.classifier(keyword, k=k, label_by_genre=label_by_genre)
        else:
            raise ValueError("method must be one of ['baseline', 'word2vec', 'classifier']")

    def recommend_top_global(self, k: int = 250) -> pd.DataFrame:
        """Recommend top tracks globally (non-personalized)."""
        if self.non_personalized is None:
            raise ValueError("Non-personalized recommender not initialized (genres_df missing).")
        return self.non_personalized.top_k_global(k)

    def recommend_top_genre(self, genre: str, k: int = 100) -> pd.DataFrame:
        """Recommend top tracks for a specific genre."""
        if self.non_personalized is None:
            raise ValueError("Non-personalized recommender not initialized (genres_df missing).")
        return self.non_personalized.top_k_by_genre(genre, k)

    # ---------------- Evaluation ----------------
    def evaluate_user_based(self, k: int = 10, users: Optional[List[int]] = None) -> dict:
        return self.user_based.precision_recall_at_k(k=k, users=users)

    def evaluate_item_based(self, k: int = 10, tracks: Optional[List[int]] = None) -> dict:
        return self.item_based.precision_recall_at_k(k=k, tracks=tracks)
