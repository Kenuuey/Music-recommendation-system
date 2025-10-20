from .recommenders.base import MusicRecommender
import pandas as pd

class NonPersonalizedRecommender(MusicRecommender):
    """Recommend most popular tracks overall or by genre."""

    def __init__(self, interactions_df: pd.DataFrame, tracks_df: pd.DataFrame, genres_df: pd.DataFrame):
        self.interactions_df = interactions_df
        self.tracks_df = tracks_df
        self.genres_df = genres_df

    def fit(self):
        """No training needed for non-personalized recommender."""
        pass

    def top_k_global(self, k: int = 250) -> pd.DataFrame:
        """Return Top k tracks overall."""
        track_playcounts = (
            self.interactions_df.groupby("song_id")["play_count"]
            .sum().reset_index()
            .sort_values("play_count", ascending=False)
            .head(k)
        )
        top_k = track_playcounts.merge(self.tracks_df, on="song_id", how="left")
        top_k.index = top_k.index + 1
        top_k.index.name = "rank"
        return top_k["artist_name", "track_title", "play_count"]

    def top_k_by_genre(self, genre: str, k: int = 100) -> pd.DataFrame:
        """
        Return top-k tracks for a given genre.

        Args:
            genre (str): Genre to filter by.
            k (int): Number of top tracks to return.

        Returns:
            pd.DataFrame: Columns ['artist_name', 'track_title', 'play_count'], indexed by rank.
        """
        if genre not in self.genres_df["majority_genre"].unique():
            raise ValueError(f"Genre '{genre}' not found in dataset")
        merged = (
            self.interactions_df
            .merge(self.tracks_df[["track_id", "song_id", "artist_name", "track_title"]], 
                   on="song_id", how="left")
            .merge(self.genres_df[["track_id", "majority_genre"]], 
                   on="track_id", how="left")
        )
        
        genre_df = merged[merged["majority_genre"] == genre]
        
        top_k_by_genre = (
            genre_df.groupby(["track_id", "artist_name", "track_title"])["play_count"]
            .sum()
            .reset_index()
            .sort_values("play_count", ascending=False)
            .head(k)
        )
        top_k_by_genre.index = top_k_by_genre.index + 1
        top_k_by_genre.index.name = "rank"
        return top_k_by_genre[["artist_name", "track_title", "play_count"]]
