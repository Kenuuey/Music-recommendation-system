from src.recommenders.base import MusicRecommender
import pandas as pd

class ContentBasedRecommender(MusicRecommender):
    def __init__(self, interactions_df, tracks_df, lyrics_df):
        """
        interactions_df: DataFrame [user_id, song_id, play_count]
        tracks_df: DataFrame [track_id, song_id, artist_name, track_title]
        lyrics_df: DataFrame [track_id, mxm_track_id, bow] where bow is dict(word -> count)
        """
        self.interactions_df = interactions_df
        self.tracks_df = tracks_df
        self.lyrics_df = lyrics_df

    def collection_baseline(self, keyword : str, threshold: int = 1, k: int = 50):
        """Return Top-k tracks containing a keyword in lyrics (baseline approach)."""
        mask = self.lyrics_df["bow"].apply(lambda bow: bow.get(keyword, 0) >= threshold)
        keyword_tracks = self.lyrics_df[mask]["track_id"]

        # Merge with play counts
        merged = (
            keyword_tracks
            .merge(self.interactions_df, on="song_id", how="left")
            .merge(self.tracks_df, on="track_id", how="left")
        )

        track_playcounts = (
            merged.groupby(["track_id", "artist_name", "track_title"])["play_count"]
            .sum().reset_index()
            .sort_values("play_count", ascending=False)
            .head(k)
        )

        track_playcounts.index = track_playcounts.index + 1
        track_playcounts.index.name = "rank"
        return track_playcounts
    
    def recommend_by_word2vec(self, keyword, k=50, n_similar=10):
        # expand keyword with similar tokens & score by combined counts
        pass
    
    
    
    def recommend_by_classifier(self, keyword, k=50,): #clf):
        # train classifier on labeled songs and predict probabilities on unlabeled
        pass
