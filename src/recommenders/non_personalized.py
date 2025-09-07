from src.recommenders.base import MusicRecommender

class NonPersonalizedRecommender(MusicRecommender):
    def fit(self, interactions_df, tracks_df, genres_df=None):
        """
        interactions_df: DataFrame with [user_id, track_id, play_count]
        tracks_df:       DataFrame with [track_id, track_title, artist_name]
        genres_df:       DataFrame with [track_id, genre, (optional minority_genre)]
        """
        self.triplets_df = interactions_df
        self.tracks_df = tracks_df
        self.genres_df = genres_df

        # Precompute global playcounts
        self.track_playcounts = (
            self.triplets_df.groupby("track_id")["play_count"]
            .sum()
            .reset_index()
        )

        return self

    def recommend_top_k(self, k=250):
        """
        Returns the Top k most popular tracks (global popularity).
        Output DataFrame: [artist_name, track_title, play_count]
        """
        merged = pd.merge(self.track_playcounts, self.tracks_df, on="track_id")

        top_k = (
            merged.sort_values("play_count", ascending=False)
            .head(k)
            .reset_index(drop=True)
        )

        # Reorder columns
        top_k = top_k[["artist_name", "track_title", "play_count"]]

        return top_k

    def recommend_by_genre(self, genre, k=100):
        """
        Returns Top k tracks for a given genre.
        Requires genres_df passed in fit().
        """
        if self.genres_df is None:
            raise ValueError("Genres dataset was not provided in fit().")

        # Filter tracks for this genre
        genre_tracks = self.genres_df[self.genres_df["genre"] == genre]

        # Join with playcounts and metadata
        merged = (
            pd.merge(genre_tracks, self.track_playcounts, on="track_id")
            .merge(self.tracks_df, on="track_id")
        )

        top_k = (
            merged.sort_values("play_count", ascending=False)
            .head(k)
            .reset_index(drop=True)
        )

        return top_k[["artist_name", "track_title", "genre", "play_count"]]