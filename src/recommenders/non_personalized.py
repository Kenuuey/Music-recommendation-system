from src.recommenders.base import MusicRecommender

class NonPersonalizedRecommender(MusicRecommender):
    def __init__(self, interactions_df: pd.DataFrame, tracks_df: pd.DataFrame, genres_df: pd.DataFrame):
        self.interactions_df = interactions_df
        self.tracks_df = tracks_df
        self.genres_df = genres_df

    def top_k_global(self, k: int = 250) -> pd.DataFrame:
        """Return Top k tracks overall."""
        track_playcounts = (
            self.interactions_df.groupby("song_id")["play_count"]
            .sum().reset_index()
            .sort_values("play_count", ascending=False)
            .head(k)
        )

        top_k = (
            track_playcounts.merge(self.tracks_df, on="song_id", how="left")
            [["artist_name", "track_title", "play_count"]]
        )
        
        top_k.index = top_k.index + 1
        top_k.index.name = "rank"
        return top_k

    def top_k_by_genre(self, genre: str, k: int = 100) -> pd.DataFrame:
        """
        Return Top k tracks for a given genre.
        Output: DataFrame [index, artist_name, track_title, play_count]
        """
        if genre not in self.genres_df["majority_genre"].unique():
            raise ValueError(f"Genre '{genre}' not found in dataset")

        merged = (
            self.interactions_df
            .merge(self.tracks_df[["track_id", "song_id", "artist_name", "track_title"]], on="song_id", how="left")
            .merge(self.genres_df[["track_id", "majority_genre"]], on="track_id", how="left")
        )

        genre_df = merged[merged["majority_genre"] == genre]

        track_playcounts_by_genre = (
            genre_df.groupby(["track_id", "artist_name", "track_title"])["play_count"]
            .sum()
            .reset_index()
        )

        top_k_by_genre = (
            track_playcounts_by_genre
            .sort_values("play_count", ascending=False)
            .head(k)
            .reset_index(drop=True)
        )

        top_k_by_genre.index = top_k_by_genre.index + 1
        top_k_by_genre.index.name = "rank"
        return top_k_by_genre
