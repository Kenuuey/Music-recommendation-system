import pandas as pd
import numpy as np
import joblib
from ast import literal_eval
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MinMaxScaler

class ContentBasedRecommender():
    """Content-based music recommender using lyrics and optional Word2Vec expansion or classifier."""

    def __init__(self, interactions_df, tracks_df, lyrics_df, genres_df):
        self.interactions_df = interactions_df
        self.tracks_df = tracks_df
        self.lyrics_df = lyrics_df
        self.genres_df = genres_df

        # Ensure bow column is dict
        if isinstance(self.lyrics_df["bow"].iloc[0], str):
            self.lyrics_df["bow"] = self.lyrics_df["bow"].apply(literal_eval)

    def baseline(self, keyword : str, threshold: int = 5, k: int = 50):
        """Return top-k tracks containing a keyword in lyrics."""
        mask = self.lyrics_df["bow"].apply(lambda bow: bow.get(keyword, 0) >= threshold)
        keyword_tracks = self.lyrics_df[mask][["track_id"]]
        
        if keyword_tracks.empty:
            print(f"No tracks contain the keyword '{keyword}'. Returning empty results.")
            return pd.DataFrame()

        merged = (
            keyword_tracks
            .merge(self.tracks_df, on="track_id", how="left") \
            .merge(self.interactions_df, on="song_id", how="left")
        )

        track_playcounts = (
            merged.groupby(["track_id", "artist_name", "track_title"])["play_count"]
            .sum()
            .reset_index()
        )

        top_k = track_playcounts.sort_values("play_count", ascending=False).head(k).reset_index(drop=True)
        top_k.index = top_k.index + 1
        top_k.index.name = "rank"
        return top_k
    
    def word2vec(self, keyword: str, model, topn: int = 10, threshold: int = 5, k: int = 50):
        """Use Word2Vec expansion to find tracks related to a keyword."""
        if keyword not in model.wv.key_to_index:
            print(f"Keyword '{keyword}' not in Word2Vec vocabulary. Returning empty results.")
            return pd.DataFrame()

        similar_words = [w for w, _ in model.wv.most_similar(keyword, topn=topn)]
        all_keywords = [keyword] + similar_words

        mask = self.lyrics_df["bow"].apply(lambda bow: any(bow.get(w, 0) >= threshold for w in all_keywords))
        keyword_tracks = self.lyrics_df[mask][["track_id"]]

        if keyword_tracks.empty:
            print(f"No tracks contain the keyword '{keyword}' or similar words. Returning empty results.")
            return pd.DataFrame()
    
        merged = (
            keyword_tracks
            .merge(self.tracks_df, on="track_id", how="left")
            .merge(self.interactions_df, on="song_id", how="left")
        )

        track_playcounts = (
            merged.groupby(["track_id", "artist_name", "track_title"])["play_count"]
            .sum()
            .reset_index()
        )

        top_k = track_playcounts.sort_values("play_count", ascending=False).head(k).reset_index(drop=True)
        top_k.index = top_k.index + 1
        top_k.index.name = "rank"
        return top_k
    
    def classifier(self, keyword: str, k: int = 50, label_by_genre=False):
        """Predict tracks about a keyword using a trained classifier."""
        
        # Merge genre info
        if label_by_genre:
            self.lyrics_df = self.lyrics_df.merge(
                self.genres_df[['track_id', 'majority_genre']], 
                on='track_id', 
                how='left'
            )
        
        # Create labels
        if label_by_genre:
            if "majority_genre" not in self.lyrics_df.columns:
                print("Warning: 'majority_genre' column not found. Using keyword labeling instead.")
                label_by_genre = False

        if label_by_genre:
            y = (self.lyrics_df["majority_genre"] == keyword).astype(int)
        else:
            y = self.lyrics_df["bow"].apply(lambda bow: 1 if bow.get(keyword, 0) > 0 else 0).astype(int)

        if y.sum() == 0:
            print(f"No tracks contain the keyword '{keyword}'. Returning empty results.")
            return pd.DataFrame()
        
        # Vectorize features
        vec = DictVectorizer(sparse=True)
        X = vec.fit_transform(self.lyrics_df['bow'])

        # Cross-validated probabilities
        clf = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced', random_state=42)
        probs = cross_val_predict(clf, X, y, cv=5, method='predict_proba')[:, 1]

        scored = pd.DataFrame({
            "track_id": self.lyrics_df["track_id"].values,
            "score": probs
        })

        # Aggregate play counts
        song_plays = self.interactions_df.groupby("song_id")["play_count"].sum().reset_index()
        tracks_and_plays = self.tracks_df.merge(song_plays, on="song_id", how="left")
        tracks_and_plays["play_count"] = tracks_and_plays["play_count"].fillna(0)
        
        merged = scored.merge(tracks_and_plays, on="track_id", how="left")
        merged["play_count_log"] = np.log1p(merged["play_count"])

        # Normalize scores
        scaler = MinMaxScaler()
        merged[["score_norm", "plays_norm"]] = scaler.fit_transform(merged[["score", "play_count_log"]])

        alpha = 0.6
        merged["final_score"] = alpha * merged["score_norm"] + (1 - alpha) * merged["plays_norm"]

        topk = merged.sort_values("final_score", ascending=False).head(k).reset_index(drop=True)
        topk.index = topk.index + 1
        topk.index.name = "rank"
        return topk[["track_id", "artist_name", "track_title", "play_count", "score", "final_score"]]