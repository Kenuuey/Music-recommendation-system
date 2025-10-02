from .base import MusicRecommender
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MinMaxScaler

class ContentBasedRecommender(MusicRecommender):
    """Content-based music recommender using lyrics and optional Word2Vec expansion or classifier."""

    def __init__(self, interactions_df, tracks_df, lyrics_df):
        """
        interactions_df: DataFrame [user_id, song_id, play_count]
        tracks_df: DataFrame [track_id, song_id, artist_name, track_title]
        lyrics_df: DataFrame [track_id, mxm_track_id, bow] where bow is dict(word -> count)
        """
        self.interactions_df = interactions_df
        self.tracks_df = tracks_df
        self.lyrics_df = lyrics_df

    def fit(self):
        """No training needed for baseline content-based methods."""
        pass

    def baseline(self, keyword : str, threshold: int = 5, k: int = 50):
        """Return top-k tracks containing a keyword in lyrics."""
        mask = self.lyrics_df["bow"].apply(lambda bow: bow.get(keyword, 0) >= threshold)
        keyword_tracks = self.lyrics_df[mask]["track_id"]

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
    
    def word2vec(self, keyword: str, model, topn: int = 10, threshold: int = 5, k: int = 50):
        """
        Use Word2Vec expansion to find tracks related to a keyword.

        Args:
            keyword (str): Seed word.
            model: Trained gensim Word2Vec model.
            topn (int): Number of similar words to include.
            threshold (int): Minimum count in bow to consider.
            k (int): Number of top tracks to return.
        """
        # if keyword not in model.wv.vocab:
        #     raise ValueError(f"Keyword {keyword} not in vocabulary")
        
        similar_words = [w for w, _ in model.wv.most_similar(keyword, topn=topn)]
        all_keywords = [keyword] + similar_words

        mask = self.lyrics_df["bow"].apply(lambda bow: any(bow.get(w, 0) >= threshold for w in all_keywords))
        keyword_tracks = self.lyrics_df[mask][["track_id"]]
    
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
        """
        Predict tracks about a keyword using a trained classifier.

        Args:
            keyword (str): Genre/label to predict (must exist in lyrics_df).
            k (int): Number of top tracks to return.
            label_by_genre (bool): Whether to label by genre or by keyword presence.
        """
        # create labels
        if label_by_genre:
            y = (self.lyrics_df["majority_genre"] == keyword).astype(int)
        else:
            y = self.lyrics_df["bow"].apply(lambda bow: 1 if bow.get(keyword, 0) > 0 else 0).astype(int)
        
        # vectorize
        vec = DictVectorizer(sparse=True)
        X = vec.fit_transform(self.lyrics_df['bow'])

        # classifier and get out-of-fold probabilities
        clf = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced', random_state=42)
        # get out-of-fold probabilities for realistic scoring
        probs = cross_val_predict(clf, X, y, cv=5, method='predict_proba')[:, 1]

        scored = pd.DataFrame({
            "track_id": self.lyrics_df["track_id"].values,
            "score": probs
        })

        # aggregate play counts per song_id
        song_plays = self.interactions_df.groupby("song_id")["play_count"].sum().reset_index()
        tracks_and_plays = self.tracks_df.merge(song_plays, on="song_id", how="left")
        tracks_and_plays["play_count"] = tracks_and_plays["play_count"].fillna(0)

        # merge scored (per-track) with tracks_and_plays (per-track)
        merged = scored.merge(tracks_and_plays, on="track_id", how="left")

        # log scale plays
        merged["play_count_log"] = np.log1p(merged["play_count"])
        
        # normalize score and play_count_log to [0,1]
        scaler = MinMaxScaler()
        merged[["score_norm", "plays_norm"]] = scaler.fit_transform(merged[["score", "play_count_log"]])

        # combine with tunable weight (alpha for score importance)
        alpha = 0.6
        merged["final_score"] = alpha * merged["score_norm"] + (1 - alpha) * merged["plays_norm"]

        topk = merged.sort_values("final_score", ascending=False).head(k).reset_index(drop=True)
        topk.index = topk.index + 1
        topk.index.name = "rank"
        return topk[["track_id", "artist_name", "track_title", "play_count", "score", "final_score"]]