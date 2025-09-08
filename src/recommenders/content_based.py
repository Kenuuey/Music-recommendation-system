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

    def collection_baseline(self, keyword : str, threshold: int = 5, k: int = 50):
        """Return Top-k tracks containing a keyword in lyrics (baseline approach)."""
        mask = self.lyrics_df["bow"].apply(
            lambda bow: bow.get(keyword, 0) >= threshold
        )
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

        top_k_by_keyword_baseline = (
            track_playcounts
            .sort_values("play_count", ascending=False)
            .head(k)
            .reset_index(drop=True)
        )

        top_k_by_keyword_baseline.index = top_k_by_keyword_baseline.index + 1
        top_k_by_keyword_baseline.index.name = "rank"
        return top_k_by_keyword_baseline
    
    def collection_word2vec(self, keyword: str, model, topn: int = 10, threshold: int = 5, k: int = 50):
        """
        Use Word2Vec expansion of keyword.
        model: trained gensim Word2Vec model
        """
        similar_words = [w for w, _ in model.wv.most_similar(keyword, topn=topn)]
        all_keywords = [keyword] + similar_words

        mask = self.lyrics_df["bow"].apply(
            lambda bow: any(bow.get(w, 0) >= threshold for w in all_keywords)
        )
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

        top_k_by_keyword_word2vec = (
            track_playcounts
            .sort_values(by="play_count", ascending=False)
            .head(k)
            .reset_index(drop=True)
        )

        top_k_by_keyword_word2vec.index = top_k_by_keyword_word2vec.index + 1
        top_k_by_keyword_word2vec.index.name = "rank"
        return top_k_by_keyword_word2vec
    
    def collection_classifier(self, keyword: str, k: int = 50, vectorizer_type="count"):
        """
        Predict tracks about a keyword using a trained classifier.
         keyword: str, the genre/label to predict (must exist in lyrics_df['genre'] or similar)
        k: int, number of top tracks to return
        vectorizer_type: "count" for CountVectorizer or "tfidf" for TfidfVectorizer
        """
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        # Prepare features
        texts = self.lyrics_df["bow"].apply(
            lambda bow: " ".join([w for w, c in bow.items() for _ in range(c)])
        )

        # Create a binary target column for the keyword
        y = (self.lyrics_df["majority_genre"] == keyword).astype(int)  # 1 if track is the keyword genre, else 0

        # Fit vectorizer
        if vectorizer_type == "count":
            vectorizer = CountVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
        elif vectorizer_type == "tfidf":
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
        else:
            raise ValueError("vectorizer_type must be 'count' or 'tfidf'")
        
        X = vectorizer.fit_transform(texts)

        # Train classifier
        classifier = LogisticRegression(max_iter=500, solver='liblinear', random_state=42)
        classifier.fit(X, y)
        
        # Predict probabilities
        probs = classifier.predict_proba(X)[:, 1]  # probability of positive class

        scored = pd.DataFrame({
            "track_id": self.lyrics_df["track_id"],
            "score": probs
        })

        # Merge metadata
        merged = (
            scored
            .merge(self.tracks_df, on="track_id", how="left")
            .merge(self.interactions_df, on="song_id", how="left")
        )
        
        merged["play_count"] = merged["play_count"].fillna(0)
        
        ranked = (
            merged.groupby(["track_id", "artist_name", "track_title"])[["play_count", "score"]]
            .sum()
            .reset_index()
        )
        
        # Final ranking
        ranked["final_score"] = ranked["score"] * ranked["play_count"]
        ranked = ranked.sort_values("final_score", ascending=False).head(k)
        
        ranked.index = ranked.index + 1
        ranked.index.name = "rank"
        
        return ranked