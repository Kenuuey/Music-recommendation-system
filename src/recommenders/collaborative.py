from src.recommenders.base import MusicRecommender
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import faiss

class Collaborative(MusicRecommender):
    def __init__(self, interactions_df, tracks_df):
        self.interactions_df = interactions_df
        self.tracks_df = tracks_df
        self.train_df = None
        self.test_df = None

    def train_test_split(self, test_size=0.2, random_state=42):
        self.train_df, self.test_df = train_test_split(
            self.interactions_df, test_size=test_size, random_state=random_state
        )

class UserBasedRecommender(Collaborative):
    def __init__(self, interactions_df, tracks_df, top_k_neighbors=50):
        super().__init__(interactions_df, tracks_df)
        self.top_k_neighbors = top_k_neighbors
        self.user_map = None
        self.song_map = None
        self.user_item_sparse = None
        self.user_ids = None
        self.user_sim = None  # stores top-K neighbors
    
    def fit(self):
        # 1. Map user_id and song_id to indices
        self.user_ids = self.train_df['user_id'].unique()
        self.song_ids = self.train_df['song_id'].unique()
        self.user_map = {user: idx for idx, user in enumerate(self.user_ids)}
        self.song_map = {song: idx for idx, song in enumerate(self.song_ids)}
        
        # 2. Build sparse user-item matrix
        rows = self.train_df['user_id'].map(self.user_map).to_numpy()
        cols = self.train_df['song_id'].map(self.song_map).to_numpy()
        data = self.train_df['play_count'].to_numpy()
        self.user_item_sparse = csr_matrix(
            (data, (rows, cols)), shape=(len(self.user_ids), len(self.song_ids))
        )

        # 3. Compute latent user vectors via SVD
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
        svd = TruncatedSVD(n_components=128)
        user_latent = svd.fit_transform(self.user_item_sparse)  # shape: (num_users, 128)

        # 4. Normalize for cosine similarity (dot product = cosine)
        user_latent = normalize(user_latent.astype(np.float32))

        # 5. Build IVF index
        dim = user_latent.shape[1]
        nlist = 4096  # number of clusters
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT) 
        
        # Train the index on a subset
        np.random.seed(42)
        train_sample = user_latent[np.random.choice(user_latent.shape[0], size=100000, replace=False)]
        index.train(train_sample)

        # Add all users to the index
        index.add(user_latent)

        # How many clusters to search? (tradeoff: higher = more accurate, slower)
        index.nprobe = 20

        # 6. Query all users at once
        D, I = index.search(user_latent, self.top_k_neighbors + 1)  # +1 = includes self

        # Store neighbors
        self.user_sim = {}
        for i, user_id in enumerate(self.user_ids):
            neighbors = [(self.user_ids[j], float(d)) 
                         for j, d in zip(I[i], D[i]) if j != i]
            self.user_sim[user_id] = dict(neighbors[:self.top_k_neighbors])

    def recommend_for_user(self, user_id, k=10):
        """Return top-k recommended tracks for a given user based on co-listening"""
        if user_id not in self.user_map:
            return pd.DataFrame(columns=['rank', 'artist_name', 'track_title'])

        # 1. Get neighbors and similarities
        neighbors = self.user_sim.get(user_id, {})
        if not neighbors:
            return pd.DataFrame(columns=['rank', 'artist_name', 'track_title'])

        neighbor_ids, sim_scores = zip(*neighbors.items())
        neighbor_indices = [self.user_map[u] for u in neighbor_ids]

        # 2. Get neighbor interaction matrix
        neighbor_matrix = self.user_item_sparse[neighbor_indices]

        # 3. Weighted sum of neighbor play counts
        sim_scores = np.array(sim_scores).reshape(-1, 1)
        weighted_ratings = neighbor_matrix.T.dot(sim_scores).flatten()

        # 4. Remove already listened items
        user_index = self.user_map[user_id]
        listened = self.user_item_sparse[user_index].toarray().flatten() > 0
        weighted_ratings[listened] = 0

        # 5. Get top-k recommendations
        top_idx = np.argpartition(-weighted_ratings, k)[:k]
        top_scores = weighted_ratings[top_idx]

        recommended_song_ids = [self.song_ids[i] for i in top_idx]
        recs = pd.DataFrame({
            'song_id': recommended_song_ids,
            'score': top_scores
        }).sort_values(by="score", ascending=False).reset_index(drop=True)

        # Optional: merge with track metadata for readability
        recs = recs.merge(self.tracks_df, on="song_id", how="left")
        recs.insert(0, "rank", range(1, len(recs) + 1))
        return recs.head(k)   

    def precision_recall_at_k(self, k=10, users=None):
        """
        Compute average Precision@k and Recall@k across users in the test set.

        Args:
            k (int): number of recommendations to evaluate
            users (list, optional): specific user_ids to evaluate.
                                    If None, evaluates all test users.

        Returns:
            dict: {'precision_at_k': float, 'recall_at_k': float}
        """
        if users is None:
            users = self.test_df['user_id'].unique()

        precisions = []
        recalls = []

        for user_id in users:
            # 1. Ground truth items from test set
            actual_tracks = set(self.test_df[self.test_df['user_id'] == user_id]['song_id'])
            if not actual_tracks:
                continue

            # 2. Recommended items
            recs = self.recommend_for_user(user_id, k)
            if recs.empty:
                continue
            rec_tracks = set(recs['song_id'])

            # 3. Hits and metrics
            hits = len(actual_tracks & rec_tracks)
            precisions.append(hits / k)
            recalls.append(hits / len(actual_tracks))

        return {
            'precision_at_k': float(np.mean(precisions)) if precisions else 0.0,
            'recall_at_k': float(np.mean(recalls)) if recalls else 0.0
        }
class ItemBasedRecommender(Collaborative):
    def __init__(self, interactions_df, tracks_df, top_k_neighbors=50):
        super().__init__(interactions_df, tracks_df)
        self.top_k_neighbors = top_k_neighbors
        self.song_map = None
        self.user_map = None
        self.user_item_sparse = None
        self.song_ids = None
        self.item_sim = None  # stores top-K similar items
    
    def fit(self):
        # 1. Map user_id and song_id to indices
        self.user_ids = self.train_df['user_id'].unique()
        self.song_ids = self.train_df['song_id'].unique()
        self.user_map = {user: idx for idx, user in enumerate(self.user_ids)}
        self.song_map = {song: idx for idx, song in enumerate(self.song_ids)}

        # 2. Build sparse user-item matrix
        rows = self.train_df['user_id'].map(self.user_map).to_numpy()
        cols = self.train_df['song_id'].map(self.song_map).to_numpy()
        data = self.train_df['play_count'].to_numpy()
        self.user_item_sparse = csr_matrix(
            (data, (rows, cols)), shape=(len(self.user_ids), len(self.song_ids))
        )

        # 3. Compute item-item similarities (cosine similarity)
        # Transpose: items x users
        from sklearn.preprocessing import normalize
        item_matrix = self.user_item_sparse.T  # shape: (num_songs, num_users)
        item_matrix = normalize(item_matrix.astype(np.float32))
        # Cosine similarity
        self.item_sim = cosine_similarity(item_matrix, dense_output=False)  # sparse matrix
        # Optional: zero out self-similarity diagonal
        self.item_sim.setdiag(0)

    def recommend_for_track(self, track_id, k=10):
        """Return top-k recommended tracks similar to a given track"""
        if track_id not in self.song_map:
            return pd.DataFrame(columns=['rank', 'artist_name', 'track_title'])

        track_index = self.song_map[track_id]
        sim_scores = self.item_sim[track_index].toarray().flatten()  # similarity with all songs

        # 1. Get top-k similar tracks
        top_idx = np.argpartition(-sim_scores, k)[:k]
        top_scores = sim_scores[top_idx]
        recommended_song_ids = [self.song_ids[i] for i in top_idx]

        recs = pd.DataFrame({
            'song_id': recommended_song_ids,
            'score': top_scores
        }).sort_values(by="score", ascending=False).reset_index(drop=True)

        # 2. Merge with track metadata
        recs = recs.merge(self.tracks_df, on="song_id", how="left")
        recs.insert(0, "rank", range(1, len(recs) + 1))
        return recs.head(k)

    def precision_recall_at_k(self, k=10, tracks=None):
        """
        Compute average Precision@k and Recall@k across tracks in the test set.

        Args:
            k (int): number of recommendations to evaluate.
            tracks (list, optional): specific track_ids to evaluate. 
                                     If None, evaluates all tracks in test set.

        Returns:
            dict: {'precision_at_k': float, 'recall_at_k': float}
        """
        if tracks is None:
            tracks = self.test_df['song_id'].unique()

        precisions = []
        recalls = []

        for track_id in tracks:
            # 1. Ground truth: users who listened to this track in test set
            actual_users = set(self.test_df[self.test_df['song_id'] == track_id]['user_id'])
            if not actual_users:
                continue

            # 2. Recommended tracks
            recs = self.recommend_for_track(track_id, k)
            if recs.empty:
                continue
            rec_tracks = set(recs['song_id'])

            # 3. Compute hits
            # A hit: a recommended track that was listened to by the same users as the original track
            hits = 0
            for rec_track in rec_tracks:
                users_who_listened = set(self.test_df[self.test_df['song_id'] == rec_track]['user_id'])
                hits += len(actual_users & users_who_listened)

            # Precision@k: fraction of recommended tracks relevant to users
            precisions.append(hits / k)
            # Recall@k: fraction of actual users covered
            recalls.append(hits / len(actual_users))

        return {
            'precision_at_k': float(np.mean(precisions)) if precisions else 0.0,
            'recall_at_k': float(np.mean(recalls)) if recalls else 0.0
        }
