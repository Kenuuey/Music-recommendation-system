import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from sklearn.preprocessing import normalize

class CollaborativeRecommender():
    """Base class for collaborative recommenders."""
    def __init__(self, interactions_df, tracks_df):
        self.interactions_df = interactions_df
        self.tracks_df = tracks_df
        self.train_df = None
        self.test_df = None

    def train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        """Split interactions into train and test sets."""
        self.train_df, self.test_df = train_test_split(
            self.interactions_df, test_size=test_size, random_state=random_state
        )

    def precision_recall_at_k(self, k: int = 10, mode: str = 'user', entities=None, sample_size: int = 1000) -> dict:
        """
        Fast computation of average Precision@k and Recall@k.

        Parameters:
            k (int): Number of recommendations to consider.
            mode (str): 'user' for user-based, 'track' for track-based evaluation.
            entities (list or None): Specific users or tracks to evaluate. If None, sample from test set.
            sample_size (int): Max number of users/tracks to evaluate (random sampling for speed).

        Returns:
            dict: {'precision_at_k': float, 'recall_at_k': float}
        """
        precisions, recalls = [], []

        if mode == 'user':
            if entities is None:
                entities = self.test_df['user_id'].unique()
            if len(entities) > sample_size:
                entities = np.random.choice(entities, sample_size, replace=False)

            # Precompute user-item interactions in test set
            test_user_items = self.test_df.groupby('user_id')['song_id'].apply(set).to_dict()

            for user_id in entities:
                actual_tracks = test_user_items.get(user_id, set())
                if not actual_tracks:
                    continue
                recs = self.recommend_for_user(user_id, k)
                if recs.empty:
                    continue
                rec_tracks = set(recs['song_id'])
                hits = len(actual_tracks & rec_tracks)
                precisions.append(hits / k)
                recalls.append(hits / len(actual_tracks))

        elif mode == 'track':
            if entities is None:
                entities = self.test_df['song_id'].unique()
            if len(entities) > sample_size:
                entities = np.random.choice(entities, sample_size, replace=False)

            # Precompute track-user interactions in test set
            test_track_users = self.test_df.groupby('song_id')['user_id'].apply(set).to_dict()

            for track_id in entities:
                actual_users = test_track_users.get(track_id, set())
                if not actual_users:
                    continue
                recs = self.recommend_for_track(track_id, k)
                if recs.empty:
                    continue
                rec_tracks = set(recs['song_id'])
                hits = sum(len(actual_users & test_track_users.get(rec, set())) for rec in rec_tracks)
                precisions.append(hits / k)
                recalls.append(hits / len(actual_users))

        else:
            raise ValueError("mode must be 'user' or 'track'")

        return {
            'precision_at_k': float(np.mean(precisions)) if precisions else 0.0,
            'recall_at_k': float(np.mean(recalls)) if recalls else 0.0
        }
class UserBasedRecommender(CollaborativeRecommender):
    """User-based collaborative filtering using FAISS for approximate nearest neighbors."""
    def __init__(self, interactions_df, tracks_df, top_k_neighbors=50):
        super().__init__(interactions_df, tracks_df)
        self.top_k_neighbors = top_k_neighbors
        self.user_map = None
        self.song_map = None
        self.user_item_sparse = None
        self.user_ids = None
        self.user_sim = None  # stores top-K neighbors
    
    def fit(self):
        # Map user_id and song_id to indices
        self.user_ids = self.train_df['user_id'].unique()
        self.song_ids = self.train_df['song_id'].unique()
        self.user_map = {user: idx for idx, user in enumerate(self.user_ids)}
        self.song_map = {song: idx for idx, song in enumerate(self.song_ids)}
        
        # Build sparse user-item matrix
        rows = self.train_df['user_id'].map(self.user_map).to_numpy()
        cols = self.train_df['song_id'].map(self.song_map).to_numpy()
        data = self.train_df['play_count'].to_numpy()
        self.user_item_sparse = csr_matrix(
            (data, (rows, cols)), shape=(len(self.user_ids), len(self.song_ids))
        )

        # Compute latent user vectors via SVD
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize

        n_components = min(64, self.user_item_sparse.shape[1])
        svd = TruncatedSVD(n_components=128)
        user_latent = svd.fit_transform(self.user_item_sparse)  # shape: (num_users, 128)
        user_latent = normalize(user_latent.astype(np.float32))

        # Choose FAISS index type based on dataset size
        dim = user_latent.shape[1]
        num_users = len(self.user_ids)

        if num_users < 5000:
            # Small dataset → exact search, no clustering needed
            index = faiss.IndexFlatIP(dim)
            index.add(user_latent)
        else:
            # Large dataset → IVF index
            nlist = min(4096, max(1, num_users // 10))  # adapt clusters to size
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

            # Train index on subset (at most all users)
            train_sample_size = min(100000, num_users)
            np.random.seed(42)
            train_sample = user_latent[np.random.choice(num_users, train_sample_size, replace=False)]
            index.train(train_sample)
            index.add(user_latent)
            index.nprobe = min(20, nlist)

        # Query all users at once
        D, I = index.search(user_latent, self.top_k_neighbors + 1)  # +1 = includes self

        # Store top-K neighbors
        self.user_sim = {}
        for i, user_id in enumerate(self.user_ids):
            neighbors = [(self.user_ids[j], float(d)) 
                         for j, d in zip(I[i], D[i]) if j != i]
            self.user_sim[user_id] = dict(neighbors[:self.top_k_neighbors])

    def recommend_for_user(self, user_id, k=10):
        """Return top-k recommended tracks for a given user."""
        if user_id not in self.user_map:
            return pd.DataFrame(columns=['rank', 'artist_name', 'track_title'])

        neighbors = self.user_sim.get(user_id, {})
        if not neighbors:
            return pd.DataFrame(columns=['rank', 'artist_name', 'track_title'])

        neighbor_ids, sim_scores = zip(*neighbors.items())
        neighbor_indices = [self.user_map[u] for u in neighbor_ids]

        neighbor_matrix = self.user_item_sparse[neighbor_indices]
        sim_scores = np.array(sim_scores).reshape(-1, 1)
        weighted_ratings = neighbor_matrix.T.dot(sim_scores).flatten()

        # Remove already listened items
        user_index = self.user_map[user_id]
        listened = self.user_item_sparse[user_index].toarray().flatten() > 0
        weighted_ratings[listened] = 0

        # Top-k recommendations
        top_idx = np.argpartition(-weighted_ratings, k)[:k]
        recommended_song_ids = [self.song_ids[i] for i in top_idx]
        top_scores = weighted_ratings[top_idx]

        recs = pd.DataFrame({'song_id': recommended_song_ids, 'score': top_scores})
        recs = recs.merge(self.tracks_df, on="song_id", how="left")
        recs.insert(0, "rank", range(1, len(recs) + 1))
        return recs.sort_values(by="score", ascending=False).head(k)
class ItemBasedRecommender(CollaborativeRecommender):
    """Item-based collaborative filtering using cosine similarity."""
    def __init__(self, interactions_df, tracks_df, top_k_neighbors: int = 50):
        super().__init__(interactions_df, tracks_df)
        self.top_k_neighbors = top_k_neighbors
        self.song_map = None
        self.user_map = None
        self.user_item_sparse = None
        self.song_ids = None
        self.item_sim = None  # stores top-K similar items
    
    def fit(self):
        # Map user_id and song_id to indices
        self.user_ids = self.train_df['user_id'].unique()
        self.song_ids = self.train_df['song_id'].unique()
        self.user_map = {user: idx for idx, user in enumerate(self.user_ids)}
        self.song_map = {song: idx for idx, song in enumerate(self.song_ids)}

        # Build sparse user-item matrix
        rows = self.train_df['user_id'].map(self.user_map).to_numpy()
        cols = self.train_df['song_id'].map(self.song_map).to_numpy()
        data = self.train_df['play_count'].to_numpy()
        self.user_item_sparse = csr_matrix((data, (rows, cols)), shape=(len(self.user_ids), len(self.song_ids)))

        # Compute item-item similarities
        item_matrix = normalize(self.user_item_sparse.T.astype(np.float32))
        self.item_sim = cosine_similarity(item_matrix, dense_output=False)
        self.item_sim.setdiag(0)

    def recommend_for_track(self, track_id: int, k: int = 10) -> pd.DataFrame:
        """Return top-k recommended tracks similar to a given track."""
        if track_id not in self.song_map:
            return pd.DataFrame(columns=['rank', 'artist_name', 'track_title'])

        track_index = self.song_map[track_id]
        sim_scores = self.item_sim[track_index].toarray().flatten()

        top_idx = np.argpartition(-sim_scores, k)[:k]
        recommended_song_ids = [self.song_ids[i] for i in top_idx]
        top_scores = sim_scores[top_idx]

        recs = pd.DataFrame({'song_id': recommended_song_ids, 'score': top_scores})
        recs = recs.merge(self.tracks_df, on="song_id", how="left")
        recs.insert(0, "rank", range(1, len(recs) + 1))
        return recs.sort_values(by="score", ascending=False).head(k)