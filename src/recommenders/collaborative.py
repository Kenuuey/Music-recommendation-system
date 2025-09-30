from src.recommenders.base import MusicRecommender


import pandas as pd
from sklearn.model_selection import train_test_split



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

    

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import faiss

class UserBasedRecommender(Collaborative):
    def __init__(self, interactions_df, tracks_df, top_k_neighbors=50):
        super().__init__(interactions_df, tracks_df)
        self.top_k_neighbors = top_k_neighbors
        self.user_map = None
        self.song_map = None
        self.user_item_sparse = None
        self.user_ids = None
        self.user_sim = None  # stores top-K neighbors
    
    def fit(self, top_k_neighbors=50):
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

        """
        30.09 13:25
        
        NEED TO COMPUTE SIMILARITY BETWEEN USERS
        FOR USERS IN TRAIN SET
        TO GET TOP-K NEIGHBORS AND RECOMMEND SONGS
        FOR USERS IN TEST SET 
        """


        svd = TruncatedSVD(n_components=128)
    
        from sklearn.preprocessing import normalize
        user_latent = svd.fit_transform(self.user_item_sparse)  # shape: (num_users, 128)
        user_latent = normalize(user_latent.astype(np.float32))  # normalize for cosine similarity

        # Build FAISS index
        dim = user_latent.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product = cosine for normalized vectors
        index.add(user_latent)

        # Query all users at once
        D, I = index.search(user_latent, 51)  # 50 neighbors + self

        # Build dictionary of top-K neighbors
        user_sim = {}
        for i, user_id in enumerate(ub_rec.user_ids):
            neighbors = [(ub_rec.user_ids[j], float(d)) for j, d in zip(I[i], D[i]) if j != i]
            user_sim[user_id] = dict(neighbors[:50])



        # # Compute top-K neighbors efficiently
        # self.user_sim = {}
        # for i in tqdm(range(user_latent.shape[0])):
        #     sims = user_latent @ user_latent[i].T  # vectorized dot product = cosine
        #     sims[i] = 0  # ignore self
        #     top_idx = np.argpartition(sims, -self.top_k_neighbors)[-self.top_k_neighbors:]
        #     top_users = self.user_ids[top_idx]
        #     top_sims = sims[top_idx]
        #     self.user_sim[self.user_ids[i]] = dict(zip(top_users, top_sims))

        
        # # 6. Query top-K neighbors
        # D, I = index.search(user_dense, self.top_k_neighbors + 1)  # +1 to skip self

        # # 7. Store neighbors
        # self.user_sim = {}
        # for i, user_id in enumerate(self.user_ids):
        #     neighbors = [(self.user_ids[idx], float(score)) for idx, score in zip(I[i], D[i]) if idx != i]
        #     self.user_sim[user_id] = dict(neighbors[:self.top_k_neighbors])


        # # 3. Compute top-K user similarities efficiently
        # self.user_sim = {}
        # print("Computing top-K user similarities...")
        # for i in tqdm(range(self.user_item_sparse.shape[0])):
        #     user_vector = self.user_item_sparse[i]
        #     sim = cosine_similarity(user_vector, self.user_item_sparse).flatten()
        #     sim[i] = 0  # ignore self-similarity
            
        #     # Get top-K indices
        #     if self.top_k_neighbors < len(sim):
        #         top_idx = np.argpartition(sim, -self.top_k_neighbors)[-self.top_k_neighbors:]
        #     else:
        #         top_idx = np.argsort(sim)[::-1]
            
        #     # Store as dictionary: {neighbor_user_id: similarity}
        #     top_user_ids = [self.user_ids[j] for j in top_idx]
        #     top_sims = sim[top_idx]
        #     self.user_sim[self.user_ids[i]] = dict(zip(top_user_ids, top_sims))


    def recommend_for_user(self, user_id, k=10):
        """Return top-k recommended tracks for a given user based on co-listening"""
        if user_id not in self.user_ids:
            return pd.DataFrame(columns=['rank', 'artist_name', 'track_title'])

        # TODO: 30.09 10:46
        neighbors = user_sim[user_id]  # top-K neighbors
        neighbor_ids, sim_scores = zip(*neighbors)
        neighbor_indices = [user_map[u] for u in neighbor_ids]
        
        # Get neighbor ratings
        neighbor_matrix = user_item_sparse[neighbor_indices]
        
        # Weighted sum
        sim_scores = np.array(sim_scores).reshape(-1,1)
        weighted_ratings = neighbor_matrix.T.dot(sim_scores).flatten()
        
        # Remove already listened items
        listened = user_item_sparse[user_map[user_id]].toarray().flatten() > 0
        weighted_ratings[listened] = 0
        
        top_idx = np.argpartition(-weighted_ratings, k)[:k]
        recommended_song_ids = [song_ids[i] for i in top_idx]
        return recommended_song_ids
        
        
        # # 1. Aggregate weighted ratings from top-K similar users
        # neighbors = self.user_sim[user_id]
        # weighted_ratings = np.zeros(len(self.item_ids))
        # for neighbor_id, sim_score in neighbors.items():
        #     neighbor_vector = self.user_item_matrix.loc[neighbor_id].values
        #     weighted_ratings += sim_score * neighbor_vector

        # weighted_ratings = pd.Series(weighted_ratings, index=self.item_ids)

        # # 2. Filter out already listened items
        # listened = self.user_item_matrix.loc[user_id]
        # weighted_ratings = weighted_ratings[listened == 0].sort_values(ascending=False).head(k)

        # # 3. Format recommendations
        # recommended = weighted_ratings.reset_index()
        # recommended.columns = ['song_id', 'score']
        # recommended = recommended.merge(self.tracks_df, on='song_id', how='left')
        # recommended = recommended[['artist_name', 'track_title', 'score']]
        # recommended.index = recommended.index + 1
        # recommended.index.name = 'rank'
        # return recommended

    def precision_at_k(self, k=10):
        """Average p@k across all users in test set"""
        precisions = []
        for user_id in self.test_df['user_id'].unique():
            actual_tracks = set(self.test_df[self.test_df['user_id'] == user_id]['song_id'])
            rec_tracks = set(self.recommend_for_user(user_id, k)['song_id'])
            if not rec_tracks:
                continue
            precisions.append(len(actual_tracks & rec_tracks) / k)
        return np.mean(precisions)

class ItemBasedRecommender(MusicRecommender):
    def __init__(self, interactions_df, tracks_df):
        self.interactions_df = interactions_df
        self.tracks_df = tracks_df

    def fit(self):
        """Create track-user matrix for item-based similarity"""
        self.track_user_matrix = self.interactions_df.pivot_table(
            index='song_id', columns='user_id', values='play_count', fill_value=0
        )

    def recommend_for_track(self, track_id, k=10):
        """Return top-k similar tracks for a given track"""
        if track_id not in self.track_user_matrix.index:
            return pd.DataFrame(columns=['rank', 'artist_name', 'track_title'])

        track_vector = self.track_user_matrix.loc[track_id]
        sim_scores = self.track_user_matrix.dot(track_vector)
        sim_scores = sim_scores.drop(track_id, errors='ignore').sort_values(ascending=False).head(k)

        recommended = pd.DataFrame(sim_scores).reset_index()
        recommended.columns = ['song_id', 'score']
        recommended = recommended.merge(self.tracks_df, on='song_id', how='left')
        recommended = recommended[['artist_name', 'track_title', 'score']]
        recommended.index = recommended.index + 1
        recommended.index.name = 'rank'
        return recommended
