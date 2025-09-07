from src.recommenders.base import MusicRecommender

class UserBasedRecommender(MusicRecommender):
    def __init__(self, interactions_df, tracks_df):
        self.interactions_df = interactions_df
        self.tracks_df = tracks_df

    def train_test_split(self, test_size=0.2, random_state=42):
        self.train_df, self.test_df = train_test_split(
            self.interactions_df, test_size=test_size, random_state=random_state
        )
    
    def fit(self):
        """Create user-item matrix from train set"""
        self.user_item_matrix = self.train_df.pivot_table(
            index='user_id', columns='song_id', values='play_count', fill_value=0
        )
    
    def recommend_for_user(self, user_id, k=10):
        """Return top-k recommended tracks for a given user based on co-listening"""
        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame(columns=['rank', 'artist_name', 'track_title'])

        user_vector = self.user_item_matrix.loc[user_id]
        # Compute similarity with other users (cosine or simple co-listen count)
        sim_scores = self.user_item_matrix.dot(user_vector)
        sim_scores = sim_scores.drop(user_id, errors='ignore')

        # Aggregate scores to tracks
        track_scores = self.user_item_matrix.T.dot(sim_scores)
        track_scores = track_scores.sort_values(ascending=False).head(k)

        recommended = pd.DataFrame(track_scores).reset_index()
        recommended.columns = ['song_id', 'score']
        recommended = recommended.merge(self.tracks_df, on='song_id', how='left')
        recommended = recommended[['artist_name', 'track_title', 'score']]
        recommended.index = recommended.index + 1
        recommended.index.name = 'rank'
        return recommended

    def precision_at_k(self, k=10):
        """Average p@k across all users in test set"""
        precisions = []
        for user_id in self.test_df['user_id'].unique():
            actual_tracks = self.test_df[self.test_df['user_id'] == user_id]['song_id'].tolist()
            rec_df = self.recommend_for_user(user_id, k)
            rec_tracks = rec_df.merge(self.tracks_df[['song_id']], left_on='track_title', right_on='track_title', how='left')['song_id'].tolist()
            n_hits = len(set(actual_tracks) & set(rec_tracks))
            precisions.append(n_hits / k)
        return sum(precisions) / len(precisions)

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
