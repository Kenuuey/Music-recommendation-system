from src.recommenders.base import MusicRecommender

class UserBasedCF(MusicRecommender):
    def fit(self, interactions_sparse):
        # compute user-user similarities (cosine/Jaccard) or use matrix factorization
        pass
    def recommend(self, user_id, k=10):
        # produce top-k with predicted scores   
        pass

class ItemBasedCF(MusicRecommender):
    def fit(self, interactions_sparse):
        # item-item similarity matrix
        pass
    def recommend(self, track_id, k=10):
        # item->items recommendations
        pass
