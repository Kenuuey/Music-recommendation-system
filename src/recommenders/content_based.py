from src.recommenders.base import MusicRecommender

class ContentBasedRecommender(MusicRecommender):
    def fit(self, lyrics_sparse, track_meta):
        # compute TF-IDF or normalized counts; optionally train word2vec embeddings
        pass
    def recommend_by_keyword_baseline(self, keyword, k=50, threshold=1):
        # filter tracks with keyword counts >= threshold, sort by play_count
        pass
    def recommend_by_word2vec(self, keyword, k=50, n_similar=10):
        # expand keyword with similar tokens & score by combined counts
        pass
    def recommend_by_classifier(self, keyword, k=50,): #clf):
        # train classifier on labeled songs and predict probabilities on unlabeled
        pass
