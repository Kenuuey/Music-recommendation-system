import numpy as np

def precision_at_k(actual, predicted, k):
    actual, predicted = set(actual), predicted[:k]
    return len(set(predicted) & actual) / k if k else 0.0

def recall_at_k(actual, predicted, k):
    actual, predicted = set(actual), predicted[:k]
    return len(set(predicted) & actual) / len(actual) if actual else 0.0

def mean_precision_recall_at_k(model, test_df, k=10):
    precisions, recalls = [], []
    for user_id in test_df['user_id'].unique():
        actual = set(test_df[test_df['user_id'] == user_id]['song_id'])
        recs = model.recommend_for_user(user_id, k)
        if recs.empty: 
            continue
        predicted = recs['song_id'].tolist()
        precisions.append(precision_at_k(actual, predicted, k))
        recalls.append(recall_at_k(actual, predicted, k))
    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0
    }
