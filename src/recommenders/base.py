import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MinMaxScaler

class MusicRecommender:
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def recommend(self, *args, **kwargs):
        raise NotImplementedError