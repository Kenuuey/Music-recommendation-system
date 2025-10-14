# src/recommenders/base.py
from abc import ABC, abstractmethod
import pandas as pd

class MusicRecommender(ABC):
    """
    Base abstract class for all recommenders.
    All recommenders should inherit from this.
    """

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def recommend_for_user(self, user_id, k=10):
        raise NotImplementedError

    @abstractmethod
    def precision_recall_at_k(self, k=10, users=None):
        raise NotImplementedError