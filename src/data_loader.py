# src/data_loading.py
import pandas as pd
from pathlib import Path

import os
import pandas as pd


class SampleDataLoader:
    def __init__(self, data_path, frac=0.01, random_state=42):
        self.data_path = data_path
        self.interactions_df = None
        self.tracks_df = None
        self.genres_df = None
        self.lyrics_df = None
        self.frac = frac
        self.random_state = random_state

    def load_data(self):
        self.load_interactions()
        self.load_tracks()
        self.load_genres()
        self.load_lyrics()

    def load_interactions(self):
        interactions_df = pd.read_csv(
            os.path.join(self.data_path, "interactions_sample.csv"),
        )
        self.interactions_df = interactions_df.sample(frac=self.frac, random_state=self.random_state)
        return self.interactions_df