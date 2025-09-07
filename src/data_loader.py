class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.interactions_df = None
        self.lyrics_df = None
        self.genres_df = None
        self.tracks_df = None

    def load_data(self):
        self.load_interactions()
        self.load_lyrics()
        self.load_genres()
        self.load_tracks()

    def load_interactions(self):
        self.interactions_df = pd.read_csv(
            os.path.join(self.data_path, "interactions.csv"),
            sep='\t',
            header=None,
            names=["user_id", "song_id", "play_count"]
        )
        return self.interactions_df
    
    def load_lyrics(self):
        with open(os.path.join(self.data_path, "mxm_dataset_train.txt")) as f:
            lines = f.readlines()

        vocab_line = [l for l in lines if l.startswith('%')][0]
        vocab = vocab_line[1:].strip().split(',')

        records = []
        for line in lines:
            if line.startswith('#') or line.startswith('%'):
                continue
            parts = line.strip().split(',')
            track_id, mxm_track_id, *word_counts = parts
            bow = {}
            for wc in word_counts:
                idx, count = wc.split(':')
                bow[vocab[int(idx)-1]] = int(count)
            records.append((track_id, mxm_track_id, bow))

        self.lyrics_df = pd.DataFrame(records, columns=['track_id', 'mxm_track_id', 'bow'])
        return self.lyrics_df
    
    def load_genres(self):
        self.genres_df = pd.read_csv(
            os.path.join(self.data_path, "p02_msd_tagtraum_cd2.cls"),
            sep="\t", comment="#", header=None,
            names=["track_id", "majority_genre", "minority_genre"]
        )
        return self.genres_df

    def load_tracks(self):
        self.tracks_df = pd.read_csv(
            os.path.join(self.data_path, "p02_unique_tracks.txt"),
            sep="<SEP>", header=None, engine="python",
            names=["track_id", "song_id", "artist_name", "track_title"]
        )
        return self.tracks_df

