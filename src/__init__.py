from data_loader import DataLoader
from recommenders.non_personalized import NonPersonalizedRecommender

# Load data
loader = DataLoader("./drive/MyDrive/S21/data/My_Spotify/")
interactions = loader.load_interactions()
lyrics = loader.load_lyrics()
genres = loader.load_genres()
tracks = loader.load_tracks()

# Initialize recommender
recommender = NonPersonalizedRecommender(interactions, tracks, genres)

# Top 250 global
top_250 = recommender.top_k_global(250)
print(top_250.head())

# Top 100 Rock
top_rock = recommender.top_k_by_genre("Rock", 100)
print(top_rock.head())
