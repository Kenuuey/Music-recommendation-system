# main.py
import pandas as pd
from gensim.models import Word2Vec
from src.non_personalized import NonPersonalizedRecommender
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.collaborative import UserBasedRecommender
from src.collaborative import ItemBasedRecommender


# ------------------ Load data samples ------------------
def load_data():
    """Load small sample datasets (CSV)."""
    interactions = pd.read_csv("data_samples/interactions_sample.csv")
    tracks = pd.read_csv("data_samples/tracks_sample.csv")
    genres = pd.read_csv("data_samples/genres_sample.csv")
    lyrics = pd.read_csv("data_samples/lyrics_sample.csv")
    return interactions, tracks, genres, lyrics


# ------------------ Main interactive menu ------------------
def main():
    print("\n🎵 Welcome to the Music Recommender System 🎵")
    print("Select a recommender type:")
    print("1. Non-personalized (Top tracks)")
    print("2. Content-based (Keyword search)")
    print("3. Collaborative filtering (User/Track based)")

    choice = input("Enter choice [1-3]: ").strip()
    interactions, tracks, genres, lyrics = load_data()


    if choice == "1":
        recommender = NonPersonalizedRecommender(interactions, tracks, genres)
        print("\n1. Top 250 tracks\n2. Top 100 by genre")
        sub_choice = input("Enter choice [1-2]: ").strip()

        if sub_choice == "1":
            result = recommender.top_k_global(250)
        else:
            genre = input(
                "Enter genre (Rock, Rap, Latin, Jazz, Electronic, Punk, Pop, "
                "New Age, Metal, RnB, Country, Reggae, Folk, Blues, World): "
            )
            result = recommender.top_k_by_genre(genre, 100)

        print("\n🎧 Recommended tracks:")
        print(result)

    elif choice == "2":
        recommender = ContentBasedRecommender(interactions, tracks, lyrics, genres)
        print("\nSelect method:")
        print("1. Baseline keyword search")
        print("2. Word2Vec expansion search")
        print("3. Classifier-based ranking")
        method_choice = input("Enter choice [1-3]: ")

        keyword = input("Enter a keyword (or genre for classifier): ")

        if method_choice == "1":
            result = recommender.baseline(keyword, k=50)

        elif method_choice == "2":
            w2v_model = Word2Vec.load("models/w2v_model.model")
            result = recommender.word2vec(keyword, model=w2v_model, k=50)
        
        elif method_choice == "3":
            label_by_genre_input = input("Label by genre? (y/n): ").lower()
            label_by_genre = label_by_genre_input == "y"
            
            result = recommender.classifier(keyword, k=50, label_by_genre=label_by_genre)
            
            if result.empty:
                print(f"No tracks found with classifier for '{keyword}'. Falling back to baseline search.")
                result = recommender.baseline(keyword, k=50)

        print("\n🎧 Recommended tracks:")
        print(result)

    elif choice == "3":
        recommender = CollaborativeRecommender(interactions, tracks)
        print("\n1. Recommend for a user (User-based)")
        print("2. Recommend similar tracks (Item-based)")
        sub_choice = input("Enter choice [1-2]: ").strip()

        if sub_choice == "1":
            recommender = UserBasedRecommender(interactions, tracks)
            recommender.train_test_split()
            recommender.fit()
            user_id = input("Enter user_id: ")
            result = recommender.recommend_for_user(user_id, k=10)
        else:
            recommender = ItemBasedRecommender(interactions, tracks)
            recommender.train_test_split()
            recommender.fit()
            track_id = input("Enter track_id: ")
            result = recommender.recommend_for_track(track_id, k=10)

        print("\n🎧 Recommended tracks:")
        print(result.head(10))

    else:
        print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()