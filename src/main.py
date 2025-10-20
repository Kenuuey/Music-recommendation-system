# main.py
import pandas as pd
from non_personalized import NonPersonalizedRecommender
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender


# ------------------ Load data samples ------------------
def load_data():
    """Load small sample datasets (CSV)."""
    interactions = pd.read_csv("../data/samples/interactions_sample.csv")
    tracks = pd.read_csv("../data/samples/tracks_sample.csv")
    genres = pd.read_csv("../data/samples/genres_sample.csv")
    lyrics = pd.read_csv("../data/samples/lyrics_sample.csv")
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
        recommender = ContentBasedRecommender(interactions, tracks, lyrics)
        keyword = input("Enter a keyword (e.g., love, war, happiness): ")
        result = recommender.baseline(keyword, k=10)

        print("\n🎧 Recommended tracks:")
        print(result.head(10))

    elif choice == "3":
        recommender = CollaborativeRecommender(interactions, tracks)
        print("\n1. Recommend for a user\n2. Recommend similar tracks")
        sub_choice = input("Enter choice [1-2]: ").strip()

        if sub_choice == "1":
            user_id = input("Enter user_id: ")
            result = recommender.recommend_for_user(user_id, k=10)
        else:
            track_id = input("Enter track_id: ")
            result = recommender.recommend_for_track(track_id, k=10)

        print("\n🎧 Recommended tracks:")
        print(result.head(10))

    else:
        print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()