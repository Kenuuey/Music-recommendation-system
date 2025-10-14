# Recommender systems. Music recommendations

This project introduces algorithms used for recommendation: non-personalized, content-based, and collaborative filtering

### Introduction

Recommendation systems simplify decision-making by guiding users toward relevant content or products, just as people historically relied on friends, family, or popularity indicators like bestsellers. They benefit users by reducing cognitive effort and offering trusted suggestions, while also benefiting businesses through increased engagement, cross-selling, and revenue growth. This mutual value makes recommender systems essential to modern digital products.

**Types of Recommendation systems:**

1. **Non-Personalized** recommenders show the **same suggestions to everyone**, regardless of individual preferences or history. Instead of tailoring recommendations to a specific person, they rely on **aggregate data** such as popularity or trends.
2. **Content-based** recommenders suggest items that are **similar to items the user already liked**, based on item features (descriptions, attributes, metadata).
3. **Collaborative filtering** recommends items by analyzing **user behavior** (ratings, clicks, purchases). The idea is that users with similar preferences in the past will like similar things in the future.
4. **Matrix factorization** is an advanced collaborative filtering technique. It reduces the large user-item interaction matrix into two smaller matrices that capture **latent factors** (hidden features). These factors represent underlying dimensions, such as genres, themes, or user preferences.


The goal is to develop a music recommendation system using various approaches:

1. Top 250 Tracks (Non-Personalized Approach):
- Recommend the 250 most popular tracks globally based on play counts.
- Example: "Top 250 most streamed tracks worldwide."

2. Top 100 Tracks by Genre (Non-Personalized Approach):
- Identify the 100 most popular tracks in each specified genre: Rock, Rap, Jazz, Electronic, Pop, Blues, Country, Reggae, New Age.
- Example: "Top 100 Pop tracks."

3. Collections (Content-Based Approach):
- Create thematic playlists based on lyrical content using the musiXmatch dataset:
- 50 songs about love, 50 songs about war, 50 songs about happiness, 50 songs about loneliness, 50 songs about money
- Method: Analyze lyrics to identify themes (e.g., frequent use of “love” words assigns a song to the Love collection).

4. People Similar to You Listening (Collaborative Filtering – User-Based):
- Recommend 10 songs per user based on the listening habits of similar users.
- Example: If User A and User B share similar music tastes and User A listens to a new song, recommend that song to User B.

5. People Who Listen to This Track Also Listen to (Collaborative Filtering – Item-Based):
- Recommend 10 songs for each track based on co-listening patterns.
- Example: If many users who listen to Track X also listen to Track Y, recommend Track Y to others who listen to Track X.


### **Dataset**

1. The Echo Nest Taste Profile Subset (User-Song Interactions):
- Format: `(user_id, song_id, play_count)`
- Use: for popularity counts and collaborative filtering.

2. The musiXmatch Dataset (Lyrics Bag-of-Words):
- Format: `(track_id, mxm_track_id, word counts)`
- Use: for content-based filtering (lyrics similarity, collections).

3. Tagtraum Genre Annotations:
- Format: `(track_id, majority_genre, minority_genre)`
- Use: for Top tracks by genre.

4. Mapping between track_id and song_id:
- Format: `(track_id, song_id, artist, title)`
- Use: links datasets together → helps connect user plays, lyrics, and genres.

**References**

- [Million Song Dataset](https://labrosa.ee.columbia.edu/millionsong/)
- [musiXmatch Dataset](https://www.musixmatch.com/)
- [Tagtraum Genre Annotations](https://www.tagtraum.com/)


### **Project Structure**
Explain what files/folders exist and what they contain.

```markdown
music-recommender/
|
├── data/
│   ├── raw/
│   ├── processed/
|   ├── samples/
│   └── README.md
│
├── notebooks/
│   ├── 01_research.ipynb
│
├── src/
│   ├── __init__.py
│   ├── base.py                # Abstract Base class (fit(), recommend())
│   ├── non_personalized.py    # Top tracks & Top tracks by genre
│   ├── content_based.py       # Keyword collections, Word2Vec, classifier
│   ├── collaborative.py       # User-based & Item-based filtering
│   └── utils.py
│
├── scripts/ # Python scripts to run recommendations and evaluation
│   ├── run_all.py             # CLI script: builds all recommenders, saves outputs
│   ├── evaluate.py            # Evaluates using precision@k
│
├── tests/                     # Unit tests (pytest)
│   ├── test_non_personalized.py
│   ├── test_collaborative.py
│   └── sample_data.csv
│
├── requirements.txt           # — Python dependencies
├── environment.yml            # optional: conda version of environment
├── README.md                  # how to run, architecture, results
└── main.py                    # entry point (optional CLI interface)
```


### Virtual environment setup

1. Create virtual environment:
```bash
# Using Python venv (recommended for lightweight setup)
python -m venv music-recommender_env

# OR using Conda (recommended if you already use Anaconda/Miniconda)
conda create -n music-recommender_env python=3.10
```

2. Activate environment:
```bash
# Windows (Command Prompt or PowerShell)
music-recommender_env\Scripts\activate

 # macOS/Linux
source music-recommender_env/bin/activate

# OR (if using Conda)
conda activate music-recommender_env
```
3. Install dependencies:
```bash
# Using pip
pip install -r requirements.txt

# OR using Conda (recommended if you have environment.yml)
conda env create -f environment.yml
```

4. Register Jupyter kernel:
```bash
python -m ipykernel install --user --name=music-recommender --display-name "Python (music-recommender)"
```
5. Launch Jupyter Notebook:
```bash
jupyter notebook
```