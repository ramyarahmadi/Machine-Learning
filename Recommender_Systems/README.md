# Recommender Systems

Two content-based recommender systems: one for movies, one for music. Both suggest similar
items purely from item features (genre, cast, audio characteristics, etc.) — no user
interaction history is used.

## 🎬 01-Recommender_System_Imdb_v1.ipynb

Content-based movie recommender on the [IMDB Top 1000](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)
dataset — suggests the 5 most similar movies to a given title.

**Pipeline:**
1. Clean the data and split into numeric features (`Rating`, `Runtime`, `Votes`, `Year`) and
   categorical features (`Genre`, `Director`, `Star1`–`Star4`).
2. Scale numeric features — `StandardScaler` or `MinMaxScaler`, chosen per column based on
   outlier analysis (detected with both Local Outlier Factor and the 3-sigma rule).
3. Encode the categorical features two different ways and compare them:
   - **Approach 1**: `OneHotEncoder` on each categorical column
   - **Approach 2**: `CountVectorizer` bag-of-words on the combined text
4. Recommend similar movies with `cosine_similarity` (both approaches), plus a
   `NearestNeighbors` implementation for Approach 2.
5. Evaluate all three implementations with a genre-overlap consistency metric and compare
   them visually.

**Bug found and fixed:** Approach 1 originally one-hot encoded the *entire* combined
genre+director+cast string as a single category, which made almost every movie its own unique
category with zero feature overlap with any other movie — so its recommendations looked
essentially random. The fix was to one-hot encode `Genre`, `Director`, and `Star1`–`Star4`
**separately** and concatenate the results, so movies that share a director, actor, or genre
actually overlap in feature space.

**Result:** `CountVectorizer` (Approach 2) scored highest on the genre-overlap metric, since
it gives partial credit for sharing even one genre word instead of requiring an identical
genre combination. `NearestNeighbors` on Approach 2's feature space returns the same
recommendations as its `cosine_similarity` counterpart, but without building the full O(n²)
similarity matrix — relevant once the catalog grows beyond 1,000 titles.

## 🎵 02-Spotify_Music_Recommender_System_v1.ipynb

Content-based music recommender on a Spotify audio-features dataset (`tracks_features.csv`)
— suggests similar tracks from audio characteristics and artist identity.

**Pipeline:**
1. Exploratory data analysis: missing values, memory footprint (reduced via dtype downcasting),
   distributions, and correlations across 9 numeric audio features (`danceability`, `energy`,
   `loudness`, `speechiness`, `acousticness`, `valence`, `tempo`, `duration_ms`, `year`).
2. Outlier handling on `duration_ms`: extreme values are **capped** (via a wide 3×IQR bound),
   not dropped, so `MinMaxScaler`'s range isn't distorted by a handful of very long tracks.
3. Feature engineering: cleaned artist names, vectorized with `CountVectorizer` (kept sparse);
   scaled numeric audio features with `MinMaxScaler` and up-weighted them before combining with
   the sparse artist matrix — without this, the thousands of sparse artist dimensions
   completely overpowered the 9 audio features in the distance calculation.
4. Two `NearestNeighbors` models compared: **cosine distance** (on the full combined matrix)
   vs. **Euclidean distance**, with results visualized side by side.

**Result:** After weighting the audio features, the cosine model recommends based on actual
sound similarity rather than shared-artist overlap alone. Noted next steps: genre metadata as
an extra signal, and an approximate nearest-neighbor index (FAISS/Annoy) for production-scale
catalogs.

## Data

- `imdb_top_1000.csv` — IMDB Top 1000 Movies dataset (used by the movie recommender)
- Spotify track features CSV (used by the music recommender; not committed here — see the
  notebook for the source)
