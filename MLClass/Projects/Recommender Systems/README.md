# 🎬🎵 Recommender Systems — Content-Based Filtering

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

This repository contains **three implementations of a content-based recommender system**, each built on a different dataset (or a revisited version of one), evolving in technique from simple to more advanced as the datasets get larger and messier.

| Notebook | Dataset | Domain | Scale | Core Technique |
|---|---|---|---|---|
| `01-IMDB_Movie_Recommender_System(v1).ipynb` | IMDB Top 1000 | 🎬 Movies | ~1,000 rows | `MultiLabelBinarizer` + `LabelEncoder` + KNN & Cosine |
| `02-IMDB_Movie_Recommender_System(v2).ipynb` | Movie Metadata (Kaggle) | 🎬 Movies | ~5,043 rows | `CountVectorizer` + `OrdinalEncoder`/`OneHotEncoder` + Cosine |
| `03-Spotify_Music_Recommender_System(v1).ipynb` | Spotify Tracks | 🎵 Music | ~1,204,025 rows | `CountVectorizer` + `MinMaxScaler` + `NearestNeighbors` |

> 💡 All three solve the same underlying problem — *"given an item the user likes, find similar items"* — using **content-based filtering** (comparing item attributes, not user behavior). The technique changes across notebooks based on dataset size and feature complexity.

---

## 📌 The Core Idea: Content-Based Filtering

```
Item Metadata → Encode/Vectorize → Scale (if numeric) → Similarity/Distance → Top-N Closest Items
```

---

## 🎬 01 — IMDB Movie Recommender (v1)

### 🗂️ Features used
`Genre`, `Director`, `Star1-4`

### 🧠 Method

1. **Genre encoding — why `MultiLabelBinarizer`, not `OneHotEncoder`?**
   A movie can belong to *multiple* genres at once (e.g. "Action, Crime, Drama"). `OneHotEncoder` would treat "Action, Crime, Drama" as a single unique category — meaning a movie tagged only "Action" would look completely unrelated to one tagged "Action, Crime, Drama", even though they share a genre. `MultiLabelBinarizer` instead creates **one binary column per individual genre**, so shared genres are correctly captured as overlap.

2. **Director/Star encoding — why `LabelEncoder`?**
   Unlike genre, each movie has exactly *one* director and one actor per `Star` column — a single value, not a multi-label field — so a simple integer code per name (`LabelEncoder`) is sufficient here.

3. **Scaling — why `MinMaxScaler`?**
   Label-encoded columns can range up to ~700 (hundreds of unique director/actor names), while the one-hot genre columns are strictly 0 or 1. Without scaling, director/actor codes would completely dominate any distance calculation, making genre nearly irrelevant. `MinMaxScaler` brings every column into the same [0, 1] range.

4. **Train/test split**
   80/20 split — mainly to demonstrate the recommender can generalize to movies it wasn't "trained" on, since this is an unsupervised task with no real train/test requirement in the traditional sense.

5. **Two models compared side-by-side: `NearestNeighbors` (Euclidean) vs `cosine_similarity`**
   Euclidean distance measures *absolute* closeness in feature space; cosine similarity measures the *angle/direction* of overlap between vectors, ignoring magnitude. Comparing both lets you see whether the two metrics agree or disagree on top recommendations — useful validation for an unsupervised system with no ground-truth labels.

6. **Evaluation**
   Since there's no labeled "correct" recommendation, the notebook evaluates visually — scatter plots highlighting the query movie (⭐) against its top-5 recommendations (●) on scaled Director/Star1 axes, for both models.

### ⚙️ Pipeline
```
CSV → Drop irrelevant cols → MultiLabelBinarizer(genre) + LabelEncoder(director/stars)
                    ↓
              MinMaxScaler
                    ↓
         Combined feature matrix → train/test split
                    ↓
      NearestNeighbors (Euclidean)  |  cosine_similarity
                    ↓
         Top-5 recommendations + scatter plot visualization
```

---

## 🎬 02 — Movie Metadata Recommender (v2)

### 🗂️ Features used
Text: `genres`, `director_name`, `actor_1_name`, `actor_2_name`, `actor_3_name`
Categorical: `language`, `content_rating`
Numeric: `imdb_score`, `title_year`, `duration`

### 🧠 Method — why it's different from v1

This dataset (`movie_metadata.csv`, ~5,043 rows) has more diverse column types than v1, so the encoding strategy changes per column type instead of using one blanket approach.

1. **Combined text feature — why `CountVectorizer` here instead of `MultiLabelBinarizer`/`LabelEncoder`?**
   Genres, director, and all three actors are merged into one `Combined_Features` text string per movie (with prefixes like `genre_`, `director_`, `star1_` to keep categories from colliding, and `_` joining multi-word names into single tokens). `CountVectorizer` then treats every genre/director/actor as a token and counts presence — a single vectorizer handles all these categorical text fields at once, instead of encoding each column separately.

2. **`content_rating` — why `OrdinalEncoder`?**
   Content ratings (G, PG, PG-13, R, ...) have a natural, meaningful order (increasing maturity level), so encoding them as ordered integers preserves that relationship — unlike genres or names, which have no inherent order.

3. **`language` — why `OneHotEncoder`?**
   Language has no meaningful order (English vs. French vs. Mandarin aren't "closer" or "farther" from each other numerically), so it's one-hot encoded to avoid implying a false ordinal relationship.

4. **`imdb_score`, `title_year`, `duration` — why `MinMaxScaler`?**
   These are continuous numeric features on very different scales (score 0–10, year ~1920–2016, duration in minutes) — scaled to [0, 1] so none dominates the similarity calculation.

5. **Combining everything — `pd.concat`**
   The vectorized text matrix, ordinal-encoded rating, one-hot encoded language, and scaled numeric columns are concatenated into one final feature matrix (`Data`), all aligned by movie title as the index.

6. **Similarity — `cosine_similarity` on the full matrix**
   At ~5,043 movies, a full N×N cosine similarity matrix (~25 million cells) is small enough to compute directly and cache as `cosine_sim_df` — no need for approximate/on-demand nearest-neighbor search at this scale.

7. **Lookup robustness**
   Movie titles are cleaned of stray non-breaking space characters (`\xa0`) before being used as a lookup key — a common artifact in scraped movie-title data that would otherwise cause "movie not found" errors on titles that look identical but aren't.

### ⚙️ Pipeline
```
CSV (5,043 rows) → Select relevant columns → Drop rows with missing values
                    ↓
   CountVectorizer(genres+director+actors) → text matrix
   OrdinalEncoder(content_rating)          → ordered codes
   OneHotEncoder(language)                 → one-hot matrix
   MinMaxScaler(imdb_score, year, duration)→ scaled numeric
                    ↓
        pd.concat → final feature matrix
                    ↓
       cosine_similarity (full N×N matrix)
                    ↓
      Clean title strings → Top-5 lookup via input()
```

---

## 🎵 03 — Spotify Music Recommender (v1)

### 🗂️ Features used
Text: `artists`
Numeric: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `valence`, `tempo`, `duration_ms`, `year`

### 🧠 Method — why it had to change again

At **1,204,025 rows**, this dataset is ~240x larger than the movie-metadata dataset, which rules out the "build one big dense matrix" approach used in notebooks 01 and 02.

1. **Artist name cleaning — why regex instead of `ast.literal_eval`?**
   The raw `artists` column is a string that *looks* like a Python list (`"['Daryl Hall & John Oates']"`). Parsing 1.2M such strings as actual Python code (`ast.literal_eval`) is far slower than a simple regex (`re.findall(r"'([^']+)'", ...)`) that just extracts the quoted names directly. Multi-word names are then joined with `_` — same tokenization reasoning as notebooks 01 and 02 — so `CountVectorizer` treats a full name as one token.

2. **`CountVectorizer` on cleaned artist names — same logic as v2's text handling**, treating each unique artist as a token to capture shared-artist similarity.

3. **`MinMaxScaler` on audio features — why?**
   `duration_ms` (hundreds of thousands) and `tempo` (up to ~250) are on wildly different scales than `danceability` or `valence` (0–1). Without scaling, those large-magnitude columns would dominate similarity almost entirely.

4. **Combining artist vectors + numeric features — sparse `hstack`, not `pd.concat`**
   `CountVectorizer` returns a **sparse** matrix by design, since the vast majority of artist columns are 0 for any given song. Converting that to a dense DataFrame (`.toarray()`) before concatenating — as v2 does at a much smaller scale — would require allocating roughly **1.24 TiB of memory** at this row count (confirmed by hitting exactly this `MemoryError` during development). `scipy.sparse.hstack` combines the sparse artist matrix with the numeric matrix while staying in sparse format the whole time.

5. **Why `NearestNeighbors` instead of a precomputed `cosine_similarity` matrix?**
   A full N×N similarity matrix for 1.2M songs would need ~1.4 trillion cells — impossible to hold in memory regardless of sparsity tricks, since a similarity matrix between *all pairs* is inherently dense. `NearestNeighbors(metric='cosine')` instead computes similarity **on demand**, only between the one queried song and the rest, which is exactly what a real-time recommendation lookup needs.

### 🐞 Bug fixed in this pass
Cell 8 previously contained:
```python
df['artists'] = df['artists'].str.replace
```
This assigns the **unbound `.str.replace` method object itself** to the column (missing `()` to actually call it), silently corrupting `artists` before the real cleaning function (`clean_artist_names`, cell 9) ever runs — causing a `TypeError` when the notebook is run top to bottom on a fresh kernel. The line was dead code with no effect on downstream cells besides breaking them, so it's been **removed**. `clean_artist_names` already operates directly and correctly on the raw string column.

### ⚙️ Pipeline
```
CSV (1,204,025 rows) → Select relevant columns → Drop rows with missing name
                    ↓
    Clean artist names (regex, not ast.literal_eval)
                    ↓
   CountVectorizer(artists) → sparse text matrix
   MinMaxScaler(audio features) → scaled numeric matrix
                    ↓
       scipy.sparse.hstack → combined sparse matrix
                    ↓
    NearestNeighbors(metric='cosine') → fit once
                    ↓
      Top-N recommendations via .kneighbors() on demand
```

---

## 🔑 Key Takeaways Across All Three Notebooks

| Challenge | 01 (IMDB v1) | 02 (Movie Metadata v2) | 03 (Spotify v1) |
|---|---|---|---|
| Multi-label categorical (genre) | `MultiLabelBinarizer` | Merged into text + `CountVectorizer` | — |
| Single-value categorical (director/actor) | `LabelEncoder` | Merged into text + `CountVectorizer` | `CountVectorizer` (artists) |
| Ordered categorical | — | `OrdinalEncoder` (content_rating) | — |
| Unordered categorical | — | `OneHotEncoder` (language) | — |
| Numeric scaling | `MinMaxScaler` | `MinMaxScaler` | `MinMaxScaler` |
| Combining feature groups | `pd.concat` | `pd.concat` | `scipy.sparse.hstack` (sparse-safe) |
| Similarity computation | Full matrix (`NearestNeighbors` + `cosine_similarity`, compared) | Full `cosine_similarity` matrix | `NearestNeighbors` — on demand |
| Why this choice? | Small dataset (~1K) → cheap to compute, good for comparing 2 metrics | Medium dataset (~5K) → still cheap for a dense N×N matrix | Huge dataset (~1.2M) → dense matrix is physically impossible (~1.4 TB); must query on demand |

**The overall arc:** as dataset size and feature diversity grow across the three notebooks, encoding moves from *per-column, hand-picked encoders* (01) → *unified text vectorization for categorical fields + separate encoders for ordinal/nominal fields* (02) → *sparse-matrix-safe pipeline required by scale* (03).

---

## 🚀 How to Run

```bash
pip install pandas numpy scikit-learn scipy seaborn matplotlib
jupyter notebook
```

Place the matching CSV (`imdb_top_1000.csv`, `movie_metadata.csv`, or `tracks_features.csv`) next to its notebook, run all cells top to bottom, and enter a movie/song title when prompted to get recommendations.

---

## 📁 Repo Structure
```
├── 01-IMDB_Movie_Recommender_System(v1).ipynb
├── 02-IMDB_Movie_Recommender_System(v2).ipynb
├── 03-Spotify_Music_Recommender_System(v1).ipynb
├── imdb_top_1000.csv
├── movie_metadata.csv
├── tracks_features.csv
└── README.md
```
