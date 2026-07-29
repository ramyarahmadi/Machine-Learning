# 🎬🎵 Recommender Systems — Content-Based Filtering

This repository contains **two implementations of a content-based recommender system**, each built on a different dataset and using a slightly different technical approach to solve the same core problem: *"given an item the user likes, find similar items."*

| Notebook | Dataset | Domain | Scale |
|---|---|---|---|
| `01-IMDB_Movie_Recommender_System(v1).ipynb` | IMDB Top 1000 | 🎬 Movies | ~1,000 rows, 16 columns |
| `02-IMDB_Movie_Recommender_System(v2).ipynb` | Movie Metadata (Kaggle) | 🎬 Movies | ~5,043 rows, 28 columns |

> 💡 Both notebooks tackle the **same domain (movies)** and the **same problem (content-based recommendation)**, but v2 works with a richer, messier, real-world dataset — more metadata fields, missing values, multiple actors, and numeric popularity signals — which pushes the feature engineering a step further than v1.

---

## 📌 The Core Idea: Content-Based Filtering

Content-based filtering recommends items by comparing their **attributes/metadata** (not user behavior). If two items share similar genres, creators, or characteristics, they're considered "close" to each other in feature space.

```
Item Metadata → Vectorization → Similarity Matrix → Top-N Closest Items
```

---

## 🎬 v1 — IMDB Movie Recommender

### 🗂️ Features used
`Genre`, `Director`, `Star1-4`

### 🧠 Method
1. **Feature Combination** — All categorical metadata is merged into a single string per movie, using prefixes (`genre_`, `director_`, `star1_`, ...) so identical names across categories (e.g. an actor who's also a director) don't collide.
2. **Text Vectorization — why `CountVectorizer`?**
   Since every feature here is essentially a *label* (genre name, person's name), `CountVectorizer` (Bag-of-Words) is a natural fit: it treats each unique genre/director/actor as a token and counts its presence. Multi-word names are joined with `_` first (e.g. `Robert_Downey_Jr.`) so they're treated as **one token**, not split into meaningless separate words.
3. **Similarity — why `cosine_similarity`?**
   With Bag-of-Words vectors, what matters is the *overlap direction* of shared attributes, not raw magnitude — two movies sharing 3 out of 5 features should be "close" regardless of how many total features exist. Cosine similarity measures the angle between vectors, making it ideal for sparse, high-dimensional, count-based data like this.
4. **Full similarity matrix** — With only 1,000 movies, computing a dense N×N (1,000 × 1,000) cosine similarity matrix is trivial and lets us directly `argsort` a row to get the top-N recommendations.

### ⚙️ Pipeline
```
CSV → Clean & merge text features → CountVectorizer → cosine_similarity (N×N) → argsort → Top-5 movies
```

---

## 🎬 v2 — Movie Metadata Recommender (richer features)

### 🗂️ Features used
Categorical/text: `genres`, `director_name`, `actor_1_name`, `actor_2_name`, `actor_3_name`, `plot_keywords`
Numeric: `imdb_score`, `num_voted_users`, `budget`, `gross`, `title_year` (and similar popularity/financial signals)

### 🧠 Method — and why it goes further than v1

`movie_metadata.csv` has **28 columns** and real-world messiness (missing values, inconsistent casing, multiple actor columns) — so the pipeline needs an extra cleaning and combining step compared to v1.

1. **Missing value handling** — Several fields (`plot_keywords`, `director_name`, actor names) contain nulls that must be filled (e.g. with an empty string) before they can be merged into a single text blob; otherwise concatenation breaks or `NaN` pollutes the combined feature string.
2. **Multi-word / multi-value tokenization — same reasoning as v1** — Director and actor names are joined with `_` (e.g. `James_Cameron`) so `CountVectorizer` treats a full name as one token instead of splitting it into separate, meaningless words. `plot_keywords` (already pipe-separated, e.g. `"culture clash|future|space war"`) is split and rejoined with spaces so each keyword becomes its own token.
3. **Feature combination** — Genres, director, all three actors, and plot keywords are merged into one `Combined_Features` text column per movie — the same "prefix and concatenate" strategy from v1, extended to more fields for a richer content signal.
4. **`CountVectorizer` — why still Bag-of-Words?**
   Same logic as v1: every field here is a categorical label or keyword, not free-flowing prose, so raw word/phrase presence (not frequency weighting) is what defines similarity. `CountVectorizer` keeps this simple and interpretable.
5. **Should numeric columns (`imdb_score`, `budget`, `gross`) be included?**
   Unlike v1, this dataset has meaningful numeric signals. If included, they should be scaled (e.g. `MinMaxScaler`) before combining with the text-based vectors — otherwise a column like `gross` (up to hundreds of millions) would completely dominate cosine similarity over genre/cast overlap.
6. **Similarity — why `cosine_similarity` still works here**
   At ~5,043 movies, a full N×N similarity matrix (~25 million cells) is small enough to compute and hold in memory directly — unlike the 1.2M-row case that forced a switch to `NearestNeighbors` in an earlier version of this project. So v2 keeps the same `cosine_similarity` + `argsort` approach as v1, just on more, and richer, features.

### ⚙️ Pipeline
```
CSV (5,043 rows, 28 columns) → Fill missing values → Clean & merge text features
      (genres + director + actor1-3 + plot_keywords)
                    ↓
              CountVectorizer
                    ↓
       cosine_similarity (N×N, ~5K x 5K)
                    ↓
                 argsort → Top-5 movies
```

---

## 🔑 Key Takeaways: v1 vs v2

| Challenge | v1 Solution | v2 Solution |
|---|---|---|
| Missing values | Not present in this curated dataset | Explicit fillna step before merging text |
| Number of content fields | 5 fields (genre, director, star1-4) | 6+ fields (genres, director, 3 actors, plot_keywords) |
| Multi-word tokens | `_` joining for names | `_` joining for names + keyword re-splitting |
| Numeric signals | Not used | Optionally scaled and combined (imdb_score, budget, gross) |
| Text encoding | `CountVectorizer` | `CountVectorizer` |
| Similarity computation | Full `cosine_similarity` matrix (N×N) | Full `cosine_similarity` matrix (N×N) — still feasible at ~5K rows |
| Why the difference? | Small, clean, curated dataset | Larger, real-world, messier dataset — needs more preprocessing, but still small enough for a dense similarity matrix |

---

## 🚀 How to Run

```bash
pip install pandas numpy scikit-learn scipy matplotlib
jupyter notebook
```

Open either notebook, run all cells top to bottom, and enter a movie title when prompted to get 5 recommendations.

---

## 📁 Repo Structure
```
├── 01-IMDB_Movie_Recommender_System(v1).ipynb
├── 02-IMDB_Movie_Recommender_System(v2).ipynb
├── imdb_top_1000.csv
├── movie_metadata.csv
└── README.md
```
