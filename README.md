# Machine Learning & Data Science Projects

This repository is a collection of small, self-contained machine learning and data analysis
projects. Each project lives in its own folder with the code/notebook, a short description of
the goal and approach, the dataset used (or a reference to it), and results/takeaways.

## 📁 Classification_Projects

Classification tasks across different domains — medical imaging data, financial time series,
and Persian-language text.

- **BTC next-day direction XGBoost project** — Predicts the direction (up/down) of Bitcoin's
  next-day candle using `XGBClassifier`, with technical indicators (RSI, MACD, moving
  averages) and macro features (S&P 500, oil returns), tuned via `GridSearchCV` with
  time-series-aware cross-validation.
- **luna16-nodule-classification** — Lung nodule (benign vs. malignant) classification on the
  LUNA16 CT scan dataset. A classical ML pipeline (ROI extraction → feature extraction →
  feature selection → model comparison) with careful scan-level data leakage detection and a
  leakage-free evaluation using `GroupKFold`. Best result: ROC-AUC 0.854 ± 0.072 (Logistic
  Regression).

  **In The Feature:**
- **Digikala Reviews** — Sentiment analysis on Persian-language product reviews from Digikala.
- **Snapfood Reviews** — Sentiment analysis on Persian-language customer feedback for
  Snappfood food delivery orders.
- **Amazon Reviews** — Text classification and sentiment scoring on Amazon product reviews.

## 📁 Recommender_Systems

Recommendation engines using collaborative filtering, content-based filtering, and hybrid
approaches.

- **IMDb Movie Recommender** — Suggests movies based on user ratings and content similarity
  (genre, cast, plot).
- **Spotify Music Recommender** — Recommends songs and playlists based on audio features and
  listening history.

## 📁 Regression_Projects

Regression models predicting continuous values and fitting functions to data.

- **Bitcoin Price Estimation** — Predicting Bitcoin prices using historical data, feature
  engineering, and regularized polynomial regression.
- **Sinusoidal Function Fitting** — Fitting different models (linear, polynomial, etc.) to
  sinusoidal data to compare performance and accuracy.

## 🎯 Purpose

This repository serves as a personal playground for practicing and showcasing different
machine learning techniques — from recommendation systems and NLP to regression and medical
image-derived feature classification — using real-world and well-known datasets.

## 🛠️ Tech Stack

Common tools and libraries used across projects:
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn

## License

See [LICENSE](./LICENSE) — applies to the whole repository unless a subfolder states
otherwise.
