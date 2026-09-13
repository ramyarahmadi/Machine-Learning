# Classification Projects

Classification tasks across two very different domains: financial time series and medical
imaging-derived tabular data.

## 📈 BTC next-day direction XGBoost project

Predicts the **direction** of Bitcoin's next-day candle (`1` = up, `0` = down) using
`XGBClassifier`.

- **Data**: daily OHLCV for `BTC-USD`, plus macro context — S&P 500 (`^GSPC`) and WTI crude
  oil (`CL=F`) daily returns — all via `yfinance`.
- **Features**: core price-action features (`open - close`, `low - high`, daily % change),
  technical indicators (SMA, EMA, MACD, RSI, volatility, momentum), and the macro returns
  above — all strictly backward-looking, no future leakage.
- **Model**: `StandardScaler` + `XGBClassifier` in a `Pipeline`, tuned with `GridSearchCV`
  (`scoring="f1_macro"`) using `TimeSeriesSplit` cross-validation, so hyperparameter search
  never trains on future data to predict the past.
- **Result**: realistic accuracy/F1 in the ~50–55% range on held-out data — consistent with
  BTC's next-day direction being close to a random walk when predicted from price/volume data
  alone. A separate, borrowed notebook that claimed 96–99% accuracy was found to have data
  leakage (a feature that accidentally used tomorrow's closing price) and was removed from
  this project.

See [`README.md`](./BTC%20next-day%20direction%20XGBoost%20project/README.md) inside the
project folder for full details.

## 🫁 luna16-nodule-classification

Lung nodule (benign vs. malignant) classification on the [LUNA16](https://luna16.grand-challenge.org/)
CT scan dataset.

- **Pipeline**: nodule annotations → ROI extraction → feature extraction → feature selection →
  model comparison → cross-validation → scan-level leakage-free evaluation.
- **Features**: 7 first-order intensity features (mean, median, std, min, max, range,
  variance), ranked by importance with a Decision Tree.
- **Models compared**: Logistic Regression, KNN, SVM, and Random Forest.
- **Data leakage found and fixed**: 200 samples came from only 102 unique CT scans (30 scans
  contributed multiple samples each), so a naive random split let the same scan appear in both
  train and test. Final evaluation uses `GroupKFold` grouped by Scan ID, with feature selection
  performed inside each training fold.
- **Result** (5-fold, scan-level, leakage-free): best ROC-AUC 0.854 ± 0.072 (Logistic
  Regression); best accuracy 0.755 ± 0.045 (SVM).
