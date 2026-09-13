# Bitcoin Next-Day Candle Direction Prediction (XGBoost Classifier)

Predicts the **direction** of BTC-USD's next-day candle (`1` = up / bullish, `0` = down /
bearish) using `XGBClassifier`, tuned with `GridSearchCV` inside a scikit-learn `Pipeline`.

## Pipeline overview

1. **Data**: Daily OHLCV data for `BTC-USD`, plus two macro series for context — the S&P 500
   index (`^GSPC`) and WTI crude oil futures (`CL=F`) — all via `yfinance`.
2. **Feature engineering**:
   - Core: `open_close = Open - Close`, `low_high = Low - High`, `pct_change`, `volume_change`
   - Technical indicators: `sma_7`, `sma_14`, `ema_12`, `ema_26`, `macd`, `macd_signal`,
     `rsi_14`, `volatility_7`, `momentum_10`
   - Macro: `sp500_return`, `oil_return` (aligned onto BTC's daily calendar, forward-filled
     over weekends/holidays when those markets are closed)
3. **Target**: binary candle direction — `1` if tomorrow's close > today's close, else `0`.
4. **Split**: chronological train/test split (no shuffling) to avoid look-ahead leakage.
5. **Preprocessing**: `StandardScaler`, fit on the training set only.
6. **Model**: `XGBClassifier`, hyperparameters tuned with `GridSearchCV` (scoring = `f1_macro`)
   using `TimeSeriesSplit` cross-validation. GPU (`device="cuda"`) supported for Colab T4 runs —
   set `n_jobs=1` in that case, since parallel CPU folds competing for one GPU tends to be
   slower, not faster.
7. **Evaluation**: accuracy, precision, recall, F1-score, classification report, confusion
   matrix, and feature importance on the held-out test set.

## Usage

```bash
pip install -r requirements.txt
jupyter notebook btc_next_day_direction_xgboost.ipynb
```

## Debugging history / fixes applied

1. **Print bug**: an early version printed `-grid_search.best_score_`, a leftover from a
   regression setup that used `neg_mean_absolute_error` scoring. Once scoring became a normal
   positive metric, that leading minus sign made a positive score print as negative. Fixed by
   printing `grid_search.best_score_` directly.
2. **Degenerate model (always predicts one class)**: `scoring="accuracy"` let `GridSearchCV`
   settle on a model that just predicts the majority class. Fixed by scoring on `"f1_macro"`,
   which penalizes ignoring either class.
3. **Data leakage (found in a separate/borrowed notebook — removed from this project)**: a
   feature computed as `np.log(Close / Close.shift(-1))` used `shift(-1)` on a *feature*,
   which pulls tomorrow's price backward into today's row and leaks the answer. That version
   reported 96–99% accuracy as a result — not a real predictive result. The rule going forward:
   `shift(-1)` is only ever used to build the **target**, never inside a feature.

## Notes

- This predicts direction only, not magnitude — it is not a trading strategy on its own, and
  doesn't account for transaction costs, slippage, or position sizing.
- Financial time series are non-stationary; re-fit periodically on recent data for real use.
- Realistic F1/accuracy on held-out data is modest (often 50–58%). Treat a much higher number
  as a signal to check for leakage before celebrating.
