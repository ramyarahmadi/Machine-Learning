# 📁 Project Showcase: Regression & Feature Engineering

This folder contains **three Jupyter notebooks** that demonstrate different approaches to **polynomial regression**, **feature engineering**, and **model selection** using `scikit-learn`. Each project builds on the previous one, starting from a simple synthetic example and progressing to a real-world financial dataset.

---

## 📂 Contents

| File | Description |
|------|-------------|
| **`01-Bitcoin_Feature_Engineering.ipynb`** | Real Bitcoin price data. Adds **custom engineered features** (e.g., price lags, rolling statistics, technical indicators) alongside polynomial features. |
| **`02-Sine_Polynomial_Regression.ipynb`** | Synthetic sine wave data. Simple polynomial features only. Baseline model for underfitting/overfitting analysis. |
| **`03-Traffic_Polynomial_Regression.ipynb`** | Simulated traffic dataset. Uses only polynomial features, like the sine project, but on a more complex pattern. |

---

## 🧠 Key Concepts Covered

### 1. **Polynomial Regression**
- Using `PolynomialFeatures` to capture non-linear relationships.
- Tuning polynomial degree via `GridSearchCV`.

### 2. **Feature Engineering**
- **Sine & Traffic projects:** Only polynomial features.
- **Bitcoin project:** Combines polynomial features with **domain-specific engineered features**:
  - Lagged prices
  - Rolling means / standard deviations
  - Price differences (momentum)
  - Technical indicators (e.g., RSI-like ratios)

### 3. **Model Comparison**
- Linear Regression
- Ridge (L2 regularization)
- Lasso (L1 regularization)
- Elastic Net (L1 + L2)

### 4. **Diagnostics**
- Cross-validated RMSE vs polynomial degree.
- Train/test performance comparison.
- Residual analysis.
- Coefficient importance plots.

### 5. **Underfitting vs Overfitting**
- Visual demonstration of bias-variance tradeoff.
- Why more features ≠ always better (Bitcoin vs Sine).

---

## 📊 Quick Summary

| Project | Data Type | Features Used | Best Model | Key Insight |
|---------|-----------|---------------|------------|-------------|
| **Sine** | Synthetic | Polynomial only | Ridge/Lasso (degree 3) | Moderate polynomial works well when underlying function is non-linear and clean. |
| **Traffic** | Simulated | Polynomial only | — | Similar to sine; shows how polynomial models handle periodic patterns. |
| **Bitcoin** | Real-world (price) | Polynomial + Engineered | Linear (degree 1) | Real data is noisy and near-random-walk; more features & higher degrees cause overfitting. |

---

## 🛠️ How to Run

1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

2. Open any notebook:
   ```bash
   jupyter notebook <filename>.ipynb
   ```

3. Run all cells (or step through).

---

## 🎯 Purpose

This folder was created to:

1. Show **how polynomial regression behaves** on different types of data.
2. Demonstrate the **importance of feature engineering** (sine vs Bitcoin).
3. Provide a **teaching toolkit** for understanding:
   - Bias-variance tradeoff
   - Regularization (Ridge, Lasso, Elastic Net)
   - Cross-validation
   - Overfitting detection

---

## 📌 Note

- The **sine** and **traffic** notebooks are controlled experiments with known ground truth.
- The **Bitcoin** notebook is a real-world case study where the best model is often simpler than expected.
