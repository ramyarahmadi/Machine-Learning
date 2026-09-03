import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

# Load Dataset
data_path = r"D:\lung\features.csv"
df = pd.read_csv(data_path)
print("Dataset shape:", df.shape)
# Features

feature_columns = [
    "Mean",
    "Median",
    "Std",
    "Min",
    "Max",
    "Range",
    "Variance"
]
X = df[feature_columns]
y = df["Label"]
# 5-Fold Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ذخیره Importance هر Fold
all_importances = []
print("\n==============================")
print("5-Fold Cross Validation")
print("==============================")

for fold, (train_index, test_index) in enumerate(cv.split(X, y), start=1):
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]
    # Decision Tree
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    all_importances.append(importance)
    print(f"\nFold {fold}")
    for feature, value in zip(feature_columns,importance):
        print(f"{feature:10s}: {value:.6f}")
# Convert to NumPy
all_importances = np.array(all_importances)
# Mean Importance
mean_importance = np.mean(all_importances,axis=0)
# Standard Deviation
std_importance = np.std(all_importances,axis=0)
# Create Results Table
results = pd.DataFrame({
    "Feature": feature_columns,
    "Mean_Importance": mean_importance,
    "Std_Importance": std_importance
})
# مرتب‌سازی بر اساس میانگین Importance
results = results.sort_values(by="Mean_Importance",ascending=False).reset_index(drop=True)
# Print Final Ranking
print("\n==============================")
print("Final Feature Ranking")
print("==============================")
print(results.to_string(index=False))
# Save Results
output_path = r"D:\lung\feature_importance_cv.csv"
results.to_csv(output_path,index=False)
print("\nSaved to:")
print(output_path)