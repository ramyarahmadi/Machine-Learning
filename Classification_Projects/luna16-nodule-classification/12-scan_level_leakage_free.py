# ============================================================
# LUNA16 - Leakage-Free Scan-Level Cross Validation
# Feature Selection + Classical Machine Learning
# ============================================================

import os
import re
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix)

warnings.filterwarnings("ignore")
INPUT_FILE = r"D:\lung\features.csv"
OUTPUT_DIR = r"D:\lung"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
print(df.columns.tolist())

def extract_scan_id(filename):
    match = re.search(r"(1\.3\.6\.1\.4\.1\.14519\.5\.2\.1\.6279\.6001\.\d+)",str(filename))
    if match:
        return match.group(1)
    return None

df["Scan_ID"] = df["Filename"].apply(extract_scan_id)
# Check missing Scan IDs
missing_scan_ids = df["Scan_ID"].isna().sum()
print("Missing Scan IDs:", missing_scan_ids)
if missing_scan_ids > 0:
    print("\nRows with missing Scan_ID:")
    print(df[df["Scan_ID"].isna()][["Filename"]])
    raise ValueError("Some samples do not have a valid Scan_ID.")
print("Unique Scan IDs:", df["Scan_ID"].nunique())
print()


# Define Features and Label
FEATURES = ["Mean","Median","Std","Min","Max","Range","Variance"]
TARGET = "Label"
GROUP = "Scan_ID"
# Check columns
required_columns = FEATURES + [TARGET, GROUP]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in dataset."
        )

X = df[FEATURES].copy()
y = df[TARGET].astype(int)
groups = df[GROUP]

print("=" * 60)
print("FEATURES")
print("=" * 60)
print(FEATURES)
print()
print("Class distribution:")
print(y.value_counts())
print()

# Models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000,random_state=42))]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf",C=1.0,probability=True,random_state=42))]),
    "Random Forest": RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced")
}


# Group K-Fold
N_SPLITS = 5
group_kfold = GroupKFold(
    n_splits=N_SPLITS
)

# Storage
results = []
feature_importance_results = []
fold_feature_selection = []
# Cross Validation
for fold, (train_idx, test_idx) in enumerate(
        group_kfold.split(X, y, groups), 1):
    print("\n")
    print("=" * 60)
    print(f"FOLD {fold}")
    print("=" * 60)
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]
    # Check leakage
    intersection = set(train_groups).intersection(set(test_groups))

    print("Training samples:", len(train_idx))
    print("Testing samples :", len(test_idx))

    print("Training scans  :", train_groups.nunique())
    print("Testing scans   :", test_groups.nunique())

    print("Common Scan IDs :", len(intersection))

    if len(intersection) > 0:
        raise ValueError("DATA LEAKAGE DETECTED! ""A Scan_ID exists in both train and test.")
    print("Leakage check    : PASSED")
    # Feature Selection
    # ONLY Training Data
    selector = DecisionTreeClassifier(random_state=42,max_depth=4)
    selector.fit(X_train,y_train)
    importances = selector.feature_importances_
    importance_df = pd.DataFrame({"Feature": FEATURES,"Importance": importances}).sort_values(by="Importance",ascending=False)
    print("\nFeature Importance:")
    print(importance_df.to_string(index=False))
    # Save feature importance
    for _, row in importance_df.iterrows():
        feature_importance_results.append({
            "Fold": fold,
            "Feature": row["Feature"],
            "Importance": row["Importance"]
        })
    # Select Top 3
    TOP_N = 3
    top_features = (
        importance_df
        .head(TOP_N)["Feature"]
        .tolist()
    )
    print("\nSelected Top 3 Features:")
    print(top_features)
    fold_feature_selection.append({
        "Fold": fold,
        "Top_1": top_features[0],
        "Top_2": top_features[1],
        "Top_3": top_features[2]

    })

    # Prepare Top 3 Dataset
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]
    # Train and Evaluate Models
    for model_name, model in models.items():
        print(f"\n{model_name}")
        # Train
        model.fit(X_train_top,y_train)
        # Prediction
        y_pred = model.predict(X_test_top)
        # Probability
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_top)[:, 1]
        else:
            y_prob = model.decision_function( X_test_top)
        # Metrics
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred,zero_division=0)
        recall = recall_score(y_test,y_pred,zero_division=0)
        f1 = f1_score(y_test,y_pred,zero_division=0)
        auc = roc_auc_score(y_test,y_prob)
        cm = confusion_matrix(y_test,y_pred)
        print(f"Accuracy : {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall   : {recall:.3f}")
        print(f"F1-Score : {f1:.3f}")
        print(f"ROC-AUC  : {auc:.3f}")
        print("Confusion Matrix:")
        print(cm)
        # Save
        results.append({"Fold": fold,"Feature_Set": "Top 3","Features": ", ".join(top_features),"Model": model_name,"Accuracy": accuracy,"Precision": precision,"Recall": recall,"F1-Score": f1,"ROC-AUC": auc})

#Results DataFrame
results_df = pd.DataFrame(results)

# Calculate Mean ± Std
summary = (results_df.groupby(["Feature_Set", "Model"] ).agg({
        "Accuracy": ["mean", "std"],
        "Precision": ["mean", "std"],
        "Recall": ["mean", "std"],
        "F1-Score": ["mean", "std"],
        "ROC-AUC": ["mean", "std"]
})
)

# Flatten columns
summary.columns = [f"{col[0]} {col[1]}"for col in summary.columns]
summary = summary.reset_index()

# Print Final Results
print("\n")
print("=" * 70)
print("FINAL SCAN-LEVEL LEAKAGE-FREE RESULTS")
print("=" * 70)
print(summary.to_string(index=False))

#Feature Importance Across Folds
importance_df_all = pd.DataFrame(feature_importance_results)
importance_summary = (importance_df_all.groupby("Feature").agg({"Importance": ["mean", "std"]}).reset_index())
importance_summary.columns = ["Feature","Mean_Importance","Std_Importance"]
importance_summary = (importance_summary.sort_values("Mean_Importance",ascending=False))
print("\n")
print("=" * 70)
print("FEATURE IMPORTANCE ACROSS FOLDS")
print("=" * 70)
print(importance_summary.to_string(index=False))

# Feature Selection Per Fold
selection_df = pd.DataFrame(fold_feature_selection)
print("\n")
print("=" * 70)
print("FEATURE SELECTION PER FOLD")
print("=" * 70)
print(selection_df.to_string(index=False))
# Save Results
fold_results_path = os.path.join(OUTPUT_DIR,"scan_level_fold_results.csv")
summary_path = os.path.join(OUTPUT_DIR,"scan_level_final_results.csv")
importance_path = os.path.join(OUTPUT_DIR,"scan_level_feature_importance.csv")
selection_path = os.path.join(OUTPUT_DIR,"scan_level_feature_selection.csv")
results_df.to_csv(fold_results_path,index=False)
summary.to_csv(summary_path,index=False)
importance_summary.to_csv(importance_path,index=False)
selection_df.to_csv(selection_path,index=False)
# Final Best Model
best_row = summary.sort_values("ROC-AUC mean",ascending=False).iloc[0]
print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)
print("Model:",best_row["Model"])
print("Accuracy:",f"{best_row['Accuracy mean']:.4f}")
print("F1:",f"{best_row['F1-Score mean']:.4f}")
print("ROC-AUC:",f"{best_row['ROC-AUC mean']:.4f}")
print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)
print(fold_results_path)
print(summary_path)
print(importance_path)
print(selection_path)
print("\n")
print("SCAN-LEVEL LEAKAGE-FREE EVALUATION COMPLETED.")