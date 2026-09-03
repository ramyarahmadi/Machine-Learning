#without data Leakage by feature selection
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

data_path = r"D:\lung\features.csv"
df = pd.read_csv(data_path)
print("Dataset shape:", df.shape)


feature_columns = ["Mean","Median","Std","Min","Max","Range","Variance"]
X = df[feature_columns]
y = df["Label"]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000,random_state=42))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf",probability=True,random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=200,random_state=42)
}
results = []
feature_selection_results = []
#Cross Validation
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y),start=1):
    print("\n")
    print("==========================================")
    print(f"FOLD {fold}")
    print("==========================================")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    #Feature Selection
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train,y_train)

    # Importance
    importances = tree.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_columns,"Importance": importances})
    importance_df = importance_df.sort_values(by="Importance",ascending=False).reset_index(drop=True)

    #Select Top 3 Features
    top_3 = importance_df.head(3)["Feature"].tolist()
    print("\nFeature Importance:")
    print(importance_df.to_string(index=False))
    print("\nSelected Top 3 Features:")
    print(top_3)
    # ذخیره Feature Selection
    for _, row in importance_df.iterrows():
        feature_selection_results.append({
            "Fold": fold,
            "Feature": row["Feature"],
            "Importance": row["Importance"]})
    # Create Feature Sets
    feature_sets = {"All Features": feature_columns,"Top 3 Features": top_3}


    for feature_set_name, selected_features in feature_sets.items():
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        for model_name, model in models.items():
            # Train
            model.fit(X_train_selected,y_train)
            # Prediction
            y_pred = model.predict(X_test_selected)
            # Score for ROC-AUC
            if hasattr(model,"predict_proba"):
                y_score = model.predict_proba(X_test_selected)[:, 1]
            else:
                y_score = model.decision_function(X_test_selected)

            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred,zero_division=0)
            recall = recall_score(y_test,y_pred,zero_division=0)
            f1 = f1_score(y_test,y_pred,zero_division=0)
            auc = roc_auc_score(y_test,y_score)

            results.append({
                "Fold": fold,
                "Feature Set": feature_set_name,
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "ROC-AUC": auc
            })

            print(
                f"{feature_set_name:18s} | "
                f"{model_name:22s} | "
                f"Accuracy={accuracy:.3f} | "
                f"F1={f1:.3f} | "
                f"AUC={auc:.3f}"
            )


# Results DataFrame
results_df = pd.DataFrame(results)
# Calculate Mean and Std
summary = (results_df.groupby(["Feature Set", "Model"]).agg({
        "Accuracy": ["mean", "std"],
        "Precision": ["mean", "std"],
        "Recall": ["mean", "std"],
        "F1-Score": ["mean", "std"],
        "ROC-AUC": ["mean", "std"]
    })
)
#Flatten Columns
summary.columns = [
    "Accuracy Mean",
    "Accuracy Std",
    "Precision Mean",
    "Precision Std",
    "Recall Mean",
    "Recall Std",
    "F1 Mean",
    "F1 Std",
    "ROC-AUC Mean",
    "ROC-AUC Std"
]
summary = summary.reset_index()
#Sort
summary = summary.sort_values(by="F1 Mean",ascending=False).reset_index(drop=True)
#Print Final Results
print("\n")
print("==========================================")
print("FINAL LEAKAGE-FREE RESULTS")
print("==========================================")
print(summary.to_string(index=False))

#Feature Selection Summary
feature_df = pd.DataFrame(feature_selection_results)
feature_summary = (feature_df.groupby("Feature").agg({"Importance": ["mean", "std"]}))
feature_summary.columns = ["Mean Importance","Std Importance"]
feature_summary = feature_summary.reset_index()
feature_summary = feature_summary.sort_values(by="Mean Importance",ascending=False).reset_index(drop=True)
print("\n")
print("==========================================")
print("FEATURE IMPORTANCE ACROSS FOLDS")
print("==========================================")
print(feature_summary.to_string(index=False))
# Save Results
results_path = r"D:\lung\leakage_free_results.csv"
summary.to_csv(results_path,index=False)
feature_path = r"D:\lung\feature_importance_leakage_free.csv"
feature_summary.to_csv(feature_path,index=False)
print("\nResults saved to:")
print(results_path)
print("\nFeature importance saved to:")
print(feature_path)