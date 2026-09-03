import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

data_path = r"D:\lung\features.csv"
df = pd.read_csv(data_path)
print("Dataset shape:", df.shape)

all_features = ["Mean","Median","Std","Min","Max","Range","Variance"]
top_3_features = ["Mean","Max","Median"]
top_5_features = ["Mean","Max","Median","Variance","Min"]

feature_sets = {
    "All Features": all_features,
    "Top 3 Features": top_3_features,
    "Top 5 Features": top_5_features
}

y = df["Label"]

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


#Cross Validation
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

scoring = {"accuracy": "accuracy","precision": "precision","recall": "recall","f1": "f1","roc_auc": "roc_auc"}
results = []
for feature_set_name, features in feature_sets.items():
    print("\n")
    print("==========================================")
    print(feature_set_name)
    print("==========================================")

    X = df[features]

    for model_name, model in models.items():

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        accuracy_mean = scores["test_accuracy"].mean()
        accuracy_std = scores["test_accuracy"].std()

        precision_mean = scores["test_precision"].mean()
        precision_std = scores["test_precision"].std()

        recall_mean = scores["test_recall"].mean()
        recall_std = scores["test_recall"].std()

        f1_mean = scores["test_f1"].mean()
        f1_std = scores["test_f1"].std()

        auc_mean = scores["test_roc_auc"].mean()
        auc_std = scores["test_roc_auc"].std()


        # ذخیره
        results.append({

            "Feature Set": feature_set_name,
            "Model": model_name,
            "Accuracy Mean": accuracy_mean,
            "Accuracy Std": accuracy_std,
            "Precision Mean": precision_mean,
            "Precision Std": precision_std,
            "Recall Mean": recall_mean,
            "Recall Std": recall_std,
            "F1 Mean": f1_mean,
            "F1 Std": f1_std,
            "ROC-AUC Mean": auc_mean,
            "ROC-AUC Std": auc_std
        })

        print("\nModel:", model_name)
        print(f"Accuracy : {accuracy_mean:.4f} ± {accuracy_std:.4f}")
        print(f"Precision: {precision_mean:.4f} ± {precision_std:.4f}")
        print(f"Recall   : {recall_mean:.4f} ± {recall_std:.4f}")
        print(f"F1-Score : {f1_mean:.4f} ± {f1_std:.4f}")
        print(f"ROC-AUC  : {auc_mean:.4f} ± {auc_std:.4f}")

results_df = pd.DataFrame(results)

# Sort by F1
results_df = results_df.sort_values(by="F1 Mean",ascending=False).reset_index(drop=True)


# Final Results
print("\n")
print("==========================================")
print("FINAL 5-FOLD CROSS-VALIDATION RESULTS")
print("==========================================")
print(results_df.to_string(index=False))

output_path = r"D:\lung\cross_validation_results.csv"
results_df.to_csv(output_path,index=False)
print("\nResults saved to:")
print(output_path)