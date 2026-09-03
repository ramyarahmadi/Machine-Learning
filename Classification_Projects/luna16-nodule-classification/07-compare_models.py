import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, roc_auc_score, confusion_matrix

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
train_index, test_index = train_test_split(df.index,test_size=0.20,random_state=42,stratify=y)
y_train = y.loc[train_index]
y_test = y.loc[test_index]

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

#Run Experiments
results = []
for feature_set_name, features in feature_sets.items():
    print("\n")
    print("==========================================")
    print(feature_set_name)
    print("==========================================")

    X = df[features]

    X_train = X.loc[train_index]
    X_test = X.loc[test_index]

    print("Features:", features)

    for model_name, model in models.items():
        # Train
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        # Probability / Score
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        # Metrics
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred,zero_division=0)
        recall = recall_score(y_test,y_pred,zero_division=0)
        f1 = f1_score(y_test,y_pred,zero_division=0)
        auc = roc_auc_score(y_test,y_score)
        cm = confusion_matrix(y_test,y_pred)
        # Save result
        results.append({
            "Feature Set": feature_set_name,
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": auc
        })

        # Verbose
        print("\nModel:", model_name)
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        print("Confusion Matrix:")
        print(cm)

#Results DataFrame
results_df = pd.DataFrame(results)

#Sort by F1
results_df = results_df.sort_values(by="F1-Score",ascending=False).reset_index(drop=True)

#Display Final Results
print("\n")
print("==========================================")
print("FINAL MODEL COMPARISON")
print("==========================================")
print(results_df.to_string(index=False))

#Save Results
output_path = r"D:\lung\model_comparison.csv"
results_df.to_csv(output_path,index=False)
print("\nResults saved to:")
print(output_path)