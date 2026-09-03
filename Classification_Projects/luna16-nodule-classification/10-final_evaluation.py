import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay
)


data_path = r"D:\lung\features.csv"
df = pd.read_csv(data_path)
print("Dataset shape:", df.shape)


features = ["Mean","Median","Std","Min","Max","Range","Variance"]
X = df[features]
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
roc_data = {}

for model_name, model in models.items():
    print("\n")
    print("==========================================")
    print(model_name)
    print("==========================================")

    all_true = []
    all_pred = []
    all_score = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y),start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        # Train
        model.fit(X_train,y_train)
        # Prediction
        y_pred = model.predict(X_test)
        # Score
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        # Save predictions
        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())
        all_score.extend(y_score.tolist())
    # Convert to numpy
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_score = np.array(all_score)
    # Metrics
    accuracy = accuracy_score(all_true,all_pred)
    precision = precision_score(all_true,all_pred,zero_division=0)
    recall = recall_score(all_true,all_pred,zero_division=0)
    f1 = f1_score(all_true,all_pred,zero_division=0)
    auc = roc_auc_score(all_true,all_score)
    #Confusion Matrix
    cm = confusion_matrix(all_true,all_pred)
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4))
    print("\nConfusion Matrix:")
    print(cm)
    # Save Metrics
    results.append({"Model": model_name,"Accuracy": accuracy,"Precision": precision,"Recall": recall,"F1-Score": f1,"ROC-AUC": auc})
    # ROC data
    fpr, tpr, thresholds = roc_curve(all_true,all_score)
    roc_data[model_name] = {"fpr": fpr,"tpr": tpr,"auc": auc}
# Results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1-Score",ascending=False).reset_index(drop=True)
print("\n")
print("==========================================")
print("FINAL MODEL RESULTS")
print("==========================================")
print(results_df.to_string(index=False))
# Save CSV
output_path = r"D:\lung\final_model_results.csv"
results_df.to_csv(output_path,index=False)
print("\nResults saved to:")
print(output_path)

#ROC Curve
plt.figure(figsize=(8, 6))
for model_name, data in roc_data.items():
    plt.plot(data["fpr"],data["tpr"],label=f"{model_name} (AUC = {data['auc']:.3f})")
plt.plot([0, 1],[0, 1],linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - 5-Fold Cross Validation")
plt.legend()
plt.grid()
plt.tight_layout()
roc_path = r"D:\lung\roc_curves.png"
plt.savefig(roc_path,dpi=300)
plt.show()
print("\nROC curve saved to:")
print(roc_path)
# Confusion Matrix
for model_name, model in models.items():
    # Recalculate predictions
    all_true = []
    all_pred = []
    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        model.fit(X_train,y_train)
        pred = model.predict(X_test)
        all_true.extend(y_test.tolist())
        all_pred.extend(pred.tolist())

    cm = confusion_matrix(all_true,all_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Negative","Positive"])
    disp.plot(ax=ax)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    filename = model_name.replace(" ","_")
    cm_path = (f"D:\\lung\\confusion_matrix_{filename}.png")
    plt.savefig(cm_path,dpi=300)
    plt.show()
    print(f"Saved: {cm_path}")

# Accuracy Comparison

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"],results_df["Accuracy"])
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.title("Accuracy Comparison")
plt.ylim(0,1)
plt.xticks(rotation=20)
plt.grid(axis="y")
plt.tight_layout()
accuracy_path = (r"D:\lung\accuracy_comparison.png")
plt.savefig(accuracy_path,dpi=300)
plt.show()

# F1 Comparison
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"],results_df["F1-Score"])
plt.ylabel("F1-Score")
plt.xlabel("Model")
plt.title("F1-Score Comparison")
plt.ylim(0,1)
plt.xticks(rotation=20)
plt.grid(axis="y")
plt.tight_layout()
f1_path = (r"D:\lung\f1_comparison.png")
plt.savefig(f1_path,dpi=300)
plt.show()

# ROC-AUC Comparison

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"],results_df["ROC-AUC"])
plt.ylabel("ROC-AUC")

plt.xlabel("Model")
plt.title("ROC-AUC Comparison")
plt.ylim(0,1)
plt.xticks(rotation=20)
plt.grid(axis="y")
plt.tight_layout()
auc_path = (r"D:\lung\auc_comparison.png")
plt.savefig(auc_path,dpi=300)
plt.show()

# Final Message
print("\n")
print("==========================================")
print("EVALUATION COMPLETED")
print("==========================================")
print("Generated files:")
print("1.", output_path)
print("2.", roc_path)
print("3. Confusion matrices")
print("4.", accuracy_path)
print("5.", f1_path)
print("6.", auc_path)