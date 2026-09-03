import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix



data_path = r"D:\lung\features.csv"
df = pd.read_csv(data_path)
print("Dataset shape:", df.shape)

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


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
# Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)
# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("\n==============================")
print("Decision Tree Evaluation")
print("==============================")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-Score :", f1)
print("\nConfusion Matrix:")
print(cm)

# Feature Importance
importance = model.feature_importances_
feature_importance = pd.DataFrame({"Feature": feature_columns,"Importance": importance})
feature_importance = feature_importance.sort_values(by="Importance",ascending=False)
print("\n==============================")
print("Feature Importance")
print("==============================")
print(feature_importance.to_string(index=False))

# Save Feature Importance
output_path = r"D:\lung\feature_importance.csv"
feature_importance.to_csv(output_path,index=False)
print("\nFeature importance saved to:")
print(output_path)