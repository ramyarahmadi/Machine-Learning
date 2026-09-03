import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


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
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Feature Importance
importance = model.feature_importances_
feature_importance = pd.DataFrame({"Feature": feature_columns,"Importance": importance})
# مرتب‌سازی
feature_importance = feature_importance.sort_values(by="Importance",ascending=False)
print("\n==============================")
print("Feature Importance")
print("==============================")
print(feature_importance)

# Save Results
output_path = r"D:\lung\feature_importance.csv"
feature_importance.to_csv(output_path,index=False)
print("\nSaved to:")
print(output_path)

plt.figure(figsize=(9, 5))
plt.bar(feature_importance["Feature"],feature_importance["Importance"])
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Decision Tree Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()