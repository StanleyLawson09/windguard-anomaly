# Machine Learning failure prediction model training with a Random Forest Classifier. The model is trained on the AI4I 2020 Predictive Maintenance dataset.
import pandas as pd
import numpy as np
import os
import joblib
from requests.packages import target
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

os.makedirs("model", exist_ok=True)

# Load the dataset
df = pd.read_csv("data/ai4i2020.csv")

# Drop columns that are not relevant for the model training to avoid data leakage. The columns to be dropped are specified in the drop_columns list
drop_columns = ["twf", "hdf", "pwf", "osf", "rnf"]
df = df.drop(columns=[col for col in drop_columns if col in df.columns])

# Map the categorical "type" column to numerical values (L=0, M=1, H=2) and fill any missing values with 0. This is done to prepare the data for model training, as machine learning algorithms typically require numerical input.
df["type"] = df["type"].map({"L": 0, "M": 1, "H": 2}).fillna(0)

# Create new features based on existing columns to enhance the model's predictive power
df["temp_diff"]= df["process_temperature"] - df["air_temperature"]
df["power_w"]= df["torque"] * (df["rotational_speed"] * 2 * 3.14159 / 60)
df["wear_torque"] = df["tool_wear"] * df["torque"]

# Define the features and target variable for the model
features = ["type", "air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear", "temp_diff", "power_w", "wear_torque"]
target = "machine_failure"

X=df[features].values
y=df[target].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier.
model = RandomForestClassifier(n_estimators=200, class_weight="balanced", max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate the model on the test set and print the classification report, confusion matrix, ROC AUC score and feature importances.
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"]))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print(f"\nROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print(f"\nF1 Score (Failure class): {f1_score(y_test, y_pred):.4f}")

print("\n=== Feature Importances ===")
importance_list = sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)
for feature, importance in importance_list:
    print(f"{feature:<25s} {importance:.4f}")

# Save the trained model, scaler, and features to disk using joblib for later use in predictions.
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(features, "model/features.pkl")
print("\nModel, scaler, and features saved successfully.")