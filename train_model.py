# train_model.py
# Part 1: retrieve the dataset  |  Part 2: preprocess, train, evaluate, save

import json
import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# PART 1 - Retrieve and inspect the dataset (UCI Student Performance, ID 320)
# ---------------------------------------------------------------------------
student = fetch_ucirepo(id=320)

# Combine student features and grade targets
df = pd.concat([student.data.features, student.data.targets], axis=1)
df.to_csv("student_data.csv", index=False)

print("Rows and columns:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())

# Checkpoint: G1, G2 and G3 must exist
for col in ["G1", "G2", "G3"]:
    if col not in df.columns:
        raise SystemExit(f"Column {col} is missing. Check the dataset download.")
print("Checkpoint OK: G1, G2 and G3 are present.")

# ---------------------------------------------------------------------------
# PART 2 - Preprocess, train and evaluate
# ---------------------------------------------------------------------------
# Target: 1 = at risk (G3 < 10); 0 = not at risk
df = df.drop_duplicates()
y = (df["G3"] < 10).astype(int)

# Never use G3 as a predictor (prevents target leakage)
X = df.drop(columns=["G3"])

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", round(accuracy, 3))
print(classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0,
))

# Save the model
joblib.dump(model, "student_model.pkl")
print("Saved student_model.pkl")

# Save the evaluation results so the Streamlit app can display them
report = classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0,
    output_dict=True,
)
metrics = {
    "accuracy": accuracy,
    "report": report,
    "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved metrics.json")
