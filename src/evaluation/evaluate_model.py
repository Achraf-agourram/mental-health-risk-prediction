import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

# 1. Charger les données d'entraînement et de test prétraitées
X_train = pd.read_csv("data/processed/X_train_processed.csv")
X_test = pd.read_csv("data/processed/X_test_processed.csv")

target_map = {"Low": 0, "Medium": 1, "High": 2}
y_train = pd.read_csv("data/processed/y_train.csv")["Mental_Health_Risk"].map(target_map).values
y_test = pd.read_csv("data/processed/y_test.csv")["Mental_Health_Risk"].map(target_map).values


df_raw = pd.read_csv("data/raw/student_mental_health.csv")
X_raw = df_raw.drop(columns=["Student_ID", "Mental_Health_Risk"])
y_raw = df_raw["Mental_Health_Risk"]
_, X_test_orig, _, y_test_orig = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw)


model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="saga")
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("=== ÉVALUATION DU MODÈLE ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score (Macro) : {f1_score(y_test, y_pred, average='macro'):.4f}\n")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

