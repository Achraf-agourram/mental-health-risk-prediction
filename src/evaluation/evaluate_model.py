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


plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low", "Medium", "High"],
    yticklabels=["Low", "Medium", "High"],
)
plt.title("Matrice de Confusion")
plt.xlabel("Prédiction")
plt.ylabel("Réalité")
plt.tight_layout()
plt.savefig("reports/figures/confusion_matrix.png", dpi=300)
plt.show()


inv_map = {0: "Low", 1: "Medium", 2: "High"}
errors_df = X_test_orig.copy()
errors_df["Actual"] = y_test_orig.values
errors_df["Predicted"] = pd.Series(y_pred, index=X_test_orig.index).map(inv_map)

misclassified = errors_df[errors_df["Actual"] != errors_df["Predicted"]]
print(f"\nNombre d'erreurs : {len(misclassified)} / {len(y_test)}")
print("\nTop des confusions :")
print(misclassified.groupby(["Actual", "Predicted"]).size().sort_values(ascending=False))


joblib.dump(model, "models/best_mental_health_model.joblib")
print("\nModèle sauvegardé !")