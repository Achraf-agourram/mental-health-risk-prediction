import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
)
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

X_train = pd.read_csv("data/processed/X_train_processed.csv")
X_test = pd.read_csv("data/processed/X_test_processed.csv")

target_map = {'Low': 0, 'Medium': 1, 'High': 2}
y_train = pd.read_csv('data/processed/y_train.csv')['Mental_Health_Risk'].map(target_map).values
y_test = pd.read_csv('data/processed/y_test.csv')['Mental_Health_Risk'].map(target_map).values


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss')
}


results = {}
detailed_reports = []

for name, model in models.items():
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  results[name] = {
      "Accuracy": accuracy_score(y_test, y_pred),
      "Precision (Macro)": precision_score(y_test, y_pred, average="macro"),
      "Recall (Macro)": recall_score(y_test, y_pred, average="macro"),
      "F1-score (Macro)": f1_score(y_test, y_pred, average="macro"),
  }


  report_dict = classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"], output_dict=True)
  report_df = pd.DataFrame(report_dict).transpose()
  report_df["Model"] = name
  detailed_reports.append(report_df)


metrics_df = pd.DataFrame(results).T
metrics_df.to_csv("reports/metrics/models_performance_summary.csv", index=True)


all_reports_df = pd.concat(detailed_reports)
all_reports_df.to_csv("reports/metrics/models_detailed_classification_reports.csv", index=True)

print("All evaluation reports and summary tables successfully saved !")