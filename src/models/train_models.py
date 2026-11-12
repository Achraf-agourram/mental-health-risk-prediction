import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

X_train = pd.read_csv("data/processed/X_train_processed.csv")
X_test = pd.read_csv("data/processed/X_test_processed.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
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
metrics_df.to_csv("reports/models_performance_summary.csv", index=True)


all_reports_df = pd.concat(detailed_reports)
all_reports_df.to_csv("reports/models_detailed_classification_reports.csv", index=True)

print("All evaluation reports and summary tables successfully saved !")