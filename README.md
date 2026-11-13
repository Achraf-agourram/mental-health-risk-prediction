# Student Mental Health Risk Prediction

This repository contains an exploratory and machine-learning workflow for classifying a student's self-reported mental-health risk as **Low**, **Medium**, or **High**. It includes notebooks for data exploration and preprocessing, scripts for model training and evaluation, an optional Airflow DAG, and a Streamlit survey app.

> **Important:** This is an educational prediction project, not a clinical screening or diagnostic tool. Its output must not be used as a diagnosis or as a substitute for advice from a qualified mental-health professional. The model's scores are not clinical probabilities.

## Project structure

```text
airflow/dags/                    Optional Airflow orchestration DAG
app/                              Streamlit application
config/                           Project configuration
data/
  raw/                            Input survey CSV (not committed)
  processed/                      Generated train/test CSVs (not committed)
notebooks/
  eda.ipynb                       Exploratory data analysis
  spliting_and_preprocessing.ipynb Train/test split and preprocessing
models/                           Generated model artifact (not committed)
reports/
  figures/                        Generated evaluation figures
  metrics/                        Generated model reports
src/
  models/train_models.py          Train and compare baseline classifiers
  models/optimize_models.py       Tune classifier hyperparameters
  evaluation/evaluate_model.py    Evaluate and save the app's model
requirements.txt                  Python dependencies for the ML workflow and app
```

## Dataset

Place the survey dataset at `data/raw/student_mental_health.csv`. The CSV is expected to contain 2,000 rows and these columns:

`Student_ID`, `Age`, `Gender`, `Year_of_Study`, `Sleep_Hours`, `Study_Hours_Per_Week`, `GPA`, `Social_Media_Hours`, `Screen_Time_Hours`, `Physical_Activity`, `Financial_Stress`, `Family_Support`, `Attendance_Rate`, `Part_Time_Job`, `Diet_Quality`, `Previous_Mental_Health_History`, and `Mental_Health_Risk`.

The target column, `Mental_Health_Risk`, uses the labels `Low`, `Medium`, and `High`. The current model excludes `Student_ID`, `Age`, and `Gender`; it uses the remaining 13 survey features. The raw and generated data files are ignored by Git, so provide the input CSV locally before running the workflow.

## Setup

Use Python 3.10 or newer and run commands from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents virtual-environment activation, follow your machine's Python/PowerShell policy or run the environment's Python directly as `.venv\Scripts\python.exe`.

## Run the workflow

1. **Explore the data** (optional): open `notebooks/eda.ipynb` in Jupyter and run its cells.
2. **Prepare the data:** execute `notebooks/spliting_and_preprocessing.ipynb` from Jupyter with the kernel working directory set to `notebooks/` (the notebook reads and writes data using paths relative to that directory). It creates `X_train_processed.csv`, `X_test_processed.csv`, `y_train.csv`, and `y_test.csv` in `data/processed/`. The notebook uses a stratified 80/20 split, imputes missing values, caps numeric outliers, scales numeric features, and ordinal-encodes categories.
3. **Train and compare baseline models:**

   ```powershell
   python src/models/train_models.py
   ```

   This trains Logistic Regression, Random Forest, and XGBoost models and writes performance summaries under `reports/metrics/`.
4. **Tune model hyperparameters** (optional):

   ```powershell
   python src/models/optimize_models.py
   ```

   This runs randomized cross-validation searches and writes a tuned-model summary under `reports/metrics/`.
5. **Evaluate and create the app model:**

   ```powershell
   python src/evaluation/evaluate_model.py
   ```

   This evaluates a Logistic Regression model, saves a confusion-matrix figure under `reports/figures/`, and writes `models/best_mental_health_model.joblib`. Run this step before starting the app.

## Run the Streamlit app

After generating the model artifact:

```powershell
python -m streamlit run app/streamlit_app.py
```

The app accepts the 13 survey features used by the model and displays its predicted category and model scores. It does not collect age or gender.

## Optional: Apache Airflow

`airflow/dags/student_mental_health_dag.py` defines a sequential DAG for preprocessing, baseline training, tuning, and evaluation. Apache Airflow is intentionally not included in `requirements.txt`; install it separately using the [official Airflow installation instructions](https://airflow.apache.org/docs/apache-airflow/stable/installation/index.html) and the constraints file for your Python and Airflow versions. Configure Airflow to load `airflow/dags/` and ensure task commands run with the repository root as their working directory. The DAG's preprocessing task also uses `nbconvert`, which is included in the requirements.

## Generated files and reproducibility

Processed data, trained model files, reports, and Airflow runtime files are excluded from version control. Re-run the workflow to regenerate them. Invoke the training and evaluation scripts from the repository root; the notebooks use paths relative to `notebooks/`.
