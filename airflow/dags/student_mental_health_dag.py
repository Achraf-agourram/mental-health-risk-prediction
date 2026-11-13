from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id='student_mental_health_pipeline',
    default_args={'retries': 1, 'retry_delay': timedelta(minutes=5)},
    description='Pipeline end-to-end pour la prédiction du risque de santé mentale des étudiants',
    schedule_interval=None,
    start_date=datetime(2026, 11, 13),
    catchup=False,
) as dag:

    preprocess_task = BashOperator(
        task_id='preprocessing_step',
        bash_command='python -m nbconvert --to notebook --execute --inplace notebooks/spliting_and_preprocessing.ipynb',
    )

    train_task = BashOperator(
        task_id='train_models_step',
        bash_command='python src/models/train_models.py',
    )

    optimize_task = BashOperator(
        task_id='optimize_models_step',
        bash_command='python src/models/optimize_models.py',
    )

    evaluate_task = BashOperator(
        task_id='evaluate_model_step',
        bash_command='python src/evaluation/evaluate_model.py',
    )

    preprocess_task >> train_task >> optimize_task >> evaluate_task