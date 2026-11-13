import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score


X_train = pd.read_csv('data/processed/X_train_processed.csv')
X_test = pd.read_csv('data/processed/X_test_processed.csv')

target_map = {'Low': 0, 'Medium': 1, 'High': 2}
y_train = pd.read_csv('data/processed/y_train.csv')['Mental_Health_Risk'].map(target_map).values
y_test = pd.read_csv('data/processed/y_test.csv')['Mental_Health_Risk'].map(target_map).values


models_to_tune = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'saga']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='mlogloss'),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    }
}


tuned_results = {}
best_estimators = {}

for name, config in models_to_tune.items():
    print(f"\n--- Tuning Hyperparameters for {name} ---")
    
    search = RandomizedSearchCV(
        estimator=config['model'],
        param_distributions=config['params'],
        n_iter=10,
        cv=5,      
        scoring='f1_macro',
        random_state=42,
        n_jobs=-1           
    )
    
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    best_estimators[name] = best_model
    
    print(f"Best Parameters: {search.best_params_}")
    
    y_pred = best_model.predict(X_test)
    
    tuned_results[name] = {
        'Best CV Score (F1 Macro)': search.best_score_,
        'Test Accuracy': accuracy_score(y_test, y_pred),
        'Test F1-score (Macro)': f1_score(y_test, y_pred, average='macro')
    }
    
tuned_summary_df = pd.DataFrame(tuned_results).T
tuned_summary_df.to_csv('reports/metrics/tuned_models_performance_summary.csv', index=True)

print("\n=== TUNED MODELS PERFORMANCE COMPARISON ===")
print(tuned_summary_df)