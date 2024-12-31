import os
import pandas as pd
from scipy.stats import ks_2samp
from sqlalchemy import create_engine
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import train_test_split


# Load DSN from environment variable
def load_secrets():
    dsn = os.getenv('DSN')
    if not dsn:
        raise ValueError("DSN environment variable not found.")
    return dsn

# Fetch new data from the database
def fetch_new_data(dsn, query="SELECT * FROM predictions;"):
    try:
        engine = create_engine(dsn)
        df = pd.read_sql_query(query, engine)
        # Extract input_data JSON column into a DataFrame
        json_data = pd.json_normalize(df['input_data'])
        return json_data
    except Exception as e:
        print(f"Error querying or processing the database data: {e}")
        return None

# Perform KS Test
def perform_ks_test(original_data, new_data, columns):
    ks_results = []
    for column in columns:
        # Apply the KS test
        stat, p_value = ks_2samp(original_data[column], new_data[column])
        ks_results.append({
            "Feature": column,
            "KS Statistic": stat,
            "P-Value": p_value,
            "Drift Detected": p_value < 0.05
        })
    return pd.DataFrame(ks_results)

# Check for overall data drift
def check_overall_drift(ks_results_df, drift_threshold=0.2):
    proportion_drift = ks_results_df['Drift Detected'].mean()
    drift_detected = proportion_drift >= drift_threshold
    return drift_detected, proportion_drift

# Retraining the models if drift is detected
def retrain_and_save_models(X_train, y_train):
    models = {
        'logistic_regression': LogisticRegression(),
        'gradient_boosting': GradientBoostingClassifier(),
        'lightgbm': lgb.LGBMClassifier(),
        'xgboost': xgb.XGBClassifier(),
        'catboost': cb.CatBoostClassifier(learning_rate=0.1, iterations=1000, depth=6, cat_features=[], verbose=0)
    }

    model_paths = {
        'logistic_regression': 'deployment/models/logistic_regression.pkl',
        'gradient_boosting': 'deployment/models/gradient_boosting.pkl',
        'lightgbm': 'deployment/models/lightgbm.pkl',
        'xgboost': 'deployment/models/xgboost.pkl',
        'catboost': 'deployment/models/catboost.pkl'
    }

    for model_name, model in models.items():
        print(f"Retraining {model_name} model...")
        model.fit(X_train, y_train)
        joblib.dump(model, model_paths[model_name])
        print(f"{model_name} model saved.")

if __name__ == "__main__":
    # Load secrets and fetch DSN
    dsn = load_secrets()

    # Load original data
    original_data = pd.read_csv('./data/processed/data_cleaned.csv')

    # Fetch new data from the database
    new_data = fetch_new_data(dsn)

    if new_data is not None:
        # Columns to test
        test_columns = [
            'area_mean', 'area_se', 'area_worst', 'compactness_mean', 'compactness_worst',
            'concave points_mean', 'concave points_worst', 'concavity_mean', 'concavity_worst',
            'perimeter_mean', 'perimeter_se', 'perimeter_worst', 'radius_mean', 'radius_se',
            'radius_worst', 'smoothness_worst', 'symmetry_worst', 'texture_mean', 'texture_worst'
        ]

        new_data_features = new_data[test_columns].copy()

        # Perform KS Test
        ks_results_df = perform_ks_test(original_data, new_data_features, test_columns)

        # Check for overall data drift with drift threshold as 20%
        drift_threshold = 0.2
        drift_detected, proportion_drift = check_overall_drift(ks_results_df, drift_threshold)

        # Display results
        print("\nKolmogorov-Smirnov Test Results:")
        print(ks_results_df)

        if drift_detected:
            print("\nOverall data drift detected.")
            # Retrain models if drift is detected
            X = original_data[test_columns]
            y = original_data['diagnosis']
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            retrain_and_save_models(X_train, y_train)
        else:
            print("\nNo overall data drift detected.")

        print(f"\nProportion of features with drift: {proportion_drift:.2%}")
