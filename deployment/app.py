import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import json


# Models paths
model_paths = {
    'Logistic Regression': 'models/logistic_regression.pkl',
    'Gradient Boosting': 'models/gradient_boosting.pkl',
    'LightGBM': 'models/lightgbm.pkl',
    'XGBoost': 'models/xgboost.pkl',
    'Catboost': 'models/catboost.pkl',
}

model_names = list(model_paths.keys())

# Streamlit app setup
st.set_page_config(page_title="Breast Cancer Prediction App", layout="wide")
st.title("Breast Cancer Prediction App")
st.markdown("This app predicts whether a tumor is **benign** or **malignant** using a variety of models.")

# Navigation buttons at the top
col1, col2 = st.columns([1, 3])
with col1:
    manual_input_button = st.button("Manual Input")
with col2:
    csv_upload_button = st.button("Upload CSV")

## Page Navigation Logic
if 'page' not in st.session_state:
    st.session_state.page = "Manual Input"

if manual_input_button:
    st.session_state.page = "Manual Input"
elif csv_upload_button:
    st.session_state.page = "Upload CSV"

# Model selection
selected_model_name = st.sidebar.selectbox("Select Model", model_names)
model = joblib.load(model_paths[selected_model_name])

# Feature names
features = [
    'area_mean', 'area_se', 'area_worst', 'compactness_mean',
    'compactness_worst', 'concave points_mean', 'concave points_worst',
    'concavity_mean', 'concavity_worst', 'perimeter_mean', 'perimeter_se',
    'perimeter_worst', 'radius_mean', 'radius_se', 'radius_worst',
    'smoothness_worst', 'symmetry_worst', 'texture_mean', 'texture_worst'
]

# PostgreSQL DSN
dsn = st.secrets["DSN"]

# Connect to PostgreSQL database
conn = psycopg2.connect(dsn)
cursor = conn.cursor()

# Create table to store predictions and input data
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50),
    prediction VARCHAR(10),
    prediction_proba_benign FLOAT,
    prediction_proba_malignant FLOAT,
    input_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")
# Commit table creation
conn.commit()

# Manual Input Page
if st.session_state.page == "Manual Input":
    # Create two columns for visualizations
    left_column, right_column = st.columns(2)

    # Display Global Feature Importance
    with left_column:
        st.subheader("Global Feature Importance")

        # For Logistic Regression
        if selected_model_name == 'Logistic Regression':
            global_importance = np.abs(model.coef_[0])
        # For tree-based models
        elif selected_model_name in ['Gradient Boosting', 'LightGBM', 'XGBoost', 'Catboost']:
            global_importance = model.feature_importances_

        # Create a DF for visualization
        global_importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': global_importance
        })
        # Sort the global importance
        global_importance_df_sorted = global_importance_df.sort_values(by='Importance', ascending=True)

        # Plot the global feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(global_importance_df_sorted['Feature'], global_importance_df_sorted['Importance'], color='lightblue')
        plt.xlabel('Feature Importance')
        plt.title(f'Global Feature Importance ({selected_model_name})')
        st.pyplot(plt.gcf())

    # Model Explanation
    with right_column:
        st.subheader("Model Explanation")
        st.markdown("Enter feature values in the sidebar and click **Predict** to see an explanation.")

    # Sidebar input form
    st.sidebar.header("Input Tumor Features")
    input_data = {}
    for feature in features:
        input_data[feature] = st.sidebar.number_input(
            f"Enter {feature}:", value=0.0, format="%.2f"
        )

    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction and display result
    if st.sidebar.button("Predict"):
        # Get prediction and prediction probabilities
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df) if hasattr(model, "predict_proba") else None

        with right_column:
            st.subheader("Prediction Results")
            if prediction[0] == 1:
                prediction_result = "Malignant"
                st.success("The tumor is **Malignant**.")
            else:
                prediction_result = "Benign"
                st.success("The tumor is **Benign**.")

            if prediction_proba is not None:
                prediction_proba_benign = prediction_proba[0][0]
                prediction_proba_malignant = prediction_proba[0][1]
                # Display the prediction probabilities for both classes (Benign and Malignant)
                st.write(f"Prediction Probability: {prediction_proba_benign:.2f} (Benign)")
                st.write(f"Prediction Probability: {prediction_proba_malignant:.2f} (Malignant)")

            # Logistic Regression Explanation
            if selected_model_name == 'Logistic Regression':
                coefs = model.coef_[0]
                intercept = model.intercept_[0]
                log_odds = intercept + np.dot(input_df, coefs)
                probability = 1 / (1 + np.exp(-log_odds))

                # Plot feature contributions for Logistic Regression
                feature_contributions = pd.DataFrame({
                    'Feature': features,
                    'Coefficient': coefs,
                    'Value': input_df.iloc[0].values,
                    'Contribution': coefs * input_df.iloc[0].values
                })

                # Plot feature contributions
                plt.figure(figsize=(10, 6))
                feature_contributions_sorted = feature_contributions.sort_values(by='Contribution', ascending=False)
                plt.barh(feature_contributions_sorted['Feature'], feature_contributions_sorted['Contribution'])
                plt.xlabel('Feature Contribution')
                plt.title('Feature Contributions to the Malignant Prediction (Logistic Regression)')
                st.pyplot(plt.gcf())

            # For Tree-Based Models
            else:
                # Plot feature contributions for Tree-Based Models
                feature_importances = model.feature_importances_
                feature_contributions = pd.DataFrame({
                    'Feature': features,
                    'Importance': feature_importances,
                    'Value': input_df.iloc[0].values
                })
                feature_contributions['Contribution'] = feature_contributions['Importance'] * feature_contributions[
                    'Value']

                # Plot feature contributions
                plt.figure(figsize=(10, 6))
                feature_contributions_sorted = feature_contributions.sort_values(by='Contribution', ascending=False)
                plt.barh(feature_contributions_sorted['Feature'], feature_contributions_sorted['Contribution'])
                plt.xlabel('Feature Contribution')
                plt.title('Feature Contributions to the Malignant Prediction (Tree-Based Models)')
                st.pyplot(plt.gcf())

        # Store prediction in PostgreSQL
        cursor.execute(
            """
            INSERT INTO predictions (model_name, prediction, prediction_proba_benign, prediction_proba_malignant, input_data)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (
                selected_model_name,
                prediction_result,
                float(prediction_proba_benign) if prediction_proba_benign is not None else None,
                float(prediction_proba_malignant) if prediction_proba_malignant is not None else None,
                json.dumps(input_data),
            )
        )
        # Commit the data insertion
        conn.commit()

# Page for CSV upload
elif st.session_state.page == "Upload CSV":
    st.subheader("Upload CSV for Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Check the uploaded file correct
        if all(col in df.columns for col in features):
            # Make predictions
            predictions = model.predict(df[features])
            prediction_proba = model.predict_proba(df[features]) if hasattr(model, "predict_proba") else None

            # Add the predictions and probabilities to the df
            df['prediction'] = predictions
            df['prediction_label'] = df['prediction'].apply(lambda x: 'Malignant' if x == 1 else 'Benign')
            df['model'] = selected_model_name

            if prediction_proba is not None:
                df['prediction_proba_benign'] = prediction_proba[:, 0]
                df['prediction_proba_malignant'] = prediction_proba[:, 1]

            # Display the df with predictions
            st.write(df)

            # Button to download predictions as a CSV file
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

            # Display feature importance visualization
            st.subheader("Global Feature Importance")
            if selected_model_name in ['Gradient Boosting', 'LightGBM', 'XGBoost', 'Catboost']:
                global_importance = model.feature_importances_
            else:
                global_importance = np.abs(model.coef_[0])

            global_importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': global_importance
            })
            global_importance_df_sorted = global_importance_df.sort_values(by='Importance', ascending=True)

            plt.figure(figsize=(10, 6))
            plt.barh(global_importance_df_sorted['Feature'], global_importance_df_sorted['Importance'],
                     color='lightblue')
            plt.xlabel('Feature Importance')
            plt.title(f'Global Feature Importance ({selected_model_name})')
            st.pyplot(plt.gcf())

            # Save features and predictions to the PostgreSQL database
            for idx, row in df.iterrows():
                input_data = row[features].to_dict()
                cursor.execute(
                    """
                    INSERT INTO predictions (model_name, prediction, prediction_proba_benign, prediction_proba_malignant, input_data)
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    (
                        selected_model_name,
                        row['prediction_label'],
                        float(row['prediction_proba_benign']) if 'prediction_proba_benign' in row else None,
                        float(row['prediction_proba_malignant']) if 'prediction_proba_malignant' in row else None,
                        json.dumps(input_data),
                    )
                )
            # Commit data insertion
            conn.commit()

# Close database connection
conn.close()
