# Author: Gherga Ciprian Vlad

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import os
from typing import List

# --- Configuration ---
# Constants should match those used in train_model.py for consistency
MODEL_SAVE_PATH = 'predictive_maintenance_model.joblib'
TIMESTAMP_COLUMN = 'timestamp'
MACHINE_ID_COLUMN = 'machine_id'
NUMERICAL_FEATURES_BASE = ['sensor_1', 'sensor_2', 'sensor_3', 'pressure_setting', 'temperature_setting']
ROLLING_WINDOW_SIZE = 10 # Note: Actual rolling logic is simplified for single-point inference
LAG_PERIODS = [1, 3, 5] # Note: Actual lag logic is simplified for single-point inference

# --- Model Loading ---
# Load the trained scikit-learn pipeline model
model_pipeline = None
try:
    # Ensure the model file exists
    if not os.path.exists(MODEL_SAVE_PATH):
        # Raise a specific error if the file is not found
        raise FileNotFoundError(f"Model file not found at {MODEL_SAVE_PATH}")
    
    # Load the model pipeline
    model_pipeline = joblib.load(MODEL_SAVE_PATH)
    print(f"Model loaded successfully from {MODEL_SAVE_PATH}.")

except FileNotFoundError as e:
    print(f"FATAL ERROR: {e}")
    print("The model could not be loaded. The API /predict endpoint will not be functional.")
    print("Please ensure 'predictive_maintenance_model.joblib' exists in the same directory.")

except Exception as e:
    # Catch any other exceptions during loading
    print(f"FATAL ERROR: An unexpected error occurred while loading the model: {e}")
    print("The API /predict endpoint will not be functional.")

# --- Pydantic Model for Request Body ---
# Defines the structure and types of the incoming request payload
class SensorInput(BaseModel):
    timestamp: datetime
    machine_id: str
    sensor_1: float
    sensor_2: float
    sensor_3: float
    pressure_setting: float
    temperature_setting: float

# --- FastAPI App Instance ---
app = FastAPI()

# --- Prediction Endpoint ---
@app.post("/predict")
async def predict_failure(input_data: SensorInput):
    """
    Receives sensor data for a single point in time for a machine,
    engineers features (simplified for single-point inference),
    and predicts the probability of a failure event using the loaded model.
    """
    # Check if the model was loaded successfully during startup
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")

    # --- Data Preparation ---
    # Convert the incoming Pydantic model data into a pandas DataFrame
    # This structure is needed for feature engineering and model prediction
    data_dict = {
        TIMESTAMP_COLUMN: [input_data.timestamp],
        MACHINE_ID_COLUMN: [input_data.machine_id],
        'sensor_1': [input_data.sensor_1],
        'sensor_2': [input_data.sensor_2],
        'sensor_3': [input_data.sensor_3],
        'pressure_setting': [input_data.pressure_setting],
        'temperature_setting': [input_data.temperature_setting],
    }
    input_df = pd.DataFrame(data_dict)

    # Ensure the timestamp column is datetime type for feature extraction
    # Pydantic typically handles this, but explicitly ensuring here
    if not pd.api.types.is_datetime64_any_dtype(input_df[TIMESTAMP_COLUMN]):
         input_df[TIMESTAMP_COLUMN] = pd.to_datetime(input_df[TIMESTAMP_COLUMN])

    # --- Feature Engineering (Simplified) ---
    # Replicate the feature names created during training ('train_model.py'),
    # but calculate them based *only* on the single incoming data point
    # as per the specified simplified logic for API inference.

    # 1. Time-Based Features
    input_df['hour_of_day'] = input_df[TIMESTAMP_COLUMN].dt.hour
    input_df['day_of_week'] = input_df[TIMESTAMP_COLUMN].dt.dayofweek
    input_df['month'] = input_df[TIMESTAMP_COLUMN].dt.month

    # 2. Simplified Rolling Window Statistics (mean, std, min, max)
    # As per instruction, for a single point:
    # mean = current value
    # std = 0.0
    # min = current value
    # max = current value
    for col in NUMERICAL_FEATURES_BASE:
        input_df[f'{col}_roll_mean'] = input_df[col]
        input_df[f'{col}_roll_std'] = 0.0
        input_df[f'{col}_roll_min'] = input_df[col]
        input_df[f'{col}_roll_max'] = input_df[col]

    # 3. Simplified Lag Features
    # As per instruction, for a single point, set lag features to the current value
    for col in NUMERICAL_FEATURES_BASE:
        for lag in LAG_PERIODS:
            input_df[f'{col}_lag_{lag}'] = input_df[col]


    # --- Prepare DataFrame for Prediction ---
    # The trained pipeline (specifically the ColumnTransformer within it)
    # expects the input DataFrame to have columns with specific names.
    # Reconstruct the list of feature names that the pipeline was trained on.
    # The order should match how features were added in 'train_model.py' after
    # creating the initial dataframe from CSV, excluding timestamp and target.
    # The typical order in the final X DataFrame from train_model.py would be:
    # MACHINE_ID_COLUMN, NUMERICAL_FEATURES_BASE, Time Features, Rolling Features, Lag Features.

    expected_feature_order: List[str] = []
    expected_feature_order.append(MACHINE_ID_COLUMN) # Categorical feature
    expected_feature_order.extend(NUMERICAL_FEATURES_BASE) # Base numerical features
    expected_feature_order.extend(['hour_of_day', 'day_of_week', 'month']) # Time-based numerical features

    # Rolling features - matching the names generated in train_model.py (e.g., sensor_1_roll_mean)
    # The loop order in train_model.py was: `for col in NUMERICAL_FEATURES_BASE:` then add all stats for that col.
    for col in NUMERICAL_FEATURES_BASE:
         expected_feature_order.extend([f'{col}_roll_{stat}' for stat in ['mean', 'std', 'min', 'max']])

    # Lag features - matching the names generated in train_model.py (e.g., sensor_1_lag_1)
    # The loop order in train_model.py was: `for col in NUMERICAL_FEATURES_BASE:` then `for lag in LAG_PERIODS:`.
    for col in NUMERICAL_FEATURES_BASE:
         expected_feature_order.extend([f'{col}_lag_{lag}' for lag in LAG_PERIODS])

    # Select and reorder columns in the engineered DataFrame to match the expected order
    # This ensures the ColumnTransformer receives data in the format it expects.
    # If any generated column name doesn't match, this will raise a KeyError.
    try:
        engineered_df = input_df[expected_feature_order]
    except KeyError as e:
        print(f"Feature engineering mismatch: Missing column {e}.")
        print(f"Expected feature columns: {expected_feature_order}")
        print(f"Generated DataFrame columns: {list(input_df.columns)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: Feature mismatch ({e}).")
    except Exception as e:
         print(f"An unexpected error occurred while preparing data for prediction: {e}")
         raise HTTPException(status_code=500, detail=f"Internal server error: Data preparation failed ({e}).")


    # --- Prediction ---
    try:
        # Use the loaded pipeline to make a prediction.
        # The pipeline handles preprocessing (scaling, one-hot encoding) internally.
        prediction = model_pipeline.predict(engineered_df)[0]

        # Get the probability of the positive class (failure event, which is class 1)
        # predict_proba returns probabilities for all classes, slice to get probability of class 1
        probability = model_pipeline.predict_proba(engineered_df)[:, 1][0]

    except Exception as e:
        # Catch any exceptions during the prediction step
        print(f"Error during model prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {e}")

    # --- Return Response ---
    # Return the prediction and the probability of failure in the specified JSON format
    return {
        "prediction": int(prediction),
        "probability_failure": float(probability)
    }

# --- Optional: Root Endpoint ---
# A simple endpoint to check if the API is running
@app.get("/")
async def read_root():
    """
    Root endpoint. Returns a simple message indicating the API status.
    """
    status = "operational" if model_pipeline is not None else "model_loading_failed"
    return {"message": "Predictive Maintenance API", "status": status}

# --- How to Run ---
# 1. Make sure you have installed the necessary libraries:
#    pip install fastapi uvicorn pydantic joblib pandas numpy scikit-learn
# 2. Save the trained model file ('predictive_maintenance_model.joblib') in the same directory as this script ('main.py').
# 3. Run the application from your terminal using uvicorn:
#    uvicorn main:app --reload
# The API will be available at http://127.0.0.1:8000
# The interactive documentation (Swagger UI) will be available at http://127.0.0.1:8000/docs
