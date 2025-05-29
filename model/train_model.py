# Author: Gherga Ciprian Vlad

"""
train_model.py

This script loads predictive maintenance sensor data, preprocesses it,
engineers features, trains a machine learning model, evaluates it,
and saves the trained model.

It is conceptually aligned with the findings and ideas from the
accompanying Exploratory Data Analysis (EDA) notebook.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import os

# --- Configuration ---
DATA_PATH = 'sensor_data.csv'
MODEL_SAVE_PATH = 'predictive_maintenance_model.joblib'
TARGET_COLUMN = 'failure_event_occurred'
TIMESTAMP_COLUMN = 'timestamp'
MACHINE_ID_COLUMN = 'machine_id'

# Define sensor columns (example based on EDA; adjust if actual data differs)
# Assuming all other numerical columns not listed as timestamp/machine_id/target are sensors/operational
NUMERICAL_FEATURES_BASE = ['sensor_1', 'sensor_2', 'sensor_3', 'pressure_setting', 'temperature_setting'] # Example list
CATEGORICAL_FEATURES = [MACHINE_ID_COLUMN]

# Feature Engineering Parameters
ROLLING_WINDOW_SIZE = 10 # Number of data points for rolling window (adjust based on time frequency)
LAG_PERIODS = [1, 3, 5] # Number of previous data points for lag features

# --- Data Loading ---
print(f"Loading data from {DATA_PATH}...")
try:
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: {DATA_PATH} not found. Creating a dummy DataFrame for demonstration.")
    # Create a dummy DataFrame based on EDA description if file is missing
    data = {
        TIMESTAMP_COLUMN: pd.to_datetime(pd.date_range(start='2023-01-01', periods=5000, freq='10min')),
        MACHINE_ID_COLUMN: np.random.choice(['M1', 'M2', 'M3', 'M4', 'M5'], 5000),
        'sensor_1': np.random.rand(5000) * 100 + np.sin(np.arange(5000)/100) * 20,
        'sensor_2': np.random.rand(5000) * 50 + 20 + np.cos(np.arange(5000)/50) * 15,
        'sensor_3': np.random.rand(5000) * 10 + 5,
        'pressure_setting': np.random.rand(5000) * 10 + 1,
        'temperature_setting': np.random.rand(5000) * 30 + 150,
        TARGET_COLUMN: np.random.choice([0, 1], 5000, p=[0.95, 0.05]) # Simulate imbalance
    }
    df = pd.DataFrame(data)
    # Introduce some missing values and potential outliers in dummy data
    for col in NUMERICAL_FEATURES_BASE:
         df.loc[np.random.choice(df.index, int(len(df)*0.01)), col] = np.nan
    df.loc[np.random.choice(df.index, int(len(df)*0.005)), 'sensor_1'] = np.random.rand(int(len(df)*0.005)) * 500 + 200 # Outliers
    print("Created a dummy DataFrame for demonstration.")


# --- Data Preprocessing: Initial Steps ---
print("Performing initial data preprocessing...")

# 1. Ensure timestamp is datetime and sort data
if pd.api.types.is_datetime64_any_dtype(df[TIMESTAMP_COLUMN]):
    print(f"'{TIMESTAMP_COLUMN}' column is already datetime type.")
else:
    print(f"Converting '{TIMESTAMP_COLUMN}' column to datetime type.")
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])

# Sort data by timestamp and machine_id for time-series operations
df.sort_values(by=[MACHINE_ID_COLUMN, TIMESTAMP_COLUMN], inplace=True)
print("Data sorted by machine ID and timestamp.")

# 2. Handle missing values (using ffill within each machine group)
print("Handling missing values using forward fill within machine groups...")
# Identify numerical columns including the target if it's int/float
numerical_cols_all = df.select_dtypes(include=np.number).columns.tolist()
# Ensure target is handled if it's a numerical type
if TARGET_COLUMN in numerical_cols_all:
    numerical_cols_all.remove(TARGET_COLUMN)

for col in numerical_cols_all:
    df[col] = df.groupby(MACHINE_ID_COLUMN)[col].fillna(method='ffill')

# After ffill, there might still be NaNs at the beginning of each machine's data
# or if a machine has only NaNs. Handle remaining NaNs, e.g., with backward fill
for col in numerical_cols_all:
    df[col] = df.groupby(MACHINE_ID_COLUMN)[col].fillna(method='bfill')

# If any NaNs still exist (e.g., if a column is all NaNs), fill with 0 or mean as a fallback
for col in numerical_cols_all:
     if df[col].isnull().any():
         df[col].fillna(df[col].mean(), inplace=True)
         print(f"Fallback: Filled remaining NaNs in {col} with mean.")

print(f"Missing values after initial handling:\n{df.isnull().sum()}")


# --- Feature Engineering ---
print("\nPerforming feature engineering...")

# 1. Time-Based Features (Extract from timestamp)
df['hour_of_day'] = df[TIMESTAMP_COLUMN].dt.hour
df['day_of_week'] = df[TIMESTAMP_COLUMN].dt.dayofweek
df['month'] = df[TIMESTAMP_COLUMN].dt.month
print("Created time-based features (hour, day of week, month).")

# 2. Rolling Window Statistics (Grouped by machine)
print(f"Creating rolling window features (window={ROLLING_WINDOW_SIZE})...")
for col in NUMERICAL_FEATURES_BASE:
    if col in df.columns:
        df[f'{col}_roll_mean'] = df.groupby(MACHINE_ID_COLUMN)[col].transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean())
        df[f'{col}_roll_std'] = df.groupby(MACHINE_ID_COLUMN)[col].transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).std())
        df[f'{col}_roll_min'] = df.groupby(MACHINE_ID_COLUMN)[col].transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).min())
        df[f'{col}_roll_max'] = df.groupby(MACHINE_ID_COLUMN)[col].transform(lambda x: x.rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).max())
print("Created rolling window features for base numerical columns.")

# 3. Lag Features (Grouped by machine)
print(f"Creating lag features (lags={LAG_PERIODS})...")
for col in NUMERICAL_FEATURES_BASE:
    if col in df.columns:
        for lag in LAG_PERIODS:
            df[f'{col}_lag_{lag}'] = df.groupby(MACHINE_ID_COLUMN)[col].shift(lag)
print("Created lag features for base numerical columns.")

# --- Prepare Data for Modeling ---
print("\nPreparing data for modeling...")

# Drop rows that have NaNs as a result of rolling window/lag features at the beginning
# Also drop any remaining NaNs (should be minimal after ffill/bfill)
initial_rows = df.shape[0]
df.dropna(subset=[TARGET_COLUMN] + [col for col in df.columns if '_roll_' in col or '_lag_' in col], inplace=True)
rows_dropped_fe = initial_rows - df.shape[0]
print(f"Dropped {rows_dropped_fe} rows due to NaNs introduced by feature engineering.")
print(f"Remaining rows: {df.shape[0]}")

# Define features (X) and target (y)
# Exclude original timestamp and target column from features
feature_columns = [col for col in df.columns if col not in [TIMESTAMP_COLUMN, TARGET_COLUMN]]

X = df[feature_columns]
y = df[TARGET_COLUMN]

# Identify final numerical and categorical columns after feature engineering
# Time-based features are numerical, new rolling/lag features are numerical
final_numerical_features = X.select_dtypes(include=np.number).columns.tolist()
final_categorical_features = X.select_dtypes(include='object').columns.tolist() # Should only contain machine_id

print(f"\nFeatures ({X.shape[1]}): {list(X.columns)}")
print(f"Target variable: '{TARGET_COLUMN}'")
print(f"Numerical features ({len(final_numerical_features)}): {final_numerical_features}")
print(f"Categorical features ({len(final_categorical_features)}): {final_categorical_features}")

# --- Data Splitting ---
print("\nSplitting data into training and testing sets (time-based split)...")

# Sort data by timestamp before splitting to ensure time-based separation
# Data is already sorted from initial steps, but re-sorting here confirms
df.sort_values(by=TIMESTAMP_COLUMN, inplace=True)
split_index = int(len(df) * 0.8) # 80% for training, 20% for testing

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"Training set size: {len(X_train)} rows")
print(f"Testing set size: {len(X_test)} rows")
print(f"Training set time range: {df[TIMESTAMP_COLUMN].iloc[0]} to {df[TIMESTAMP_COLUMN].iloc[split_index-1]}")
print(f"Testing set time range: {df[TIMESTAMP_COLUMN].iloc[split_index]} to {df[TIMESTAMP_COLUMN].iloc[-1]}")


# --- Preprocessing Pipeline ---
# Create a column transformer to apply different steps to different column types
# StandardScaler for numerical, OneHotEncoder for categorical
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), final_numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), final_categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any, though there shouldn't be after defining features)
)

# --- Model Selection and Training ---
print("\nTraining the model (RandomForestClassifier)...")

# Create the full pipeline: preprocess data then train the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')) # Added class_weight for imbalance
])

# Train the model on the training data
model_pipeline.fit(X_train, y_train)
print("Model training completed.")

# --- Model Evaluation ---
print("\nEvaluating the model on the test set...")

y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1] # Probability of the positive class (1)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n--- Model Evaluation Metrics ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("----------------------------")

# Check prevalence in test set
test_prevalence = y_test.sum() / len(y_test)
print(f"Prevalence of failure event in test set: {test_prevalence:.4f}")


# --- Model Saving ---
print(f"\nSaving the trained model to {MODEL_SAVE_PATH}...")

# Ensure the directory exists
model_dir = os.path.dirname(MODEL_SAVE_PATH)
if model_dir and not os.path.exists(model_dir):
    os.makedirs(model_dir)

joblib.dump(model_pipeline, MODEL_SAVE_PATH)
print("Model saved successfully.")

print("\nScript execution finished.")
