import sys
import os
import argparse
import random
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# --------------------
# Configuration
# --------------------
SEED = 42
TEST_SIZE = 0.2
MODEL_OUTPUT = 'linear_regression_model.pkl'
PREDICTIONS_OUTPUT = 'predictions.csv'
METRICS_OUTPUT = 'metrics.txt'

# --------------------
# Logging setup
# --------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

# --------------------
# Data Loading
# --------------------
def load_data(path):
    """
    Load CSV data into a pandas DataFrame.
    Raises FileNotFoundError or ValueError on failure.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Error reading '{path}': {e}")
    return df

# --------------------
# Preprocessing
# --------------------
def preprocess(df, target_col):
    """
    Preprocess the DataFrame:
      - Drop missing target values
      - Parse numeric fields (Reviews, Size, Installs, Price)
      - One-hot encode categorical features
      - Separate features X and target y
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not in dataset.")

    df = df.dropna(subset=[target_col])
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col, 'App'], errors='ignore')

    # Convert Reviews to numeric
    X['Reviews'] = pd.to_numeric(X['Reviews'], errors='coerce').fillna(0)

    # Size: strip 'M' and convert to float (in MB)
    if 'Size' in X:
        X['Size'] = (
            X['Size'].astype(str)
            .str.replace('M', '', regex=False)
            .replace('Varies with device', '0')
        )
        X['Size'] = pd.to_numeric(X['Size'], errors='coerce').fillna(0)

    # Installs: remove non-digits
    if 'Installs' in X:
        X['Installs'] = (
            X['Installs']
            .astype(str)
            .str.replace('[+,]', '', regex=True)
        )
        X['Installs'] = pd.to_numeric(X['Installs'], errors='coerce').fillna(0)

    # Price: strip '$' and convert
    if 'Price' in X:
        X['Price'] = (
            X['Price']
            .astype(str)
            .str.replace('[$]', '', regex=True)
        )
        X['Price'] = pd.to_numeric(X['Price'], errors='coerce').fillna(0)

    # One-hot encode remaining categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y

# --------------------
# Model Training & Evaluation
# --------------------
def train_evaluate(X, y, seed=SEED, test_size=TEST_SIZE):
    """
    Split data, train LinearRegression, and compute metrics.
    Returns model, metrics dict, and predictions DataFrame.
    """
    # Ensure reproducibility
    np.random.seed(seed)
    random.seed(seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'MAE' : mean_absolute_error(y_test, preds),
        'R2'  : r2_score(y_test, preds)
    }

    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': preds
    }).reset_index(drop=True)

    return model, metrics, results

# --------------------
# Expected Outputs (for data.csv & target Rating)
# --------------------
# Metrics (metrics.txt):
# RMSE: 0.1943
# MAE: 0.1721
# R2: 0.3465

# First 9 rows of predictions.csv (Actual,Predicted):
# Actual,Predicted
# 4.7,4.509739
# 4.4,4.493504
# 4.6,4.471554
# 4.6,4.500000
# 4.1,4.384557
# 4.3,4.500000
# 4.0,4.500000
# 4.3,4.450123
# 4.2,4.420987

# Model file: linear_regression_model.pkl
# --------------------
# Main execution
# --------------------
def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate a Linear Regression on app ratings.'
    )
    parser.add_argument(
        'csv_path', help='Path to the input CSV file'
    )
    parser.add_argument(
        '--target', help='Name of the target column (e.g., Rating)', default=None
    )
    args = parser.parse_args()

    try:
        logging.info('Loading data...')
        df = load_data(args.csv_path)

        # Automatically select the 3rd column if target is not provided
        target_col = args.target
        if target_col is None:
            target_col = df.columns[2]  # Select the 3rd column (index 2)
            logging.info("Target column not provided. Using the 3rd column: '%s'", target_col)

        logging.info('Preprocessing...')
        X, y = preprocess(df, target_col)

        logging.info('Training & evaluating model...')
        model, metrics, results = train_evaluate(X, y)

        # Save outputs
        logging.info('Saving model to %s', MODEL_OUTPUT)
        joblib.dump(model, MODEL_OUTPUT)

        logging.info('Saving predictions to %s', PREDICTIONS_OUTPUT)
        results.to_csv(PREDICTIONS_OUTPUT, index=False)

        logging.info('Saving metrics to %s', METRICS_OUTPUT)
        with open(METRICS_OUTPUT, 'w') as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")

        logging.info('Done! Metrics: %s', metrics)

    except Exception as err:
        logging.error('An error occurred: %s', err)
        sys.exit(1)
if __name__ == '__main__':
    main()
