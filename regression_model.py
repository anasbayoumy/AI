# import sys
# import os
# import argparse
# import random
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import joblib

# # --------------------
# # Configuration
# # --------------------
# SEED = 42
# TEST_SIZE = 0.2
# MODEL_OUTPUT = 'linear_regression_model.pkl'
# PREDICTIONS_OUTPUT = 'predictions.csv'
# METRICS_OUTPUT = 'metrics.txt'
# PLOTS_DIR = 'plots'

# # Create plots directory if it doesn't exist
# os.makedirs(PLOTS_DIR, exist_ok=True)

# # --------------------
# # Logging setup
# # --------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(levelname)s] %(message)s'
# )

# # --------------------
# # Data Loading
# # --------------------
# def load_data(path):
#     """Load CSV data into a pandas DataFrame."""
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"File not found: {path}")
#     try:
#         df = pd.read_csv(path)
#         logging.info(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
#         return df
#     except Exception as e:
#         raise ValueError(f"Error reading '{path}': {e}")

# # --------------------
# # Preprocessing
# # --------------------
# def preprocess(df, target_col):
#     """
#     Enhanced preprocessing for better linear regression performance:
#     - Handle missing values properly
#     - Transform skewed features with log transformation
#     - Create interaction features
#     - Scale numeric features
#     - Optimize categorical encodings
#     """
#     if target_col not in df.columns:
#         raise KeyError(f"Target column '{target_col}' not in dataset.")

#     # Create a copy to avoid modifying the original
#     df = df.copy()
    
#     # Verify target column looks like ratings (0-5 range)
#     target_values = pd.to_numeric(df[target_col], errors='coerce')
#     valid_ratings = target_values.between(0, 5)
    
#     if valid_ratings.sum() / len(df) < 0.5:  # If less than 50% of values are in 0-5 range
#         logging.warning(f"WARNING: Target column '{target_col}' doesn't look like ratings (0-5 range).")
#         logging.warning(f"Target min: {target_values.min()}, max: {target_values.max()}")
        
#         # Find column that looks more like ratings
#         for col in df.columns:
#             if col != target_col:
#                 test_values = pd.to_numeric(df[col], errors='coerce')
#                 if test_values.between(0, 5).sum() / len(df) > 0.7:  # 70% in range
#                     logging.warning(f"Column '{col}' looks more like ratings (0-5 range).")
#                     logging.warning(f"Consider using: --target {col}")
    
#     # Drop missing target values
#     df = df.dropna(subset=[target_col])
#     y = df[target_col].astype(float)
    
#     # Ensure target is within 0-5 range
#     if y.min() < 0 or y.max() > 5:
#         logging.warning(f"Target values outside expected rating range: min={y.min()}, max={y.max()}")
#         logging.warning("Constraining target values to 0-5 range")
#         y = y.clip(0, 5)  # Constrain values to 0-5 range
    
#     logging.info(f"Target values - min: {y.min()}, max: {y.max()}, mean: {y.mean():.2f}")
    
#     # Drop non-feature columns
#     X = df.drop(columns=[target_col, 'App'], errors='ignore')
    
#     # --- Feature Engineering for numeric columns ---
    
#     # Reviews: Log transformation to handle skewness
#     if 'Reviews' in X:
#         X['Reviews'] = pd.to_numeric(X['Reviews'], errors='coerce').fillna(0)
#         X['Log_Reviews'] = np.log1p(X['Reviews'])  # log(1+x) handles zeros well
    
#     # Size: Handle size properly
#     if 'Size' in X:
#         # Extract numeric size and convert to MB
#         X['Size'] = (
#             X['Size'].astype(str)
#             .str.replace('M', '', regex=False)
#             .replace('Varies with device', '0')
#         )
#         X['Size'] = pd.to_numeric(X['Size'], errors='coerce').fillna(0)
    
#     # Installs: Log transformation for better distribution
#     if 'Installs' in X:
#         X['Installs'] = (
#             X['Installs']
#             .astype(str)
#             .str.replace('[+,]', '', regex=True)
#         )
#         X['Installs'] = pd.to_numeric(X['Installs'], errors='coerce').fillna(0)
#         X['Log_Installs'] = np.log1p(X['Installs'])
    
#     # Price: Create binary feature for free vs paid
#     if 'Price' in X:
#         X['Price'] = (
#             X['Price']
#             .astype(str)
#             .str.replace('[$]', '', regex=True)
#         )
#         X['Price'] = pd.to_numeric(X['Price'], errors='coerce').fillna(0)
#         X['Is_Free'] = (X['Price'] == 0).astype(int)
    
#     # --- Extract meaningful info from categorical variables ---
    
#     # Content Rating: Map to age restrictions
#     if 'Content Rating' in X:
#         content_rating_map = {
#             'Everyone': 0,
#             'Everyone 10+': 10,
#             'Teen': 13,
#             'Mature 17+': 17,
#             'Adults only 18+': 18,
#             'Unrated': 0
#         }
#         X['Age_Restriction'] = X['Content Rating'].map(
#             lambda x: content_rating_map.get(str(x), 0)
#         )
    
#     # Android Version: Extract major version number
#     if 'Android Ver' in X:
#         X['Android_Major_Ver'] = X['Android Ver'].str.extract(r'(\d+\.\d+)').astype(float)
#         X['Android_Major_Ver'] = X['Android_Major_Ver'].fillna(X['Android_Major_Ver'].median())
    
#     # --- Create interaction features ---
    
#     # Reviews per Install (engagement ratio)
#     if all(col in X.columns for col in ['Reviews', 'Installs']):
#         X['Reviews_per_Install'] = X['Reviews'] / (X['Installs'] + 1)
    
#     # --- One-hot encode remaining categorical columns ---
#     cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
#     # For categories with many levels, keep only the top N to avoid dimensionality explosion
#     for col in cat_cols:
#         if col == 'Category' and col in X.columns:
#             # Keep top categories, group others
#             top_categories = X[col].value_counts().nlargest(15).index
#             X[col] = np.where(X[col].isin(top_categories), X[col], 'Other')
    
#     # One-hot encode categorical features
#     X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
#     # Drop any remaining NaN values
#     X = X.fillna(0)  
    
#     logging.info(f"Preprocessing complete. Features shape: {X.shape}")
#     return X, y

# # --------------------
# # Model Training & Evaluation
# # --------------------
# def train_evaluate(X, y, seed=SEED, test_size=TEST_SIZE):
#     """
#     Split data, train LinearRegression with a proper pipeline, and compute metrics.
#     Returns model, metrics dict, and predictions DataFrame.
#     """
#     # Ensure reproducibility
#     np.random.seed(seed)
#     random.seed(seed)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=seed
#     )
    
#     # Use a pipeline to properly scale features first
#     model = Pipeline([
#         ('scaler', StandardScaler()),  # Scale features for better performance
#         ('regression', LinearRegression())
#     ])
    
#     # Train model
#     model.fit(X_train, y_train)
    
#     # Generate predictions
#     preds = model.predict(X_test)
    
#     # Constrain predictions to valid range for app ratings (0-5)
#     preds = np.clip(preds, 0, 5)
    
#     # Calculate metrics
#     metrics = {
#         'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
#         'MAE': mean_absolute_error(y_test, preds),
#         'R2': r2_score(y_test, preds)
#     }
    
#     # Create results DataFrame
#     results = pd.DataFrame({
#         'Actual': y_test,
#         'Predicted': preds
#     }).reset_index(drop=True)
    
#     # Visualize results
#     visualize_results(y_test, preds)
    
#     return model, metrics, results

# # --------------------
# # Visualization
# # --------------------
# def visualize_results(y_true, y_pred):
#     """Create helpful visualizations of model results."""
#     # 1. Actual vs Predicted Plot
#     plt.figure(figsize=(10, 6))
#     plt.scatter(y_true, y_pred, alpha=0.5)
    
#     # Add perfect prediction line
#     min_val = min(y_true.min(), y_pred.min())
#     max_val = max(y_true.max(), y_pred.max())
#     plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
#     plt.title('Actual vs Predicted App Ratings')
#     plt.xlabel('Actual Rating (0-5)')
#     plt.ylabel('Predicted Rating (0-5)')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"{PLOTS_DIR}/actual_vs_predicted.png")
#     plt.close()
    
#     # 2. Residual Plot
#     residuals = y_true - y_pred
#     plt.figure(figsize=(10, 6))
#     plt.scatter(y_pred, residuals, alpha=0.5)
#     plt.axhline(y=0, color='r', linestyle='--')
#     plt.title('Residuals vs Predicted Values')
#     plt.xlabel('Predicted Rating')
#     plt.ylabel('Residuals')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"{PLOTS_DIR}/residuals.png")
#     plt.close()
    
#     # 3. Residual Distribution
#     plt.figure(figsize=(10, 6))
#     sns.histplot(residuals, kde=True)
#     plt.title('Distribution of Residuals')
#     plt.xlabel('Residual Value')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"{PLOTS_DIR}/residual_distribution.png")
#     plt.close()
    
#     logging.info(f"Saved visualizations to {PLOTS_DIR}/")

# # --------------------
# # Main execution
# # --------------------
# def main():
#     parser = argparse.ArgumentParser(
#         description='Train and evaluate a Linear Regression on app ratings (0-5 scale).'
#     )
#     parser.add_argument(
#         'csv_path', help='Path to the input CSV file'
#     )
#     parser.add_argument(
#         '--target', help='Name of the target column containing app ratings (0-5 scale)', default=None
#     )
#     parser.add_argument(
#         '--show-columns', action='store_true', help='Show all column names in the dataset'
#     )
#     args = parser.parse_args()

#     try:
#         logging.info('Loading data...')
#         df = load_data(args.csv_path)
        
#         if args.show_columns:
#             logging.info("Columns in dataset:")
#             for i, col in enumerate(df.columns):
#                 # For each column, show basic stats if numeric
#                 try:
#                     numeric_vals = pd.to_numeric(df[col], errors='coerce')
#                     stats = f"min={numeric_vals.min():.2f}, max={numeric_vals.max():.2f}, mean={numeric_vals.mean():.2f}"
#                 except:
#                     stats = "(non-numeric)"
#                 logging.info(f"{i}: {col} - {stats}")
#             return

#         # Automatically select the column named 'Rating' if target is not provided
#         target_col = args.target
#         if target_col is None:
#             if 'Rating' in df.columns:
#                 target_col = 'Rating'
#                 logging.info(f"Target column not provided. Using column named 'Rating'")
#             else:
#                 # Try to find a column with values mostly in the 0-5 range
#                 for col in df.columns:
#                     try:
#                         vals = pd.to_numeric(df[col], errors='coerce')
#                         if vals.between(0, 5).sum() / len(df) > 0.7:  # 70% of values in range
#                             target_col = col
#                             logging.info(f"Target column not provided. Using column '{col}' with values in 0-5 range")
#                             break
#                     except:
#                         continue
                
#                 # If still no target, use the 3rd column
#                 if target_col is None:
#                     if len(df.columns) > 2:
#                         target_col = df.columns[2]
#                         logging.info(f"Target column not provided. Using the 3rd column: '{target_col}'")
#                     else:
#                         raise ValueError("Could not automatically determine the target column")

#         logging.info('Preprocessing...')
#         X, y = preprocess(df, target_col)

#         logging.info('Training & evaluating model...')
#         model, metrics, results = train_evaluate(X, y)

#         # Save outputs
#         logging.info(f"Saving model to {MODEL_OUTPUT}")
#         joblib.dump(model, MODEL_OUTPUT)

#         logging.info(f"Saving predictions to {PREDICTIONS_OUTPUT}")
#         results.to_csv(PREDICTIONS_OUTPUT, index=False)

#         logging.info(f"Saving metrics to {METRICS_OUTPUT}")
#         with open(METRICS_OUTPUT, 'w') as f:
#             for k, v in metrics.items():
#                 f.write(f"{k}: {v:.4f}\n")

#         logging.info(f"Done! Metrics: {metrics}")

#     except Exception as err:
#         logging.error(f"An error occurred: {err}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)

# if __name__ == '__main__':
#     main()

