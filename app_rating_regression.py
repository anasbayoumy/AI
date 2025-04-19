import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)

def preprocess_size(size_str):
    """Convert size string to numeric value in MB"""
    if pd.isna(size_str):
        return np.nan
    size_str = str(size_str).upper()
    if 'M' in size_str:
        return float(size_str.replace('M', ''))
    elif 'K' in size_str:
        return float(size_str.replace('K', '')) / 1024
    else:
        return np.nan

def preprocess_installs(installs_str):
    """Convert installs string to numeric value"""
    if pd.isna(installs_str):
        return np.nan
    installs_str = str(installs_str)
    installs_str = installs_str.replace(',', '').replace('+', '')
    if installs_str.isdigit():
        return float(installs_str)
    return np.nan

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Basic preprocessing steps
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Convert Size to numeric
    df['Size'] = df['Size'].apply(preprocess_size)
    
    # Convert Installs to numeric
    df['Installs'] = df['Installs'].apply(preprocess_installs)
    
    # Convert Price to numeric (handle the case where Price is in the wrong column)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    # Handle missing values
    df = df.dropna()
    
    # Convert categorical variables
    categorical_features = ['Category', 'Type', 'Content Rating']
    numerical_features = ['Reviews', 'Size', 'Installs', 'Price']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return df, preprocessor

def train_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate multiple regression models
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=SEED),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=SEED),
        'XGBoost': xgb.XGBRegressor(random_state=SEED),
        'LightGBM': lgb.LGBMRegressor(random_state=SEED),
        'CatBoost': cb.CatBoostRegressor(random_state=SEED, verbose=False)
    }
    
    results = {}
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'R2': r2,
            'model': model
        }
        
        print(f"{name}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  R2 Score: {r2:.4f}")
    
    return results

def create_visualizations(df, results):
    """
    Create visualizations for analysis
    """
    # Create directory for visualizations
    import os
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Distribution of ratings
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Rating'], kde=True)
    plt.title('Distribution of App Ratings')
    plt.savefig('visualizations/rating_distribution.png')
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('visualizations/correlation_heatmap.png')
    plt.close()
    
    # Model performance comparison
    plt.figure(figsize=(12, 6))
    model_names = list(results.keys())
    mse_scores = [results[name]['MSE'] for name in model_names]
    plt.bar(model_names, mse_scores)
    plt.xticks(rotation=45)
    plt.title('Model Performance (MSE)')
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.savefig('visualizations/model_performance.png')
    plt.close()

def main():
    # Load and preprocess data
    df, preprocessor = load_and_preprocess_data('test_googleplaystore.csv')
    
    # Prepare features and target
    X = df.drop(['Rating', 'App'], axis=1)  # Drop App name as it's not a feature
    y = df['Rating']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    
    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train and evaluate models
    results = train_models(X_train_processed, y_train, X_test_processed, y_test)
    
    # Create visualizations
    create_visualizations(df, results)
    
    # Save the best models
    sorted_results = sorted(results.items(), key=lambda x: x[1]['MSE'])
    for i, (name, result) in enumerate(sorted_results[:2], 1):
        import joblib
        joblib.dump(result['model'], f'best_model_{i}_{name.replace(" ", "_")}.joblib')
        joblib.dump(preprocessor, f'preprocessor_{i}.joblib')

if __name__ == "__main__":
    main() 