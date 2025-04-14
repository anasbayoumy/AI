# App Rating Prediction Competition

This project aims to predict app ratings on the Google Play Store using various regression models. The solution includes data preprocessing, feature engineering, and multiple regression models to achieve the best prediction accuracy.

## Project Structure

- `app_rating_regression.py`: Main script containing the implementation
- `requirements.txt`: Required Python packages
- `visualizations/`: Directory containing generated plots and analysis
- `best_model_*.joblib`: Saved best performing models
- `preprocessor_*.joblib`: Saved preprocessing pipelines

## Features

- Multiple regression models:
  - Linear Regression
  - Ridge Regression
  - Random Forest
  - XGBoost
  - LightGBM
  - CatBoost
- Data preprocessing and feature engineering
- Model evaluation and comparison
- Visualization of results
- Reproducible results with fixed random seeds

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your dataset file (`googleplaystore.csv`) in the project directory
2. Run the main script:
```bash
python app_rating_regression.py
```

## Output

The script will:
1. Preprocess the data
2. Train multiple regression models
3. Generate visualizations in the `visualizations/` directory
4. Save the best performing models and their preprocessing pipelines

## Model Selection

The script automatically selects and saves the top 2 performing models based on Mean Squared Error (MSE). These models can be used for future predictions.

## Reproducibility

All models use a fixed random seed (SEED = 42) to ensure reproducible results. 