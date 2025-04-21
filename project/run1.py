# app_rating_stacked_linear.py

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

SEED = 42

# ---------- Parsing Helpers ----------
def parse_size(s):
    if pd.isna(s) or s=='Varies with device': return np.nan
    if s.endswith('M'): return float(s[:-1]) * 1024
    if s.endswith('k'): return float(s[:-1])
    return np.nan

# ---------- Cleaning ----------
def clean_train(df):
    df = df.dropna(subset=['X1','X2','X3','X4','X5','X7','X8','Y'])
    df['X3'] = df['X3'].apply(parse_size)
    df = df.dropna(subset=['X3'])
    df['X2'] = df['X2'].astype(int)
    df['X4'] = df['X4'].str.replace('[+,]','',regex=True).astype(int)
    df['X6'] = df['X6'].str.replace('$','',regex=False).astype(float)
    df = df[(df['Y']>=1.0)&(df['Y']<=5.0)]
    df = df[df['X2']<=df['X4']]
    df.loc[df['X5']=='Free','X6'] = 0.0
    return df[df['X6']>=0.0].reset_index(drop=True)

def clean_test(df, med_size, med_price):
    df['X3'] = df['X3'].apply(parse_size)
    df['X3'].fillna(med_size, inplace=True)
    df['X2'] = df['X2'].astype(int)
    df['X4'] = df['X4'].str.replace('[+,]','',regex=True).astype(int)
    df['X6'] = df['X6'].str.replace('$','',regex=False).astype(float)
    df['X6'].fillna(med_price, inplace=True)
    return df

# ---------- Feature Engineering ----------
def feature_engineer(train, test):
    tr = clean_train(train.copy())
    med_size, med_price = tr['X3'].median(), tr['X6'].median()

    te = clean_test(test.copy(), med_size, med_price)
    y = tr['Y'].astype(float)
    tr = tr.drop(columns=['Y'])
    tr['__is_train'] = 1
    te['__is_train'] = 0
    full = pd.concat([tr, te], ignore_index=True)

    # Log-transforms
    full['log_reviews'] = np.log1p(full['X2'])
    full['log_installs'] = np.log1p(full['X4'])

    # One-hot encode all categorical columns
    categorical_cols = ['X1', 'X5', 'X7', 'X8']
    full = pd.get_dummies(full, columns=categorical_cols, drop_first=True)

    # Drop unused columns
    drop_cols = ['X0', 'X2', 'X3', 'X4', 'X9', 'X10', 'X11']
    full.drop(columns=[c for c in drop_cols if c in full.columns], inplace=True)

    X_tr = full[full['__is_train'] == 1].drop(columns=['__is_train'])
    X_te = full[full['__is_train'] == 0].drop(columns=['__is_train'])
    return X_tr, y, X_te

# ---------- Training & Stacking ----------
def train_and_predict(X, y, X_test):
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # scale numeric
        scaler = StandardScaler()
        num_cols = ['log_reviews','log_installs','X6']
        X_tr[num_cols]  = scaler.fit_transform(X_tr[num_cols])
        X_val[num_cols] = scaler.transform(X_val[num_cols])
        X_test[num_cols]= scaler.transform(X_test[num_cols])

        # define base learners
        base_learners = [
            ('lr', LinearRegression()),  # :contentReference[oaicite:6]{index=6}
            ('rf', RandomForestRegressor(n_estimators=200, random_state=SEED)),
            ('gbr', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=SEED)),
            ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.2)),
            ('mlp', MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=SEED))
        ]

        # stacking regressor with linear meta-model
        stack = StackingRegressor(
            estimators=base_learners,
            final_estimator=LinearRegression(),  # :contentReference[oaicite:7]{index=7}
            cv=5,
            passthrough=False,
            n_jobs=-1
        )

        stack.fit(X_tr, y_tr)
        oof_preds[val_idx] = stack.predict(X_val)
        test_preds       += stack.predict(X_test) / kf.n_splits

    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"OOF RMSE: {rmse:.4f}")
    return np.clip(test_preds, 1.0, 5.0)

# ---------- Main ----------
def main():
    # train = pd.read_csv('train.csv')
    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')
    sub   = pd.read_csv('SampleSubmission.csv')

    X_tr, y_tr, X_te = feature_engineer(train, test)
    sub['Y'] = train_and_predict(X_tr, y_tr, X_te)
    sub.to_csv('submission.csv', index=False)
    print("submission.csv generated.")

if __name__ == "__main__":
    main()
