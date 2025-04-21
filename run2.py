# app_rating_competition_linear_advanced.py

"""
Advanced Linear‐Model Pipeline for Kaggle App Rating Competition:
- Loads train/test CSVs + SampleSubmission
- Applies original parsing & cleaning (X0–X11)
- Log-transforms + one-hot for small‐cardinality cats
- Fold-wise target‐encoding for high‐cardinality cats (X1,X8)
- Standardizes + enriches numeric feats with polynomials (degree=2)
- Trains ElasticNetCV within 5‐fold CV, outputs OOF RMSE
- Clips predictions to [1.0,5.0] and writes submission.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import ElasticNetCV
from category_encoders import TargetEncoder

# ---------- Parsing Helpers (unchanged) ----------
def parse_size(s):
    if pd.isna(s) or s=='Varies with device': return np.nan
    try:
        if s.endswith('M'): return float(s[:-1]) * 1024
        if s.endswith('k'): return float(s[:-1])
    except: return np.nan
    return np.nan

# ---------- Cleaning Helpers (unchanged) ----------
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

# ---------- Feature Engineering (unchanged) ----------
def feature_engineer(train, test):
    tr = clean_train(train.copy())
    med_size, med_price = tr['X3'].median(), tr['X6'].median()
    te = clean_test(test.copy(), med_size, med_price)

    y = tr['Y'].astype(float)
    tr = tr.drop(columns=['Y'])
    tr['__is_train'] = 1
    te['__is_train'] = 0
    full = pd.concat([tr, te], ignore_index=True)

    # log transforms
    full['log_reviews']  = np.log1p(full['X2'])
    full['log_installs'] = np.log1p(full['X4'])

    # one-hot small cats
    for col in ['X5','X7']:
        d = pd.get_dummies(full[col], prefix=col, drop_first=True)
        full = pd.concat([full.drop(columns=[col]), d], axis=1)

    drop_cols = ['X0','X2','X3','X4','X9','X10','X11']
    full.drop(columns=[c for c in drop_cols if c in full.columns], inplace=True)

    tr_fe = full[ full['__is_train']==1 ].drop(columns=['__is_train'])
    te_fe = full[ full['__is_train']==0 ].drop(columns=['__is_train'])
    return tr_fe, y, te_fe

# ---------- Training & Prediction ----------
def train_and_predict(X, y, X_test):
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for tr_idx, val_idx in folds.split(X):
        X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # 1) Fold‐wise target encoding on X1, X8
        te = TargetEncoder(cols=['X1','X8'], smoothing=0.2)
        X_tr[['X1','X8']] = te.fit_transform(X_tr[['X1','X8']], y_tr)  # :contentReference[oaicite:4]{index=4}
        X_val[['X1','X8']] = te.transform(X_val[['X1','X8']])
        X_test_fold = X_test.copy()
        X_test_fold[['X1','X8']] = te.transform(X_test_fold[['X1','X8']])

        # 2) Numeric pipeline: scale + polynomial
        num_feats = ['log_reviews','log_installs','X6','X1','X8']
        scaler = StandardScaler().fit(X_tr[num_feats])             # :contentReference[oaicite:5]{index=5}
        poly   = PolynomialFeatures(degree=2, include_bias=False)  # :contentReference[oaicite:6]{index=6}

        X_tr_num = poly.fit_transform(scaler.transform(X_tr[num_feats]))
        X_val_num= poly.transform(   scaler.transform(X_val[num_feats]))
        X_te_num = poly.transform(   scaler.transform(X_test_fold[num_feats]))

        # 3) Categorical remainder (one-hot columns)
        cat_feats = [c for c in X_tr.columns if c not in num_feats]

        X_tr_final = np.hstack([X_tr_num,  X_tr[cat_feats].values])
        X_val_final= np.hstack([X_val_num, X_val[cat_feats].values])
        X_te_final = np.hstack([X_te_num,  X_test_fold[cat_feats].values])

        # 4) ElasticNetCV: L1+L2 with internal CV to pick alpha & l1_ratio
        model = ElasticNetCV(
            l1_ratio=[.1, .5, .9, 1.0],
            alphas=np.logspace(-4, 0, 50),
            cv=5,
            max_iter=5_000,
            n_jobs=-1,
            random_state=42
        )  # :contentReference[oaicite:7]{index=7}

        model.fit(X_tr_final, y_tr)
        oof[val_idx]     = model.predict(X_val_final)
        test_preds      += model.predict(X_te_final) / folds.n_splits

    rmse = np.sqrt(mean_squared_error(y, oof))
    print(f"OOF RMSE: {rmse:.4f}")
    return np.clip(test_preds, 1.0, 5.0)

def main():
    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')
    sub   = pd.read_csv('SampleSubmission.csv')

    X_tr, y_tr, X_te = feature_engineer(train, test)
    sub['Y'] = train_and_predict(X_tr, y_tr, X_te)
    sub.to_csv('submission.csv', index=False)
    print("submission.csv generated successfully.")

if __name__=='__main__':
    main()
