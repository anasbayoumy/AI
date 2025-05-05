# import pandas as pd
# import numpy as np
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
# from sklearn.linear_model import LinearRegression
# from scipy.stats import mstats

# SEED = 2025
# NFOLDS = 5
# np.random.seed(SEED)

# # ─── Parsing Helpers ───────────────────────────────────────────────────────────
# def parse_size(s):
#     nums = s.astype(str).str.replace(r'[^\d.]', '', regex=True)
#     vals = pd.to_numeric(nums, errors='coerce')
#     is_m = s.astype(str).str.lower().str.endswith('m')
#     return vals.mul(np.where(is_m, 1024, 1))

# def parse_installs(s):
#     cleaned = s.astype(str).str.replace(r'[^0-9]', '', regex=True)
#     return pd.to_numeric(cleaned, errors='coerce')

# def parse_price(s):
#     cleaned = s.astype(str).str.replace(r'[$]', '', regex=True)
#     return pd.to_numeric(cleaned, errors='coerce')

# def parse_version(s):
#     vs = (s.astype(str)
#            .fillna('')
#            .replace("Varies with device","",regex=False)
#            .str.replace(r'[^0-9\.]','',regex=True)
#          )
#     def norm(v):
#         if v.count('.')==0: return "0.0.0"
#         if v.count('.')==1: return v + ".0"
#         return v
#     vs = vs.map(norm)
#     parts = vs.str.split('.', expand=True).iloc[:,:3].fillna('0')
#     for col in parts.columns:
#         parts[col] = pd.to_numeric(parts[col], errors='coerce').fillna(0).astype(int)
#     parts.columns = ['ver_major','ver_minor','ver_patch']
#     return parts

# # ─── Cleaning & Feature Engineering ──────────────────────────────────────────
# def clean_and_engineer(df, is_train=True):
#     df = df.copy()
#     if is_train:
#         df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
#         df.dropna(subset=['Y'], inplace=True)

#     # Numeric parsing
#     df['reviews']  = pd.to_numeric(df['X2'], errors='coerce').fillna(0)
#     df['installs'] = parse_installs(df['X4']).fillna(0)
#     df['size_kb']  = parse_size(df['X3']).fillna(0)
#     df['price']    = parse_price(df['X6']).fillna(0)

#     # Winsorize & log transform
#     df['reviews']      = mstats.winsorize(df['reviews'],  limits=[0.01,0.01])
#     df['installs']     = mstats.winsorize(df['installs'], limits=[0.01,0.01])
#     df['log_reviews']  = np.log1p(df['reviews'])
#     df['log_installs'] = np.log1p(df['installs'])

#     # Paid/free & ratios
#     df['is_paid']           = (df['X5'].astype(str).str.lower()!='free').astype(int)
#     df['rev_install_ratio'] = df['reviews'] / (df['installs'] + 1)
#     df['price_per_install'] = df['price']   / (df['installs'] + 1)

#     # Date features
#     dates = pd.to_datetime(df['X9'], format='%B %d, %Y', errors='coerce')
#     df['days_since_update'] = (dates.max() - dates).dt.days.fillna(0).astype(int)
#     df['month']             = dates.dt.month.fillna(1).astype(int)
#     df['day_of_week']       = dates.dt.dayofweek.fillna(0).astype(int)

#     # Version parts
#     ver = parse_version(df['X10'])
#     df = pd.concat([df, ver], axis=1)

#     # Text features
#     names = df['X0'].astype(str)
#     df['name_len']   = names.str.len()
#     df['name_words'] = names.str.split().str.len()
#     df['has_pro']    = names.str.contains(r'\bPro\b', case=False).astype(int)
#     df['has_free']   = names.str.contains(r'\bFree\b', case=False).astype(int)

#     # Genre count
#     df['genre_count'] = df['X8'].fillna('').str.count(';') + np.where(df['X8'].notna(), 1, 0)

#     # Categorical fields
#     df['X1'] = df['X1'].fillna('Unknown')
#     df['X7'] = df['X7'].fillna('Unknown')

#     # Drop raw columns
#     drop_cols = ['X0','X2','X3','X4','X5','X6','X8','X9','X10','X11']
#     df.drop(columns=[c for c in drop_cols if c in df], inplace=True)

#     return df

# # ─── Training & Prediction ───────────────────────────────────────────────────
# def train_and_predict(train_df, test_df):
#     y = train_df.pop('Y').astype(float)

#     numeric_features = [
#         'size_kb','price','log_reviews','log_installs',
#         'is_paid','rev_install_ratio','price_per_install',
#         'days_since_update','month','day_of_week',
#         'ver_major','ver_minor','ver_patch',
#         'name_len','name_words','has_pro','has_free',
#         'genre_count'
#     ]
#     categorical_features = ['X1','X7']

#     numeric_transformer = Pipeline([
#         ('imputer', SimpleImputer(strategy='median')),
#         ('scaler',  StandardScaler()),
#         ('poly',    PolynomialFeatures(degree=2, include_bias=False))
#     ])
#     categorical_transformer = Pipeline([
#         ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
#         ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
#     ])

#     preprocessor = ColumnTransformer([
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

#     pipe = Pipeline([
#         ('pre', preprocessor),
#         ('lr',  LinearRegression())
#     ])

#     oof_preds = np.zeros(len(train_df))
#     kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
#     for fold, (train_idx, val_idx) in enumerate(kf.split(train_df), 1):
#         X_tr, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
#         y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

#         pipe.fit(X_tr, y_tr)
#         oof_preds[val_idx] = pipe.predict(X_val)
#         rmse_fold = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
#         print(f"Fold {fold} RMSE: {rmse_fold:.5f}")

#     overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
#     print(f"Overall OOF RMSE: {overall_rmse:.5f}")

#     pipe.fit(train_df, y)
#     preds = pipe.predict(test_df)
#     return np.clip(preds, 1.0, 5.0)

# def main():
#     print("Loading data...")
#     train = pd.read_csv('train.csv')
#     test  = pd.read_csv('test.csv')
#     sub   = pd.read_csv('SampleSubmission.csv')

#     print("Cleaning & engineering train data...")
#     tr = clean_and_engineer(train, is_train=True)
#     print("Cleaning & engineering test data...")
#     te = clean_and_engineer(test,  is_train=False)

#     print("Training Linear Regression and predicting...")
#     sub['Y'] = train_and_predict(tr, te)

#     sub.to_csv('submission.csv', index=False)
#     print("✅ submission.csv generated.")

# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures,
    OneHotEncoder
)
from sklearn.linear_model import Ridge
from scipy.stats import mstats

SEED = 2025
NFOLDS = 5
np.random.seed(SEED)

# ─── Parsing Helpers ───────────────────────────────────────────────────────────
def parse_size(s):
    nums = s.astype(str).str.replace(r'[^\d.]', '', regex=True)
    vals = pd.to_numeric(nums, errors='coerce')
    is_m = s.astype(str).str.lower().str.endswith('m')
    return vals.mul(np.where(is_m, 1024, 1))

def parse_installs(s):
    cleaned = s.astype(str).str.replace(r'[^0-9]', '', regex=True)
    return pd.to_numeric(cleaned, errors='coerce')

def parse_price(s):
    cleaned = s.astype(str).str.replace(r'[$]', '', regex=True)
    return pd.to_numeric(cleaned, errors='coerce')

def parse_version(s):
    vs = (s.astype(str)
           .fillna('')
           .replace("Varies with device","",regex=False)
           .str.replace(r'[^0-9\.]','',regex=True)
         )
    def norm(v):
        if v.count('.')==0: return "0.0.0"
        if v.count('.')==1: return v + ".0"
        return v
    vs = vs.map(norm)
    parts = vs.str.split('.', expand=True).iloc[:,:3].fillna('0')
    for col in parts.columns:
        parts[col] = pd.to_numeric(parts[col], errors='coerce').fillna(0).astype(int)
    parts.columns = ['ver_major','ver_minor','ver_patch']
    return parts

# ─── Cleaning & Feature Engineering ──────────────────────────────────────────
def clean_and_engineer(df, is_train=True):
    df = df.copy()
    if is_train:
        df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
        df.dropna(subset=['Y'], inplace=True)

    df['reviews']      = pd.to_numeric(df['X2'], errors='coerce').fillna(0)
    df['installs']     = parse_installs(df['X4']).fillna(0)
    df['size_kb']      = parse_size(df['X3']).fillna(0)
    df['price']        = parse_price(df['X6']).fillna(0)
    df['reviews']      = mstats.winsorize(df['reviews'],  limits=[0.01,0.01])
    df['installs']     = mstats.winsorize(df['installs'], limits=[0.01,0.01])
    df['log_reviews']  = np.log1p(df['reviews'])
    df['log_installs'] = np.log1p(df['installs'])
    df['is_paid']           = (df['X5'].astype(str).str.lower()!='free').astype(int)
    df['rev_install_ratio'] = df['reviews'] / (df['installs']+1)
    df['price_per_install'] = df['price']   / (df['installs']+1)
    dates = pd.to_datetime(df['X9'], format='%B %d, %Y', errors='coerce')
    df['days_since_update'] = (dates.max()-dates).dt.days.fillna(0).astype(int)
    df['month']             = dates.dt.month.fillna(1).astype(int)
    df['day_of_week']       = dates.dt.dayofweek.fillna(0).astype(int)
    df = pd.concat([df, parse_version(df['X10'])], axis=1)
    names = df['X0'].astype(str)
    df['name_len']   = names.str.len()
    df['name_words'] = names.str.split().str.len()
    df['has_pro']    = names.str.contains(r'\bPro\b', case=False).astype(int)
    df['has_free']   = names.str.contains(r'\bFree\b', case=False).astype(int)
    df['genre_count'] = df['X8'].fillna('').str.count(';') + np.where(df['X8'].notna(),1,0)
    df['X1'] = df['X1'].fillna('Unknown')
    df['X7'] = df['X7'].fillna('Unknown')
    drop = ['X0','X2','X3','X4','X5','X6','X8','X9','X10','X11']
    df.drop(columns=[c for c in drop if c in df.columns], inplace=True)
    return df

# ─── Train & Hyperparameter Search ───────────────────────────────────────────
def train_and_search(tr_df, te_df):
    y = tr_df.pop('Y').astype(float)

    numeric_features = [
        'size_kb','price','log_reviews','log_installs',
        'is_paid','rev_install_ratio','price_per_install',
        'days_since_update','month','day_of_week',
        'ver_major','ver_minor','ver_patch',
        'name_len','name_words','has_pro','has_free',
        'genre_count'
    ]
    categorical_features = ['X1','X7']

    best_rmse = np.inf
    best_cfg  = None
    best_oof  = None

    # Grid of degrees and alphas (alpha=0 → OLS)
    degrees = [1,2,3]
    alphas = [0.0, 0.01, 0.1, 1.0, 10.0]

    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

    print("Searching over polynomial degrees and Ridge alphas...")
    for deg in degrees:
        for alpha in alphas:
            oof_preds = np.zeros(len(tr_df))

            # build preprocess→model pipeline
            num_pipe = Pipeline([
                ('imp',   SimpleImputer(strategy='median')),
                ('scale', StandardScaler()),
                ('poly',  PolynomialFeatures(degree=deg, include_bias=False))
            ])
            cat_pipe = Pipeline([
                ('imp',    SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            pre = ColumnTransformer([
                ('num', num_pipe, numeric_features),
                ('cat', cat_pipe, categorical_features)
            ])
            model = Ridge(alpha=alpha, random_state=SEED)
            pipe = Pipeline([('pre', pre), ('model', model)])

            # CV
            for train_idx, val_idx in kf.split(tr_df):
                X_tr = tr_df.iloc[train_idx]
                X_val= tr_df.iloc[val_idx]
                y_tr = y.iloc[train_idx]
                y_val= y.iloc[val_idx]

                pipe.fit(X_tr, y_tr)
                oof_preds[val_idx] = pipe.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y, oof_preds))
            print(f"  degree={deg} α={alpha:<5} → OOF RMSE={rmse:.5f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_cfg  = (deg, alpha)
                best_oof  = oof_preds.copy()

    print(f"\nBest config: degree={best_cfg[0]}, alpha={best_cfg[1]} → RMSE={best_rmse:.5f}\n")

    # Retrain on full data
    deg, alpha = best_cfg
    num_pipe = Pipeline([
        ('imp',   SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('poly',  PolynomialFeatures(degree=deg, include_bias=False))
    ])
    cat_pipe = Pipeline([
        ('imp',    SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    pre = ColumnTransformer([
        ('num', num_pipe, numeric_features),
        ('cat', cat_pipe, categorical_features)
    ])
    final_pipe = Pipeline([('pre', pre), ('model', Ridge(alpha=alpha, random_state=SEED))])
    final_pipe.fit(tr_df, y)
    preds = final_pipe.predict(te_df)

    return best_rmse, preds

def main():
    print("Loading data...")
    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')
    sub   = pd.read_csv('SampleSubmission.csv')

    print("Cleaning features...")
    tr_df = clean_and_engineer(train, is_train=True)
    te_df = clean_and_engineer(test,  is_train=False)

    best_rmse, preds = train_and_search(tr_df, te_df)
    sub['Y'] = np.clip(preds, 1.0, 5.0)
    sub.to_csv('submission.csv', index=False)
    print(f"✅ Done. Best OOF RMSE={best_rmse:.5f}, submission.csv generated.")

if __name__ == "__main__":
    main()
