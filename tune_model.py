import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
import optuna
import gc
import warnings
warnings.filterwarnings('ignore')

# 1. Feature Engineering function to keep things clean
def create_features(df_path):
    print("Loading data for Optuna tuning...")
    df = pd.read_csv(df_path)
    
    if 'session_id' in df.columns:
        df.drop('session_id', axis=1, inplace=True)
    if 'DateTime' in df.columns:
        df.drop('DateTime', axis=1, inplace=True)
        
    target = 'is_click'
    
    print("Applying Count Encoding...")
    count_cols = ['user_id', 'campaign_id', 'webpage_id', 'product']
    for col in count_cols:
        count_map = df[col].value_counts().to_dict()
        df[f'{col}_count'] = df[col].map(count_map)

    print("Applying Target Encoding with K-Fold...")
    target_encode_cols = ['campaign_id', 'user_id', 'webpage_id', 'product_category_1']
    for col in target_encode_cols:
        df[f'{col}_target_enc'] = np.nan

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(df):
        X_train, X_val = df.iloc[train_idx], df.iloc[val_idx]
        for col in target_encode_cols:
            target_mean = X_train.groupby(col)[target].mean()
            df.loc[val_idx, f'{col}_target_enc'] = X_val[col].map(target_mean)

    for col in target_encode_cols:
        global_mean = df[target].mean()
        df[f'{col}_target_enc'] = df[f'{col}_target_enc'].fillna(global_mean)

    cat_features = ['product', 'campaign_id', 'webpage_id', 'product_category_1', 
                    'product_category_2', 'user_group_id', 'gender', 'age_level', 'user_depth', 'hour', 'day_of_week']
                    
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    features = [c for c in df.columns if c not in [target, 'user_id']]
    return df[features], df[target], [c for c in cat_features if c in features]

data_path = r"D:\projects\Ad Click Pridiction\archive\processed_train.csv"
X, y, cat_features_list = create_features(data_path)

# Split once to use a consistent validation set during tuning
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features_list, free_raw_data=False)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, free_raw_data=False)

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': 42,
        
        # Hyperparameters to tune
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 128),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 300),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True)
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000, # Using slightly fewer rounds for faster tuning
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    y_pred_prob = model.predict(X_valid, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_valid, y_pred_prob)
    return auc

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.INFO)
    print("\nStarting Optuna Hyperparameter Optimization (Running 15 Trials)...")
    
    study = optuna.create_study(direction='maximize', study_name='LGBM_CTR')
    study.optimize(objective, n_trials=15)

    print("\n--- Best Trial ---")
    print(f"Best AUC: {study.best_value:.4f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  '{key}': {value},")
