import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, log_loss
import gc

data_path = r"D:\projects\Ad Click Pridiction\archive\processed_train.csv"
print("Loading data...")
df = pd.read_csv(data_path)

if 'session_id' in df.columns:
    df.drop('session_id', axis=1, inplace=True)
if 'DateTime' in df.columns:
    df.drop('DateTime', axis=1, inplace=True)

target = 'is_click'
cat_features = ['product', 'campaign_id', 'webpage_id', 'product_category_1', 
                'product_category_2', 'user_group_id', 'gender', 'age_level', 'user_depth', 'hour', 'day_of_week']

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

print("Preparing features for LightGBM...")
for col in cat_features:
    if col in df.columns:
        df[col] = df[col].astype('category')

features = [c for c in df.columns if c not in [target, 'user_id']]

X = df[features]
y = df[target]

print("Splitting data...")
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training Advanced LightGBM model...")
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=[c for c in cat_features if c in features])
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': 8,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 100,
    'verbose': -1,
    'random_state': 42
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1500,
    valid_sets=[train_data, valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=100)]
)

print("\n--- Advanced Model Evaluation ---")
y_pred_prob = model.predict(X_valid, num_iteration=model.best_iteration)

auc = roc_auc_score(y_valid, y_pred_prob)
logloss = log_loss(y_valid, y_pred_prob)

print(f"ROC AUC Score: {auc:.4f} (Previous baseline was ~0.5957)")
print(f"Log Loss:      {logloss:.4f} (Previous baseline was ~0.2437)")

importance = pd.DataFrame({
    'Feature': model.feature_name(),
    'Importance': model.feature_importance(importance_type='gain')
}).sort_values(by='Importance', ascending=False)
print("\nTop 10 Important Features:")
print(importance.head(10))
