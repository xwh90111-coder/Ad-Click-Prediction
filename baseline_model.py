import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

data_path = r"D:\projects\Ad Click Pridiction\archive\processed_train.csv"
print("Loading processed data...")
df = pd.read_csv(data_path)

if 'DateTime' in df.columns:
    df = df.drop(columns=['DateTime'])
if 'session_id' in df.columns:
    df = df.drop(columns=['session_id'])

string_cols = df.select_dtypes(include=['object']).columns
for col in string_cols:
    df[col] = df[col].astype('category')

target = 'is_click'
features = [c for c in df.columns if c != target]

cat_features = ['product', 'campaign_id', 'webpage_id', 'product_category_1', 
                'product_category_2', 'user_group_id', 'gender', 'age_level', 'user_depth']
for col in cat_features:
    if col in df.columns:
        df[col] = df[col].astype('category')

X = df[features]
y = df[target]

print("Splitting data...")
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training LightGBM baseline...")
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=[c for c in cat_features if c in features])
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=100)]
)

print("\n--- Baseline Model Evaluation ---")
y_pred_prob = model.predict(X_valid, num_iteration=model.best_iteration)
y_pred_class = (y_pred_prob > 0.5).astype(int)

auc = roc_auc_score(y_valid, y_pred_prob)
logloss = log_loss(y_valid, y_pred_prob)
acc = accuracy_score(y_valid, y_pred_class)
baseline_acc = 1 - y.mean()

print(f"ROC AUC Score: {auc:.4f} (The closer to 1.0, the better)")
print(f"Log Loss:      {logloss:.4f} (The lower, the better)")
print(f"Model Accuracy:{acc:.4f}")
print(f"Dumb Baseline Accuracy (always predict 0): {baseline_acc:.4f}")

importance = pd.DataFrame({
    'Feature': model.feature_name(),
    'Importance': model.feature_importance(importance_type='gain')
}).sort_values(by='Importance', ascending=False)
print("\nTop 5 Important Features:")
print(importance.head(5))
