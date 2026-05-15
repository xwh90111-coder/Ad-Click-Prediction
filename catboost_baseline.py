import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

data_path = r"D:\projects\Ad Click Pridiction\archive\processed_train.csv"
print("Loading processed data...")
df = pd.read_csv(data_path)

if 'DateTime' in df.columns:
    df = df.drop(columns=['DateTime'])
if 'session_id' in df.columns:
    df = df.drop(columns=['session_id'])

# CatBoost prefers string or integer types for categorical features
cat_features = ['product', 'campaign_id', 'webpage_id', 'product_category_1', 
                'product_category_2', 'user_group_id', 'gender', 'age_level', 'user_depth']

# Convert object types to string and fill nans
for col in cat_features:
    if col in df.columns:
        df[col] = df[col].astype(str)

target = 'is_click'
features = [c for c in df.columns if c != target]

X = df[features]
y = df[target]

print("Splitting data...")
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training CatBoost baseline...")
cat_features_indices = [X.columns.get_loc(col) for col in cat_features if col in X.columns]

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    bagging_temperature=0.2,
    od_type='Iter',
    od_wait=50,
    verbose=100
)

model.fit(
    X_train, y_train,
    cat_features=cat_features_indices,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

print("\n--- CatBoost Baseline Model Evaluation ---")
y_pred_prob = model.predict_proba(X_valid)[:, 1]
y_pred_class = model.predict(X_valid)

auc = roc_auc_score(y_valid, y_pred_prob)
logloss = log_loss(y_valid, y_pred_prob)
acc = accuracy_score(y_valid, y_pred_class)
baseline_acc = 1 - y.mean()

print(f"ROC AUC Score: {auc:.4f} (The closer to 1.0, the better)")
print(f"Log Loss:      {logloss:.4f} (The lower, the better)")
print(f"Model Accuracy:{acc:.4f}")
print(f"Dumb Baseline Accuracy (always predict 0): {baseline_acc:.4f}")

importance = pd.DataFrame({
    'Feature': model.feature_names_,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nTop 5 Important Features:")
print(importance.head(5))
