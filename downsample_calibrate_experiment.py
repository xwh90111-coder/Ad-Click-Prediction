# -*- coding: utf-8 -*-
import sys
import io
# 强制 stdout 使用 UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
负样本下采样 + 概率校准 对照实验
=====================================
核心思路：
  1. 训练时对负样本做下采样 → 让模型多看正样本，改善排序能力（AUC）
  2. 预测时用 Platt Scaling 把概率校准回自然分布 → 恢复绝对概率的保真度
  3. 验证集/测试集始终保持原始分布，不做任何修改

对比的采样比例：原始(1:13) / 1:1 / 1:2 / 1:4 / 1:8
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# ── 0. 配置 ──────────────────────────────────────────────
data_path = r"D:\projects\Ad Click Pridiction\archive\processed_train.csv"
target = 'is_click'
POS_LABEL = 1
NEG_LABEL = 0

# ── 1. 特征工程（与 advanced_model.py 完全一致）───────────
print("=" * 60)
print("1. 加载数据 & 特征工程")
print("=" * 60)

df = pd.read_csv(data_path)
if 'session_id' in df.columns:
    df.drop('session_id', axis=1, inplace=True)
if 'DateTime' in df.columns:
    df.drop('DateTime', axis=1, inplace=True)

print(f"原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
print(f"正样本占比: {df[target].mean():.4f} ({df[target].sum()} 条)")

# --- Count Encoding ---
print("\n→ Count Encoding...")
count_cols = ['user_id', 'campaign_id', 'webpage_id', 'product']
for col in count_cols:
    count_map = df[col].value_counts().to_dict()
    df[f'{col}_count'] = df[col].map(count_map)

# --- Target Encoding (K-Fold, 防泄漏) ---
print("→ Target Encoding (5-Fold)...")
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

# --- 类别特征转 category dtype ---
cat_features = ['product', 'campaign_id', 'webpage_id', 'product_category_1',
                'product_category_2', 'user_group_id', 'gender', 'age_level',
                'user_depth', 'hour', 'day_of_week']
for col in cat_features:
    if col in df.columns:
        df[col] = df[col].astype('category')

features = [c for c in df.columns if c not in [target, 'user_id']]
print(f"特征数: {len(features)}")

# ── 2. 数据划分（Train / Calib / Test）─────────────────────
print("\n" + "=" * 60)
print("2. 数据划分: Train(60%) / Calib(20%) / Test(20%)")
print("=" * 60)

# 先分出 20% test
train_val, test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[target]
)
# 再从剩余中分出 25%（即总量的 20%）作为校准集
train_full, calib = train_test_split(
    train_val, test_size=0.25, random_state=42, stratify=train_val[target]
)

print(f"训练集(用于下采样): {train_full.shape[0]} 行, 正样本率: {train_full[target].mean():.4f}")
print(f"校准集(原始分布):   {calib.shape[0]} 行, 正样本率: {calib[target].mean():.4f}")
print(f"测试集(原始分布):   {test.shape[0]} 行, 正样本率: {test[target].mean():.4f}")

# ── 3. 下采样函数 ─────────────────────────────────────────
def downsample_negative(df, pos_ratio_target):
    """
    对负样本进行随机下采样，使正负比例达到目标值。
    pos_ratio_target = 0.5 表示 1:1
    pos_ratio_target = 0.2 表示 1:4
    pos_ratio_target = None 表示不采样
    """
    if pos_ratio_target is None:
        return df.copy()

    df_pos = df[df[target] == POS_LABEL]
    df_neg = df[df[target] == NEG_LABEL]

    n_pos = len(df_pos)
    n_neg_needed = int(n_pos * (1 - pos_ratio_target) / pos_ratio_target)

    if n_neg_needed >= len(df_neg):
        print(f"  !! 负样本不足, 使用全部 {len(df_neg)} 条")
        return df.copy()

    df_neg_sampled = df_neg.sample(n=n_neg_needed, random_state=42)
    result = pd.concat([df_pos, df_neg_sampled], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    actual_ratio = result[target].mean()
    print(f"  下采样后: {len(result)} 条 (正样本 {n_pos}, 负样本 {n_neg_needed}), 正样本率 {actual_ratio:.4f}")
    return result


# ── 4. 实验主循环 ─────────────────────────────────────────
print("\n" + "=" * 60)
print("3. 对照实验：不同下采样比例")
print("=" * 60)

# 实验配置：(标签, 正样本目标比例)
experiments = [
    ("无下采样(原始)", None),
    ("1:8 下采样", 1/9),      # 正:负 = 1:8, 正样本率 ≈ 11.1%
    ("1:4 下采样", 1/5),      # 正:负 = 1:4, 正样本率 = 20%
    ("1:2 下采样", 1/3),      # 正:负 = 1:2, 正样本率 ≈ 33.3%
    ("1:1 下采样", 0.5),      # 正:负 = 1:1, 正样本率 = 50%
]

results = []

for label, target_ratio in experiments:
    print(f"\n{'─' * 50}")
    print(f"【{label}】正样本目标率 = {target_ratio}")
    print(f"{'─' * 50}")

    # 4a. 对训练集做下采样
    if target_ratio is not None:
        train_sampled = downsample_negative(train_full, target_ratio)
    else:
        train_sampled = train_full.copy()
        print(f"  保持原始分布: {len(train_sampled)} 条, 正样本率 {train_sampled[target].mean():.4f}")

    X_train = train_sampled[features]
    y_train = train_sampled[target]

    # 4b. 训练 LightGBM
    train_data = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=[c for c in cat_features if c in features]
    )

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

    # 使用校准集做 early stopping
    X_calib = calib[features]
    y_calib = calib[target]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1500,
        valid_sets=[lgb.Dataset(X_calib, label=y_calib, reference=train_data)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )

    # 4c. 在测试集上评估「未校准」的预测
    X_test = test[features]
    y_test = test[target]
    y_pred_raw = model.predict(X_test, num_iteration=model.best_iteration)

    auc_raw = roc_auc_score(y_test, y_pred_raw)
    logloss_raw = log_loss(y_test, y_pred_raw)
    brier_raw = brier_score_loss(y_test, y_pred_raw)

    # 4d. Platt Scaling 校准（在 Calib 集上拟合）
    from sklearn.linear_model import LogisticRegression

    # 用校准集上的预测值作为单一特征，拟合 sigmoid
    y_calib_pred = model.predict(X_calib, num_iteration=model.best_iteration)
    calib_lr = LogisticRegression(penalty=None)  # 无正则化的 LR = Platt Scaling
    calib_lr.fit(y_calib_pred.reshape(-1, 1), y_calib)

    # 对测试集预测做校准
    y_pred_calibrated = calib_lr.predict_proba(y_pred_raw.reshape(-1, 1))[:, 1]

    auc_cal = roc_auc_score(y_test, y_pred_calibrated)
    logloss_cal = log_loss(y_test, y_pred_calibrated)
    brier_cal = brier_score_loss(y_test, y_pred_calibrated)

    # 记录结果
    results.append({
        '实验组': label,
        '训练集正样本率': train_sampled[target].mean(),
        '训练集样本量': len(train_sampled),
        'AUC(未校准)': f"{auc_raw:.4f}",
        'LogLoss(未校准)': f"{logloss_raw:.4f}",
        'AUC(校准后)': f"{auc_cal:.4f}",
        'LogLoss(校准后)': f"{logloss_cal:.4f}",
        'Brier(未校准)': f"{brier_raw:.4f}",
        'Brier(校准后)': f"{brier_cal:.4f}",
        '最佳迭代轮': model.best_iteration,
    })

    print(f"\n  >> 未校准: AUC={auc_raw:.4f}, LogLoss={logloss_raw:.4f}, Brier={brier_raw:.4f}")
    print(f"  >> 校准后: AUC={auc_cal:.4f}, LogLoss={logloss_cal:.4f}, Brier={brier_cal:.4f}")

# ── 5. 汇总对比 ────────────────────────────────────────────
print("\n" + "=" * 85)
print("最终结果汇总")
print("=" * 85)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# 找出最佳组合
best_auc_idx = results_df['AUC(校准后)'].astype(float).idxmax()
best_logloss_idx = results_df['LogLoss(校准后)'].astype(float).idxmin()
print(f"\n>> 最高 AUC:  {results_df.loc[best_auc_idx, '实验组']} (AUC={results_df.loc[best_auc_idx, 'AUC(校准后)']})")
print(f">> 最低 LogLoss: {results_df.loc[best_logloss_idx, '实验组']} (LogLoss={results_df.loc[best_logloss_idx, 'LogLoss(校准后)']})")

# 论文级别的讨论素材
print("\n" + "=" * 60)
print("论文讨论要点")
print("=" * 60)
print("""
1. AUC 是排序指标 → 下采样能否提升 AUC 取决于模型是否因数据不平衡
   而"忽略"了正样本的细微模式。本实验中可观察 AUC 随采样比例的变化。

2. LogLoss / Brier 是概率校准指标 → 未经校准的下采样模型 LogLoss
   会极差（概率被系统性高估），但 Platt Scaling 可以恢复。

3. "下采样 + 校准" vs "不用下采样"的对比，本身就是论文中有价值的讨论：
   - 如果 AUC 提高 → 下采样改善了排序学习
   - 如果 AUC 不变 + LogLoss 恢复 → 说明校准有效，但下采样无额外收益
   - 如果 AUC 下降 → GBDT 已经能处理不平衡，强行采样反而丢失信息

4. Brier Score 衡量概率预测的均方误差，校准后应显著降低。
""")
