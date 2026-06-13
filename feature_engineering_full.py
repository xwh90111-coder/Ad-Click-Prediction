# -*- coding: utf-8 -*-
"""
全面特征工程 —— 广告CTR预测
===============================
5 层特征递增，目标突破 AUC 0.65+

Layer 1: 笛卡尔积交叉特征 (7组) + Target Encoding
Layer 2: 群体聚合统计 (14个) — 每实体 click_rate 等
Layer 3: 用户时序行为特征 (4个)
Layer 4: 比率/交互特征 (4个)
Layer 5: 共现计数特征 (2个)

防泄漏: 所有涉及 target 的统计在 5-Fold 内计算
"""

import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, log_loss
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 配置
# ============================================================
data_path = r"D:\projects\Ad Click Pridiction\archive\processed_train.csv"
TARGET = 'is_click'
N_FOLDS = 5
RANDOM_SEED = 42

# ============================================================
# 1. 加载 & 基础预处理
# ============================================================
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

df = pd.read_csv(data_path)
print(f"原始: {df.shape[0]} 行, {df.shape[1]} 列")

if 'session_id' in df.columns:
    df.drop('session_id', axis=1, inplace=True)
if 'DateTime' in df.columns:
    df.drop('DateTime', axis=1, inplace=True)

global_click_rate = df[TARGET].mean()
print(f"全局点击率: {global_click_rate:.4f} | 正样本: {df[TARGET].sum()}")

# ============================================================
# 2. Layer 5 先行: 共现计数（无泄漏，全数据计算）
# ============================================================
print("\n" + "=" * 60)
print("2. Layer 5 — 共现计数特征 (无泄漏)")
print("=" * 60)

df['user_campaign_count'] = df.groupby(['user_id', 'campaign_id'])['user_id'].transform('count')
df['user_product_count']  = df.groupby(['user_id', 'product'])['user_id'].transform('count')
print(f"  user_campaign_count: {df['user_campaign_count'].max()} 最大共现次数")
print(f"  user_product_count:  {df['user_product_count'].max()} 最大共现次数")

# ============================================================
# 2.5. Layer 6: 用户时间行为特征（从原始 CSV 提取 DateTime）
# ============================================================
print("\n" + "=" * 60)
print("2.5. Layer 6 — 用户时序行为 (从原始DateTime提取)")
print("=" * 60)

raw_path = r"D:\projects\Ad Click Pridiction\archive\Ad_click_prediction_train (1).csv"
raw = pd.read_csv(raw_path, usecols=['user_id', 'DateTime'])
raw['DateTime'] = pd.to_datetime(raw['DateTime'])
raw = raw.sort_values(['user_id', 'DateTime'])

print(f"  原始数据: {len(raw)} 行")

# 为每个用户计算时间聚合特征
def compute_user_temporal(grp):
    times = grp['DateTime'].values
    n = len(times)
    if n == 1:
        return pd.Series({
            'user_avg_gap_hours': 0.0,
            'user_max_gap_hours': 0.0,
            'user_active_span_days': 0.0,
            'user_night_ratio': 1.0 if 0 <= pd.Timestamp(times[0]).hour < 6 else 0.0,
            'user_morning_ratio': 1.0 if 6 <= pd.Timestamp(times[0]).hour < 12 else 0.0,
            'user_afternoon_ratio': 1.0 if 12 <= pd.Timestamp(times[0]).hour < 18 else 0.0,
            'user_evening_ratio': 1.0 if 18 <= pd.Timestamp(times[0]).hour < 24 else 0.0,
        })

    # 时间间隔（小时）
    gaps = np.diff(times).astype('timedelta64[h]').astype(np.float64)
    avg_gap = gaps.mean()
    max_gap = gaps.max()

    # 活跃跨度（天）
    span_days = (times[-1] - times[0]) / np.timedelta64(1, 'D')

    # 时段分布
    hours = np.array([pd.Timestamp(t).hour for t in times])
    night = (hours < 6).mean()
    morning = ((hours >= 6) & (hours < 12)).mean()
    afternoon = ((hours >= 12) & (hours < 18)).mean()
    evening = (hours >= 18).mean()

    return pd.Series({
        'user_avg_gap_hours': avg_gap,
        'user_max_gap_hours': max_gap,
        'user_active_span_days': span_days,
        'user_night_ratio': night,
        'user_morning_ratio': morning,
        'user_afternoon_ratio': afternoon,
        'user_evening_ratio': evening,
    })

print("  计算用户时序聚合 (按 user_id 分组)...")
user_temporal = raw.groupby('user_id').apply(compute_user_temporal).reset_index()
user_temporal = user_temporal.astype({
    'user_avg_gap_hours': np.float32,
    'user_max_gap_hours': np.float32,
    'user_active_span_days': np.float32,
    'user_night_ratio': np.float32,
    'user_morning_ratio': np.float32,
    'user_afternoon_ratio': np.float32,
    'user_evening_ratio': np.float32,
})

# 合并进主表
df = df.merge(user_temporal, on='user_id', how='left')

# 填充（新用户没有时间历史）
for col in ['user_avg_gap_hours', 'user_max_gap_hours', 'user_active_span_days']:
    df[col] = df[col].fillna(0.0)
for col in ['user_night_ratio', 'user_morning_ratio', 'user_afternoon_ratio', 'user_evening_ratio']:
    df[col] = df[col].fillna(0.0)

print(f"  user_avg_gap_hours: mean={df['user_avg_gap_hours'].mean():.1f}h")
print(f"  user_active_span_days: mean={df['user_active_span_days'].mean():.1f}d")
print(f"  user_night_ratio: mean={df['user_night_ratio'].mean():.3f}")

del raw, user_temporal

# ============================================================
# 3. Layer 1 预备: 创建交叉特征列（字符串拼接）
# ============================================================
print("\n" + "=" * 60)
print("3. Layer 1 — 构建笛卡尔积交叉特征")
print("=" * 60)

cross_pairs = [
    # 2阶交叉
    ('gender',        'product_category_1', 'gender_x_pcat1'),
    ('gender',        'product_category_2', 'gender_x_pcat2'),
    ('user_depth',    'product_category_1', 'udepth_x_pcat1'),
    ('age_level',     'product_category_1', 'age_x_pcat1'),
    ('hour',          'day_of_week',        'hour_x_dow'),
    ('user_group_id', 'campaign_id',        'ugroup_x_camp'),
    ('product_category_1', 'product_category_2', 'pcat1_x_pcat2'),
    # 3阶交叉
    ('user_group_id', 'hour',          'ugroup_x_hour'),
    ('user_group_id', 'day_of_week',   'ugroup_x_dow'),
    ('gender',        'hour',          'gender_x_hour'),
    ('age_level',     'hour',          'age_x_hour'),
    ('hour',          'product_category_1', 'hour_x_pcat1'),
    ('user_depth',    'product_category_2', 'udepth_x_pcat2'),
]

cross_col_names = []
for col_a, col_b, new_name in cross_pairs:
    df[new_name] = df[col_a].astype(str) + '_x_' + df[col_b].astype(str)
    cross_col_names.append(new_name)
    n_unique = df[new_name].nunique()
    print(f"  {new_name}: {n_unique} 个唯一组合")

# ============================================================
# 4. 5-Fold 循环: 一次性编码所有 target 相关特征
# ============================================================
print("\n" + "=" * 60)
print("4. 5-Fold 编码 (防泄漏) — Target Enc + Group Aggregations")
print("=" * 60)

# -- 定义所有需要 target encoding 的列 ------------------------
# 原有个体编码
target_encode_cols = [
    'campaign_id', 'user_id', 'webpage_id', 'product_category_1'
] + cross_col_names  # 包含所有2阶+3阶交叉

# -- 初始化 target encoding 列 -------------------------------
for col in target_encode_cols:
    df[f'{col}_target_enc'] = np.nan

# -- 定义需要做群体 CTR 聚合的实体 ---------------------------
agg_entities = [
    'user_id', 'campaign_id', 'product', 'webpage_id',
    'product_category_1', 'product_category_2',
    'user_group_id', 'gender', 'age_level', 'user_depth'
]

# -- 初始化群体聚合列 ----------------------------------------
for entity in agg_entities:
    df[f'{entity}_ctr'] = np.float32(np.nan)
    df[f'{entity}_imp_count'] = np.int32(0)  # 不计 target，仅计数

# user 时间聚合列
df['user_avg_hour'] = np.float32(np.nan)
df['user_hour_std'] = np.float32(np.nan)
df['user_active_days'] = np.float32(1.0)

# -- 5-Fold 循环 ---------------------------------------------
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"  Fold {fold_idx+1}/{N_FOLDS}...", end=' ')
    X_train_fold = df.iloc[train_idx]
    X_val_fold = df.iloc[val_idx]

    # --- 4a. Target Encoding (个体 & 交叉) ---
    for col in target_encode_cols:
        target_mean = X_train_fold.groupby(col)[TARGET].mean()
        df.loc[val_idx, f'{col}_target_enc'] = X_val_fold[col].map(target_mean)

    # --- 4b. 群体 CTR ---
    for entity in agg_entities:
        ctr_map = X_train_fold.groupby(entity)[TARGET].mean()
        df.loc[val_idx, f'{entity}_ctr'] = X_val_fold[entity].map(ctr_map).astype(np.float32)

    # --- 4c. 群体曝光计数 (无泄漏，可直接用全量) ---
    # 在循环外统一计算一次即可，这里跳过

    # --- 4d. user 时间聚合 ---
    user_hour_map = X_train_fold.groupby('user_id')['hour'].mean()
    user_hour_std_map = X_train_fold.groupby('user_id')['hour'].std()
    user_days_map = X_train_fold.groupby('user_id')['day'].nunique()

    df.loc[val_idx, 'user_avg_hour'] = X_val_fold['user_id'].map(user_hour_map).astype(np.float32)
    df.loc[val_idx, 'user_hour_std'] = X_val_fold['user_id'].map(user_hour_std_map).astype(np.float32)
    df.loc[val_idx, 'user_active_days'] = X_val_fold['user_id'].map(user_days_map).fillna(1).astype(np.float32)

    print("Done")

# -- 4e. 群体曝光计数 (无泄漏，全量计算一次) ------------------
print("\n  计算群体曝光计数 (无泄漏)...")
for entity in agg_entities:
    imp_map = df[entity].value_counts().to_dict()
    df[f'{entity}_imp_count'] = df[entity].map(imp_map).astype(np.int32)

# -- 4f. 填充缺失值（训练集中未出现的实体） --------------------
print("  填充缺失值...")
for col in target_encode_cols:
    null_count = df[f'{col}_target_enc'].isnull().sum()
    if null_count > 0:
        df[f'{col}_target_enc'] = df[f'{col}_target_enc'].fillna(global_click_rate)
        print(f"    {col}_target_enc: {null_count} 条缺失 -> 全局均值")

for entity in agg_entities:
    null_count = df[f'{entity}_ctr'].isnull().sum()
    if null_count > 0:
        df[f'{entity}_ctr'] = df[f'{entity}_ctr'].fillna(global_click_rate)
        # print(f"    {entity}_ctr: {null_count} 条缺失 -> 全局均值")

# user 时间特征缺失填充
df['user_avg_hour'] = df['user_avg_hour'].fillna(df['hour'].median())
df['user_hour_std'] = df['user_hour_std'].fillna(0)
df['user_active_days'] = df['user_active_days'].fillna(1.0)

# ============================================================
# 5. Layer 3: 用户时序行为特征（在 fold 编码基础上计算）
# ============================================================
print("\n" + "=" * 60)
print("5. Layer 3 — 用户时序行为特征")
print("=" * 60)

# user_avg_hour, user_hour_std, user_active_days 已在 fold 内计算

# 日均曝光量
df['user_impressions_per_day'] = (
    df['user_id_imp_count'] / df['user_active_days'].clip(lower=1)
).astype(np.float32)

# 工作日 vs 周末活跃比 (day_of_week: 0=Mon ... 6=Sun)
user_weekday_imp = df[df['day_of_week'] < 5].groupby('user_id').size()
user_weekend_imp = df[df['day_of_week'] >= 5].groupby('user_id').size()
df['user_weekday_ratio'] = df['user_id'].map(
    (user_weekday_imp / (user_weekend_imp + 1)).to_dict()
).fillna(0.5).astype(np.float32)
# clip to reasonable range
df['user_weekday_ratio'] = df['user_weekday_ratio'].clip(0, 10)

print(f"  user_impressions_per_day: mean={df['user_impressions_per_day'].mean():.1f}")
print(f"  user_weekday_ratio: mean={df['user_weekday_ratio'].mean():.2f}")

# ============================================================
# 6. Layer 4: 比率/交互特征
# ============================================================
print("\n" + "=" * 60)
print("6. Layer 4 — 比率/交互特征")
print("=" * 60)

eps = 0.001

df['user_vs_campaign_ctr'] = (
    df['user_id_ctr'] / (df['campaign_id_ctr'] + eps)
).astype(np.float32).clip(0, 50)

df['user_vs_global_ctr'] = (
    df['user_id_ctr'] / global_click_rate
).astype(np.float32).clip(0, 50)

df['campaign_vs_global_ctr'] = (
    df['campaign_id_ctr'] / global_click_rate
).astype(np.float32).clip(0, 50)

# user 曝光量分位数
df['user_imp_rank'] = df['user_id_imp_count'].rank(pct=True).astype(np.float32)

print(f"  user_vs_campaign_ctr:  mean={df['user_vs_campaign_ctr'].mean():.3f}")
print(f"  user_vs_global_ctr:    mean={df['user_vs_global_ctr'].mean():.3f}")
print(f"  campaign_vs_global_ctr: mean={df['campaign_vs_global_ctr'].mean():.3f}")

# ============================================================
# 7. Count Encoding（沿用 advanced_model 方案）
# ============================================================
print("\n" + "=" * 60)
print("7. Count Encoding")
print("=" * 60)

count_cols = ['user_id', 'campaign_id', 'webpage_id', 'product']
# 注意: 上面已经有了 imp_count，但用的是相同逻辑，避免重复
# 保留已有的 imp_count，不再重复创建
# 这里只补充 user_id_count（如果尚未存在）
if 'user_id_count' not in df.columns:
    for col in count_cols:
        count_map = df[col].value_counts().to_dict()
        df[f'{col}_count'] = df[col].map(count_map)
    print("  已创建 count 特征")
else:
    print("  count 特征已存在，跳过")

# ============================================================
# 8. 准备特征矩阵
# ============================================================
print("\n" + "=" * 60)
print("8. 特征汇总 & 数据划分")
print("=" * 60)

# 类别特征转 category dtype
cat_features = [
    'product', 'campaign_id', 'webpage_id', 'product_category_1',
    'product_category_2', 'user_group_id', 'gender', 'age_level',
    'user_depth', 'hour', 'day_of_week'
]
for col in cat_features:
    if col in df.columns:
        df[col] = df[col].astype('category')

# 构建特征列表
exclude_cols = [TARGET, 'user_id'] + cross_col_names  # 交叉原始列只保留 target_enc
features = [c for c in df.columns if c not in exclude_cols]

print(f"总特征数: {len(features)}")
# 分类统计
enc_cols = [c for c in features if c.endswith('_target_enc')]
ctr_cols = [c for c in features if c.endswith('_ctr')]
imp_cols = [c for c in features if c.endswith('_imp_count')]
count_cols_final = [c for c in features if c.endswith('_count') or c.endswith('_count_x')]

print(f"  - Target Encoding: {len(enc_cols)}")
print(f"  - CTR 聚合: {len(ctr_cols)}")
print(f"  - 曝光计数: {len(imp_cols)}")
print(f"  - 频次编码: {len(count_cols_final)}")
print(f"  - 其他: {len(features) - len(enc_cols) - len(ctr_cols) - len(imp_cols) - len(count_cols_final)}")

# 数据划分
X = df[features]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
print(f"\n训练集: {X_train.shape[0]} 行, 测试集: {X_test.shape[0]} 行")

# ============================================================
# 9. LightGBM 训练
# ============================================================
print("\n" + "=" * 60)
print("9. LightGBM 训练")
print("=" * 60)

cat_feat_names = [c for c in cat_features if c in features]

train_data = lgb.Dataset(
    X_train, label=y_train,
    categorical_feature=cat_feat_names
)
valid_data = lgb.Dataset(
    X_test, label=y_test,
    reference=train_data
)

base_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'learning_rate': 0.0213, 'num_leaves': 235, 'max_depth': 8,
    'min_data_in_leaf': 106, 'feature_fraction': 0.675,
    'bagging_fraction': 0.893, 'bagging_freq': 4,
    'lambda_l1': 1.9e-5, 'lambda_l2': 0.00166,
    'min_split_gain': 0.002, 'path_smooth': 2.5,
    'feature_pre_filter': False,
    'verbose': -1
}

# 多 seed 平均: 训练 5 个不同种子的模型, 平均预测
N_SEEDS = 5
all_test_preds = []

for seed_i in range(N_SEEDS):
    seed = RANDOM_SEED + seed_i * 10
    print(f"\n  训练模型 {seed_i+1}/{N_SEEDS} (seed={seed})...")
    params = {**base_params, 'random_state': seed}

    model = lgb.train(
        params, train_data,
        num_boost_round=2000,
        valid_sets=[train_data, valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=0)  # 安静模式
        ]
    )
    pred = model.predict(X_test, num_iteration=model.best_iteration)
    all_test_preds.append(pred)
    print(f"    AUC(单模型) = {roc_auc_score(y_test, pred):.4f}")

# 平均预测
y_pred_avg = np.mean(all_test_preds, axis=0)

# 单模型评估 (seed=42)
y_pred = all_test_preds[0]
y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)

test_auc = roc_auc_score(y_test, y_pred)
test_auc_avg = roc_auc_score(y_test, y_pred_avg)
test_logloss = log_loss(y_test, y_pred)
train_auc = roc_auc_score(y_train, y_train_pred)

print(f"\n{'=' * 60}")
print(f"10. 最终评估")
print(f"{'=' * 60}")
print(f"训练集 AUC:         {train_auc:.4f}")
print(f"测试集 AUC (单模型): {test_auc:.4f}")
print(f"测试集 AUC (5-平均): {test_auc_avg:.4f}")
print(f"测试集 LogLoss:     {test_logloss:.4f}")
print(f"过拟合差距:         {train_auc - test_auc:.4f}")

# 对比基线
baseline_auc = 0.6376
print(f"\n相较基线 (0.6376):")
print(f"  单模型提升: {'+' if test_auc - baseline_auc > 0 else ''}{test_auc - baseline_auc:.4f}")
print(f"  5-平均提升: {'+' if test_auc_avg - baseline_auc > 0 else ''}{test_auc_avg - baseline_auc:.4f}")
print(f"  历史最佳 0.6482: {'↑ 突破!' if test_auc_avg > 0.6482 else '✗'}")

# ============================================================
# 11. 特征重要性
# ============================================================
print("\n" + "=" * 60)
print("11. Top 30 特征重要性")
print("=" * 60)

importance = pd.DataFrame({
    'Feature': model.feature_name(),
    'Importance': model.feature_importance(importance_type='gain')
}).sort_values(by='Importance', ascending=False)

print(importance.head(30).to_string(index=False))

# 统计各层特征在 Top 30 中的占比
print("\n--- 各层特征在 Top 30 中的分布 ---")
for layer_name, pattern in [
    ('Layer 1 - 交叉 Target Enc', '_x__target_enc'),
    ('Layer 2 - CTR 聚合', '_ctr'),
    ('Layer 2 - 曝光计数', '_imp_count'),
    ('Layer 3 - 时序行为', 'user_avg_hour|user_hour_std|user_active_days|impressions_per_day|weekday'),
    ('Layer 4 - 比率', 'user_vs_|user_imp_rank'),
    ('Layer 5 - 共现', 'user_campaign_count|user_product_count'),
    ('原有 count 编码', '_count'),
    ('原有 target 编码', '_target_enc'),
]:
    top30 = importance.head(30)
    if '|' in pattern:
        import re
        count = sum(1 for f in top30['Feature'] if re.search(pattern, f))
    else:
        count = sum(1 for f in top30['Feature'] if pattern in f)
    if count > 0:
        print(f"  {layer_name}: {count} 个")

print("\nDone!")
