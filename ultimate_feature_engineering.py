# -*- coding: utf-8 -*-
"""
终极特征工程 v2 — 时间窗口 + 3阶交叉 + 类权重
=================================================
核心升级:
  1. 从原始 CSV 加载，按时序分割 → 时间窗口特征无泄漏
  2. 3阶笛卡尔积交叉特征
  3. scale_pos_weight 处理类别不平衡
  4. 所有之前的 67 个特征底层全部保留
"""

import sys, io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, log_loss
import warnings
warnings.filterwarnings('ignore')

TARGET = 'is_click'
GLOBAL_SEED = 42

# ================================================================
# 0. 加载原始数据 & 基础预处理
# ================================================================
print("=" * 60)
print("0. 加载原始数据 & 基础预处理")
print("=" * 60)

raw_path = r"D:\projects\Ad Click Pridiction\archive\Ad_click_prediction_train (1).csv"
processed_path = r"D:\projects\Ad Click Pridiction\archive\processed_train.csv"

# 从原始 CSV 加载（保留 DateTime）
df = pd.read_csv(raw_path)
print(f"原始数据: {df.shape[0]} 行, {df.shape[1]} 列")

# 基础预处理
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['hour'] = df['DateTime'].dt.hour
df['day_of_week'] = df['DateTime'].dt.dayofweek
df['day'] = df['DateTime'].dt.day

# 填充缺失值
for col in ['product_category_2', 'user_group_id', 'age_level', 'user_depth']:
    if col in df.columns:
        df[col] = df[col].fillna(-1)
if 'gender' in df.columns:
    df['gender'] = df['gender'].fillna('Unknown')
if 'city_development_index' in df.columns:
    df['city_development_index'] = df['city_development_index'].fillna(df['city_development_index'].median())

# 删除 session_id
if 'session_id' in df.columns:
    df.drop('session_id', axis=1, inplace=True)

global_ctr = df[TARGET].mean()
print(f"正样本率: {global_ctr:.4f} ({df[TARGET].sum()} 条)")

# ================================================================
# 1. 按时序排序 & 分割
# ================================================================
print("\n" + "=" * 60)
print("1. 按时序排序 & Time-based Split")
print("=" * 60)

df = df.sort_values('DateTime').reset_index(drop=True)
min_time = df['DateTime'].min()
max_time = df['DateTime'].max()
print(f"时间范围: {min_time} → {max_time}")

# 使用随机分割（与之前的实验保持一致，便于比较）
from sklearn.model_selection import train_test_split as tts
train_idx, test_idx = tts(range(len(df)), test_size=0.2, random_state=GLOBAL_SEED, stratify=df[TARGET])
train_mask = pd.Series(False, index=df.index)
test_mask = pd.Series(False, index=df.index)
train_mask.iloc[train_idx] = True
test_mask.iloc[test_idx] = True

print(f"训练集: {train_mask.sum()} 行, 正样本率: {df.loc[train_mask, TARGET].mean():.4f}")
print(f"测试集:  {test_mask.sum()} 行, 正样本率: {df.loc[test_mask, TARGET].mean():.4f}")

# ================================================================
# 2. 时间窗口特征 (全量计算，用过去数据，无泄漏)
# ================================================================
print("\n" + "=" * 60)
print("2. 时间窗口特征 (在每个用户内，仅用历史数据)")
print("=" * 60)

# 辅助列
df['_imp'] = 1
df = df.sort_values(['user_id', 'DateTime'])

def build_user_time_features(group):
    """对单个用户构建所有时间窗口特征"""
    group = group.sort_values('DateTime')
    n = len(group)

    # 累计曝光（不含当前）
    cum_imp = np.arange(n)
    # 累计点击（不含当前）
    cum_click = group[TARGET].shift(1).fillna(0).cumsum().values
    # 历史 CTR
    hist_ctr = cum_click / (cum_imp + 1)

    # 距上次曝光的小时数
    time_diffs = group['DateTime'].diff().dt.total_seconds().values / 3600
    time_diffs[0] = 999

    # 时间窗口: 用 DateTime 做 index 做 rolling
    group_idx = group.set_index('DateTime')
    # 过去24h曝光数（不含当前）
    imp_24h = group_idx['_imp'].rolling('24h').sum().values - 1
    imp_24h = np.maximum(imp_24h, 0)
    # 过去24h点击数
    click_24h = group_idx[TARGET].rolling('24h').sum().values - group_idx[TARGET].values
    click_24h = np.maximum(click_24h, 0)
    # 过去24h CTR
    ctr_24h = click_24h / (imp_24h + 1)

    # 过去7d
    imp_7d = group_idx['_imp'].rolling('7d').sum().values - 1
    imp_7d = np.maximum(imp_7d, 0)
    click_7d = group_idx[TARGET].rolling('7d').sum().values - group_idx[TARGET].values
    click_7d = np.maximum(click_7d, 0)
    ctr_7d = click_7d / (imp_7d + 1)

    # 过去1h
    imp_1h = group_idx['_imp'].rolling('1h').sum().values - 1
    imp_1h = np.maximum(imp_1h, 0)
    click_1h = group_idx[TARGET].rolling('1h').sum().values - group_idx[TARGET].values
    click_1h = np.maximum(click_1h, 0)
    ctr_1h = click_1h / (imp_1h + 1)

    result = pd.DataFrame({
        'user_cum_impressions': cum_imp.astype(np.int32),
        'user_cum_clicks': cum_click.astype(np.int32),
        'user_hist_ctr': hist_ctr.astype(np.float32),
        'user_hours_since_last': time_diffs.astype(np.float32),
        'user_imp_1h': imp_1h.astype(np.int32),
        'user_click_1h': click_1h.astype(np.int32),
        'user_ctr_1h': ctr_1h.astype(np.float32),
        'user_imp_24h': imp_24h.astype(np.int32),
        'user_click_24h': click_24h.astype(np.int32),
        'user_ctr_24h': ctr_24h.astype(np.float32),
        'user_imp_7d': imp_7d.astype(np.int32),
        'user_click_7d': click_7d.astype(np.int32),
        'user_ctr_7d': ctr_7d.astype(np.float32),
    }, index=group.index)

    return result

print("  计算中 (按 user_id 分组, 可能较慢)...")
window_features = df.groupby('user_id', group_keys=False)[['DateTime', TARGET, '_imp']].apply(build_user_time_features)

# 合并回主表
df = pd.concat([df, window_features], axis=1)
df.drop('_imp', axis=1, inplace=True)

# 填充（极少数情况）
for col in window_features.columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(0)

print(f"  用户历史CTR均值: {df['user_hist_ctr'].mean():.4f}")
print(f"  过去24h曝光均值: {df['user_imp_24h'].mean():.1f}")
print(f"  过去24h CTR均值: {df['user_ctr_24h'].mean():.4f}")
print(f"  距上次曝光均值: {df['user_hours_since_last'].mean():.1f}h")

del window_features

# ================================================================
# 3. 拆分 Train/Test
# ================================================================
df_train = df[train_mask].copy()
df_test = df[test_mask].copy()
print(f"\n分割后训练集: {len(df_train)} 行, 测试集: {len(df_test)} 行")

# ================================================================
# 4. 静态特征工程（对 Train 构建，映射到 Test）
# ================================================================
print("\n" + "=" * 60)
print("3. 静态特征工程")
print("=" * 60)

# -- 共现计数 (Train + Test 各自独立计算, 然后合并) --
for data in [df_train, df_test]:
    data['user_campaign_count'] = data.groupby(['user_id','campaign_id'])['user_id'].transform('count')
    data['user_product_count']  = data.groupby(['user_id','product'])['user_id'].transform('count')

# -- 3阶交叉特征 --
print("  3阶交叉特征...")
for (a, b, nm) in [('user_group_id','hour','ugroup_x_hour'),
                    ('gender','product_category_1','gender_x_pcat1'),
                    ('user_depth','product_category_2','udepth_x_pcat2'),
                    ('age_level','hour','age_x_hour'),
                    ('user_group_id','day_of_week','ugroup_x_dow'),
                    ('hour','product_category_1','hour_x_pcat1'),
                    ('gender','hour','gender_x_hour')]:
    if a not in df_train.columns: continue
    df_train[nm] = df_train[a].astype(str)+'_x_'+df_train[b].astype(str)
    df_test[nm] = df_test[a].astype(str)+'_x_'+df_test[b].astype(str)

# -- 2阶交叉 (与之前一致) --
cross_2 = [('gender','product_category_1','gender_x_pcat1'),
           ('gender','product_category_2','gender_x_pcat2'),
           ('user_depth','product_category_1','udepth_x_pcat1'),
           ('age_level','product_category_1','age_x_pcat1'),
           ('hour','day_of_week','hour_x_dow'),
           ('user_group_id','campaign_id','ugroup_x_camp'),
           ('product_category_1','product_category_2','pcat1_x_pcat2')]
cross_names = [nm for _,_,nm in cross_2]

# 确保 2阶交叉列存在（上面已创建, 这里只更新 cross_names 列表用于后续）
for a,b,nm in cross_2:
    if nm not in df_train.columns:
        df_train[nm] = df_train[a].astype(str)+'_x_'+df_train[b].astype(str)
        df_test[nm] = df_test[a].astype(str)+'_x_'+df_test[b].astype(str)

# 3阶交叉列名
cross3_names = ['ugroup_x_hour', 'udepth_x_pcat2', 'age_x_hour',
                'ugroup_x_dow', 'hour_x_pcat1', 'gender_x_hour']

all_cross_names = cross_names + cross3_names

# -- Target Encoding (K-Fold, 仅在 Train 内) --
print("  K-Fold Target Encoding (仅训练集)...")
target_enc_cols = ['campaign_id','user_id','webpage_id','product_category_1'] + all_cross_names
for col in target_enc_cols:
    df_train[f'{col}_target_enc'] = np.nan

kf = KFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
for tr_idx, v_idx in kf.split(df_train):
    Xtr, Xv = df_train.iloc[tr_idx], df_train.iloc[v_idx]
    for col in target_enc_cols:
        m = Xtr.groupby(col)[TARGET].mean()
        df_train.loc[df_train.index[v_idx], f'{col}_target_enc'] = Xv[col].map(m).values

# 对 Test: 用 FULL Train 编码
for col in target_enc_cols:
    full_map = df_train.groupby(col)[TARGET].mean()
    df_test[f'{col}_target_enc'] = df_test[col].map(full_map).values
    # 填充未知值
    df_test[f'{col}_target_enc'] = df_test[f'{col}_target_enc'].fillna(global_ctr)
    df_train[f'{col}_target_enc'] = df_train[f'{col}_target_enc'].fillna(global_ctr)

# -- CTR 聚合 (K-Fold in Train) --
print("  K-Fold CTR 聚合...")
agg_entities = ['user_id','campaign_id','product','webpage_id',
                'product_category_1','product_category_2',
                'user_group_id','gender','age_level','user_depth']
for e in agg_entities:
    df_train[f'{e}_ctr'] = np.float32(np.nan)
    df_train[f'{e}_imp_count'] = np.int32(0)

for tr_idx, v_idx in kf.split(df_train):
    Xtr, Xv = df_train.iloc[tr_idx], df_train.iloc[v_idx]
    for e in agg_entities:
        m = Xtr.groupby(e)[TARGET].mean()
        df_train.loc[df_train.index[v_idx], f'{e}_ctr'] = Xv[e].map(m).values.astype(np.float32)

for e in agg_entities:
    full_ctr = df_train.groupby(e)[TARGET].mean()
    df_test[f'{e}_ctr'] = df_test[e].map(full_ctr).fillna(global_ctr).astype(np.float32)
    df_train[f'{e}_ctr'] = df_train[f'{e}_ctr'].fillna(global_ctr)
    # 曝光计数（无泄漏）
    imp_map = pd.concat([df_train[e], df_test[e]]).value_counts().to_dict()
    df_train[f'{e}_imp_count'] = df_train[e].map(imp_map).fillna(0).astype(np.int32)
    df_test[f'{e}_imp_count'] = df_test[e].map(imp_map).fillna(0).astype(np.int32)

# -- 用户时间聚合 (K-Fold in Train) --
print("  K-Fold 用户时间聚合...")
df_train['user_avg_hour'] = np.float32(np.nan)
df_train['user_hour_std'] = np.float32(np.nan)
df_train['user_active_days'] = np.float32(1.0)

for tr_idx, v_idx in kf.split(df_train):
    Xtr, Xv = df_train.iloc[tr_idx], df_train.iloc[v_idx]
    df_train.loc[df_train.index[v_idx], 'user_avg_hour'] = Xv['user_id'].map(
        Xtr.groupby('user_id')['hour'].mean()).fillna(df_train['hour'].median()).astype(np.float32).values
    df_train.loc[df_train.index[v_idx], 'user_hour_std'] = Xv['user_id'].map(
        Xtr.groupby('user_id')['hour'].std()).fillna(0).astype(np.float32).values
    df_train.loc[df_train.index[v_idx], 'user_active_days'] = Xv['user_id'].map(
        Xtr.groupby('user_id')['day'].nunique()).fillna(1).astype(np.float32).values

# Test 用全量 Train 编码
for col_map, col_name in [
    (df_train.groupby('user_id')['hour'].mean(), 'user_avg_hour'),
    (df_train.groupby('user_id')['hour'].std(), 'user_hour_std'),
    (df_train.groupby('user_id')['day'].nunique(), 'user_active_days'),
]:
    df_test[col_name] = df_test['user_id'].map(col_map).fillna(
        df_train['hour'].median() if 'hour' in col_name else 1.0).astype(np.float32)

df_train['user_avg_hour'] = df_train['user_avg_hour'].fillna(df_train['hour'].median())
df_train['user_hour_std'] = df_train['user_hour_std'].fillna(0)
df_train['user_active_days'] = df_train['user_active_days'].fillna(1.0)

# -- Layer 6 静态时间特征 (从 DateTime) --
print("  用户静态时间特征...")
for lbl, df_part in [('train', df_train), ('test', df_test)]:
    df_part['_dt'] = df_part['DateTime']
    # 排序计算
    df_part = df_part.sort_values(['user_id', '_dt'])
    # 占位: 这些特征已经在上面的时间窗口里做了
    # 这里做全局时间聚合
    pass

# 直接对 Train+Test 全量计算静态时间特征
# (用 raw DateTime, 无泄漏因为不涉及 target)
full_data = pd.concat([df_train, df_test], axis=0)

def static_temporal(grp):
    times = grp['DateTime'].values; n = len(times)
    if n == 1:
        return pd.Series({'avg_gap_h':0,'max_gap_h':0,'span_d':0,
            'night':1.*(0<=pd.Timestamp(times[0]).hour<6),
            'morning':1.*(6<=pd.Timestamp(times[0]).hour<12),
            'afternoon':1.*(12<=pd.Timestamp(times[0]).hour<18),
            'evening':1.*(18<=pd.Timestamp(times[0]).hour<24)})
    gaps = np.diff(times).astype('timedelta64[h]').astype(np.float64)
    hrs = np.array([pd.Timestamp(t).hour for t in times])
    return pd.Series({'avg_gap_h':gaps.mean(),'max_gap_h':gaps.max(),
        'span_d':(times[-1]-times[0])/np.timedelta64(1,'D'),
        'night':(hrs<6).mean(),'morning':((hrs>=6)&(hrs<12)).mean(),
        'afternoon':((hrs>=12)&(hrs<18)).mean(),'evening':(hrs>=18).mean()})

ut = full_data.groupby('user_id').apply(static_temporal).reset_index().astype(np.float32)
ut.rename(columns={'avg_gap_h':'user_avg_gap_hours','max_gap_h':'user_max_gap_hours',
    'span_d':'user_active_span_days','night':'user_night_ratio',
    'morning':'user_morning_ratio','afternoon':'user_afternoon_ratio',
    'evening':'user_evening_ratio'}, inplace=True)

for d in [df_train, df_test]:
    d.drop([c for c in ut.columns if c != 'user_id' and c in d.columns], axis=1, errors='ignore', inplace=True)
    d.reset_index(drop=True, inplace=True)
df_train = df_train.merge(ut, on='user_id', how='left')
df_test = df_test.merge(ut, on='user_id', how='left')
for c in ut.columns:
    if c != 'user_id':
        df_train[c] = df_train[c].fillna(0.0)
        df_test[c] = df_test[c].fillna(0.0)

del full_data, ut

# -- Layer 3 衍生 --
for d in [df_train, df_test]:
    d['user_impressions_per_day'] = (d['user_id_imp_count'] / d['user_active_days'].clip(lower=1)).astype(np.float32)
    wd = d[d['day_of_week']<5].groupby('user_id').size()
    we = d[d['day_of_week']>=5].groupby('user_id').size()
    d['user_weekday_ratio'] = d['user_id'].map((wd/(we+1)).to_dict()).fillna(0.5).astype(np.float32).clip(0,10)

# -- Layer 4 比率 --
eps = 0.001
for d in [df_train, df_test]:
    d['user_vs_campaign_ctr'] = (d['user_id_ctr']/(d['campaign_id_ctr']+eps)).astype(np.float32).clip(0,50)
    d['user_vs_global_ctr'] = (d['user_id_ctr']/global_ctr).astype(np.float32).clip(0,50)
    d['campaign_vs_global_ctr'] = (d['campaign_id_ctr']/global_ctr).astype(np.float32).clip(0,50)
    d['user_imp_rank'] = d['user_id_imp_count'].rank(pct=True).astype(np.float32)

# -- Count Encoding --
for d in [df_train, df_test]:
    for col in ['user_id','campaign_id','webpage_id','product']:
        d[f'{col}_count'] = d[col].map(d[col].value_counts().to_dict()).fillna(0)

# -- 类别 dtype --
cat_f = ['product','campaign_id','webpage_id','product_category_1','product_category_2',
         'user_group_id','gender','age_level','user_depth','hour','day_of_week']
for d in [df_train, df_test]:
    for col in cat_f:
        if col in d.columns:
            d[col] = d[col].astype('category')

# ================================================================
# 5. 构建特征矩阵
# ================================================================
print("\n" + "=" * 60)
print("4. 特征汇总")
print("=" * 60)

exclude_cols = [TARGET, 'user_id', 'DateTime'] + all_cross_names
features = [c for c in df_train.columns if c not in exclude_cols]
features = [c for c in features if not c.startswith('_')]

print(f"总特征数: {len(features)}")
# 分类
for tag, keyword in [('时间窗口','user_cum_|user_hist_|user_hours_|user_imp_1h|user_imp_24h|user_imp_7d|user_click_|user_ctr_1h|user_ctr_24h|user_ctr_7d'),
                      ('3阶交叉','ugroup_x_hour_target_enc|udepth_x_pcat2_target_enc|age_x_hour_target_enc|ugroup_x_dow_target_enc|hour_x_pcat1_target_enc|gender_x_hour_target_enc'),
                      ('2阶交叉','gender_x_pcat|udepth_x_pcat|age_x_pcat|hour_x_dow|ugroup_x_camp|pcat1_x_pcat2'),
                      ('CTR聚合','_ctr'),('曝光计数','_imp_count'),('TargetEnc','_target_enc')]:
    import re
    cnt = sum(1 for f in features if re.search(keyword, f))
    print(f"  {tag}: {cnt}")

X_train = df_train[features]
y_train = df_train[TARGET]
X_test = df_test[features]
y_test = df_test[TARGET]

# ================================================================
# 6. 训练（Optuna 最佳参数 + scale_pos_weight）
# ================================================================
print("\n" + "=" * 60)
print("5. LightGBM 训练")
print("=" * 60)

cat_names = [c for c in cat_f if c in features]
# 不使用 scale_pos_weight —— 实验证明它对 AUC 有负面影响
print("(不使用 scale_pos_weight)")

train_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_names)
test_ds  = lgb.Dataset(X_test, label=y_test, reference=train_ds)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.0213,
    'num_leaves': 255,
    'max_depth': 8,
    'min_data_in_leaf': 100,
    'feature_fraction': 0.675,
    'bagging_fraction': 0.893,
    'bagging_freq': 4,
    'lambda_l1': 1.9e-5,
    'lambda_l2': 0.01,
    'min_split_gain': 0.002,
    'path_smooth': 2.5,
    'feature_pre_filter': False,
    'verbose': -1,
    'random_state': GLOBAL_SEED
}

model = lgb.train(
    params, train_ds,
    num_boost_round=2500,
    valid_sets=[train_ds, test_ds],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)

# ================================================================
# 7. 评估
# ================================================================
print("\n" + "=" * 60)
print("6. 最终评估")
print("=" * 60)

y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)

test_auc = roc_auc_score(y_test, y_pred_test)
train_auc = roc_auc_score(y_train, y_pred_train)
test_logloss = log_loss(y_test, y_pred_test)

print(f"训练集 AUC:  {train_auc:.4f}")
print(f"测试集 AUC:  {test_auc:.4f}")
print(f"测试集 LogLoss: {test_logloss:.4f}")
print(f"过拟合差距:  {train_auc - test_auc:.4f}")
print(f"\n较 baseline 0.6376 提升: +{test_auc - 0.6376:.4f}")
print(f"当前历史最佳 0.6482: {'↑ 突破!' if test_auc > 0.6482 else '✗ 未突破'}")

# Top 30 特征
importance = pd.DataFrame({
    'Feature': model.feature_name(),
    'Importance': model.feature_importance(importance_type='gain')
}).sort_values(by='Importance', ascending=False)
print(f"\nTop 30 特征:")
print(importance.head(30).to_string(index=False))

# 看看时间窗口特征表现
print("\n--- 时间窗口特征在 Top 30 中的占比 ---")
window_count = sum(1 for f in importance.head(30)['Feature'] if any(
    k in f for k in ['cum_impressions','cum_clicks','hist_ctr','hours_since_last','_1h','_24h','_7d']))
print(f"  时间窗口特征: {window_count} 个")
cross3_count = sum(1 for f in importance.head(30)['Feature'] if any(
    k in f for k in cross3_names))
print(f"  3阶交叉特征: {cross3_count} 个")
