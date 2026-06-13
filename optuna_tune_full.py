# -*- coding: utf-8 -*-
"""
Optuna 超参数调优 —— 在全面特征工程基础上搜索最优 LGB 参数
=============================================================
特征工程部分与 feature_engineering_full.py 完全一致。
特征构建一次，Optuna 搜索 25 轮。
"""

import sys, io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
import optuna
import warnings
warnings.filterwarnings('ignore')

TARGET = 'is_click'
GLOBAL_SEED = 42

# ================================================================
# 特征工程（同 feature_engineering_full.py）
# ================================================================
def build_features(data_path, raw_csv_path):
    print("=" * 60)
    print("构建特征集...")
    print("=" * 60)

    df = pd.read_csv(data_path)
    if 'session_id' in df.columns: df.drop('session_id', axis=1, inplace=True)
    if 'DateTime' in df.columns: df.drop('DateTime', axis=1, inplace=True)

    global_ctr = df[TARGET].mean()
    print(f"数据: {df.shape[0]} 行, 正样本率: {global_ctr:.4f}")

    # -- Layer 5: 共现计数 --
    df['user_campaign_count'] = df.groupby(['user_id','campaign_id'])['user_id'].transform('count')
    df['user_product_count']  = df.groupby(['user_id','product'])['user_id'].transform('count')

    # -- Layer 6: 时间行为 --
    raw = pd.read_csv(raw_csv_path, usecols=['user_id','DateTime'])
    raw['DateTime'] = pd.to_datetime(raw['DateTime'])
    raw = raw.sort_values(['user_id','DateTime'])

    def _temporal(grp):
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

    ut = raw.groupby('user_id').apply(_temporal).reset_index().astype(np.float32)
    ut.rename(columns={'avg_gap_h':'user_avg_gap_hours','max_gap_h':'user_max_gap_hours',
        'span_d':'user_active_span_days','night':'user_night_ratio',
        'morning':'user_morning_ratio','afternoon':'user_afternoon_ratio',
        'evening':'user_evening_ratio'}, inplace=True)
    df = df.merge(ut, on='user_id', how='left')
    for c in ut.columns:
        if c != 'user_id': df[c] = df[c].fillna(0.0)
    del raw, ut

    # -- Layer 1: 交叉特征 --
    cross = [('gender','product_category_1','gender_x_pcat1'),
             ('gender','product_category_2','gender_x_pcat2'),
             ('user_depth','product_category_1','udepth_x_pcat1'),
             ('age_level','product_category_1','age_x_pcat1'),
             ('hour','day_of_week','hour_x_dow'),
             ('user_group_id','campaign_id','ugroup_x_camp'),
             ('product_category_1','product_category_2','pcat1_x_pcat2')]
    cross_names = []
    for a,b,nm in cross:
        df[nm] = df[a].astype(str)+'_x_'+df[b].astype(str); cross_names.append(nm)

    # -- 5-Fold 编码 --
    target_enc_cols = ['campaign_id','user_id','webpage_id','product_category_1'] + cross_names
    agg_entities = ['user_id','campaign_id','product','webpage_id',
                    'product_category_1','product_category_2',
                    'user_group_id','gender','age_level','user_depth']

    for col in target_enc_cols: df[f'{col}_target_enc'] = np.nan
    for e in agg_entities:
        df[f'{e}_ctr'] = np.float32(np.nan); df[f'{e}_imp_count'] = np.int32(0)
    df['user_avg_hour'] = np.float32(np.nan)
    df['user_hour_std'] = np.float32(np.nan)
    df['user_active_days'] = np.float32(1.0)

    kf = KFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
    for tr_idx, v_idx in kf.split(df):
        Xtr, Xv = df.iloc[tr_idx], df.iloc[v_idx]
        for col in target_enc_cols:
            m = Xtr.groupby(col)[TARGET].mean()
            df.loc[v_idx, f'{col}_target_enc'] = Xv[col].map(m)
        for e in agg_entities:
            m = Xtr.groupby(e)[TARGET].mean()
            df.loc[v_idx, f'{e}_ctr'] = Xv[e].map(m).astype(np.float32)
        # 时间聚合
        df.loc[v_idx,'user_avg_hour'] = Xv['user_id'].map(Xtr.groupby('user_id')['hour'].mean()).astype(np.float32)
        df.loc[v_idx,'user_hour_std'] = Xv['user_id'].map(Xtr.groupby('user_id')['hour'].std()).astype(np.float32)
        df.loc[v_idx,'user_active_days'] = Xv['user_id'].map(Xtr.groupby('user_id')['day'].nunique()).fillna(1).astype(np.float32)

    for e in agg_entities:
        df[f'{e}_imp_count'] = df[e].map(df[e].value_counts().to_dict()).fillna(0).astype(np.int32)

    for col in target_enc_cols: df[f'{col}_target_enc'] = df[f'{col}_target_enc'].fillna(global_ctr)
    for e in agg_entities: df[f'{e}_ctr'] = df[f'{e}_ctr'].fillna(global_ctr)
    df['user_avg_hour'] = df['user_avg_hour'].fillna(df['hour'].median())
    df['user_hour_std'] = df['user_hour_std'].fillna(0)
    df['user_active_days'] = df['user_active_days'].fillna(1.0)

    # -- Layer 3: 时序衍生 --
    df['user_impressions_per_day'] = (df['user_id_imp_count'] / df['user_active_days'].clip(lower=1)).astype(np.float32)
    wd = df[df['day_of_week']<5].groupby('user_id').size()
    we = df[df['day_of_week']>=5].groupby('user_id').size()
    df['user_weekday_ratio'] = df['user_id'].map((wd/(we+1)).to_dict()).fillna(0.5).astype(np.float32).clip(0,10)

    # -- Layer 4: 比率 --
    eps=0.001
    df['user_vs_campaign_ctr'] = (df['user_id_ctr']/(df['campaign_id_ctr']+eps)).astype(np.float32).clip(0,50)
    df['user_vs_global_ctr'] = (df['user_id_ctr']/global_ctr).astype(np.float32).clip(0,50)
    df['campaign_vs_global_ctr'] = (df['campaign_id_ctr']/global_ctr).astype(np.float32).clip(0,50)
    df['user_imp_rank'] = df['user_id_imp_count'].rank(pct=True).astype(np.float32)

    # -- Count Encoding --
    for col in ['user_id','campaign_id','webpage_id','product']:
        df[f'{col}_count'] = df[col].map(df[col].value_counts().to_dict())

    # -- 类别转 category --
    cat_f = ['product','campaign_id','webpage_id','product_category_1','product_category_2',
             'user_group_id','gender','age_level','user_depth','hour','day_of_week']
    for col in cat_f:
        if col in df.columns: df[col] = df[col].astype('category')

    # -- 特征列表 --
    exclude = [TARGET, 'user_id'] + cross_names
    ft = [c for c in df.columns if c not in exclude]
    print(f"特征总数: {len(ft)}")

    return df, ft, cat_f, global_ctr


# ================================================================
# 主流程
# ================================================================
data_path = r"D:\projects\Ad Click Pridiction\archive\processed_train.csv"
raw_path = r"D:\projects\Ad Click Pridiction\archive\Ad_click_prediction_train (1).csv"

df, features, cat_features, global_ctr = build_features(data_path, raw_path)

# 划分
X = df[features]; y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=GLOBAL_SEED, stratify=y)
print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

# 为 Optuna 准备固定数据集
cat_names = [c for c in cat_features if c in features]
train_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_names)

# Optuna 内部用 80/20 分验证
from sklearn.model_selection import train_test_split as tts
X_tune, X_val, y_tune, y_val = tts(X_train, y_train, test_size=0.2, random_state=GLOBAL_SEED, stratify=y_train)
tune_ds = lgb.Dataset(X_tune, label=y_tune, categorical_feature=cat_names)
val_ds = lgb.Dataset(X_val, label=y_val, reference=tune_ds)

# ================================================================
# Optuna Objective
# ================================================================
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': GLOBAL_SEED,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 500),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 5.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 5.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 0.1, log=True),
        'path_smooth': trial.suggest_float('path_smooth', 0, 5.0),
        'feature_pre_filter': False,  # 避免跨 trial 的 min_data_in_leaf 冲突
    }

    try:
        model = lgb.train(
            params, tune_ds,
            num_boost_round=2000,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)]
        )

        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        return roc_auc_score(y_val, y_pred)
    except Exception as e:
        print(f"  Trial failed: {e}")
        return 0.0  # 失败返回最低分


# ================================================================
# 运行 Optuna
# ================================================================
optuna.logging.set_verbosity(optuna.logging.INFO)
print("\n" + "=" * 60)
print("启动 Optuna 超参数搜索 (25 Trials)")
print("=" * 60)

study = optuna.create_study(direction='maximize', study_name='LGB_FullFeatures')
study.optimize(objective, n_trials=25, show_progress_bar=True)

print(f"\n最佳 AUC (验证集): {study.best_value:.4f}")
print("最佳参数:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# ================================================================
# 用最佳参数重新训练并评估
# ================================================================
print("\n" + "=" * 60)
print("用最佳参数在完整训练集上重新训练...")
print("=" * 60)

best_params = study.best_params.copy()
best_params.update({
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'random_state': GLOBAL_SEED,
})

test_ds = lgb.Dataset(X_test, label=y_test, reference=train_ds)

final_model = lgb.train(
    best_params, train_ds,
    num_boost_round=2000,
    valid_sets=[train_ds, test_ds],
    callbacks=[
        lgb.early_stopping(stopping_rounds=80, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)

y_pred_test = final_model.predict(X_test, num_iteration=final_model.best_iteration)
y_pred_train = final_model.predict(X_train, num_iteration=final_model.best_iteration)

test_auc = roc_auc_score(y_test, y_pred_test)
train_auc = roc_auc_score(y_train, y_pred_train)

print(f"\n=== 最终评估 ===")
print(f"训练集 AUC:  {train_auc:.4f}")
print(f"测试集 AUC:  {test_auc:.4f}")
print(f"过拟合差距:  {train_auc - test_auc:.4f}")
print(f"较 baseline 0.6376 提升: +{test_auc - 0.6376:.4f}")
