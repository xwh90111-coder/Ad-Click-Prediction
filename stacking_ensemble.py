# -*- coding: utf-8 -*-
"""
Stacking 集成模型 — LGB + XGBoost + CatBoost → LogisticRegression
==================================================================
5-Fold OOF 预测作为元特征，Logistic Regression 做 Meta-Learner。
"""

import sys, io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

TARGET = 'is_click'
GLOBAL_SEED = 42
N_FOLDS = 5

# ================================================================
# 特征工程（复用 optuna_tune_full.py 的 build_features）
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

    df['user_campaign_count'] = df.groupby(['user_id','campaign_id'])['user_id'].transform('count')
    df['user_product_count']  = df.groupby(['user_id','product'])['user_id'].transform('count')

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

    df['user_impressions_per_day'] = (df['user_id_imp_count'] / df['user_active_days'].clip(lower=1)).astype(np.float32)
    wd = df[df['day_of_week']<5].groupby('user_id').size()
    we = df[df['day_of_week']>=5].groupby('user_id').size()
    df['user_weekday_ratio'] = df['user_id'].map((wd/(we+1)).to_dict()).fillna(0.5).astype(np.float32).clip(0,10)

    eps=0.001
    df['user_vs_campaign_ctr'] = (df['user_id_ctr']/(df['campaign_id_ctr']+eps)).astype(np.float32).clip(0,50)
    df['user_vs_global_ctr'] = (df['user_id_ctr']/global_ctr).astype(np.float32).clip(0,50)
    df['campaign_vs_global_ctr'] = (df['campaign_id_ctr']/global_ctr).astype(np.float32).clip(0,50)
    df['user_imp_rank'] = df['user_id_imp_count'].rank(pct=True).astype(np.float32)

    for col in ['user_id','campaign_id','webpage_id','product']:
        df[f'{col}_count'] = df[col].map(df[col].value_counts().to_dict())

    cat_f = ['product','campaign_id','webpage_id','product_category_1','product_category_2',
             'user_group_id','gender','age_level','user_depth','hour','day_of_week']
    for col in cat_f:
        if col in df.columns: df[col] = df[col].astype('category')

    exclude = [TARGET, 'user_id'] + cross_names
    ft = [c for c in df.columns if c not in exclude]
    print(f"特征总数: {len(ft)}")

    return df, ft, cat_f


# ================================================================
# 主流程
# ================================================================
data_path = r"D:\projects\Ad Click Pridiction\archive\processed_train.csv"
raw_path = r"D:\projects\Ad Click Pridiction\archive\Ad_click_prediction_train (1).csv"

df, features, cat_features = build_features(data_path, raw_path)

X = df[features]; y = df[TARGET]
# 分出最终测试集
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=GLOBAL_SEED, stratify=y
)

cat_names = [c for c in cat_features if c in features]

print(f"\n训练集: {X_train_full.shape[0]}, 测试集: {X_test.shape[0]}")

# ================================================================
# 模型配置
# ================================================================
# LGB 使用 Optuna 最佳参数
lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'learning_rate': 0.0213, 'num_leaves': 235, 'max_depth': 8,
    'min_data_in_leaf': 106, 'feature_fraction': 0.675,
    'bagging_fraction': 0.893, 'bagging_freq': 4,
    'lambda_l1': 1.9e-5, 'lambda_l2': 0.00166,
    'min_split_gain': 0.002, 'path_smooth': 2.5,
    'feature_pre_filter': False,
    'verbose': -1, 'random_state': GLOBAL_SEED
}

# XGBoost 参数
xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'learning_rate': 0.03, 'max_depth': 8,
    'min_child_weight': 100, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'reg_alpha': 0.1,
    'reg_lambda': 1.0, 'random_state': GLOBAL_SEED,
    'verbosity': 0, 'n_jobs': -1
}

# CatBoost 参数
cb_params = {
    'iterations': 1000, 'learning_rate': 0.03, 'depth': 8,
    'eval_metric': 'AUC', 'random_seed': GLOBAL_SEED,
    'bagging_temperature': 0.2, 'od_type': 'Iter', 'od_wait': 80,
    'verbose': 0, 'allow_writing_files': False
}

# ================================================================
# 5-Fold 生成 OOF 预测
# ================================================================
print("\n" + "=" * 60)
print("5-Fold 交叉验证 - 生成 Stacking 元特征")
print("=" * 60)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=GLOBAL_SEED)

# OOF 预测
oof_lgb = np.zeros(len(X_train_full))
oof_xgb = np.zeros(len(X_train_full))
oof_cb  = np.zeros(len(X_train_full))

# 测试集预测（每 fold 平均）
test_pred_lgb = np.zeros(len(X_test))
test_pred_xgb = np.zeros(len(X_test))
test_pred_cb  = np.zeros(len(X_test))

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
    print(f"\n  Fold {fold_idx+1}/{N_FOLDS}...")

    X_tr = X_train_full.iloc[train_idx]
    X_vl = X_train_full.iloc[val_idx]
    y_tr = y_train_full.iloc[train_idx]
    y_vl = y_train_full.iloc[val_idx]

    # 为 XGBoost/CatBoost 准备数据（它们需要整数或字符串类别）
    X_tr_num = X_tr.copy()
    X_vl_num = X_vl.copy()
    X_ts_num = X_test.copy()
    for c in cat_names:
        if c in X_tr_num.columns:
            # 转为整数编码
            X_tr_num[c] = X_tr_num[c].cat.codes
            X_vl_num[c] = X_vl_num[c].cat.codes
            X_ts_num[c] = X_ts_num[c].cat.codes

    # -- LightGBM (原生支持 category) --
    tr_ds = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_names)
    vl_ds = lgb.Dataset(X_vl, label=y_vl, reference=tr_ds)
    lgb_model = lgb.train(lgb_params, tr_ds,
        num_boost_round=2000, valid_sets=[vl_ds],
        callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)])
    oof_lgb[val_idx] = lgb_model.predict(X_vl, num_iteration=lgb_model.best_iteration)
    test_pred_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    print(f"    LGB  AUC={roc_auc_score(y_vl, oof_lgb[val_idx]):.4f}, best_iter={lgb_model.best_iteration}")

    # -- XGBoost --
    xgb_model = xgb.XGBClassifier(**xgb_params, n_estimators=2000,
        early_stopping_rounds=80, enable_categorical=True)
    xgb_model.fit(X_tr_num, y_tr, eval_set=[(X_vl_num, y_vl)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_vl_num)[:, 1]
    test_pred_xgb += xgb_model.predict_proba(X_ts_num)[:, 1]
    print(f"    XGB  AUC={roc_auc_score(y_vl, oof_xgb[val_idx]):.4f}, best_iter={xgb_model.best_iteration}")

    # -- CatBoost --
    cb_cat_indices = [X_tr_num.columns.get_loc(c) for c in cat_names if c in X_tr_num.columns]
    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(X_tr_num, y_tr, cat_features=cb_cat_indices,
        eval_set=(X_vl_num, y_vl), use_best_model=True, verbose=0)
    oof_cb[val_idx] = cb_model.predict_proba(X_vl_num)[:, 1]
    test_pred_cb += cb_model.predict_proba(X_ts_num)[:, 1]
    print(f"    CB   AUC={roc_auc_score(y_vl, oof_cb[val_idx]):.4f}, best_iter={cb_model.best_iteration_}")

# 平均测试集预测
test_pred_lgb /= N_FOLDS
test_pred_xgb /= N_FOLDS
test_pred_cb  /= N_FOLDS

# ================================================================
# 各模型单独评估
# ================================================================
print("\n" + "=" * 60)
print("各 Base Model 单独评估")
print("=" * 60)

print(f"LightGBM OOF AUC: {roc_auc_score(y_train_full, oof_lgb):.4f}")
print(f"XGBoost  OOF AUC: {roc_auc_score(y_train_full, oof_xgb):.4f}")
print(f"CatBoost OOF AUC: {roc_auc_score(y_train_full, oof_cb):.4f}")

print(f"\nLightGBM Test AUC: {roc_auc_score(y_test, test_pred_lgb):.4f}")
print(f"XGBoost  Test AUC: {roc_auc_score(y_test, test_pred_xgb):.4f}")
print(f"CatBoost Test AUC: {roc_auc_score(y_test, test_pred_cb):.4f}")

# ================================================================
# Meta-Learner: Logistic Regression
# ================================================================
print("\n" + "=" * 60)
print("Stacking - Logistic Regression Meta-Learner")
print("=" * 60)

# 构建元特征矩阵
meta_train = np.column_stack([oof_lgb, oof_xgb, oof_cb])
meta_test  = np.column_stack([test_pred_lgb, test_pred_xgb, test_pred_cb])

# 添加交互项：各模型预测的差异
meta_train = np.column_stack([
    meta_train,
    oof_lgb - oof_xgb,    # LGB vs XGB 差异
    oof_lgb - oof_cb,     # LGB vs CB 差异
    (oof_lgb + oof_xgb) / 2,  # LGB+XGB 平均
    (oof_lgb + oof_cb) / 2,   # LGB+CB 平均
])
meta_test = np.column_stack([
    meta_test,
    test_pred_lgb - test_pred_xgb,
    test_pred_lgb - test_pred_cb,
    (test_pred_lgb + test_pred_xgb) / 2,
    (test_pred_lgb + test_pred_cb) / 2,
])

meta_lr = LogisticRegression(penalty=None, random_state=GLOBAL_SEED, max_iter=1000)
meta_lr.fit(meta_train, y_train_full)

stacked_pred = meta_lr.predict_proba(meta_test)[:, 1]
stacked_oof = meta_lr.predict_proba(meta_train)[:, 1]

print(f"Stacking OOF AUC:  {roc_auc_score(y_train_full, stacked_oof):.4f}")
print(f"Stacking Test AUC: {roc_auc_score(y_test, stacked_pred):.4f}")
print(f"Stacking LogLoss:  {log_loss(y_test, stacked_pred):.4f}")

# 简单平均融合（无学习）
simple_avg = (test_pred_lgb + test_pred_xgb + test_pred_cb) / 3
print(f"\n简单平均 Test AUC: {roc_auc_score(y_test, simple_avg):.4f}")

# ================================================================
# Meta-Learner 权重
# ================================================================
print(f"\nMeta-Learner 系数:")
coef_names = ['LGB', 'XGB', 'CB', 'LGB-XGB', 'LGB-CB', 'Avg(LGB,XGB)', 'Avg(LGB,CB)']
for name, coef in zip(coef_names, meta_lr.coef_[0]):
    print(f"  {name}: {coef:.4f}")
print(f"  Intercept: {meta_lr.intercept_[0]:.4f}")

print(f"\n相较 baseline 0.6376 提升: +{roc_auc_score(y_test, stacked_pred) - 0.6376:.4f}")
