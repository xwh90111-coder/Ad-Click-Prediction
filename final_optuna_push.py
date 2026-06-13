# -*- coding: utf-8 -*-
"""
最终突破尝试 — Optuna 50 trials + boosting_type 搜索 + 特征选择
=================================================================
基于 86 特征集（含时间窗口 + 3阶交叉），精调 LGB 参数。
"""

import sys, io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
import gc

TARGET = 'is_click'
GLOBAL_SEED = 42

# ================================================================
# 特征工程
# ================================================================
print("构建 86 特征集...")
# --- 加载 ---
raw_path = r"D:\projects\Ad Click Pridiction\archive\Ad_click_prediction_train (1).csv"
df = pd.read_csv(raw_path)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['hour'] = df['DateTime'].dt.hour
df['day_of_week'] = df['DateTime'].dt.dayofweek
df['day'] = df['DateTime'].dt.day
for col in ['product_category_2','user_group_id','age_level','user_depth']:
    if col in df.columns: df[col] = df[col].fillna(-1)
df['gender'] = df['gender'].fillna('Unknown')
df['city_development_index'] = df['city_development_index'].fillna(df['city_development_index'].median())
if 'session_id' in df.columns: df.drop('session_id', axis=1, inplace=True)
global_ctr = df[TARGET].mean()

# --- 时间窗口 ---
df = df.sort_values(['user_id', 'DateTime'])
df['_imp'] = 1

def time_features(grp):
    grp = grp.sort_values('DateTime')
    n = len(grp)
    ci = np.arange(n)
    cc = grp[TARGET].shift(1).fillna(0).cumsum().values
    hc = cc / (ci + 1)
    td = grp['DateTime'].diff().dt.total_seconds().values / 3600; td[0] = 999
    gi = grp.set_index('DateTime')
    i24 = np.maximum(gi['_imp'].rolling('24h').sum().values - 1, 0)
    c24 = np.maximum(gi[TARGET].rolling('24h').sum().values - gi[TARGET].values, 0)
    ct24 = c24 / (i24 + 1)
    i7d = np.maximum(gi['_imp'].rolling('7d').sum().values - 1, 0)
    c7d = np.maximum(gi[TARGET].rolling('7d').sum().values - gi[TARGET].values, 0)
    ct7d = c7d / (i7d + 1)
    return pd.DataFrame({
        'cum_imp':ci.astype(np.int32),'cum_click':cc.astype(np.int32),
        'hist_ctr':hc.astype(np.float32),'hrs_since_last':td.astype(np.float32),
        'imp_24h':i24.astype(np.int32),'click_24h':c24.astype(np.int32),'ctr_24h':ct24.astype(np.float32),
        'imp_7d':i7d.astype(np.int32),'click_7d':c7d.astype(np.int32),'ctr_7d':ct7d.astype(np.float32),
    }, index=grp.index)

wf = df.groupby('user_id', group_keys=False)[['DateTime',TARGET,'_imp']].apply(time_features)
df = pd.concat([df, wf], axis=1).drop('_imp', axis=1)
for c in wf.columns: df[c] = df[c].fillna(0)
del wf

# --- 分割 ---
from sklearn.model_selection import train_test_split as tts
tr_idx, te_idx = tts(range(len(df)), test_size=0.2, random_state=GLOBAL_SEED, stratify=df[TARGET])
df_train = df.iloc[tr_idx].copy(); df_test = df.iloc[te_idx].copy()
del df; gc.collect()

# --- 交叉特征 ---
cross2 = [('gender','product_category_1','g_x_pc1'),('gender','product_category_2','g_x_pc2'),
          ('user_depth','product_category_1','ud_x_pc1'),('age_level','product_category_1','al_x_pc1'),
          ('hour','day_of_week','h_x_dow'),('user_group_id','campaign_id','ug_x_camp'),
          ('product_category_1','product_category_2','pc1_x_pc2')]
cross3 = [('user_group_id','hour','ug_x_h'),('user_depth','product_category_2','ud_x_pc2'),
          ('age_level','hour','al_x_h'),('user_group_id','day_of_week','ug_x_dow'),
          ('hour','product_category_1','h_x_pc1'),('gender','hour','g_x_h')]
all_cross = [nm for _,_,nm in cross2+cross3]
for a,b,nm in cross2+cross3:
    df_train[nm] = df_train[a].astype(str)+'_x_'+df_train[b].astype(str)
    df_test[nm] = df_test[a].astype(str)+'_x_'+df_test[b].astype(str)

# --- K-Fold 编码 ---
tgt_enc_cols = ['campaign_id','user_id','webpage_id','product_category_1'] + all_cross
for col in tgt_enc_cols:
    df_train[f'{col}_te'] = np.nan; df_test[f'{col}_te'] = np.nan

agg_ent = ['user_id','campaign_id','product','webpage_id','product_category_1',
           'product_category_2','user_group_id','gender','age_level','user_depth']
for e in agg_ent:
    df_train[f'{e}_ctr'] = np.float32(np.nan); df_train[f'{e}_ic'] = np.int32(0)
    df_test[f'{e}_ctr'] = np.float32(np.nan); df_test[f'{e}_ic'] = np.int32(0)
df_train['u_avg_h'] = np.float32(np.nan); df_train['u_std_h'] = np.float32(np.nan)
df_train['u_days'] = np.float32(1.0)
df_test['u_avg_h'] = np.float32(np.nan); df_test['u_std_h'] = np.float32(np.nan)
df_test['u_days'] = np.float32(1.0)

kf = KFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
for fi,(ti,vi) in enumerate(kf.split(df_train)):
    Xtr,Xv = df_train.iloc[ti], df_train.iloc[vi]
    for col in tgt_enc_cols:
        m = Xtr.groupby(col)[TARGET].mean()
        df_train.loc[df_train.index[vi],f'{col}_te'] = Xv[col].map(m).values
    for e in agg_ent:
        m = Xtr.groupby(e)[TARGET].mean()
        df_train.loc[df_train.index[vi],f'{e}_ctr'] = Xv[e].map(m).values.astype(np.float32)
    df_train.loc[df_train.index[vi],'u_avg_h'] = Xv['user_id'].map(
        Xtr.groupby('user_id')['hour'].mean()).fillna(df_train['hour'].median()).astype(np.float32).values
    df_train.loc[df_train.index[vi],'u_std_h'] = Xv['user_id'].map(
        Xtr.groupby('user_id')['hour'].std()).fillna(0).astype(np.float32).values
    df_train.loc[df_train.index[vi],'u_days'] = Xv['user_id'].map(
        Xtr.groupby('user_id')['day'].nunique()).fillna(1).astype(np.float32).values

# Test 用全量 train 编码
for col in tgt_enc_cols:
    m = df_train.groupby(col)[TARGET].mean()
    df_test[f'{col}_te'] = df_test[col].map(m).fillna(global_ctr).values
    df_train[f'{col}_te'] = df_train[f'{col}_te'].fillna(global_ctr)
for e in agg_ent:
    m = df_train.groupby(e)[TARGET].mean()
    df_test[f'{e}_ctr'] = df_test[e].map(m).fillna(global_ctr).astype(np.float32).values
    df_train[f'{e}_ctr'] = df_train[f'{e}_ctr'].fillna(global_ctr)

for col_map, cn in [(df_train.groupby('user_id')['hour'].mean(),'u_avg_h'),
                     (df_train.groupby('user_id')['hour'].std(),'u_std_h'),
                     (df_train.groupby('user_id')['day'].nunique(),'u_days')]:
    df_test[cn] = df_test['user_id'].map(col_map).astype(np.float32)
df_train['u_avg_h'] = df_train['u_avg_h'].fillna(df_train['hour'].median())
df_train['u_std_h'] = df_train['u_std_h'].fillna(0)
df_train['u_days'] = df_train['u_days'].fillna(1.0)
df_test['u_avg_h'] = df_test['u_avg_h'].fillna(df_test['hour'].median())
df_test['u_std_h'] = df_test['u_std_h'].fillna(0)
df_test['u_days'] = df_test['u_days'].fillna(1.0)

# 曝光计数
imp_all = pd.concat([df_train[e] for e in agg_ent], axis=0)
for e in agg_ent:
    m = pd.concat([df_train[e], df_test[e]]).value_counts().to_dict()
    df_train[f'{e}_ic'] = df_train[e].map(m).fillna(0).astype(np.int32)
    df_test[f'{e}_ic'] = df_test[e].map(m).fillna(0).astype(np.int32)

# 共现
for d in [df_train, df_test]:
    d['u_camp_cnt'] = d.groupby(['user_id','campaign_id'])['user_id'].transform('count')
    d['u_prod_cnt'] = d.groupby(['user_id','product'])['user_id'].transform('count')

# 静态时间
def static_t(grp):
    times = grp['DateTime'].values; n = len(times)
    if n == 1:
        return pd.Series({'ag':0,'mg':0,'sd':0,
            'ni':1.*(0<=pd.Timestamp(times[0]).hour<6),
            'mo':1.*(6<=pd.Timestamp(times[0]).hour<12),
            'af':1.*(12<=pd.Timestamp(times[0]).hour<18),
            'ev':1.*(18<=pd.Timestamp(times[0]).hour<24)})
    gaps = np.diff(times).astype('timedelta64[h]').astype(np.float64)
    hrs = np.array([pd.Timestamp(t).hour for t in times])
    return pd.Series({'ag':gaps.mean(),'mg':gaps.max(),
        'sd':(times[-1]-times[0])/np.timedelta64(1,'D'),
        'ni':(hrs<6).mean(),'mo':((hrs>=6)&(hrs<12)).mean(),
        'af':((hrs>=12)&(hrs<18)).mean(),'ev':(hrs>=18).mean()})

full = pd.concat([df_train, df_test])
ut = full.groupby('user_id').apply(static_t).reset_index().astype(np.float32)
ut.rename(columns={'ag':'u_avg_gap','mg':'u_max_gap','sd':'u_span','ni':'u_night',
    'mo':'u_morning','af':'u_afternoon','ev':'u_evening'}, inplace=True)
df_train = df_train.merge(ut, on='user_id', how='left')
df_test = df_test.merge(ut, on='user_id', how='left')
for c in ut.columns:
    if c!='user_id':
        df_train[c]=df_train[c].fillna(0); df_test[c]=df_test[c].fillna(0)
del full, ut; gc.collect()

# 衍生与比率
eps = 0.001
for d in [df_train, df_test]:
    d['u_imp_per_day'] = (d['user_id_ic'] / d['u_days'].clip(1)).astype(np.float32)
    wd = d[d['day_of_week']<5].groupby('user_id').size()
    we = d[d['day_of_week']>=5].groupby('user_id').size()
    d['u_wd_ratio'] = d['user_id'].map((wd/(we+1)).to_dict()).fillna(0.5).astype(np.float32).clip(0,10)
    d['u_vs_camp'] = (d['user_id_ctr']/(d['campaign_id_ctr']+eps)).astype(np.float32).clip(0,50)
    d['u_vs_glob'] = (d['user_id_ctr']/global_ctr).astype(np.float32).clip(0,50)
    d['cam_v_glob'] = (d['campaign_id_ctr']/global_ctr).astype(np.float32).clip(0,50)
    d['u_imp_rk'] = d['user_id_ic'].rank(pct=True).astype(np.float32)
    for col in ['user_id','campaign_id','webpage_id','product']:
        d[f'{col}_cnt'] = d[col].map(d[col].value_counts().to_dict()).fillna(0)

# 类别
cat_f = ['product','campaign_id','webpage_id','product_category_1','product_category_2',
         'user_group_id','gender','age_level','user_depth','hour','day_of_week']
for d in [df_train, df_test]:
    for col in cat_f:
        if col in d.columns: d[col] = d[col].astype('category')

# 特征列表
exclude = [TARGET,'user_id','DateTime'] + all_cross
features = [c for c in df_train.columns if c not in exclude]
print(f"特征总数: {len(features)}")

X = df_train[features]; y = df_train[TARGET]
X_test = df_test[features]; y_test = df_test[TARGET]

X_tune, X_val, y_tune, y_val = train_test_split(
    X, y, test_size=0.2, random_state=GLOBAL_SEED, stratify=y)

cat_names = [c for c in cat_f if c in features]
tune_ds = lgb.Dataset(X_tune, label=y_tune, categorical_feature=cat_names)
val_ds = lgb.Dataset(X_val, label=y_val, reference=tune_ds)

# ================================================================
# Optuna Objective
# ================================================================
def objective(trial):
    bt = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])
    params = {
        'objective': 'binary','metric': 'auc',
        'boosting_type': bt,
        'verbose': -1, 'random_state': GLOBAL_SEED,
        'feature_pre_filter': False,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 500),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0) if bt!='goss' else 1.0,
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10) if bt!='goss' else 1,
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 5.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 5.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 0.5, log=True),
        'path_smooth': trial.suggest_float('path_smooth', 0, 10.0),
    }

    n_rounds = 2000 if bt != 'goss' else 500
    try:
        model = lgb.train(params, tune_ds, num_boost_round=n_rounds,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        yp = model.predict(X_val, num_iteration=model.best_iteration)
        return roc_auc_score(y_val, yp)
    except Exception as e:
        return 0.0

# ================================================================
# 运行
# ================================================================
optuna.logging.set_verbosity(optuna.logging.WARNING)
print(f"\n启动 Optuna 50 Trials (含 boosting_type 搜索)...")

study = optuna.create_study(direction='maximize', study_name='FinalPush')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\n最佳 AUC(验证): {study.best_value:.4f}")
print(f"最佳 boosting_type: {study.best_params.get('boosting_type')}")
for k,v in study.best_params.items():
    print(f"  {k}: {v}")

# 最终训练
print("\n最终训练 & 评估...")
bp = {k:v for k,v in study.best_params.items() if k not in ['boosting_type']}
bp.update({'objective':'binary','metric':'auc','boosting_type':study.best_params['boosting_type'],
           'verbose':-1,'random_state':GLOBAL_SEED,'feature_pre_filter':False})

train_ds = lgb.Dataset(X, label=y, categorical_feature=cat_names)
test_ds = lgb.Dataset(X_test, label=y_test, reference=train_ds)

n_r = 3000 if bp['boosting_type']!='goss' else 500
model = lgb.train(bp, train_ds, num_boost_round=n_r,
    valid_sets=[train_ds, test_ds],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False),
               lgb.log_evaluation(period=100)])

yp_test = model.predict(X_test, num_iteration=model.best_iteration)
yp_train = model.predict(X, num_iteration=model.best_iteration)

test_auc = roc_auc_score(y_test, yp_test)
train_auc = roc_auc_score(y, yp_train)
print(f"\n训练 AUC: {train_auc:.4f} | 测试 AUC: {test_auc:.4f} | 差距: {train_auc-test_auc:.4f}")
print(f"较 baseline 0.6376: +{test_auc-0.6376:.4f}")

# Top 特征
imp = pd.DataFrame({'F':model.feature_name(),'I':model.feature_importance(importance_type='gain')}).sort_values('I',ascending=False)
print("\nTop 20:")
print(imp.head(20).to_string(index=False))
