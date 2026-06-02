#!/usr/bin/env python3
"""
gbdt.py — GBDT phase (RF / XGBoost / CatBoost / XGBoost-Optuna)
Saves checkpoint to checkpoint.json after every model.
Run BEFORE dl.py (must be separate processes — xgboost+torch conflict on macOS).
"""
import json, time, warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────
HERE  = Path(__file__).parent
DATA  = Path('/Users/buttegg/School_Projects/_thesis_env/data')
CKPT  = HERE / 'checkpoint.json'
LOGS  = HERE / 'logs'
LOGS.mkdir(exist_ok=True)

AEOLUS_FEATURES = [
    'OP_CARRIER_ENC','OP_CARRIER_FL_NUM','FL_DAY','FL_WEEK',
    'ORIGIN_INDEX','DEST_INDEX','CRS_DEP_TIME_MIN','CRS_ARR_TIME_MIN',
    'CRS_ELAPSED_TIME','O_TEMP','O_PRCP','O_WSPD',
    'D_TEMP','D_PRCP','D_WSPD','O_LATITUDE','O_LONGITUDE','D_LATITUDE','D_LONGITUDE'
]
OUR_FEATURES = [
    'OP_CARRIER_ENC','OP_CARRIER_FL_NUM','FL_YEAR','FL_MONTH','FL_DAY','FL_WEEK',
    'ORIGIN_INDEX','DEST_INDEX','CRS_DEP_TIME_MIN','CRS_ARR_TIME_MIN',
    'CRS_ELAPSED_TIME','FLIGHTS','O_TEMP','O_PRCP','O_WSPD',
    'D_TEMP','D_PRCP','D_WSPD','O_LATITUDE','O_LONGITUDE','D_LATITUDE','D_LONGITUDE'
]

# ── Checkpoint ─────────────────────────────────────────────────
def load_ckpt():
    if not CKPT.exists(): return {}
    with open(CKPT) as f: flat = json.load(f)
    d = {}
    for k, v in flat.items():
        p = k.split('__')
        if len(p)==3: d.setdefault(p[0],{}).setdefault(p[1],{})[p[2]] = v
    return d

def save_ckpt(all_res):
    out = {}
    for tgt,d in all_res.items():
        for setup,res in d.items():
            for m,v in res.items():
                out[f"{tgt}__{setup}__{m}"] = round(v,6)
    with open(CKPT,'w') as f: json.dump(out,f,indent=2)
    print(f"  [saved {len(out)} entries → {CKPT.name}]", flush=True)

# ── Models ─────────────────────────────────────────────────────
def run_rf(X_tr,y_tr,X_te,y_te):
    m = RandomForestClassifier(n_estimators=500,max_depth=15,
        min_samples_leaf=50,min_samples_split=100,
        class_weight='balanced',random_state=42,n_jobs=-1)
    m.fit(X_tr,y_tr)
    return roc_auc_score(y_te, m.predict_proba(X_te)[:,1])

def run_xgb(X_tr,y_tr,X_val,y_val,X_te,y_te):
    pw = (y_tr==0).sum()/max((y_tr==1).sum(),1)
    m  = xgb.XGBClassifier(n_estimators=1000,max_depth=6,learning_rate=0.1,
         subsample=0.8,colsample_bytree=0.8,min_child_weight=5,
         scale_pos_weight=pw,eval_metric='auc',early_stopping_rounds=50,
         random_state=42,n_jobs=-1,verbosity=0)
    m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False)
    return roc_auc_score(y_te, m.predict_proba(X_te)[:,1])

def run_catboost(X_tr,y_tr,X_val,y_val,X_te,y_te):
    m = CatBoostClassifier(iterations=1000,depth=6,learning_rate=0.05,
        auto_class_weights='Balanced',eval_metric='AUC',
        early_stopping_rounds=50,random_seed=42,verbose=0,thread_count=-1)
    m.fit(X_tr,y_tr,eval_set=(X_val,y_val),verbose=False)
    return roc_auc_score(y_te, m.predict_proba(X_te)[:,1])

def run_xgb_optuna(X_tr,y_tr,X_val,y_val,X_te,y_te,n_trials=50):
    def obj(trial):
        p = dict(n_estimators=2000,
            max_depth=trial.suggest_int('max_depth',3,12),
            learning_rate=trial.suggest_float('learning_rate',0.001,0.3,log=True),
            subsample=trial.suggest_float('subsample',0.5,1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree',0.3,1.0),
            min_child_weight=trial.suggest_int('min_child_weight',1,50),
            gamma=trial.suggest_float('gamma',0,10),
            reg_alpha=trial.suggest_float('reg_alpha',1e-8,10,log=True),
            reg_lambda=trial.suggest_float('reg_lambda',1e-8,10,log=True),
            scale_pos_weight=trial.suggest_float('scale_pos_weight',1,5),
            eval_metric='auc',early_stopping_rounds=50,
            random_state=42,n_jobs=-1,verbosity=0)
        m = xgb.XGBClassifier(**p)
        m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False)
        return roc_auc_score(y_val, m.predict_proba(X_val)[:,1])
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj,n_trials=n_trials,show_progress_bar=False)
    bp = study.best_params
    bp.update(dict(n_estimators=2000,eval_metric='auc',early_stopping_rounds=50,
                   random_state=42,n_jobs=-1,verbosity=0))
    m = xgb.XGBClassifier(**bp)
    m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False)
    return roc_auc_score(y_te, m.predict_proba(X_te)[:,1])

# ── Main ───────────────────────────────────────────────────────
def main():
    print("="*60, flush=True)
    print("GBDT Phase — RF / XGBoost / CatBoost / XGB-Optuna", flush=True)
    print("="*60, flush=True)

    df       = pd.read_parquet(DATA / 'aeolus_feat.parquet')
    all_res  = load_ckpt()
    n_cached = sum(len(v) for d in all_res.values() for v in d.values())
    print(f"  Loaded {n_cached} cached results", flush=True)

    for tgt_col,tgt_name in [('arr_delayed_15','ARR'),('dep_delayed_15','DEP')]:
        print(f"\n{'#'*55}\n  {tgt_name}_Delay\n{'#'*55}", flush=True)

        for setup_name,feat_cols,scaler_mode,do_optuna in [
            ('aeolus', AEOLUS_FEATURES, 'per_split',  False),
            ('ours',   OUR_FEATURES,    'train_only',  True),
        ]:
            tr  = df[df['split']=='train']
            val = df[df['split']=='val']
            te  = df[df['split']=='test']

            if scaler_mode == 'per_split':
                X_tr  = StandardScaler().fit_transform(tr[feat_cols].values.astype('float32'))
                X_val = StandardScaler().fit_transform(val[feat_cols].values.astype('float32'))
                X_te  = StandardScaler().fit_transform(te[feat_cols].values.astype('float32'))
            else:
                sc    = StandardScaler()
                X_tr  = sc.fit_transform(tr[feat_cols].values.astype('float32'))
                X_val = sc.transform(val[feat_cols].values.astype('float32'))
                X_te  = sc.transform(te[feat_cols].values.astype('float32'))

            y_tr  = tr[tgt_col].values.astype('float32')
            y_val = val[tgt_col].values.astype('float32')
            y_te  = te[tgt_col].values.astype('float32')

            existing = all_res.get(tgt_name,{}).get(setup_name,{})
            print(f"\n  ── {tgt_name}/{setup_name} ──", flush=True)

            def run(key, fn):
                if key in existing:
                    print(f"  [{key:<14}] AUC={existing[key]:.4f}  (cached)")
                    return existing[key]
                print(f"  [{key:<14}] ", end='', flush=True)
                t = time.time()
                v = fn()
                print(f"AUC={v:.4f}  ({time.time()-t:.0f}s)", flush=True)
                all_res.setdefault(tgt_name,{}).setdefault(setup_name,{})[key] = v
                save_ckpt(all_res)
                return v

            run('RF',       lambda: run_rf(X_tr,y_tr,X_te,y_te))
            run('XGBoost',  lambda: run_xgb(X_tr,y_tr,X_val,y_val,X_te,y_te))
            run('CatBoost', lambda: run_catboost(X_tr,y_tr,X_val,y_val,X_te,y_te))
            if do_optuna:
                run('XGB_Optuna', lambda: run_xgb_optuna(X_tr,y_tr,X_val,y_val,X_te,y_te))

    print(f"\n{'='*60}", flush=True)
    print("GBDT DONE — run dl.py next", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == '__main__':
    main()
