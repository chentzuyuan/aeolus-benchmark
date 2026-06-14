#!/usr/bin/env python3
"""
dl.py — DL phase (MLP / ResNet / AutoInt)
Reads checkpoint.json written by gbdt.py and adds DL results.

MUST run as a separate process from gbdt.py.
XGBoost + PyTorch in the same process causes segfault on macOS Apple Silicon.
"""
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────
HERE = Path(__file__).parent
DATA = Path('/path/to/School_Projects/_thesis_env/data')
CKPT = HERE / 'checkpoint.json'
LOGS = HERE / 'logs'
LOGS.mkdir(exist_ok=True)

# Use CPU — MPS causes OOM kills on long training runs
DEVICE = torch.device('cpu')

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

AEOLUS_REF = {
    'ARR': {'MLP':0.600,'AutoInt':0.623,'ResNet':0.557},
    'DEP': {'MLP':0.627},
}

# ── Models ─────────────────────────────────────────────────────
class MLP_Aeolus(nn.Module):
    def __init__(self, n, d=128, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n,d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(d,d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(d,d//2), nn.BatchNorm1d(d//2), nn.ReLU(), nn.Linear(d//2,1))
    def forward(self, x): return self.net(x).squeeze(1)

class ResNet_Aeolus(nn.Module):
    def __init__(self, n, d=128, nb=4, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(n, d)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(d,d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(d,d), nn.BatchNorm1d(d)) for _ in range(nb)])
        self.relu = nn.ReLU()
        self.head = nn.Linear(d, 1)
    def forward(self, x):
        h = self.proj(x)
        for blk in self.blocks: h = self.relu(h + blk(h))
        return self.head(h).squeeze(1)

class AutoInt_Aeolus(nn.Module):
    def __init__(self, n, d=128, nh=4, nl=4, drop=0.1):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Linear(1,d) for _ in range(n)])
        self.layers = nn.ModuleList([nn.MultiheadAttention(d,nh,dropout=drop,batch_first=True) for _ in range(nl)])
        self.norms  = nn.ModuleList([nn.LayerNorm(d) for _ in range(nl)])
        self.head   = nn.Sequential(nn.Linear(n*d,256), nn.ReLU(), nn.Dropout(drop), nn.Linear(256,1))
        self.n = n
    def forward(self, x):
        e = torch.stack([self.embeds[i](x[:,i:i+1]) for i in range(self.n)], dim=1)
        for attn, norm in zip(self.layers, self.norms):
            out, _ = attn(e,e,e); e = norm(e+out)
        return self.head(e.reshape(e.size(0),-1)).squeeze(1)

class MLP_Ours(nn.Module):
    def __init__(self, n, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(128,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x): return self.net(x).squeeze(1)

class ResNet_Ours(nn.Module):
    def __init__(self, n, d=256, nb=4, drop=0.2):
        super().__init__()
        self.proj = nn.Linear(n, d)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(d,d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(d,d), nn.BatchNorm1d(d)) for _ in range(nb)])
        self.relu = nn.ReLU()
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d,1))
    def forward(self, x):
        h = self.proj(x)
        for blk in self.blocks: h = self.relu(h + blk(h))
        return self.head(h).squeeze(1)

class AutoInt_Ours(nn.Module):
    def __init__(self, n, d=16, nh=4, nl=3, drop=0.1):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Linear(1,d) for _ in range(n)])
        self.layers = nn.ModuleList([nn.MultiheadAttention(d,nh,dropout=drop,batch_first=True) for _ in range(nl)])
        self.norms  = nn.ModuleList([nn.LayerNorm(d) for _ in range(nl)])
        self.head   = nn.Sequential(nn.Linear(n*d,256), nn.ReLU(), nn.Dropout(drop), nn.Linear(256,1))
        self.n = n
    def forward(self, x):
        e = torch.stack([self.embeds[i](x[:,i:i+1]) for i in range(self.n)], dim=1)
        for attn, norm in zip(self.layers, self.norms):
            out, _ = attn(e,e,e); e = norm(e+out)
        return self.head(e.reshape(e.size(0),-1)).squeeze(1)

# ── Training ───────────────────────────────────────────────────
def train_dl(model, X_tr, y_tr, X_val, y_val, lr=1e-3, patience=10, max_epochs=100):
    pos  = y_tr.sum(); neg = len(y_tr)-pos
    pw   = torch.tensor([neg/max(pos,1)], dtype=torch.float32)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched= torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5, min_lr=1e-5)
    ds   = TensorDataset(torch.tensor(X_tr,dtype=torch.float32),
                         torch.tensor(y_tr,dtype=torch.float32))
    loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
    Xv   = torch.tensor(X_val, dtype=torch.float32)

    best_auc, best_state, wait = 0.0, None, 0
    for epoch in range(1, max_epochs+1):
        model.train()
        for Xb,yb in loader:
            opt.zero_grad(); crit(model(Xb),yb).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vp = torch.sigmoid(model(Xv)).numpy()
        va = roc_auc_score(y_val, vp)
        sched.step(-va)
        if va > best_auc:
            best_auc = va
            best_state = {k:v.clone() for k,v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience: break
        if epoch % 10 == 0:
            print(f"      ep{epoch:3d} val={va:.4f} best={best_auc:.4f}", flush=True)
    model.load_state_dict(best_state)
    return best_auc

def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X,dtype=torch.float32))).numpy()

# ── Checkpoint ─────────────────────────────────────────────────
def load_ckpt():
    if not CKPT.exists(): return {}
    with open(CKPT) as f: flat = json.load(f)
    d = {}
    for k,v in flat.items():
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

# ── Final table ────────────────────────────────────────────────
def print_table(all_res):
    print(f"\n{'='*75}")
    print("FINAL COMPARISON TABLE  (287,845 flights, June 1-15 2024)")
    print(f"{'='*75}")
    for tgt in ['ARR','DEP']:
        ae  = all_res.get(tgt,{}).get('aeolus',{})
        ou  = all_res.get(tgt,{}).get('ours',{})
        ref = AEOLUS_REF.get(tgt,{})
        print(f"\n  {tgt}_Delay")
        print(f"  {'Model':<14} {'Aeolus setup':>14} {'Our setup':>12}  {'Δ':>8}  {'Aeolus Paper':>13}")
        print(f"  {'─'*66}")
        for m in ['RF','XGBoost','CatBoost','XGB_Optuna','MLP','ResNet','AutoInt']:
            ae_v=ae.get(m); ou_v=ou.get(m); ref_v=ref.get(m)
            ae_s  = f"{ae_v:.4f}" if ae_v else "   —   "
            ou_s  = f"{ou_v:.4f}" if ou_v else "   —   "
            diff  = (ou_v-ae_v) if (ae_v and ou_v) else None
            diff_s= f"{diff:+.4f}" if diff is not None else "   —  "
            ref_s = f"{ref_v:.3f}" if ref_v else "   —  "
            print(f"  {m:<14} {ae_s:>14} {ou_s:>12}  {diff_s:>8}  {ref_s:>13}")

        gbdt = ['RF','XGBoost','CatBoost','XGB_Optuna']
        dl   = ['MLP','ResNet','AutoInt']
        best_g_ae = max((v for k,v in ae.items() if k in gbdt), default=None)
        best_d_ae = max((v for k,v in ae.items() if k in dl),   default=None)
        best_g_ou = max((v for k,v in ou.items() if k in gbdt), default=None)
        best_d_ou = max((v for k,v in ou.items() if k in dl),   default=None)
        best_ref  = max(ref.values()) if ref else None

        def f(v): return f"{v:.4f}" if v else "   —  "
        print(f"  {'─'*66}")
        print(f"  {'Best GBDT':<14} {f(best_g_ae):>14} {f(best_g_ou):>12}")
        print(f"  {'Best DL':<14} {f(best_d_ae):>14} {f(best_d_ou):>12}")
        if best_ref: print(f"  {'Aeolus best DL':<14} {'':>14} {'':>12}  {'':>8}  {best_ref:>13.3f}")
        if best_g_ae and best_d_ae:
            g_adv_ae = best_g_ae - best_d_ae
            g_adv_ou = (best_g_ou - best_d_ou) if best_g_ou and best_d_ou else None
            print(f"  {'GBDT advantage':<14} {g_adv_ae:>+14.4f} {f(g_adv_ou):>12}")

# ── Main ───────────────────────────────────────────────────────
def main():
    torch.manual_seed(42); np.random.seed(42)
    print("="*60, flush=True)
    print("DL Phase — MLP / ResNet / AutoInt  (CPU only)", flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print("="*60, flush=True)

    df      = pd.read_parquet(DATA / 'aeolus_feat.parquet')
    all_res = load_ckpt()
    n_cached = sum(len(v) for d in all_res.values() for v in d.values())
    print(f"  Loaded {n_cached} cached results (GBDT should be here)", flush=True)

    for tgt_col,tgt_name in [('arr_delayed_15','ARR'),('dep_delayed_15','DEP')]:
        print(f"\n{'#'*55}\n  {tgt_name}_Delay\n{'#'*55}", flush=True)

        for setup_name,feat_cols,scaler_mode,patience,lr in [
            ('aeolus', AEOLUS_FEATURES, 'per_split',  5,  1e-3),
            ('ours',   OUR_FEATURES,    'train_only', 12, 5e-4),
        ]:
            nf  = len(feat_cols)
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
            print(f"\n  ── {tgt_name}/{setup_name} | {nf}f ──", flush=True)

            if setup_name == 'aeolus':
                dl_models = [('MLP',MLP_Aeolus(nf)),('ResNet',ResNet_Aeolus(nf)),('AutoInt',AutoInt_Aeolus(nf))]
            else:
                dl_models = [('MLP',MLP_Ours(nf)),('ResNet',ResNet_Ours(nf)),('AutoInt',AutoInt_Ours(nf))]

            for mname,model in dl_models:
                if mname in existing:
                    print(f"  [{mname:<10}] AUC={existing[mname]:.4f}  (cached)")
                    continue
                print(f"  [{mname:<10}] training...", flush=True)
                t0 = time.time()
                model = model.to(DEVICE)
                best_val = train_dl(model, X_tr, y_tr, X_val, y_val,
                                    lr=lr, patience=patience, max_epochs=100)
                auc = float(roc_auc_score(y_te, predict(model, X_te)))
                elapsed = time.time()-t0
                print(f"  [{mname:<10}] AUC={auc:.4f}  val={best_val:.4f}  ({elapsed:.0f}s)", flush=True)
                all_res.setdefault(tgt_name,{}).setdefault(setup_name,{})[mname] = auc
                save_ckpt(all_res)

    print_table(all_res)
    save_ckpt(all_res)
    print(f"\n✓ All done. Results in {CKPT}")

if __name__ == '__main__':
    main()
