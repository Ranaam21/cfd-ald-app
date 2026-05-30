"""
Option B — Design-space KPI surrogate with 5-fold cross-validation + ensemble.

Trains K MLP models: global_features [18] -> TMA_UI [1], one per fold.
Ensemble prediction = average of all K models → better accuracy + stable R².

Usage:
    python3 training/kpi_surrogate.py
"""

import numpy as np
import torch
import torch.nn as nn
import h5py
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

HDF5_DIR = Path(__file__).parent.parent / 'data/processed/ald_hdf5'
OUT_DIR  = Path(__file__).parent / 'eval_results'
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
K_FOLDS = 5
EPOCHS  = 2000
LR      = 3e-3

# ── 1. Load all 83 cases ─────────────────────────────────────────────────────
print('Loading KPIs + wafer-plane flow features from HDF5 files...')
X_raw, Y_raw = [], []
WAFER_FRAC = 0.05

for h5_path in sorted(HDF5_DIR.glob('*.h5')):
    with h5py.File(h5_path, 'r') as f:
        gf     = f['inputs/global'][:]
        y_fields = f['outputs/node_fields'][:]   # [N, 6] Ux,Uy,Uz,p,T,TMA
        coords = f['coords'][:]
        tma_ui = float(f['uniformity'].attrs['uniformity_index'])

    # wafer-plane mask — bottom WAFER_FRAC of z range
    z = coords[:, 2]
    z_min, z_max = z.min(), z.max()
    z_band = max(0.002, (z_max - z_min) * WAFER_FRAC)
    mask = z < (z_min + z_band)
    if mask.sum() < 10:
        mask = z < (z_min + (z_max - z_min) * 0.10)

    # Step 1 — extract wafer-plane flow statistics from CFD ground truth
    Uz_w  = y_fields[mask, 2]   # vertical velocity at wafer
    p_w   = y_fields[mask, 3]   # pressure at wafer
    T_w   = y_fields[mask, 4]   # temperature at wafer
    TMA_w = y_fields[mask, 5]   # TMA at wafer

    eps = 1e-12
    flow_feats = np.array([
        float(Uz_w.mean()),
        float(Uz_w.std()),
        float(np.clip(1 - Uz_w.std() / (abs(Uz_w.mean()) + eps), 0, 1)),  # Uz_UI
        float(p_w.mean()),
        float(p_w.std()),
        float(T_w.mean()),
        float(T_w.std()),
        float(TMA_w.mean()),
        float(TMA_w.std()),
    ], dtype=np.float32)

    # Step 2 — concatenate: 18 global + 9 wafer-plane = 27 features
    X_raw.append(np.concatenate([gf, flow_feats]))
    Y_raw.append(tma_ui)

X = np.array(X_raw, dtype=np.float32)   # [83, 27]
Y = np.array(Y_raw,  dtype=np.float32)  # [83]
print(f'  {len(Y)} cases  |  Features: {X.shape[1]} (18 global + 9 wafer-plane flow stats)')
print(f'  TMA_UI: min={Y.min():.3f} max={Y.max():.3f} mean={Y.mean():.3f} std={Y.std():.3f}')

# ── 2. MLP ───────────────────────────────────────────────────────────────────
class KpiMLP(nn.Module):
    def __init__(self, in_dim=27, hidden=64, n_layers=3):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.SiLU(), nn.LayerNorm(dims[i + 1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_fold(X_tr, Y_tr, X_va, Y_va):
    scaler = StandardScaler().fit(X_tr)
    xt = torch.from_numpy(scaler.transform(X_tr)).to(DEVICE)
    yt = torch.from_numpy(Y_tr).to(DEVICE)
    xv = torch.from_numpy(scaler.transform(X_va)).to(DEVICE)
    yv = torch.from_numpy(Y_va).to(DEVICE)

    m   = KpiMLP().to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val, best_state = 1e9, None
    for ep in range(EPOCHS):
        m.train()
        loss = nn.functional.mse_loss(m(xt), yt)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(m(xv), yv).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in m.state_dict().items()}

    m.load_state_dict(best_state)
    return m, scaler


# ── 3. K-fold cross-validation ───────────────────────────────────────────────
print(f'\nRunning {K_FOLDS}-fold cross-validation...')
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

oof_pred  = np.zeros(len(Y))   # out-of-fold predictions
fold_models, fold_scalers = [], []
fold_metrics = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    model, scaler = train_fold(X[tr_idx], Y[tr_idx], X[va_idx], Y[va_idx])

    model.eval()
    with torch.no_grad():
        xv_s = torch.from_numpy(scaler.transform(X[va_idx])).to(DEVICE)
        pv   = model(xv_s).cpu().numpy()

    oof_pred[va_idx] = pv
    fold_models.append(model)
    fold_scalers.append(scaler)

    mae = float(np.mean(np.abs(pv - Y[va_idx])))
    r2  = float(r2_score(Y[va_idx], pv))
    rho, _ = spearmanr(Y[va_idx], pv)
    fold_metrics.append((mae, r2, rho))
    print(f'  Fold {fold+1}/{K_FOLDS}  MAE={mae:.4f}  R²={r2:.4f}  ρ={rho:.4f}  '
          f'(val n={len(va_idx)})')

# ── 4. Ensemble prediction on all 83 (average across K models) ───────────────
print('\nBuilding ensemble (average of all 5 fold models)...')
ensemble_preds = np.zeros((K_FOLDS, len(Y)))
for k, (m, sc) in enumerate(zip(fold_models, fold_scalers)):
    m.eval()
    with torch.no_grad():
        xa = torch.from_numpy(sc.transform(X)).to(DEVICE)
        ensemble_preds[k] = m(xa).cpu().numpy()

ensemble_pred = ensemble_preds.mean(axis=0)   # average across folds

# ── 5. Results ───────────────────────────────────────────────────────────────
def metrics(true, pred, label):
    mae = float(np.mean(np.abs(true - pred)))
    r2  = float(r2_score(true, pred))
    rho, pval = spearmanr(true, pred)
    print(f'  {label:<22}  MAE={mae:.4f}  R²={r2:.4f}  ρ={rho:.4f}  (p={pval:.4g})')
    return mae, r2, rho

print('\n' + '='*65)
print('RESULTS — TMA_UI prediction')
print('='*65)
print('Per-fold (val only):')
for i, (mae, r2, rho) in enumerate(fold_metrics):
    print(f'    Fold {i+1}  MAE={mae:.4f}  R²={r2:.4f}  ρ={rho:.4f}')

maes  = [m[0] for m in fold_metrics]
r2s   = [m[1] for m in fold_metrics]
rhos  = [m[2] for m in fold_metrics]
print(f'  Mean ± std  MAE={np.mean(maes):.4f}±{np.std(maes):.4f}  '
      f'R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}  '
      f'ρ={np.mean(rhos):.4f}±{np.std(rhos):.4f}')

print('\nOut-of-fold (each case predicted by model that never saw it):')
mae_oof, r2_oof, rho_oof = metrics(Y, oof_pred, 'OOF (n=83)')

print('\nEnsemble (avg of 5 models) on all 83:')
mae_ens, r2_ens, rho_ens = metrics(Y, ensemble_pred, 'Ensemble (n=83)')

print()
def verdict(rho, r2):
    if rho > 0.7 and r2 > 0.5:  return '✓ GOOD — optimizer rankings trustworthy'
    if rho > 0.5:                return '⚠ ACCEPTABLE — optimizer directionally correct'
    return '✗ WEAK — rankings unreliable'
print(f'OOF verdict:      {verdict(rho_oof, r2_oof)}')
print(f'Ensemble verdict: {verdict(rho_ens, r2_ens)}')

# ── 6. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
lim = [-0.05, 0.65]

# OOF parity
ax = axes[0]
ax.scatter(Y, oof_pred, s=40, color='steelblue', edgecolors='k', lw=0.4)
ax.plot(lim, lim, 'r--', lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('CFD TMA_UI (true)'); ax.set_ylabel('MLP TMA_UI (pred)')
ax.set_title(f'OOF parity (n=83)\nR²={r2_oof:.3f}  ρ={rho_oof:.3f}')
ax.grid(True, alpha=0.3)

# Ensemble parity
ax = axes[1]
ax.scatter(Y, ensemble_pred, s=40, color='darkorange', edgecolors='k', lw=0.4)
ax.plot(lim, lim, 'r--', lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('CFD TMA_UI (true)'); ax.set_ylabel('Ensemble pred')
ax.set_title(f'Ensemble parity (n=83)\nR²={r2_ens:.3f}  ρ={rho_ens:.3f}')
ax.grid(True, alpha=0.3)

# OOF ranking — all 83 sorted by true TMA_UI
ax = axes[2]
sort_idx = np.argsort(Y)
ax.plot(range(len(Y)), Y[sort_idx],        'b-o',  ms=4, lw=1.5, label='CFD true')
ax.plot(range(len(Y)), oof_pred[sort_idx], 'r--s', ms=4, lw=1,   label='OOF pred')
ax.set_xlabel('Design rank (sorted by CFD TMA_UI)')
ax.set_ylabel('TMA_UI')
ax.set_title(f'Design ranking (OOF)\nρ={rho_oof:.3f}  — all 83 cases')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle(f'Multi-fidelity KPI Surrogate — {K_FOLDS}-Fold CV + Ensemble (27 features → TMA_UI)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out_path = OUT_DIR / 'kpi_surrogate_kfold.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\nPlot saved → {out_path}')
plt.close()

# ── 7. Save ensemble ──────────────────────────────────────────────────────────
save_path = Path(__file__).parent.parent / 'checkpoints/multihead/kpi_surrogate_multifidelity.pt'
torch.save({
    'models':      [m.state_dict() for m in fold_models],
    'scalers':     [{'mean': sc.mean_, 'std': sc.scale_} for sc in fold_scalers],
    'k_folds':     K_FOLDS,
    'in_dim':      X.shape[1],
    'oof_r2':      r2_oof,
    'oof_rho':     rho_oof,
    'feature_note': '18 global + 9 wafer-plane CFD stats (Uz,p,T,TMA mean/std + Uz_UI)',
}, save_path)
print(f'Ensemble saved → {save_path}')
