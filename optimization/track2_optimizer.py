"""
optimization/track2_optimizer.py

Track 2 VICES optimizer — searches over 4 CSG topology types (A/B/C/D)
plus continuous parameters to find designs that maximise TMA uniformity.

Pipeline:
  1. Build combined feature matrix from Track 1 + Track 2 HDF5 + metadata
  2. Train K-fold KPI MLP: features → TMA_UI
  3. Random-search 1000 candidates per topology type
  4. Build joint Pareto front (max TMA_UI, min Re proxy for Δp)
  5. Compare vs Track 1 Pareto
  6. Save results to checkpoints/optimizer/track2_optimizer_results.json

Usage:
    python3 optimization/track2_optimizer.py
"""

import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

HDF5_T1   = Path('data/processed/ald_hdf5')
HDF5_T2   = Path('data/processed/track2_hdf5')
META_T2   = Path('openfoam/track2_cases/cases')
OPT_T1    = Path('checkpoints/optimizer/optimizer_results.json')
OUT_PATH  = Path('checkpoints/optimizer/track2_optimizer_results.json')
FIG_PATH  = Path('checkpoints/optimizer/track2_pareto.png')
DEVICE    = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Physical constants (N2 at 393K)
RHO_N2 = 1.12; MU_N2 = 2.0e-5

# ── 1. Feature extraction ─────────────────────────────────────────────────────

TYPE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


def compute_re(D_m, Q_slm, n_nozzles):
    Q = Q_slm * 1e-3 / 60.0
    A = n_nozzles * math.pi * (D_m / 2) ** 2
    V = Q / (A + 1e-12)
    return RHO_N2 * V * D_m / MU_N2


def extract_features(D_mm, Q_slm, n_nozzles, H_plenum_mm, pitch_D,
                     geom_type_int, extra_param=0.0):
    """
    Build a 10-dim feature vector from design parameters.
    geom_type_int: 0=Track1/hex, 1=A, 2=B, 3=C, 4=D
    extra_param: type-specific scalar (baffle_frac, cone_r_frac, n_rings,
                 divider_r_frac — or 0 for Track 1 and types without it)
    """
    D_m = D_mm / 1000.0
    Re  = compute_re(D_m, Q_slm, n_nozzles)
    # one-hot geometry type [5]: Track1, A, B, C, D
    oh  = [0.0] * 5
    oh[geom_type_int] = 1.0
    return [
        D_mm,           # nozzle diameter [mm]
        Q_slm,          # flow rate [slm]
        n_nozzles,      # number of nozzles
        H_plenum_mm,    # plenum height [mm]
        pitch_D,        # pitch/D ratio
        Re,             # Reynolds number
        extra_param,    # type-specific scalar
    ] + oh              # one-hot type [5]
                        # total = 12 features


def load_all_cases():
    """Load Track 1 + Track 2 cases into feature matrix X and targets Y."""
    X, Y = [], []

    # Track 1
    for f in sorted(HDF5_T1.glob('case_*.h5')):
        with h5py.File(f) as h:
            gf = h['inputs/global'][:]
            ui = float(h['uniformity'].attrs['uniformity_index'])
        # extract params from global_features (indices verified empirically)
        D_mm       = float(gf[7]) * 1000.0   # feat[7] = D in metres
        pitch_D    = float(gf[8])
        H_plenum_mm = float(gf[9]) * 1000.0
        Q_slm      = max(0.1, float(gf[0]) * MU_N2 / (RHO_N2 * max(float(gf[7]), 1e-4)))
        n_nozzles  = max(1, int(round(float(gf[12]))))
        feats = extract_features(D_mm, Q_slm, n_nozzles, H_plenum_mm,
                                  pitch_D, 0, 0.0)
        X.append(feats)
        Y.append(ui)

    # Track 2 — read params from case_meta.json
    for f in sorted(HDF5_T2.glob('track2_*.h5')):
        with h5py.File(f) as h:
            ui = float(h['uniformity'].attrs['uniformity_index'])
        name = f.stem   # track2_0000_typeA_D1.5mm_Q1.0slm
        meta_path = META_T2 / name / 'case_meta.json'
        if not meta_path.exists():
            continue
        meta  = json.load(open(meta_path))
        geo   = meta.get('geometry', {})
        gtype_char = geo.get('type', 'A_baffled')[0]   # 'A','B','C','D'
        gtype_int  = TYPE_MAP.get(gtype_char, 0) + 1    # 1-4
        D_mm       = float(geo.get('D_mm', 2.0))
        Q_slm      = float(meta.get('process', {}).get('flow_rate_slm', 3.0))
        n_nozzles  = int(geo.get('n_nozzles', 19))
        H_plenum_mm = float(geo.get('H_plenum_mm', 20.0))
        pitch_D    = float(geo.get('pitch_D', 4.0))
        # type-specific extra param
        extra = {
            1: float(geo.get('baffle_frac', 0.5)),
            2: float(geo.get('cone_r_frac', 0.35)),
            3: float(geo.get('n_rings', 3)) / 5.0,   # normalise
            4: float(geo.get('divider_r_frac', 0.45)),
        }.get(gtype_int, 0.0)
        feats = extract_features(D_mm, Q_slm, n_nozzles, H_plenum_mm,
                                  pitch_D, gtype_int, extra)
        X.append(feats)
        Y.append(ui)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# ── 2. KPI MLP ────────────────────────────────────────────────────────────────

class KpiMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, n_layers=3):
        super().__init__()
        dims   = [in_dim] + [hidden] * n_layers + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers += [nn.SiLU(), nn.LayerNorm(dims[i+1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_fold(X_tr, Y_tr, X_va, Y_va, in_dim, epochs=2000):
    sc = StandardScaler().fit(X_tr)
    xt = torch.from_numpy(sc.transform(X_tr)).to(DEVICE)
    yt = torch.from_numpy(Y_tr).to(DEVICE)
    xv = torch.from_numpy(sc.transform(X_va)).to(DEVICE)
    yv = torch.from_numpy(Y_va).to(DEVICE)

    m   = KpiMLP(in_dim).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val, best_state = 1e9, None
    for ep in range(epochs):
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
    return m, sc


# ── 3. Parameter grids per topology type ─────────────────────────────────────

def candidate_grid(geom_type_int, n=300):
    """Generate n random candidates for the given geometry type."""
    rng = np.random.default_rng(42 + geom_type_int)
    D_vals   = rng.uniform(1.0, 3.0, n)
    Q_vals   = rng.uniform(0.5, 8.0, n)
    H_vals   = rng.uniform(15.0, 30.0, n)
    p_vals   = rng.uniform(3.0, 6.0, n)

    candidates = []
    for i in range(n):
        D_mm = float(D_vals[i])
        Q    = float(Q_vals[i])
        H    = float(H_vals[i])
        p    = float(p_vals[i])
        D_m  = D_mm / 1000.0

        # estimate n_nozzles from pitch and plate size
        from geometry.parametric import build_showerhead
        from geometry.grammar import NozzlePattern
        try:
            geo = build_showerhead({'D': D_m, 'pitch_over_D': p,
                                     'H_plenum': H/1000, 't_face': 0.003,
                                     'standoff': 0.020},
                                    pattern=NozzlePattern.HEX)
            n_holes = geo.n_holes
        except Exception:
            n_holes = max(1, int(math.pi * (0.075 / (D_m * p))**2))

        # type-specific extra param
        if geom_type_int == 1:   # A baffled
            extra = float(rng.uniform(0.3, 0.7))
        elif geom_type_int == 2: # B conical
            extra = float(rng.uniform(0.25, 0.55))
        elif geom_type_int == 3: # C annular
            extra = float(rng.integers(2, 5)) / 5.0
        elif geom_type_int == 4: # D two-zone
            extra = float(rng.uniform(0.35, 0.60))
        else:
            extra = 0.0

        feats = extract_features(D_mm, Q, n_holes, H, p, geom_type_int, extra)
        candidates.append({
            'features': feats,
            'D_mm': D_mm, 'Q_slm': Q, 'H_plenum_mm': H,
            'pitch_D': p, 'n_nozzles': n_holes,
            'geom_type': geom_type_int, 'extra_param': extra,
            'Re': compute_re(D_m, Q, n_holes),
        })
    return candidates


# ── 4. Pareto front ───────────────────────────────────────────────────────────

def is_dominated(a, others):
    """True if any design in others dominates a on all objectives."""
    for b in others:
        if b['TMA_UI'] >= a['TMA_UI'] and b['neg_Re'] >= a['neg_Re'] and (
           b['TMA_UI'] > a['TMA_UI'] or b['neg_Re'] > a['neg_Re']):
            return True
    return False


def pareto_front(designs):
    front = []
    for d in designs:
        if not is_dominated(d, designs):
            front.append(d)
    return sorted(front, key=lambda x: -x['TMA_UI'])


# ── 5. Main ───────────────────────────────────────────────────────────────────

def main():
    print('Loading all cases...')
    X, Y = load_all_cases()
    print(f'  {len(Y)} cases  |  TMA_UI: mean={Y.mean():.3f} std={Y.std():.3f}')
    in_dim = X.shape[1]

    # ── K-fold training ──────────────────────────────────────────────────────
    print(f'\nTraining {in_dim}-feature KPI surrogate (5-fold CV)...')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(Y))
    fold_models, fold_scalers = [], []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        m, sc = train_fold(X[tr_idx], Y[tr_idx], X[va_idx], Y[va_idx], in_dim)
        m.eval()
        with torch.no_grad():
            xv = torch.from_numpy(sc.transform(X[va_idx])).to(DEVICE)
            oof_pred[va_idx] = m(xv).cpu().numpy()
        fold_models.append(m); fold_scalers.append(sc)
        rho, _ = spearmanr(Y[va_idx], oof_pred[va_idx])
        r2     = r2_score(Y[va_idx], oof_pred[va_idx])
        print(f'  Fold {fold+1}  R²={r2:.3f}  ρ={rho:.3f}  (n={len(va_idx)})')

    rho_oof, _ = spearmanr(Y, oof_pred)
    r2_oof     = r2_score(Y, oof_pred)
    print(f'  OOF: R²={r2_oof:.3f}  ρ={rho_oof:.3f}')

    # ── Ensemble prediction helper ───────────────────────────────────────────
    def predict(feats_list):
        Xp  = np.array(feats_list, dtype=np.float32)
        out = np.zeros(len(Xp))
        for m, sc in zip(fold_models, fold_scalers):
            m.eval()
            with torch.no_grad():
                xp = torch.from_numpy(sc.transform(Xp)).to(DEVICE)
                out += m(xp).cpu().numpy()
        return out / len(fold_models)

    # ── Generate candidates for all 4 Track 2 types ─────────────────────────
    print('\nGenerating candidates for Track 2 geometry types A/B/C/D...')
    all_candidates = []
    type_names = {1: 'A_baffled', 2: 'B_conical', 3: 'C_annular', 4: 'D_twozone'}

    for gtype in [1, 2, 3, 4]:
        candidates = candidate_grid(gtype, n=300)
        feats = [c['features'] for c in candidates]
        preds = predict(feats)
        for c, p in zip(candidates, preds):
            c['TMA_UI']  = float(p)
            c['neg_Re']  = -float(c['Re'])
            c['geom_name'] = type_names[gtype]
        all_candidates.extend(candidates)
        best = max(candidates, key=lambda x: x['TMA_UI'])
        bui  = best['TMA_UI']; bdmm = best['D_mm']; bq = best['Q_slm']
        print(f'  Type {type_names[gtype]}: best TMA_UI={bui:.3f}  D={bdmm:.1f}mm Q={bq:.1f}slm')

    # ── Also score Track 1 design space for comparison ───────────────────────
    print('\nGenerating Track 1 candidates for comparison...')
    t1_candidates = candidate_grid(0, n=300)
    t1_preds = predict([c['features'] for c in t1_candidates])
    for c, p in zip(t1_candidates, t1_preds):
        c['TMA_UI'] = float(p); c['neg_Re'] = -float(c['Re'])
        c['geom_name'] = 'T1_hex'
    best_t1 = max(t1_candidates, key=lambda x: x['TMA_UI'])
    t1ui = best_t1['TMA_UI']; t1d = best_t1['D_mm']; t1q = best_t1['Q_slm']
    print(f'  Track 1 best: TMA_UI={t1ui:.3f}  D={t1d:.1f}mm Q={t1q:.1f}slm')

    # ── Pareto fronts ────────────────────────────────────────────────────────
    pareto_t2 = pareto_front(all_candidates)
    pareto_t1 = pareto_front(t1_candidates)
    print(f'\nPareto front: Track 2={len(pareto_t2)} designs, Track 1={len(pareto_t1)} designs')

    # ── Best per type ─────────────────────────────────────────────────────────
    best_per_type = {}
    for gtype, name in type_names.items():
        tc = [c for c in all_candidates if c['geom_type'] == gtype]
        best_per_type[name] = max(tc, key=lambda x: x['TMA_UI'])

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        'surrogate': {'oof_r2': r2_oof, 'oof_rho': rho_oof,
                      'n_cases': len(Y), 'in_dim': in_dim},
        'track2_pareto': [
            {k: v for k, v in d.items() if k != 'features'}
            for d in pareto_t2[:20]
        ],
        'track1_pareto': [
            {k: v for k, v in d.items() if k != 'features'}
            for d in pareto_t1[:20]
        ],
        'best_per_type': {
            k: {kk: vv for kk, vv in v.items() if kk != 'features'}
            for k, v in best_per_type.items()
        },
        'track1_best_tma_ui': float(best_t1['TMA_UI']),
        'track2_best_tma_ui': float(max(all_candidates, key=lambda x: x['TMA_UI'])['TMA_UI']),
        'improvement': float(
            max(all_candidates, key=lambda x: x['TMA_UI'])['TMA_UI'] - best_t1['TMA_UI']
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved → {OUT_PATH}')

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pareto comparison
    ax = axes[0]
    colors = {'A_baffled':'#e74c3c','B_conical':'#3498db',
              'C_annular':'#2ecc71','D_twozone':'#f39c12','T1_hex':'#95a5a6'}
    for gtype, name in {**{0:'T1_hex'}, **type_names}.items():
        grp = t1_candidates if gtype == 0 else [c for c in all_candidates if c['geom_type']==gtype]
        if not grp: continue
        ax.scatter([-c['neg_Re'] for c in grp], [c['TMA_UI'] for c in grp],
                   s=8, alpha=0.3, color=colors[name], label=name)
    # Pareto fronts
    t2_sorted = sorted(pareto_t2, key=lambda x: x['neg_Re'])
    t1_sorted = sorted(pareto_t1, key=lambda x: x['neg_Re'])
    ax.plot([-c['neg_Re'] for c in t2_sorted], [c['TMA_UI'] for c in t2_sorted],
            'r-o', ms=5, lw=2, label='Track 2 Pareto', zorder=5)
    ax.plot([-c['neg_Re'] for c in t1_sorted], [c['TMA_UI'] for c in t1_sorted],
            'k--s', ms=5, lw=2, label='Track 1 Pareto', zorder=5)
    ax.set_xlabel('Reynolds number Re'); ax.set_ylabel('TMA_UI (predicted)')
    ax.set_title('Track 2 vs Track 1 Pareto Front'); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # OOF parity
    ax = axes[1]
    t1_mask = np.array([1 if c.get('geom_type',0)==0 else 0 for c in
                         ([{}]*sum(1 for f in HDF5_T1.glob('case_*.h5'))) +
                         ([{}]*sum(1 for f in HDF5_T2.glob('track2_*.h5')))])
    colors_pts = ['steelblue' if i < sum(1 for _ in HDF5_T1.glob('case_*.h5'))
                  else 'darkorange' for i in range(len(Y))]
    ax.scatter(Y, oof_pred, s=20, c=colors_pts, edgecolors='k', lw=0.3)
    lim = [-0.05, 0.95]
    ax.plot(lim, lim, 'r--', lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('CFD TMA_UI (true)'); ax.set_ylabel('MLP TMA_UI (pred)')
    ax.set_title(f'Combined surrogate OOF\nR²={r2_oof:.3f}  ρ={rho_oof:.3f}')
    ax.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0],[0],color='steelblue',marker='o',ms=5,ls=''),
               Line2D([0],[0],color='darkorange',marker='o',ms=5,ls='')],
              ['Track 1','Track 2'], fontsize=8)

    plt.suptitle('Track 2 VICES Optimizer Results', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150, bbox_inches='tight')
    print(f'Plot saved → {FIG_PATH}')
    plt.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f'Track 1 best TMA_UI: {results["track1_best_tma_ui"]:.3f}')
    print(f'Track 2 best TMA_UI: {results["track2_best_tma_ui"]:.3f}')
    imp = results["improvement"]
    print(f'Improvement:         {imp:+.3f} ({"↑ Track 2 better" if imp > 0 else "↓ Track 1 better"})')
    print()
    print('Best design per Track 2 type:')
    for name, d in best_per_type.items():
        print(f'  {name:<15} TMA_UI={d["TMA_UI"]:.3f}  '
              f'D={d["D_mm"]:.1f}mm  Q={d["Q_slm"]:.1f}slm  '
              f'Re={d["Re"]:.0f}')


if __name__ == '__main__':
    main()
