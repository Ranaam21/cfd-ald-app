"""
Track 1 surrogate evaluation — runs inference on all 83 HDF5 cases,
compares predicted fields vs CFD ground truth, reports MAE / R² per field
and per scalar KPI (TMA_UI, T_UI, delta_p).

Usage:
    python3 training/eval_track1.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import h5py
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Config ──────────────────────────────────────────────────────────────────
CKPT_PATH  = Path(__file__).parent.parent / 'checkpoints/multihead/multihead_final.pt'
HDF5_DIR   = Path(__file__).parent.parent / 'data/processed/ald_hdf5'
OUT_DIR    = Path(__file__).parent.parent / 'training/eval_results'
OUT_DIR.mkdir(exist_ok=True)
DEVICE     = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
K          = 6       # k-NN neighbours (must match training)
N_EVAL     = 2048    # nodes to subsample per case for speed
WAFER_FRAC = 0.05    # bottom 5% of z range = wafer-plane proxy (matches postprocess.py)

FIELD_NAMES = ['Ux', 'Uy', 'Uz', 'p', 'T', 'TMA']

# ── Model definition (must match training notebook) ──────────────────────────
def mlp(in_dim, out_dim, hidden=None, n_layers=2):
    hidden = hidden or out_dim
    dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
    mods = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods += [nn.SiLU(), nn.LayerNorm(dims[i + 1])]
    return nn.Sequential(*mods)


from torch_geometric.nn import MessagePassing

class MGNProcessor(MessagePassing):
    def __init__(self, hidden):
        super().__init__(aggr='sum')
        self.edge_mlp  = mlp(3 * hidden, hidden)
        self.node_mlp  = mlp(2 * hidden, hidden)
        self.edge_norm = nn.LayerNorm(hidden)
        self.node_norm = nn.LayerNorm(hidden)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        N = x.size(0)
        new_e = self.edge_norm(
            self.edge_mlp(torch.cat([x[src], x[dst], edge_attr], -1)) + edge_attr
        )
        agg   = self.propagate(edge_index, x=x, edge_attr=new_e, size=(N, N))
        new_x = self.node_norm(self.node_mlp(torch.cat([x, agg], -1)) + x)
        return new_x, new_e

    def message(self, edge_attr):
        return edge_attr


class MultiHeadMGN(nn.Module):
    def __init__(self, node_dim, edge_dim, flow_out=4, heat_out=1,
                 species_out=1, hidden=128, n_layers=8):
        super().__init__()
        self.node_enc    = mlp(node_dim, hidden)
        self.edge_enc    = mlp(edge_dim, hidden)
        self.processors  = nn.ModuleList([MGNProcessor(hidden) for _ in range(n_layers)])
        self.flow_dec    = mlp(hidden, flow_out)
        self.heat_dec    = mlp(hidden, heat_out)
        self.species_dec = mlp(hidden, species_out)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_enc(x)
        e = self.edge_enc(edge_attr)
        for proc in self.processors:
            x, e = proc(x, edge_index, e)
        return self.flow_dec(x), self.heat_dec(x), self.species_dec(x)


# ── Load checkpoint ──────────────────────────────────────────────────────────
def load_model():
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    cfg  = ckpt['cfg']
    norm = ckpt['norm']
    m = MultiHeadMGN(
        node_dim    = cfg['node_input_dim'],
        edge_dim    = cfg['edge_input_dim'],
        flow_out    = cfg['flow_out_dim'],
        heat_out    = cfg['heat_out_dim'],
        species_out = cfg['species_out_dim'],
        hidden      = cfg['hidden_dim'],
        n_layers    = cfg['n_layers'],
    ).to(DEVICE)
    m.load_state_dict(ckpt['model'])
    m.eval()
    return m, norm, cfg


# ── Build k-NN graph from coords ─────────────────────────────────────────────
def build_graph(coords, n_nodes, k):
    idx = np.random.choice(len(coords), n_nodes, replace=False)
    coords_s = coords[idx]
    tree = cKDTree(coords_s)
    _, nn_idx = tree.query(coords_s, k=k + 1)
    nn_idx = nn_idx[:, 1:]
    src = np.repeat(np.arange(n_nodes), k)
    dst = nn_idx.flatten()
    diff = coords_s[dst] - coords_s[src]
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    med  = float(np.median(dist)) + 1e-8
    ef   = np.concatenate([diff / med, dist / med], axis=1).astype(np.float32)
    return idx, src, dst, ef


# ── Inference on one case ─────────────────────────────────────────────────────
def infer_case(h5_path, model, norm, n_nodes=N_EVAL, k=K):
    with h5py.File(h5_path, 'r') as h:
        coords = h['coords'][:]
        nf     = h['inputs/node_features'][:]
        gf     = h['inputs/global'][:]
        y_true = h['outputs/node_fields'][:]
        ui     = dict(h['uniformity'].attrs)

    node_mean = np.array(norm['node_mean'], dtype=np.float32)
    node_std  = np.array(norm['node_std'],  dtype=np.float32)
    out_mean  = np.array(norm['out_mean'],  dtype=np.float32)
    out_std   = np.array(norm['out_std'],   dtype=np.float32)

    idx, src, dst, ef = build_graph(coords, n_nodes, k)

    xi = np.concatenate([nf[idx], np.tile(gf, (n_nodes, 1))], axis=1).astype(np.float32)
    x  = torch.from_numpy((xi - node_mean) / (node_std + 1e-8)).to(DEVICE)
    ei = torch.tensor(np.stack([src, dst]), dtype=torch.long).to(DEVICE)
    ea = torch.from_numpy(ef).to(DEVICE)

    with torch.no_grad():
        fp, hp, sp = model(x, ei, ea)

    fp = fp.cpu().numpy()
    hp = hp.cpu().numpy().flatten()
    sp = sp.cpu().numpy().flatten()

    y_pred = np.zeros((n_nodes, 6), dtype=np.float32)
    y_pred[:, :4] = fp * out_std[:4] + out_mean[:4]
    y_pred[:, 4]  = hp * out_std[4]  + out_mean[4]
    y_pred[:, 5]  = sp * out_std[5]  + out_mean[5]

    y_gt = y_true[idx]

    # wafer-plane mask: bottom WAFER_FRAC of z range (matches postprocess.py logic)
    z = coords[idx, 2]
    z_min, z_max = z.min(), z.max()
    z_band = max(0.002, (z_max - z_min) * WAFER_FRAC)
    wafer_mask = z < (z_min + z_band)
    if wafer_mask.sum() < 5:          # fallback: use bottom 10%
        z_band = (z_max - z_min) * 0.10
        wafer_mask = z < (z_min + z_band)

    return y_pred, y_gt, ui, wafer_mask


# ── Main evaluation loop ──────────────────────────────────────────────────────
def main():
    print(f'Device: {DEVICE}')
    print(f'Loading checkpoint: {CKPT_PATH}')
    model, norm, cfg = load_model()
    print(f'Model: hidden={cfg["hidden_dim"]}, layers={cfg["n_layers"]}')

    h5_files = sorted(HDF5_DIR.glob('*.h5'))
    print(f'\nEvaluating on {len(h5_files)} cases...\n')

    all_pred = {f: [] for f in FIELD_NAMES}
    all_true = {f: [] for f in FIELD_NAMES}

    # scalar KPI tracking
    tma_ui_pred_list, tma_ui_true_list = [], []
    t_ui_pred_list,   t_ui_true_list   = [], []
    dp_pred_list,     dp_true_list      = [], []

    np.random.seed(42)

    for i, h5 in enumerate(h5_files):
        try:
            y_pred, y_gt, ui, wafer_mask = infer_case(h5, model, norm)

            for j, name in enumerate(FIELD_NAMES):
                all_pred[name].append(y_pred[:, j])
                all_true[name].append(y_gt[:, j])

            # --- TMA_UI: wafer-plane nodes only ---
            tma_pred_w = y_pred[wafer_mask, 5]
            tma_true_w = y_gt[wafer_mask, 5]
            tma_ui_pred = float(np.clip(
                1 - tma_pred_w.std() / (tma_pred_w.mean() + 1e-12), 0, 1))
            # use stored ground-truth uniformity_index when available
            tma_ui_true = float(ui.get(
                'uniformity_index',
                np.clip(1 - tma_true_w.std() / (tma_true_w.mean() + 1e-12), 0, 1)))

            # --- T_UI: wafer-plane nodes only ---
            T_pred_w = y_pred[wafer_mask, 4]
            T_true_w = y_gt[wafer_mask, 4]
            t_ui_pred = float(np.clip(
                1 - T_pred_w.std() / (T_pred_w.mean() + 1e-12), 0, 1))
            t_ui_true = float(np.clip(
                1 - T_true_w.std() / (T_true_w.mean() + 1e-12), 0, 1))

            # --- delta_p: full domain (max-min of pressure) ---
            dp_pred = float(abs(y_pred[:, 3].max() - y_pred[:, 3].min()))
            dp_true = float(abs(y_gt[:, 3].max()   - y_gt[:, 3].min()))

            tma_ui_pred_list.append(tma_ui_pred);  tma_ui_true_list.append(tma_ui_true)
            t_ui_pred_list.append(t_ui_pred);       t_ui_true_list.append(t_ui_true)
            dp_pred_list.append(dp_pred);           dp_true_list.append(dp_true)

            if (i + 1) % 10 == 0:
                wn = wafer_mask.sum()
                print(f'  [{i+1:3d}/{len(h5_files)}] {h5.stem[:40]} '
                      f'wafer_nodes={wn} '
                      f'TMA_UI pred={tma_ui_pred:.3f} true={tma_ui_true:.3f}')

        except Exception as e:
            print(f'  SKIP {h5.name}: {e}')

    # ── Field-level MAE and R² ─────────────────────────────────────────────
    print('\n' + '='*60)
    print('FIELD-LEVEL RESULTS')
    print('='*60)
    print(f'{"Field":<8}  {"MAE":>12}  {"R²":>8}  {"Mean|True|":>12}  {"Rel.MAE%":>10}')
    print('-'*60)

    field_results = {}
    for name in FIELD_NAMES:
        p = np.concatenate(all_pred[name])
        t = np.concatenate(all_true[name])
        mae  = float(np.mean(np.abs(p - t)))
        r2   = float(r2_score(t, p))
        mean_t = float(np.mean(np.abs(t))) + 1e-12
        rel_mae = mae / mean_t * 100
        field_results[name] = dict(mae=mae, r2=r2, rel_mae=rel_mae)
        print(f'{name:<8}  {mae:>12.4e}  {r2:>8.4f}  {mean_t:>12.4e}  {rel_mae:>9.2f}%')

    # ── Scalar KPI MAE and R² ──────────────────────────────────────────────
    print('\n' + '='*60)
    print('SCALAR KPI RESULTS')
    print('='*60)
    print(f'{"KPI":<12}  {"MAE":>10}  {"R²":>8}')
    print('-'*40)

    kpi_results = {}
    for label, pred, true in [
        ('TMA_UI',  tma_ui_pred_list,  tma_ui_true_list),
        ('T_UI',    t_ui_pred_list,    t_ui_true_list),
        ('delta_p', dp_pred_list,      dp_true_list),
    ]:
        p = np.array(pred);  t = np.array(true)
        mae = float(np.mean(np.abs(p - t)))
        r2  = float(r2_score(t, p)) if len(p) > 1 else float('nan')
        kpi_results[label] = dict(mae=mae, r2=r2)
        print(f'{label:<12}  {mae:>10.4f}  {r2:>8.4f}')

    # ── Spearman ranking correlation (compute before plots) ───────────────
    print('\n' + '='*60)
    print('DESIGN RANKING (Spearman ρ) — what the optimizer actually needs')
    print('='*60)
    print(f'{"KPI":<12}  {"Spearman ρ":>12}  {"p-value":>10}  {"Verdict"}')
    print('-'*58)

    spearman_results = {}
    for label, pred, true in [
        ('TMA_UI',  tma_ui_pred_list,  tma_ui_true_list),
        ('T_UI',    t_ui_pred_list,    t_ui_true_list),
        ('delta_p', dp_pred_list,      dp_true_list),
    ]:
        rho, pval = spearmanr(true, pred)
        spearman_results[label] = rho
        v = ('✓ GOOD' if rho > 0.7 else
             '⚠ ACCEPTABLE' if rho > 0.5 else
             '✗ WEAK')
        print(f'{label:<12}  {rho:>12.4f}  {pval:>10.4f}  {v}')

    # ── Plots ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 10))
    gs  = gridspec.GridSpec(2, 5, hspace=0.4, wspace=0.35)

    # scatter: predicted vs true for each field
    for j, name in enumerate(FIELD_NAMES):
        ax = fig.add_subplot(gs[j // 4, j % 4])
        p  = np.concatenate(all_pred[name])
        t  = np.concatenate(all_true[name])
        # subsample for plot speed
        idx = np.random.choice(len(p), min(5000, len(p)), replace=False)
        ax.scatter(t[idx], p[idx], s=1, alpha=0.3, color='steelblue')
        lims = [min(t[idx].min(), p[idx].min()), max(t[idx].max(), p[idx].max())]
        ax.plot(lims, lims, 'r--', lw=1)
        r2 = field_results[name]['r2']
        ax.set_title(f'{name}  R²={r2:.3f}', fontsize=10)
        ax.set_xlabel('CFD (true)', fontsize=8)
        ax.set_ylabel('Surrogate (pred)', fontsize=8)

    # KPI parity plots + Spearman ranking plot
    kpi_data = [
        ('TMA_UI',  tma_ui_pred_list,  tma_ui_true_list),
        ('T_UI',    t_ui_pred_list,    t_ui_true_list),
    ]
    for k_i, (label, pred, true) in enumerate(kpi_data):
        ax = fig.add_subplot(gs[1, 2 + k_i])
        ax.scatter(true, pred, s=30, color='darkorange', edgecolors='k', linewidths=0.5)
        lims = [0, 1.05]
        ax.plot(lims, lims, 'r--', lw=1)
        r2  = kpi_results[label]['r2']
        rho = spearman_results[label]
        ax.set_title(f'{label}  R²={r2:.3f}  ρ={rho:.3f}', fontsize=10)
        ax.set_xlabel('CFD (true)', fontsize=8)
        ax.set_ylabel('Surrogate (pred)', fontsize=8)
        ax.set_xlim(lims);  ax.set_ylim(lims)

    # Spearman ranking plot: sorted true TMA_UI vs predicted
    ax = fig.add_subplot(gs[1, 4])
    sort_idx = np.argsort(tma_ui_true_list)
    ax.plot(range(len(sort_idx)),
            np.array(tma_ui_true_list)[sort_idx], 'b-o', ms=4, label='CFD true')
    ax.plot(range(len(sort_idx)),
            np.array(tma_ui_pred_list)[sort_idx], 'r--s', ms=4, label='Surrogate pred')
    rho = spearman_results['TMA_UI']
    ax.set_title(f'TMA_UI ranking  ρ={rho:.3f}', fontsize=10)
    ax.set_xlabel('Design rank (by CFD)', fontsize=8)
    ax.set_ylabel('TMA_UI', fontsize=8)
    ax.legend(fontsize=7)

    plt.suptitle('Track 1 Surrogate Evaluation — All 83 Cases', fontsize=13, fontweight='bold')
    out_path = OUT_DIR / 'track1_eval.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved → {out_path}')
    plt.close()

    # ── Summary verdict ────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('VERDICT')
    print('='*60)
    tma_r2  = kpi_results['TMA_UI']['r2']
    tma_rho = spearman_results['TMA_UI']

    def r2_verdict(r2):
        if r2 > 0.90: return '✓ GOOD'
        if r2 > 0.75: return '⚠ ACCEPTABLE'
        return '✗ WEAK'

    def rho_verdict(rho):
        if rho > 0.70: return '✓ GOOD — optimizer rankings trustworthy'
        if rho > 0.50: return '⚠ ACCEPTABLE — optimizer directionally correct'
        return '✗ WEAK — optimizer rankings unreliable'

    print(f'TMA_UI  R²={tma_r2:.3f} ({r2_verdict(tma_r2)})  '
          f'Spearman ρ={tma_rho:.3f} ({rho_verdict(tma_rho)})')
    print(f'  → Most critical for ALD paper')
    print()
    if tma_rho > 0.5:
        print('Surrogate correctly ranks designs. Track 1 supports the paper.')
        print('The optimizer will find good designs even if absolute field accuracy is limited.')
    else:
        print('Design ranking unreliable. The surrogate needs more training or more cases.')
        print('Consider: focus paper on framework methodology rather than accuracy claims.')


if __name__ == '__main__':
    main()
