import sys
import os
from pathlib import Path

# ── Environment-aware repo + data root ────────────────────────────────────
# Override with env vars for local use:
#   REPO_DIR=/Users/you/Desktop/CFD/cfd-ald-app
#   CFD_BASE=/Users/you/Desktop/CFD/cfd-ald-app
# Defaults work as-is inside Colab with Drive mounted.
_REPO = Path(os.environ.get('REPO_DIR', '/content/cfd-ald-app'))
sys.path.insert(0, str(_REPO))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import streamlit as st
import numpy as np
import json
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from physics.guardrails import GuardrailEngine, GuardrailBounds
from physics.calculator import euler

# ── Constants ──────────────────────────────────────────────────────────────
_DEFAULT_BASE = '/content/drive/MyDrive/cfd-ald-app'
_LOCAL_BASE   = Path(__file__).resolve().parent   # works when app.py is run directly
DRIVE_BASE    = Path(os.environ.get('CFD_BASE', _DEFAULT_BASE))
if not DRIVE_BASE.exists():
    DRIVE_BASE = _LOCAL_BASE                      # local fallback when Drive isn't mounted
CKPT        = DRIVE_BASE / 'checkpoints' / 'multihead' / 'multihead_final.pt'
OPT_JSON    = DRIVE_BASE / 'checkpoints' / 'optimizer' / 'optimizer_results.json'
OPT2_JSON   = DRIVE_BASE / 'checkpoints' / 'optimizer' / 'track2_optimizer_results.json'
GR_JSON     = DRIVE_BASE / 'checkpoints' / 'guardrail' / 'guardrail_results.json'
GL_JSON     = DRIVE_BASE / 'checkpoints' / 'geometry_loop' / 'geometry_loop_results.json'
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
RHO_N2      = 1.145
DEFAULTS = {
    'D_mm': 2.0, 'pitch_D': 4.0, 'Q_slm': 3.0,
    'H_plenum_mm': 20.0, 't_face_mm': 3.0, 'standoff_mm': 20.0,
    're_max': 5000.0, 'ma_max': 0.3, 'eu_max': 100000000.0,
    'pr_min': 0.5, 'pr_max': 100.0, 'peh_max': 100000.0,
    'sc_min': 0.1, 'sc_max': 10.0, 'pem_max': 100000.0,
    'da_min': 0.0001, 'da_max': 100.0,
}
GLOBAL_KEYS = [
    'Re', 'Pr', 'Sc', 'Ma', 'Pe_h', 'Pe_m', 'Da',
    'D', 'pitch_over_D', 'H_plenum', 't_face', 'standoff',
    'flow_rate_slm', 'beta', 'v_th', 'D_m', 'n_holes', 'open_area_frac',
]

# ── Model ──────────────────────────────────────────────────────────────────
def mlp(in_dim, out_dim, hidden=None, n_layers=2):
    hidden = hidden or out_dim
    dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
    mods = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(nn.SiLU())
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


# ── Cached loaders ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner='Loading surrogate model...')
def load_model():
    ckpt = torch.load(CKPT, map_location='cpu')
    cfg  = ckpt['cfg']
    norm = ckpt['norm']
    m = MultiHeadMGN(
        node_dim=cfg['node_input_dim'],
        edge_dim=cfg['edge_input_dim'],
        flow_out=cfg['flow_out_dim'],
        heat_out=cfg['heat_out_dim'],
        species_out=cfg['species_out_dim'],
        hidden=cfg['hidden_dim'],
        n_layers=cfg['n_layers'],
    ).to(DEVICE)
    m.load_state_dict(ckpt['model'])
    m.float()   # bf16 → float32; CPU has no native bf16 — without this inference is ~20× slower
    m.eval()
    return m, norm, cfg



def load_json(path):
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_geom_loop():
    """Load geometry_loop_results.json produced by 09_geometry_loop.ipynb."""
    if not GL_JSON.exists():
        return [], {}
    with open(GL_JSON) as f:
        data = json.load(f)
    return data.get('ranked_designs', []), data



def run_inference_pcgm(d_mm, pitch_d, q_slm, model, norm, cfg,
                       H_plenum=0.020, t_face=0.003, standoff=0.020,
                       bounds=None):
    """Run surrogate inference on a PCGM (Physics-Constrained Geometric Morphogenesis)
    point cloud generated on-the-fly for the given design parameters.

    Returns (coords [N,3], preds [N,6], global_dict, error_str, PCGMResult).
    On guardrail rejection returns (None, None, None, reason, None).
    """
    from geometry.pcgm import generate
    from geometry.grammar import NozzlePattern

    if bounds is None:
        bounds = GuardrailBounds(Re=(0.1, 5000.0), Ma=(0.0, 0.3), Da=(1e-4, 200.0), Eu=(0.5, 1e9))
    result = generate(
        geo_params={
            'D':            d_mm / 1000.0,
            'pitch_over_D': pitch_d,
            'H_plenum':     H_plenum,
            't_face':       t_face,
            'standoff':     standoff,
        },
        process_params={'flow_rate_slm': q_slm},
        bounds=bounds,
        pattern=NozzlePattern.HEX,
        n_points=80_000,
    )
    if not result.accepted:
        return None, None, None, result.reason, None

    idata = result.inference_data
    coords = idata['coords']           # [N, 3]  float32
    nf     = idata['node_features']    # [N, 4]  float32
    gf     = idata['global_features']  # [18]    float32

    node_mean = np.array(norm['node_mean'], dtype=np.float32)
    node_std  = np.array(norm['node_std'],  dtype=np.float32)
    out_mean  = np.array(norm['out_mean'],  dtype=np.float32)
    out_std   = np.array(norm['out_std'],   dtype=np.float32)

    N = len(coords)
    K = cfg['k_neighbors']
    xi = np.concatenate([nf, np.tile(gf, (N, 1))], axis=1)

    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=K + 1)
    idx = idx[:, 1:]
    src  = np.repeat(np.arange(N), K)
    dst  = idx.flatten()
    diff = coords[dst] - coords[src]
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    med  = float(np.median(dist)) + 1e-8
    ef   = np.concatenate([diff / med, dist / med], axis=1).astype(np.float32)

    x  = torch.from_numpy((xi - node_mean) / node_std).float().to(DEVICE)
    ei = torch.tensor(np.stack([src, dst]), dtype=torch.long).to(DEVICE)
    ea = torch.from_numpy(ef).float().to(DEVICE)

    with torch.no_grad():
        fp, hp, sp = model(x, ei, ea)
    fp = fp.cpu().numpy()
    hp = hp.cpu().numpy().flatten()
    sp = sp.cpu().numpy().flatten()
    del x, ei, ea

    preds = np.zeros((N, 6), dtype=np.float32)
    preds[:, :4] = fp * out_std[:4] + out_mean[:4]
    preds[:, 4]  = hp * out_std[4]  + out_mean[4]
    preds[:, 5]  = sp * out_std[5]  + out_mean[5]

    return coords, preds, dict(zip(GLOBAL_KEYS, gf)), '', result


def plot_2d_schematic(params, nozzle_xy, dim_nums, d_metrics):
    """
    Returns (fig_cross, fig_top) — two independent figures rendered in st.columns.
    fig_cross: axisymmetric cross-section with dimension annotations
    fig_top:   top-view nozzle hole pattern with guaranteed circular wafer

    params:     ShowerheadGeometry.params dict  (D, H_plenum, t_face, standoff, D_plate in metres)
    nozzle_xy:  [N, 2]  hole centres in metres
    dim_nums:   {Re, Da, Ma, …}  from PCGMResult.dim_nums
    d_metrics:  per-design dict from geometry_loop_results ranked_designs
    """
    D   = params['D'] * 1e3          # mm
    H   = params['H_plenum'] * 1e3   # mm
    tf  = params['t_face'] * 1e3     # mm
    so  = params['standoff'] * 1e3   # mm
    Dp  = params['D_plate'] * 1e3    # mm
    r_h = D / 2.0                    # hole radius mm

    r_holes = np.sqrt(nozzle_xy[:, 0] ** 2 + nozzle_xy[:, 1] ** 2) * 1e3  # radii mm

    fig_cross = go.Figure()
    fig_top   = go.Figure()

    # ── Cross-section ──────────────────────────────────────────────────────
    wafer_h     = max(3.0, so * 0.10)
    z_face_bot  = so
    z_face_top  = so + tf
    z_plen_top  = so + tf + H
    ann_x_ext   = Dp / 2 + Dp * 0.06  # x for dimension leader lines

    # ── Cross-section shapes ────────────────────────────────────────────────
    # Wafer
    fig_cross.add_shape(type='rect', x0=0, x1=Dp / 2, y0=-wafer_h, y1=0,
                        fillcolor='#cccccc', line=dict(color='#555', width=1))
    fig_cross.add_annotation(x=Dp / 4, y=-wafer_h / 2, text='Wafer',
                             showarrow=False, font=dict(size=10))

    # Standoff gap
    fig_cross.add_shape(type='rect', x0=0, x1=Dp / 2, y0=0, y1=z_face_bot,
                        fillcolor='rgba(200,230,255,0.12)',
                        line=dict(color='#aaa', width=1, dash='dot'))

    # Faceplate body
    fig_cross.add_shape(type='rect', x0=0, x1=Dp / 2, y0=z_face_bot, y1=z_face_top,
                        fillcolor='#4472C4', opacity=0.80,
                        line=dict(color='#1a3a7a', width=1))

    # Holes cut through faceplate
    for r in r_holes:
        x0 = max(0.0, r - r_h)
        x1 = min(Dp / 2, r + r_h)
        if x1 > x0:
            fig_cross.add_shape(type='rect', x0=x0, x1=x1, y0=z_face_bot, y1=z_face_top,
                                fillcolor='white', line=dict(width=0))

    # Plenum
    fig_cross.add_shape(type='rect', x0=0, x1=Dp / 2, y0=z_face_top, y1=z_plen_top,
                        fillcolor='rgba(135,206,235,0.35)',
                        line=dict(color='#2288aa', width=1))
    fig_cross.add_annotation(x=Dp / 4, y=(z_face_top + z_plen_top) / 2,
                             text='Plenum (N\u2082 + TMA)', showarrow=False,
                             font=dict(size=10, color='#114466'))

    # Gas inlet arrow
    fig_cross.add_annotation(x=Dp / 4, y=z_plen_top + 4, ax=0, ay=-20,
                             text='Gas inlet', showarrow=True,
                             arrowhead=2, arrowcolor='navy', font=dict(size=10))

    # ── Dimension leaders ──────────────────────────────────────────────
    def _leader(y0, y1, label):
        for y in [y0, y1]:
            fig_cross.add_shape(type='line', x0=Dp / 2, x1=ann_x_ext + Dp * 0.01,
                                y0=y, y1=y, line=dict(color='black', width=0.8))
        fig_cross.add_shape(type='line', x0=ann_x_ext, x1=ann_x_ext,
                            y0=y0, y1=y1, line=dict(color='black', width=0.8))
        fig_cross.add_annotation(x=ann_x_ext + Dp * 0.02, y=(y0 + y1) / 2,
                                 text=label, showarrow=False,
                                 font=dict(size=9), xanchor='left')

    _leader(z_face_bot, z_face_top, f't_face<br>{tf:.1f} mm')
    _leader(z_face_top, z_plen_top, f'H_plenum<br>{H:.1f} mm')
    _leader(0, z_face_bot,          f'standoff<br>{so:.1f} mm')

    # D_plate label
    fig_cross.add_annotation(x=Dp / 4, y=-wafer_h - 5,
                             text=f'D_plate = {Dp:.0f} mm', showarrow=False,
                             font=dict(size=10))

    # Nozzle diameter label on first hole
    if len(r_holes):
        fig_cross.add_annotation(x=r_holes[0], y=z_face_bot - 3,
                                 text=f'D = {D:.2f} mm', showarrow=True,
                                 ay=0, ax=-28, font=dict(size=9))

    fig_cross.update_xaxes(title_text='Radius [mm]',
                           range=[-4, Dp / 2 + Dp * 0.22],
                           showgrid=True, gridcolor='rgba(0,0,0,0.07)')
    fig_cross.update_yaxes(title_text='Height from wafer [mm]',
                           range=[-wafer_h - 9, z_plen_top + 14],
                           showgrid=True, gridcolor='rgba(0,0,0,0.07)')
    fig_cross.update_layout(
        title='Cross-section (axisymmetric right half)',
        height=420, autosize=True,
        margin=dict(l=20, r=60, t=40, b=20),
        plot_bgcolor='white', paper_bgcolor='white',
    )

    # ── Top-view hole pattern ────────────────────────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 300)
    fig_top.add_trace(go.Scatter(
        x=np.cos(theta) * Dp / 2, y=np.sin(theta) * Dp / 2,
        mode='lines', line=dict(color='#1a3a7a', width=2),
        name='Faceplate edge', showlegend=True,
    ))

    marker_px = max(3, min(14, int(D * 3.5)))
    fig_top.add_trace(go.Scatter(
        x=nozzle_xy[:, 0] * 1e3, y=nozzle_xy[:, 1] * 1e3,
        mode='markers',
        marker=dict(size=marker_px, color='#4472C4', opacity=0.70,
                    line=dict(width=0.5, color='#1a3a7a')),
        name=f"{len(nozzle_xy)} nozzles  D={D:.2f}mm",
        showlegend=True,
    ))

    fig_top.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers',
        marker=dict(size=7, color='red', symbol='cross-thin',
                    line=dict(width=2, color='red')),
        name='Centre', showlegend=False,
    ))

    # Pitch indicator between two adjacent holes
    if len(nozzle_xy) >= 2:
        p1, p2 = nozzle_xy[0] * 1e3, nozzle_xy[1] * 1e3
        pitch_mm = float(np.linalg.norm(p1 - p2))
        fig_top.add_trace(go.Scatter(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]],
            mode='lines+markers',
            line=dict(color='orange', width=1.5, dash='dash'),
            marker=dict(size=4, color='orange'),
            name=f'pitch = {pitch_mm:.2f}mm',
            showlegend=True,
        ))

    _R = Dp / 2 * 1.12
    fig_top.update_xaxes(title_text='X [mm]', range=[-_R, _R],
                         showgrid=True, gridcolor='rgba(0,0,0,0.07)',
                         scaleanchor='y', scaleratio=1, constrain='domain')
    fig_top.update_yaxes(title_text='Y [mm]', range=[-_R, _R],
                         showgrid=True, gridcolor='rgba(0,0,0,0.07)',
                         constrain='domain')
    fig_top.update_layout(
        title='Top view — nozzle hole pattern',
        height=500, autosize=True,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(x=0.02, y=0.02, bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='#ccc', borderwidth=1),
    )
    return fig_cross, fig_top


def plot_3d_field(coords, preds, field_idx, field_name, nozzle_xy=None, n_sample=5000):
    """
    3D scatter of plenum point cloud coloured by a field value.
    field_idx: 0=Ux 1=Uy 2=Uz 3=p 4=T 5=TMA  None=|U|
    Subsampled to n_sample points for browser performance.
    """
    rng = np.random.default_rng(42)
    N = len(coords)
    if N > n_sample:
        sel = rng.choice(N, n_sample, replace=False)
        c = coords[sel]
        f = (np.linalg.norm(preds[sel, :3], axis=1)
             if field_idx is None else preds[sel, field_idx])
    else:
        c = coords
        f = (np.linalg.norm(preds[:, :3], axis=1)
             if field_idx is None else preds[:, field_idx])

    traces = [go.Scatter3d(
        x=c[:, 0] * 1e3, y=c[:, 1] * 1e3, z=c[:, 2] * 1e3,
        mode='markers',
        marker=dict(size=2, color=f, colorscale='Turbo', opacity=0.55,
                    colorbar=dict(title=field_name, thickness=14, len=0.7)),
        name=field_name,
        hovertemplate=(
            'X: %{x:.1f}mm  Y: %{y:.1f}mm  Z: %{z:.1f}mm<br>'
            + field_name + ': %{marker.color:.4g}<extra></extra>'
        ),
    )]

    # Mark nozzle exit positions at bottom of plenum
    if nozzle_xy is not None:
        z_exit = float(coords[:, 2].min() * 1e3)
        n_show = min(len(nozzle_xy), 500)
        traces.append(go.Scatter3d(
            x=nozzle_xy[:n_show, 0] * 1e3,
            y=nozzle_xy[:n_show, 1] * 1e3,
            z=np.full(n_show, z_exit),
            mode='markers',
            marker=dict(size=3, color='black', opacity=0.6),
            name='Nozzle exits', hoverinfo='skip',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=560,
        scene=dict(
            xaxis_title='X [mm]',
            yaxis_title='Y [mm]',
            zaxis_title='Z [mm]',
            aspectmode='data',
            bgcolor='rgb(248,248,255)',
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=f'3D field — {field_name}', x=0.5),
        legend=dict(x=0.01, y=0.98),
    )
    return fig


def scatter_slice(coords, field, title, z_frac=0.15, n=6000):
    z = coords[:, 2]
    thresh = z.min() + z_frac * (z.max() - z.min())
    m = z < thresh
    if m.sum() < 20:
        m = np.ones(len(z), dtype=bool)
    xs, ys, fs = coords[m, 0], coords[m, 1], field[m]
    if len(xs) > n:
        idx = np.random.default_rng(0).choice(len(xs), n, replace=False)
        xs, ys, fs = xs[idx], ys[idx], fs[idx]
    fig = px.scatter(
        x=xs * 1e3, y=ys * 1e3, color=fs,
        color_continuous_scale='Turbo',
        labels={'x': 'X [mm]', 'y': 'Y [mm]', 'color': title},
        title=title,
    )
    fig.update_traces(marker_size=2)
    fig.update_xaxes(scaleanchor='y', scaleratio=1, constrain='domain')
    fig.update_yaxes(constrain='domain')
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=35, b=10))
    return fig


@st.cache_data(show_spinner=False)
def _cached_inference(d_mm, pitch_d, q_slm, H_plenum, t_face, standoff,
                      re_max, ma_max, eu_max, pr_min, pr_max, peh_max,
                      sc_min, sc_max, pem_max, da_min, da_max):
    """Cache GNN inference keyed by all scalar params — same values → instant re-use."""
    model, norm, cfg = load_model()
    bounds = GuardrailBounds(
        Re=(1.0,    re_max),
        Ma=(0.0,    ma_max),
        Da=(da_min, da_max),
        Eu=(0.5,    eu_max),
        Pr=(pr_min, pr_max),
        Sc=(sc_min, sc_max),
        Pe_h=(0.1,  peh_max),
        Pe_m=(0.1,  pem_max),
    )
    return run_inference_pcgm(d_mm, pitch_d, q_slm, model, norm, cfg,
                               H_plenum=H_plenum, t_face=t_face,
                               standoff=standoff, bounds=bounds)


@st.cache_data(show_spinner=False)
def _cached_inference_vices(vices_type_char, d_mm, pitch_d, q_slm,
                             H_plenum, t_face, vices_extra,
                             re_max, ma_max, eu_max, pr_min, pr_max,
                             peh_max, sc_min, sc_max, pem_max, da_min, da_max):
    """Track 2 inference: build VICES geometry → GNN surrogate."""
    from geometry.vices.variants import (build_type_a_baffled, build_type_b_conical,
                                          build_type_c_annular, build_type_d_twozone)
    model, norm, cfg = load_model()

    # Build VICES geometry
    kw = dict(D_mm=d_mm, pitch_D=pitch_d, H_plenum_mm=H_plenum * 1000,
              t_face_mm=t_face * 1000, D_plate_mm=150.0, resolution=48)
    if vices_type_char == 'A':
        result = build_type_a_baffled(**kw, baffle_frac=vices_extra)
    elif vices_type_char == 'B':
        result = build_type_b_conical(**kw, cone_r_frac=vices_extra)
    elif vices_type_char == 'C':
        result = build_type_c_annular(**kw, n_rings=int(vices_extra))
    else:
        result = build_type_d_twozone(**kw, divider_r_frac=vices_extra)

    if not result.accepted:
        return None, None, None, result.reason, None

    # Physics guardrail check
    from physics.calculator import reynolds, mach, euler, damkohler, k_rxn_from_sticking
    n_h   = result.params.get('n_nozzles', 1)
    D_m   = d_mm / 1000.0
    Q_m3s = q_slm * 1e-3 / 60.0
    V_noz = Q_m3s / (n_h * 3.14159 * (D_m / 2) ** 2)
    Re_v  = reynolds(RHO_N2, V_noz, D_m, MU_N2)
    Ma_v  = mach(V_noz, A_SOUND)

    bounds = GuardrailBounds(Re=(1.0, re_max), Ma=(0.0, ma_max),
                              Da=(da_min, da_max), Eu=(0.5, eu_max),
                              Pr=(pr_min, pr_max), Sc=(sc_min, sc_max),
                              Pe_h=(0.1, peh_max), Pe_m=(0.1, pem_max))

    # Build k-NN graph from VICES point cloud
    import numpy as np
    from scipy.spatial import cKDTree
    coords = result.point_cloud
    nf     = result.node_features
    gf     = result.global_features
    N      = len(coords)
    K      = cfg['k_neighbors']

    node_mean = np.array(norm['node_mean'], dtype=np.float32)
    node_std  = np.array(norm['node_std'],  dtype=np.float32)
    out_mean  = np.array(norm['out_mean'],  dtype=np.float32)
    out_std   = np.array(norm['out_std'],   dtype=np.float32)

    xi   = np.concatenate([nf, np.tile(gf, (N, 1))], axis=1)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=K + 1)
    idx  = idx[:, 1:]
    src  = np.repeat(np.arange(N), K)
    dst  = idx.flatten()
    diff = coords[dst] - coords[src]
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    med  = float(np.median(dist)) + 1e-8
    ef   = np.concatenate([diff / med, dist / med], axis=1).astype(np.float32)

    import torch
    x  = torch.from_numpy((xi - node_mean) / (node_std + 1e-8)).float().to(DEVICE)
    ei = torch.tensor(np.stack([src, dst]), dtype=torch.long).to(DEVICE)
    ea = torch.from_numpy(ef).float().to(DEVICE)

    with torch.no_grad():
        fp, hp, sp = model(x, ei, ea)
    fp = fp.cpu().numpy(); hp = hp.cpu().numpy().flatten()
    sp = sp.cpu().numpy().flatten()
    del x, ei, ea

    preds = np.zeros((N, 6), dtype=np.float32)
    preds[:, :4] = fp * out_std[:4] + out_mean[:4]
    preds[:, 4]  = hp * out_std[4]  + out_mean[4]
    preds[:, 5]  = sp * out_std[5]  + out_mean[5]

    # Global dict (physics numbers) for guardrail
    gd = {'Re': Re_v, 'Ma': Ma_v, 'n_holes': n_h,
          'vices_type': vices_type_char, 'n_nozzles': n_h}
    return coords, preds, gd, '', result


@st.cache_data(show_spinner=False)
def _cached_scatter_slice(coords_bytes, field_bytes, title):
    coords = np.frombuffer(coords_bytes, dtype=np.float32).reshape(-1, 3)
    field  = np.frombuffer(field_bytes,  dtype=np.float32)
    return scatter_slice(coords, field, title)


def _slider_btns(label, min_val, max_val, default, step, key, help=None, fmt='%g'):
    """Slider with ➖ / ➕ step buttons using emoji labels — emoji render as
    coloured glyphs that cannot be hidden by CSS, and st.button callbacks work."""
    def _dec():
        cur = st.session_state.get(key, default)
        st.session_state[key] = max(min_val, round(cur - step, 10))
    def _inc():
        cur = st.session_state.get(key, default)
        st.session_state[key] = min(max_val, round(cur + step, 10))

    c_lo, c_sl, c_hi = st.columns([3, 8, 3])
    with c_lo:
        st.button('\u2796', key=f'dec_{key}', on_click=_dec, use_container_width=True)
    with c_sl:
        val = st.slider(label, min_val, max_val, default, step,
                        key=key, help=help, format=fmt)
    with c_hi:
        st.button('\u2795', key=f'inc_{key}', on_click=_inc, use_container_width=True)
    return val


# ── Page layout ────────────────────────────────────────────────────────────
st.set_page_config(page_title='ALD Showerhead CFD', layout='wide', page_icon='🌀')

st.markdown("""
<style>
/* ── Replace ? tooltip icon with bold italic i ───────────────────────────── */
[data-testid="stTooltipIcon"] {
    position: relative !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stTooltipIcon"] svg { opacity: 0 !important; width: 16px !important; height: 16px !important; }
[data-testid="stTooltipIcon"]::before {
    content: "i";
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-style: italic;
    font-weight: 900;
    font-size: 11px;
    color: #444;
    width: 15px; height: 15px;
    border: 1.5px solid #666;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    line-height: 15px;
    text-align: center;
    pointer-events: none;
    box-sizing: border-box;
}
</style>
""", unsafe_allow_html=True)

st.title('ALD Showerhead CFD Surrogate Dashboard')
st.caption(f'MultiHeadMGN  |  Device: {DEVICE}  |  BoTorch Pareto optimiser')

with st.sidebar:
    # Apply pending reset BEFORE any widget is instantiated to avoid session_state conflict
    if st.session_state.pop('_reset_requested', False):
        for _k, _v in DEFAULTS.items():
            st.session_state[_k] = _v

    # ── Track selector ────────────────────────────────────────────────────────
    st.header('Design Track')
    active_track = st.radio(
        'Geometry synthesis method',
        ['Track 1 — PCGM (Parametric)', 'Track 2 — VICES (CSG Topology)'],
        captions=[
            'Parametric hex nozzle array. Vary diameter, pitch, plenum height, '
            'faceplate thickness and standoff. Fixed topology — fast inference (~5s).',
            'CSG boolean tree synthesis via SDF + Marching Cubes. Choose from '
            '4 topology types: baffled, conical, annular, two-zone. '
            'Explores designs Track 1 cannot represent (~30s build + inference).',
        ],
        key='active_track',
        help=(
            'PCGM — Physics-Constrained Geometric Morphogenesis: '
            'generates showerhead geometry from parametric rules (nozzle diameter, '
            'pitch, plenum height). Fixed hex-array topology.\n\n'
            'VICES — Voxel-Implicit Computational Engineering Synthesis: '
            'builds geometry as Signed Distance Fields (SDF) using boolean CSG operations '
            '(union, subtract, intersect) on primitive shapes. Supports 4 distinct '
            'topology families. Design space characterised by Catalan numbers — '
            'e.g. 6 primitives yield C(5)=42 distinct tree topologies. '
            'Marching Cubes converts SDF voxel grid to triangle mesh.'
        ),
    )
    is_track2 = active_track.startswith('Track 2')

    st.divider()
    st.header('Design Parameters')

    # ── Track 2: geometry type + type-specific params ─────────────────────────
    if is_track2:
        vices_type = st.selectbox(
            'Geometry type',
            ['A — Baffled plenum', 'B — Conical diffuser',
             'C — Annular rings',  'D — Two-zone plenum'],
            key='vices_type',
            help=(
                'A — Baffled plenum: an annular baffle ring inside the plenum forces gas to '
                'redistribute radially before reaching the nozzle plate. '
                'CSG: Cylinder − BaffleAnnulus − Nozzles.\n\n'
                'B — Conical diffuser: a central cone protrudes from the inlet face, '
                'deflecting the inlet jet radially outward for more uniform distribution. '
                'CSG: Cylinder − Cone − Nozzles.\n\n'
                'C — Annular rings: nozzles arranged in concentric rings instead of hex packing, '
                'giving a different radial flux profile. '
                'CSG: Cylinder − RingNozzleUnion.\n\n'
                'D — Two-zone plenum: an annular divider ring splits the plenum into inner '
                'and outer gas zones with different residence times. '
                'CSG: Cylinder − DividerRing − Nozzles.'
            ),
        )
        vices_type_char = vices_type[0]   # 'A','B','C','D'

        with st.expander('Type-specific parameter', expanded=True):
            if vices_type_char == 'A':
                vices_extra = st.slider('Baffle position (fraction of plenum height)',
                                        0.2, 0.8, 0.5, 0.05, key='vices_extra',
                                        help='Vertical position of the baffle inside the plenum, '
                                             'expressed as a fraction of plenum height from the faceplate. '
                                             '0.3 = low (near faceplate), 0.7 = high (near inlet). '
                                             'Lower baffle → shorter mixing path, sharper redistribution.')
            elif vices_type_char == 'B':
                vices_extra = st.slider('Cone base radius (fraction of plate radius)',
                                        0.2, 0.6, 0.35, 0.05, key='vices_extra',
                                        help='Base radius of the conical diffuser as a fraction of '
                                             'the showerhead plate radius. '
                                             'Larger cone → stronger radial deflection → more uniform '
                                             'peripheral distribution. Smaller cone → more central flow.')
            elif vices_type_char == 'C':
                vices_extra = st.slider('Number of nozzle rings', 2, 5, 3, 1,
                                        key='vices_extra',
                                        help='Number of concentric nozzle rings. '
                                             'Each ring has nozzles equally spaced around its circumference. '
                                             '2 rings = sparse, 5 rings = dense. '
                                             'Nozzles per ring scales with ring circumference.')
            else:
                vices_extra = st.slider('Divider radius (fraction of plate radius)',
                                        0.3, 0.65, 0.45, 0.05, key='vices_extra',
                                        help='Inner radius of the annular divider ring as a fraction '
                                             'of the plate radius. Sets the boundary between inner and '
                                             'outer flow zones. '
                                             '0.35 = small inner zone, 0.60 = large inner zone.')

    # ── Common design params (both tracks) ────────────────────────────────────
    D_mm    = _slider_btns('Nozzle diameter D [mm]', 1.0,  3.0,   DEFAULTS['D_mm'],    0.1,
                           'D_mm',    help='Diameter of each nozzle/hole in the showerhead faceplate.')
    pitch_D = _slider_btns('Pitch / D',              3.0,  6.0,   DEFAULTS['pitch_D'], 0.25,
                           'pitch_D', help='Centre-to-centre hole spacing divided by hole diameter.')
    Q_slm   = _slider_btns('Flow rate Q [slm]',      1.0,  10.0,  DEFAULTS['Q_slm'],   0.25,
                           'Q_slm',   help='Total gas flow rate in Standard Litres per Minute.')
    with st.expander('Geometry (advanced)'):
        H_plenum_mm = _slider_btns('Plenum height H [mm]',       10.0, 40.0,  DEFAULTS['H_plenum_mm'], 1.0,
                                   'H_plenum_mm')
        t_face_mm   = _slider_btns('Faceplate thickness t [mm]',  1.0,  6.0,  DEFAULTS['t_face_mm'],   0.25,
                                   't_face_mm')
        standoff_mm = _slider_btns('Standoff gap [mm]',            5.0, 40.0,  DEFAULTS['standoff_mm'], 1.0,
                                   'standoff_mm')

    st.divider()
    st.subheader('Physics Guardrail Bounds')
    st.caption('Adjust bounds, then click Run Prediction. Violations show warnings in results.')

    st.markdown('**Momentum Transfer**')
    re_max  = _slider_btns('Re max — Reynolds number ρVD/μ',      500.0,   20000.0,  DEFAULTS['re_max'],  100.0,
                           're_max',  help='Reynolds number Re = ρVD/μ — ratio of inertial to viscous forces. '
                                          'Re < 2300: laminar. Re > 4000: turbulent. '
                                          'Surrogate trained on laminar data; high Re predictions are extrapolations.')
    ma_max  = _slider_btns('Ma max — Mach number V/a',             0.05,    1.0,      DEFAULTS['ma_max'],  0.01,
                           'ma_max',  help='Mach number Ma = V/a — ratio of flow velocity to speed of sound. '
                                          'Ma < 0.3: incompressible assumption valid. '
                                          'Ma > 0.3: compressibility effects appear; surrogate accuracy degrades.')
    eu_max  = _slider_btns('Eu max — Euler number Δp/½ρV²',       1000.0, 1e8, DEFAULTS['eu_max'],  1000.0,
                           'eu_max',  help='Euler number Eu = Δp / (½ρV²) — dimensionless pressure drop across the showerhead. '
                                          'ALD creeping-flow (low Re) naturally yields Eu ~ 1e5–1e7; this is physically correct. '
                                          'Flag only if Eu > 1e8 (unrealistic pressure drop). '
                                          'Use the raw Δp [Pa] metric for process-level pressure budget.')

    st.markdown('**Heat Transfer**')
    pr_min  = _slider_btns('Pr min — Prandtl number cpμ/k',        0.1,    2.0,      DEFAULTS['pr_min'],  0.05,
                           'pr_min',  help='Prandtl number Pr = cp·μ/k — ratio of momentum diffusivity to thermal diffusivity. '
                                          'Pr ≈ 0.71 for N₂ at process temperature. Governs thermal boundary layer thickness.')
    pr_max  = _slider_btns('Pr max — Prandtl number cpμ/k',        2.0,   500.0,     DEFAULTS['pr_max'],  10.0,
                           'pr_max',  help='Upper Pr bound. High Pr fluids have thin thermal BLs. '
                                          'N₂ carrier gas stays near 0.71; large deviation signals wrong fluid properties.')
    peh_max = _slider_btns('Pe_h max — Péclet number heat Re·Pr',  1000.0, 500000.0,  DEFAULTS['peh_max'], 1000.0,
                           'peh_max', help='Péclet number (heat) Pe_h = Re·Pr — ratio of advective to diffusive heat transport. '
                                          'High Pe_h: convection dominates, thin thermal BL. '
                                          'Low Pe_h: diffusion smooths temperature gradients.')
    st.caption('Nu (Nusselt hL/k), Bi (Biot hL/k_s) — available after CFD run only.')

    st.markdown('**Mass Transfer & Reaction**')
    sc_min  = _slider_btns('Sc min — Schmidt number μ/ρD_m',       0.05,   2.0,      DEFAULTS['sc_min'],  0.05,
                           'sc_min',  help='Schmidt number Sc = μ/(ρ·D_m) — ratio of momentum diffusivity to mass diffusivity. '
                                          'Governs the concentration boundary layer. Sc > 1 means mass diffuses slower than momentum.')
    sc_max  = _slider_btns('Sc max — Schmidt number μ/ρD_m',       2.0,   50.0,      DEFAULTS['sc_max'],  1.0,
                           'sc_max',  help='Upper Sc bound. TMA in N₂ has Sc ≈ 1–3. '
                                          'Values outside this range suggest incorrect diffusivity D_m.')
    pem_max = _slider_btns('Pe_m max — Péclet number mass Re·Sc',  1000.0, 500000.0,  DEFAULTS['pem_max'], 1000.0,
                           'pem_max', help='Péclet number (mass) Pe_m = Re·Sc — ratio of advective to diffusive mass transport. '
                                          'High Pe_m: convection dominates species transport. '
                                          'Low Pe_m: diffusion spreads precursor uniformly (favourable for ALD).')
    da_min  = _slider_btns('Da min — Damköhler number k_rxn·L/V',  0.0001, 0.1,      DEFAULTS['da_min'],  0.0001,
                           'da_min',  fmt='%g',
                           help='Damköhler number Da = k_rxn·L/V — ratio of surface reaction rate to convective transport rate. '
                                'Da >> 1: transport-limited (precursor depletes before reacting). '
                                'Da << 1: reaction-limited (ideal ALD self-limiting regime).')
    da_max  = _slider_btns('Da max — Damköhler number k_rxn·L/V',  1.0,   500.0,     DEFAULTS['da_max'],  10.0,
                           'da_max',  help='Upper Da limit. Very high Da means the precursor is consumed too fast '
                                          'before reaching the wafer centre → non-uniform deposition.')
    st.caption('Sh (Sherwood k_m L/D_m) — available after CFD run only.')

    st.divider()
    run_btn = st.button('Run Prediction', type='primary', use_container_width=True)
    if st.button('Reset to defaults', use_container_width=True):
        st.session_state['_reset_requested'] = True
        st.rerun()

    st.divider()
    with st.expander('Physics Reference'):
        st.markdown('''
**Dimensionless Numbers**

| Symbol | Full name | Formula | What it tells you |
|--------|-----------|---------|-------------------|
| **Re** | Reynolds | ρVD/μ | Flow regime: <2300 laminar, >4000 turbulent |
| **Da** | Damköhler | k_rxn·L/V | <1 reaction-limited (ideal ALD), >1 transport-limited |
| **Ma** | Mach | V/a | <0.3 incompressible, >0.3 compressibility kicks in |
| **Eu** | Euler | Δp/(½ρV²) | Pressure drop penalty; lower = less pumping power |
| **Pr** | Prandtl | cpμ/k | Heat BL thickness; ≈0.71 for N₂ |
| **Sc** | Schmidt | μ/(ρD_m) | Mass diffusion vs momentum; ≈1–3 for TMA (Trimethylaluminium) in N₂ |
| **Pe_h** | Péclet (heat) | Re·Pr | Advection vs diffusion of heat |
| **Pe_m** | Péclet (mass) | Re·Sc | Advection vs diffusion of precursor |
| **Nu** | Nusselt | hL/k | Convective vs conductive heat transfer (CFD only) |
| **Bi** | Biot | hL/k_s | Surface vs internal temperature gradient (CFD only) |
| **Sh** | Sherwood | k_m L/D_m | Convective vs diffusive mass transfer (CFD only) |

**Performance Metrics**

| Symbol | Meaning |
|--------|---------|
| **T_UI** | Temperature Uniformity Index = 1 − CV(T). Range (−∞,1]. >0.95 = excellent |
| **TMA_UI** | TMA (Trimethylaluminium, Al(CH₃)₃) Uniformity Index = 1 − CV(TMA). Negative = std > mean (non-uniform) |
| **Score** | Composite = 0.4×T_UI + 0.4×TMA_UI + 0.2×confidence |
| **conf** | Guardrail confidence [0→1]. 1.0 = all physics checks passed |

**Key terms**

- **ALD** — Atomic Layer Deposition: self-limiting surface reaction, deposits one atomic layer per cycle
- **TMA** — Trimethylaluminium Al(CH₃)₃, precursor gas for Al₂O₃ ALD
- **PCGM** — Physics-Constrained Geometric Morphogenesis: parametric showerhead geometry generator
- **slm** — Standard Litres per Minute (gas flow at 0°C, 1 atm)
- **CV** — Coefficient of Variation = std/mean (dimensionless spread)
- **plenum** — The gas distribution chamber above the faceplate
- **standoff** — Gap between faceplate and wafer surface
''')

model, norm, cfg = load_model()
opt_data = load_json(str(OPT_JSON))
gr_data  = load_json(str(GR_JSON))

tab_pred, tab_opt, tab_gr, tab_geo, tab_t2 = st.tabs(
    ['Predictions', 'Optimizer / Pareto', 'Guardrail Report',
     'Pareto Designs', 'Track 2 — VICES']
)

# ── Tab 1: Predictions ─────────────────────────────────────────────────────
with tab_pred:
    if run_btn:
        _track_label = 'VICES CSG' if is_track2 else 'PCGM'
        with st.spinner(f'Generating {_track_label} geometry + running surrogate...'):
            if is_track2:
                coords, preds, gd, err, res = _cached_inference_vices(
                    vices_type_char,
                    D_mm, pitch_D, Q_slm,
                    H_plenum_mm / 1000.0, t_face_mm / 1000.0, vices_extra,
                    re_max, ma_max, eu_max, pr_min, pr_max, peh_max,
                    sc_min, sc_max, pem_max, da_min, da_max,
                )
            else:
                coords, preds, gd, err, res = _cached_inference(
                    D_mm, pitch_D, Q_slm,
                    H_plenum_mm / 1000.0, t_face_mm / 1000.0, standoff_mm / 1000.0,
                    re_max, ma_max, eu_max, pr_min, pr_max, peh_max,
                    sc_min, sc_max, pem_max, da_min, da_max,
                )

        if coords is None:
            st.error(f'Guardrail engine rejected this design: {err}')
            st.session_state.pop('pred_results', None)
        else:
            pred_bounds = GuardrailBounds(
                Re=(1.0, re_max), Ma=(0.0, ma_max), Da=(da_min, da_max),
                Eu=(0.5, eu_max), Pr=(pr_min, pr_max), Sc=(sc_min, sc_max),
                Pe_h=(0.1, peh_max), Pe_m=(0.1, pem_max),
            )
            D_m_run   = D_mm / 1000.0
            n_h_run   = max(int(gd.get('n_holes', gd.get('n_nozzles', 1))), 1)
            Q_m3s_run = Q_slm * 1.667e-5
            V_noz_run = Q_m3s_run / (n_h_run * 3.14159 * (D_m_run / 2) ** 2)
            dp_run    = float(abs(preds[:, 3].max() - preds[:, 3].min()))
            st.session_state['pred_results'] = {
                'coords': coords, 'preds': preds, 'gd': gd, 'res': res,
                'pred_bounds': pred_bounds,
                'Eu':    euler(max(dp_run, 1e-3), RHO_N2, max(V_noz_run, 1e-3)),
                'dp_Pa': dp_run,
                'n_h': n_h_run,
                'D_mm': D_mm, 'pitch_D': pitch_D, 'Q_slm': Q_slm,
                'H_plenum_mm': H_plenum_mm, 't_face_mm': t_face_mm,
                'standoff_mm': standoff_mm if not is_track2 else 20.0,
                'track': 2 if is_track2 else 1,
                'vices_type': vices_type_char if is_track2 else None,
                'fig_3d': {},
            }

    if 'pred_results' not in st.session_state:
        st.info('Set parameters in the sidebar and click **Run Prediction**.')
    else:
        # ── Unpack from session state ─────────────────────────────────────────
        _r            = st.session_state['pred_results']
        coords        = _r['coords'];   preds  = _r['preds'];  gd   = _r['gd']
        res           = _r['res'];      Eu     = _r['Eu'];     n_h  = _r['n_h']
        dp_Pa         = _r.get('dp_Pa', float('nan'))
        pred_bounds   = _r['pred_bounds']
        D_mm_r        = _r['D_mm'];     pitch_D_r = _r['pitch_D'];  Q_slm_r = _r['Q_slm']
        H_plenum_mm_r = _r['H_plenum_mm']
        t_face_mm_r   = _r['t_face_mm']
        standoff_mm_r = _r['standoff_mm']
        _result_track = _r.get('track', 1)
        _vices_type   = _r.get('vices_type', None)

        # ── Track badge ───────────────────────────────────────────────────────
        _type_labels = {'A': 'Baffled plenum', 'B': 'Conical diffuser',
                        'C': 'Annular rings',  'D': 'Two-zone plenum'}
        if _result_track == 2 and _vices_type:
            st.info(f'🌀 **Track 2 — VICES** · Type {_vices_type}: '
                    f'{_type_labels.get(_vices_type,"")} · '
                    f'n_nozzles = {n_h} · '
                    f'CSG: {gd.get("vices_type","?")} geometry')

        # ── Key metrics ───────────────────────────────────────────────────────
        def _ui(field, z, z_lo_frac, z_hi_frac):
            z_lo = z.min() + z_lo_frac * (z.max() - z.min())
            z_hi = z.min() + z_hi_frac * (z.max() - z.min())
            m = (coords[:, 2] >= z_lo) & (coords[:, 2] <= z_hi)
            if m.sum() < 10:
                return float('nan')
            return float(1.0 - field[m].std() / (abs(field[m].mean()) + 1e-12))

        z = coords[:, 2]
        T_UI   = _ui(preds[:, 4], z, 0.0,  0.10)
        TMA_UI = _ui(preds[:, 5], z, 0.15, 0.55)

        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric('T mean [K]',  f"{preds[:, 4].mean():.1f}",
                  help='Mean gas temperature across the plenum point cloud [Kelvin]. '
                       'Target: close to process temperature (typically 150–400°C for ALD).')
        m2.metric('T range [K]', f"{preds[:, 4].max() - preds[:, 4].min():.2f}",
                  help='Temperature spread = max(T) − min(T) [Kelvin]. '
                       'Lower is better — large range means hot/cold zones on the wafer.')
        m3.metric('TMA max',     f"{preds[:, 5].max():.3e}",
                  help='Peak TMA (trimethylaluminium, Al(CH₃)₃) mass fraction in the plenum. '
                       'TMA is the precursor gas for Al₂O₃ ALD. Higher near inlets, should spread uniformly.')
        m4.metric('Eu',          f'{Eu:.2e}',
                  help='Euler number = Δp / (½ρV²) — dimensionless pressure drop across the showerhead. '
                       'ALD creeping-flow naturally gives Eu ~ 1e5–1e7 (low velocity → tiny dynamic pressure). '
                       'Computed from surrogate pressure field.')
        m5.metric('Δp [Pa]',     f'{dp_Pa:.1f}' if not np.isnan(dp_Pa) else 'n/a',
                  help='Absolute pressure drop across the showerhead [Pascal], computed from surrogate p field. '
                       'More actionable than Eu for ALD process budgeting. '
                       'Typical ALD showerhead target: < 500 Pa. Above 1000 Pa may affect precursor delivery uniformity.')
        m6.metric('T_UI',        f'{T_UI:.3f}' if not np.isnan(T_UI) else 'n/a',
                  help='Temperature Uniformity Index = 1 − CV(T) in near-wafer band (bottom 10% of plenum). '
                       'CV = std/mean. Range: (−∞, 1]. Higher is better. >0.95 is excellent.')
        m7.metric('TMA_UI',      f'{TMA_UI:.3f}' if not np.isnan(TMA_UI) else 'n/a',
                  help='TMA Uniformity Index = 1 − CV(TMA) in mid-plenum band (15–55% of plenum height). '
                       'Negative value means std > mean (highly non-uniform precursor distribution). '
                       'Target: as close to 1.0 as possible.')

        # ── Field slices ──────────────────────────────────────────────────────
        st.subheader('Field slices — bottom 15% of plenum')
        U_mag = np.linalg.norm(preds[:, :3], axis=1)
        _cb = coords.astype(np.float32).tobytes()
        _ub = U_mag.astype(np.float32).tobytes()
        c1, c2 = st.columns(2)
        c1.plotly_chart(_cached_scatter_slice(_cb, preds[:, 4].astype(np.float32).tobytes(), 'T [K]'),     use_container_width=True)
        c2.plotly_chart(_cached_scatter_slice(_cb, preds[:, 5].astype(np.float32).tobytes(), 'TMA'),        use_container_width=True)
        c3, c4 = st.columns(2)
        c3.plotly_chart(_cached_scatter_slice(_cb, preds[:, 3].astype(np.float32).tobytes(), 'p [m²/s²]'), use_container_width=True)
        c4.plotly_chart(_cached_scatter_slice(_cb, _ub, '|U| [m/s]'),                                      use_container_width=True)

        # ── Geometry + field viewer ───────────────────────────────────────────
        st.subheader('Geometry + Field Viewer')
        vis_col, ann_col = st.columns([3, 2])

        with vis_col:
            view_mode = st.radio(
                'View', ['2D Engineering Drawing', '3D Fields'],
                horizontal=True, key='pred_mode',
            )
            if view_mode == '2D Engineering Drawing':
                fig_cross, fig_top = plot_2d_schematic(
                    res.geometry.params, res.geometry.nozzle_xy, res.dim_nums, {}
                )
                _drawing = st.radio('Panel', ['Cross-section', 'Top view (wafer)'],
                                    horizontal=True, key='pred_2d_panel')
                if _drawing == 'Cross-section':
                    st.plotly_chart(fig_cross, use_container_width=True)
                else:
                    st.plotly_chart(fig_top, use_container_width=True)
            else:
                _field_map = {'T [K]': 4, 'TMA': 5, 'p [Pa]': 3,
                              '|U| [m/s]': None, 'Ux': 0, 'Uy': 1, 'Uz': 2}
                sel_field = st.selectbox(
                    'Field to display',
                    list(_field_map.keys()),
                    key='pred_field',
                )
                if sel_field not in _r['fig_3d']:
                    with st.spinner(f'Building 3D view for {sel_field}...'):
                        _r['fig_3d'][sel_field] = plot_3d_field(
                            coords, preds, _field_map[sel_field], sel_field,
                            res.geometry.nozzle_xy,
                        )
                st.plotly_chart(_r['fig_3d'][sel_field], use_container_width=True)

        with ann_col:
            p_geo = res.geometry.params
            dim   = res.dim_nums

            st.markdown('#### Design Specification')
            st.dataframe(pd.DataFrame({
                'Parameter': [
                    'D — nozzle diameter', 'pitch / D', 'Q — flow rate',
                    'n_holes', 'open area %', 'H_plenum', 't_face',
                    'standoff', 'D_plate', 'pattern',
                ],
                'Value': [
                    f"{D_mm_r:.2f} mm",
                    f"{pitch_D_r:.2f}",
                    f"{Q_slm_r:.1f} slm",
                    str(n_h),
                    f"{float(gd.get('open_area_frac', 0)) * 100:.1f}%",
                    f"{H_plenum_mm_r:.1f} mm",
                    f"{t_face_mm_r:.1f} mm",
                    f"{standoff_mm_r:.1f} mm",
                    f"{p_geo.get('D_plate', 0.3) * 1e3:.0f} mm",
                    'hex',
                ],
            }), use_container_width=True, hide_index=True)

            st.markdown('#### Dimensionless Numbers')
            _DIM_META = {
                'Re':   ('Reynolds (ρVD/μ)',      'Flow regime'),
                'Da':   ('Damköhler (k_rxn·L/V)', 'Reaction vs transport'),
                'Ma':   ('Mach (V/a)',              'Compressibility'),
                'Eu':   ('Euler (Δp/½ρV²)',         'Pressure drop'),
                'Pr':   ('Prandtl (cp·μ/k)',        'Heat BL scaling'),
                'Sc':   ('Schmidt (μ/ρD_m)',        'Diffusion scaling'),
                'Pe_h': ('Péclet heat (Re·Pr)',     'Advection vs diffusion'),
                'Pe_m': ('Péclet mass (Re·Sc)',     'Advection vs diffusion'),
            }
            all_dim = dict(dim)
            all_dim['Eu'] = Eu
            st.dataframe(pd.DataFrame([
                {'Symbol': sym,
                 'Formula': meta[0],
                 'Value': f"{all_dim.get(sym, float('nan')):.4g}",
                 'Guards against': meta[1]}
                for sym, meta in _DIM_META.items()
            ]), use_container_width=True, hide_index=True)

            st.markdown('#### Performance Metrics')
            st.dataframe(pd.DataFrame([
                ('T_UI',    f'{T_UI:.4f}'   if not np.isnan(T_UI)   else 'n/a', '1−CV(T) near wafer'),
                ('TMA_UI',  f'{TMA_UI:.4f}' if not np.isnan(TMA_UI) else 'n/a', '1−CV(TMA) mid-plenum'),
                ('T mean',  f'{preds[:, 4].mean():.1f} K', 'Mean temperature'),
                ('T range', f'{preds[:, 4].max() - preds[:, 4].min():.2f} K', 'Max − min T'),
                ('TMA max', f'{preds[:, 5].max():.3e}', 'Peak TMA mass fraction'),
                ('Eu',      f'{Eu:.4g}', 'Euler number Δp/(½ρV²)'),
                ('n_holes', str(n_h), 'Number of nozzle holes'),
            ], columns=['Metric', 'Value', 'Description']),
            use_container_width=True, hide_index=True)

        # ── Guardrail check ───────────────────────────────────────────────────
        st.subheader('Guardrail check')
        engine   = GuardrailEngine(pred_bounds)
        dim_vals = dict(res.dim_nums)
        dim_vals['Eu'] = Eu
        gr_result = engine.check(dim_vals)

        if gr_result.passed:
            st.success(f'PASS  confidence {gr_result.confidence:.3f}')
        else:
            st.error(f'FAIL  confidence {gr_result.confidence:.3f}')
            for v in gr_result.violations:
                st.warning(
                    f'**{v.symbol}** = {v.value:.4g}  '
                    f'(allowed [{v.lo:.4g}, {v.hi:.4g}])  {v.message}'
                )
        for flag in gr_result.special_flags:
            st.info(f'Flag: {flag}')
        for rec in gr_result.recommendations:
            st.caption(f'→ {rec}')

# ── Tab 2: Optimizer ───────────────────────────────────────────────────────
with tab_opt:
    if opt_data is None:
        st.warning('Run 07_optimizer.ipynb first to generate optimizer_results.json.')
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric('Cases evaluated',   opt_data.get('n_observed', '?'))
        c2.metric('Pareto designs',    opt_data.get('n_pareto',   '?'))
        c3.metric('BO suggestions',    '20  (5 rounds x 4)')

        obj_all = opt_data.get('objectives_all', [])
        if obj_all:
            df = pd.DataFrame(obj_all)
            if {'TMA_UI', 'neg_Eu'}.issubset(df.columns):
                fig = px.scatter(
                    df, x='TMA_UI', y='neg_Eu',
                    color='Re' if 'Re' in df.columns else None,
                    hover_data=[c for c in ['name', 'D', 'pitch_over_D', 'flow_rate_slm'] if c in df.columns],
                    color_continuous_scale='Plasma',
                    title='Objective space — TMA uniformity vs -Euler number',
                )
                fig.update_traces(marker_size=8)
                st.plotly_chart(fig, use_container_width=True)

            show = [c for c in ['name', 'D', 'pitch_over_D', 'flow_rate_slm',
                                 'Re', 'TMA_UI', 'T_UI', 'neg_Eu'] if c in df.columns]
            st.dataframe(
                df[show].sort_values('TMA_UI', ascending=False).reset_index(drop=True),
                use_container_width=True,
            )

        top = opt_data.get('top_candidates', [])
        if top:
            st.subheader('Top Pareto candidates')
            st.dataframe(pd.DataFrame(top), use_container_width=True)

# ── Tab 3: Guardrails ──────────────────────────────────────────────────────
with tab_gr:
    if gr_data is None:
        st.warning('Run 06_guardrail_engine.ipynb first to generate guardrail_results.json.')
    else:
        df_gr = pd.DataFrame(gr_data)
        n_pass    = int(df_gr['passed'].sum())         if 'passed'     in df_gr.columns else 0
        mean_conf = float(df_gr['confidence'].mean())  if 'confidence' in df_gr.columns else 0.0
        n_refine  = int((df_gr['confidence'] < 0.5).sum()) if 'confidence' in df_gr.columns else 0

        st.caption(
            f'Guardrail engine evaluated {len(df_gr)} designs against physics bounds '
            f'(Re, Ma, Eu, Pr, Sc, Pe_h, Pe_m, Da). '
            f'Confidence < 0.5 → recommended for full CFD (OpenFOAM reactingFoam) refinement. '
            f'0/80 passing here is expected — default bounds are conservative training-data limits; '
            f'adjust them in the sidebar to match your process conditions.'
        )

        c1, c2, c3 = st.columns(3)
        c1.metric('Passed (default bounds)', f'{n_pass} / {len(df_gr)}',
                  help='Designs within ALL physics guardrail bounds simultaneously. '
                       '0/80 is common with conservative default bounds — adjust bounds in sidebar.')
        c2.metric('Mean confidence',          f'{mean_conf:.3f}',
                  help='Average guardrail confidence score [0→1]. '
                       'Computed as 1 − (weighted violation severity). '
                       'Higher = more physically consistent design.')
        c3.metric('Need CFD refinement',      str(n_refine),
                  help='Designs with confidence < 0.5. '
                       'These are flagged for full OpenFOAM (reactingFoam) CFD validation.')

        if 'confidence' in df_gr.columns:
            fig = px.histogram(
                df_gr, x='confidence', nbins=25,
                title='Guardrail confidence distribution across all 80 designs',
                color_discrete_sequence=['steelblue'],
                labels={'confidence': 'Confidence score [0→1]', 'count': 'Number of designs'},
            )
            fig.add_vline(x=0.5, line_dash='dash', line_color='red',
                          annotation_text='CFD refinement threshold (conf < 0.5)')
            st.plotly_chart(fig, use_container_width=True)

        if 'violations' in df_gr.columns:
            from collections import Counter
            all_viols = [v for lst in df_gr['violations'] for v in (lst or [])]
            if all_viols:
                vc = Counter(all_viols).most_common()
                df_vc = pd.DataFrame(vc, columns=['Violation reason code', 'Count'])
                fig_v = px.bar(df_vc, x='Count', y='Violation reason code', orientation='h',
                               color='Count', color_continuous_scale='Reds',
                               title='Most common physics guardrail violations across all designs')
                st.plotly_chart(fig_v, use_container_width=True)
            else:
                st.success('No violations across all designs — all within guardrail bounds.')

        st.subheader('All designs — confidence & pass/fail')
        show = [c for c in ['name', 'passed', 'confidence', 'n_violations'] if c in df_gr.columns]
        st.dataframe(
            df_gr[show].sort_values('confidence', ascending=False).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

# ── Tab 4: Pareto Designs ─────────────────────────────────────────────────
with tab_geo:
    designs, gl_meta = load_geom_loop()

    if not designs:
        st.warning(
            'Run **09_geometry_loop.ipynb** first to generate '
            '`geometry_loop_results.json`, then refresh this page.'
        )

    if designs:
        # ── Summary row ────────────────────────────────────────────────────────
        w = gl_meta.get('score_weights', {'T_UI': 0.4, 'TMA_UI': 0.4, 'confidence': 0.2})
        m1, m2, m3 = st.columns(3)
        m1.metric('Ranked designs',            gl_meta.get('n_ranked', len(designs)))
        m2.metric('Flagged for CFD (OpenFOAM reactingFoam)', gl_meta.get('n_cfd_flagged', '?'))
        m3.metric(
            'Score weights',
            f"T_UI × {w.get('T_UI', 0.4)}  |  TMA_UI × {w.get('TMA_UI', 0.4)}  |  conf × {w.get('confidence', 0.2)}",
        )

        st.divider()

        # ── Design cards (3 per row) ───────────────────────────────────────────
        st.subheader('Design Cards')
        N_COLS = 3
        for row_start in range(0, len(designs), N_COLS):
            cols = st.columns(N_COLS)
            for ci, d in enumerate(designs[row_start: row_start + N_COLS]):
                with cols[ci]:
                    with st.container(border=True):
                        badge = '  `CFD`' if d.get('flag_for_cfd') else ''
                        st.markdown(f"**Rank #{d['rank']}**{badge}")
                        st.caption(
                            f"D = {d['D_mm']:.1f} mm  |  pitch/D = {d['pitch_over_D']:.1f}  |  "
                            f"Q = {d['Q_slm']:.1f} slm  |  n_holes = {d['n_holes']}  |  "
                            f"pattern = {d.get('pattern', 'hex')}"
                        )
                        r1c1, r1c2 = st.columns(2)
                        r2c1, r2c2 = st.columns(2)
                        r1c1.metric('Score',  f"{d['score']:.3f}",
                                    help='Composite optimisation score = 0.4×T_UI + 0.4×TMA_UI + 0.2×confidence. '
                                         'Range [0, 1]. Higher is better.')
                        r1c2.metric('T_UI',   f"{d['T_UI']:.3f}",
                                    help='Temperature Uniformity Index = 1 − CV(T) near wafer. '
                                         'CV = std/mean. Range (−∞, 1]. >0.95 = excellent uniformity.')
                        tma_ui = d['TMA_UI']
                        r2c1.metric('TMA_UI', f"{tma_ui:.3f}",
                                    help='TMA (trimethylaluminium precursor) Uniformity Index = 1 − CV(TMA). '
                                         'Negative means std > mean — precursor is highly non-uniform. '
                                         'Target: maximise toward 1.0.')
                        eu = d['Eu']
                        eu_str = f"{eu/1e3:.1f}k" if eu >= 1000 else f"{eu:.2f}"
                        r2c2.metric('Eu', eu_str,
                                    help='Euler number Eu = Δp/(½ρV²) — dimensionless pressure drop. '
                                         'Lower Eu = less pumping power needed. '
                                         'Very high Eu indicates flow restriction or near-choked conditions.')
                        conf = d.get('confidence_post', d.get('confidence_pre', float('nan')))
                        st.caption(
                            f"Re = {d['Re']:.1f}  ·  Reynolds number ρVD/μ (flow regime)  |  "
                            f"Da = {d['Da']:.2f}  ·  Damköhler k_rxn·L/V (reaction vs transport)  |  "
                            f"conf = {conf:.2f}"
                        )

        st.divider()

        # ── Side-by-side comparison ────────────────────────────────────────────
        st.subheader('Side-by-Side Field Comparison')
        st.caption(
            'Select any two designs to regenerate their PCGM (Physics-Constrained Geometric '
            'Morphogenesis) point clouds and run surrogate inference, then compare field slices.'
        )

        labels = [
            f"Rank {d['rank']} — D={d['D_mm']:.1f}mm  pitch/D={d['pitch_over_D']:.1f}  Q={d['Q_slm']:.1f}slm"
            for d in designs
        ]
        col_a, col_b, col_btn = st.columns([5, 5, 2])
        sel_a    = col_a.selectbox('Design A', labels, index=0, key='cmp_a')
        sel_b    = col_b.selectbox('Design B', labels, index=min(1, len(labels) - 1), key='cmp_b')
        cmp_btn  = col_btn.button('Compare', type='primary', use_container_width=True)

        if cmp_btn:
            da = designs[labels.index(sel_a)]
            db = designs[labels.index(sel_b)]

            with st.spinner(f'PCGM + surrogate — {sel_a} …'):
                coords_a, preds_a, gd_a, err_a, _ = run_inference_pcgm(
                    da['D_mm'], da['pitch_over_D'], da['Q_slm'], model, norm, cfg,
                )
            with st.spinner(f'PCGM + surrogate — {sel_b} …'):
                coords_b, preds_b, gd_b, err_b, _ = run_inference_pcgm(
                    db['D_mm'], db['pitch_over_D'], db['Q_slm'], model, norm, cfg,
                )

            if coords_a is None:
                st.error(f'Design A rejected by guardrail engine: {err_a}')
            elif coords_b is None:
                st.error(f'Design B rejected by guardrail engine: {err_b}')
            else:
                left, right = st.columns(2)
                field_defs = [
                    (4, 'T [K]'),
                    (5, 'TMA'),
                    (3, 'p [m\u00b2/s\u00b2]'),
                    (None, '|U| [m/s]'),
                ]

                with left:
                    st.markdown(f"#### {sel_a}")
                    for idx, title in field_defs:
                        f = np.linalg.norm(preds_a[:, :3], axis=1) if idx is None else preds_a[:, idx]
                        st.plotly_chart(scatter_slice(coords_a, f, title), use_container_width=True)

                with right:
                    st.markdown(f"#### {sel_b}")
                    for idx, title in field_defs:
                        f = np.linalg.norm(preds_b[:, :3], axis=1) if idx is None else preds_b[:, idx]
                        st.plotly_chart(scatter_slice(coords_b, f, title), use_container_width=True)

                # Metric delta table
                st.subheader('Metric comparison')
                metric_rows = []
                for key in ['score', 'T_UI', 'TMA_UI', 'Eu', 'Re', 'Da']:
                    va = da.get(key, float('nan'))
                    vb = db.get(key, float('nan'))
                    try:
                        delta = f'{va - vb:+.4g}'
                    except TypeError:
                        delta = '—'
                    metric_rows.append({
                        'Metric':              key,
                        f'A  (Rank {da["rank"]})': f'{va:.4g}',
                        f'B  (Rank {db["rank"]})': f'{vb:.4g}',
                        'Delta  (A − B)':      delta,
                    })
                st.dataframe(
                    pd.DataFrame(metric_rows),
                    use_container_width=True,
                    hide_index=True,
                )

        st.divider()

        # ── Design Viewer ──────────────────────────────────────────────────────
        st.subheader('Design Viewer')
        st.caption(
            'Full geometry + field visualisation for a single design. '
            'Dimension annotations, dimensionless numbers (Re, Da, Ma…), '
            'performance metrics and guardrail results are shown alongside the figure.'
        )

        dv_labels = [
            f"Rank {d['rank']} — D={d['D_mm']:.1f}mm  pitch/D={d['pitch_over_D']:.1f}  Q={d['Q_slm']:.1f}slm"
            for d in designs
        ]
        col_sel, col_mode, col_load = st.columns([4, 4, 1])
        dv_sel    = col_sel.selectbox('Select design', dv_labels, index=0, key='dv_sel')
        view_mode = col_mode.radio(
            'View mode', ['2D Engineering Drawing', '3D Fields'],
            horizontal=True, key='dv_mode',
        )
        dv_btn = col_load.button('Load', type='primary', use_container_width=True, key='dv_btn')

        if dv_btn:
            dd = designs[dv_labels.index(dv_sel)]
            with st.spinner(f'Generating PCGM point cloud + surrogate — {dv_sel} …'):
                coords_d, preds_d, gd_d, err_d, res_d = run_inference_pcgm(
                    dd['D_mm'], dd['pitch_over_D'], dd['Q_slm'], model, norm, cfg,
                )

            if coords_d is None:
                st.error(f'Guardrail engine rejected this design: {err_d}')
            else:
                geo       = res_d.geometry
                dim       = res_d.dim_nums
                nxy       = geo.nozzle_xy
                p_geo     = geo.params
                gr        = res_d.guardrail_result
                conf      = dd.get('confidence_post', dd.get('confidence_pre', float('nan')))
                gr_passed = gr.passed if gr is not None else None

                vis_col, ann_col = st.columns([3, 2])

                with vis_col:
                    if view_mode == '2D Engineering Drawing':
                        _fc, _ft = plot_2d_schematic(p_geo, nxy, dim, dd)
                        _dv_panel = st.radio('Panel', ['Cross-section', 'Top view (wafer)'],
                                             horizontal=True, key='dv_2d_panel')
                        if _dv_panel == 'Cross-section':
                            st.plotly_chart(_fc, use_container_width=True)
                        else:
                            st.plotly_chart(_ft, use_container_width=True)
                    else:
                        field_opts = ['T [K]', 'TMA', 'p [Pa]', '|U| [m/s]', 'Ux', 'Uy', 'Uz']
                        field_map  = {
                            'T [K]': 4, 'TMA': 5, 'p [Pa]': 3,
                            '|U| [m/s]': None, 'Ux': 0, 'Uy': 1, 'Uz': 2,
                        }
                        sel_field = st.selectbox('Field to display', field_opts, key='dv_field')
                        st.plotly_chart(
                            plot_3d_field(coords_d, preds_d, field_map[sel_field],
                                          sel_field, nxy),
                            use_container_width=True,
                        )

                with ann_col:
                    st.markdown('#### Design Specification')
                    st.dataframe(pd.DataFrame({
                        'Parameter': [
                            'D — nozzle diameter', 'pitch / D', 'Q — flow rate',
                            'n_holes', 'open area %', 'H_plenum', 't_face',
                            'standoff', 'D_plate', 'pattern',
                        ],
                        'Value': [
                            f"{dd['D_mm']:.2f} mm",
                            f"{dd['pitch_over_D']:.2f}",
                            f"{dd['Q_slm']:.1f} slm",
                            str(dd['n_holes']),
                            f"{dd.get('open_area_pct', float('nan')):.1f}%",
                            f"{p_geo.get('H_plenum', 0) * 1e3:.1f} mm",
                            f"{p_geo.get('t_face', 0) * 1e3:.1f} mm",
                            f"{p_geo.get('standoff', 0) * 1e3:.1f} mm",
                            f"{p_geo.get('D_plate', 0) * 1e3:.0f} mm",
                            dd.get('pattern', 'hex'),
                        ],
                    }), use_container_width=True, hide_index=True)

                    st.markdown('#### Dimensionless Numbers')
                    _DIM_META = {
                        'Re':   ('Reynolds (ρVD/μ)',       'Flow regime'),
                        'Da':   ('Damköhler (k_rxn·L/V)',  'Reaction vs transport'),
                        'Ma':   ('Mach (V/a)',              'Compressibility'),
                        'Eu':   ('Euler (Δp/½ρV²)',         'Pressure drop'),
                        'Pr':   ('Prandtl (cp·μ/k)',        'Heat BL scaling'),
                        'Sc':   ('Schmidt (μ/ρD_m)',        'Diffusion scaling'),
                        'Pe_h': ('Péclet heat (Re·Pr)',     'Advection vs diffusion'),
                        'Pe_m': ('Péclet mass (Re·Sc)',     'Advection vs diffusion'),
                    }
                    dim_rows = [
                        {'Symbol': sym,
                         'Formula / full name': meta[0],
                         'Value': f"{dim.get(sym, float('nan')):.4g}",
                         'Guards against': meta[1]}
                        for sym, meta in _DIM_META.items()
                    ]
                    st.dataframe(pd.DataFrame(dim_rows),
                                 use_container_width=True, hide_index=True)

                    st.markdown('#### Performance Metrics')
                    perf_rows = [
                        ('Score',      f"{dd['score']:.4f}",
                         '0.4×T_UI + 0.4×TMA_UI + 0.2×conf'),
                        ('T_UI',       f"{dd['T_UI']:.4f}",
                         'Temperature uniformity index  1−CV(T)'),
                        ('TMA_UI',     f"{dd['TMA_UI']:.4f}",
                         'TMA uniformity index  1−CV(TMA)'),
                        ('Eu',         f"{dd['Eu']:.4g}",
                         'Euler number  Δp/(½ρV²)'),
                        ('T_mean',     f"{dd.get('T_mean_K', float('nan')):.1f} K",
                         'Mean temperature near wafer'),
                        ('T_range',    f"{dd.get('T_range_K', float('nan')):.2f} K",
                         'Max − min temperature'),
                        ('TMA_max',    f"{dd.get('TMA_max', float('nan')):.3e}",
                         'Peak TMA mass fraction'),
                        ('Confidence', f"{conf:.4f}",
                         'Guardrail engine confidence [0→1]'),
                        ('Guardrail',
                         'PASS' if gr_passed is True else ('FAIL' if gr_passed is False else '—'),
                         ''),
                        ('CFD flag',
                         'Yes — queued for reactingFoam' if dd.get('flag_for_cfd') else 'No',
                         'OpenFOAM (reactingFoam) CFD validation'),
                    ]
                    st.dataframe(
                        pd.DataFrame(perf_rows, columns=['Metric', 'Value', 'Description']),
                        use_container_width=True, hide_index=True,
                    )

                    if gr is not None and not gr.passed and gr.violations:
                        st.markdown('#### Guardrail Violations')
                        for v in gr.violations:
                            st.warning(
                                f'**{v.symbol}** = {v.value:.4g}  '
                                f'(allowed [{v.lo:.4g}, {v.hi:.4g}])  {v.message}'
                            )
                    for flag in (gr.special_flags if gr is not None else []):
                        st.info(f'Physics flag: {flag}')
                    for rec in (gr.recommendations if gr is not None else []):
                        st.caption(f'→ {rec}')

        st.divider()

        # ── Full table ─────────────────────────────────────────────────────────
        with st.expander('All ranked designs (full table)'):
            want_cols = [
                'rank', 'score', 'D_mm', 'pitch_over_D', 'Q_slm', 'n_holes',
                'Re', 'Da', 'T_UI', 'TMA_UI', 'Eu',
                'confidence_post', 'flag_for_cfd',
            ]
            avail = [c for c in want_cols if c in designs[0]]
            st.dataframe(
                pd.DataFrame(designs)[avail],
                use_container_width=True,
                hide_index=True,
            )

# ── Tab 5: Track 2 — VICES ────────────────────────────────────────────────
with tab_t2:
    def load_track2_results():
        if OPT2_JSON.exists():
            return json.load(open(OPT2_JSON))
        return None

    t2 = load_track2_results()

    if t2 is None:
        st.info(
            'Track 2 optimizer results not found. '
            'Run `python3 optimization/track2_optimizer.py` to generate them.'
        )
    else:
        surr = t2.get('surrogate', {})
        st.markdown('## Track 2 — VICES Topology Optimizer')
        st.caption(
            'Voxel-Implicit Computational Engineering Synthesis (VICES): '
            'CSG boolean tree geometry synthesis via SDF + Marching Cubes, '
            'exploring 4 topology types across the Catalan-enumerated design space.'
        )

        # ── Key metrics ──────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Track 1 best TMA_UI', f"{t2.get('track1_best_tma_ui', 0):.3f}",
                  help='Best TMA uniformity index found by Track 1 hex-array optimizer.')
        m2.metric('Track 2 best TMA_UI', f"{t2.get('track2_best_tma_ui', 0):.3f}",
                  help='Best TMA uniformity index found by Track 2 topology optimizer.')
        imp = t2.get('improvement', 0)
        m3.metric('Improvement', f"{imp:+.3f}",
                  delta=f"{imp:+.3f}",
                  help='Track 2 best minus Track 1 best TMA_UI.')
        m4.metric('Surrogate OOF ρ', f"{surr.get('oof_rho', 0):.3f}",
                  help='Out-of-fold Spearman rank correlation of combined KPI surrogate '
                       '(trained on Track 1 + Track 2 cases).')

        st.divider()

        # ── Best design per topology type ─────────────────────────────────────
        st.subheader('Best Design per Topology Type')
        bpt = t2.get('best_per_type', {})
        type_descriptions = {
            'A_baffled':  'Internal baffle annulus redistributes flow before nozzle plate',
            'B_conical':  'Central cone deflects inlet jet radially for uniform distribution',
            'C_annular':  'Nozzles in concentric rings — different spatial distribution to hex',
            'D_twozone':  'Annular divider splits plenum into inner/outer flow zones',
        }
        type_colors = {
            'A_baffled': '#e74c3c', 'B_conical': '#3498db',
            'C_annular': '#2ecc71', 'D_twozone': '#f39c12',
        }

        cols = st.columns(4)
        for ci, (name, d) in enumerate(bpt.items()):
            with cols[ci]:
                with st.container(border=True):
                    color = type_colors.get(name, '#888')
                    st.markdown(
                        f'<div style="background:{color};color:white;'
                        f'padding:4px 8px;border-radius:4px;font-weight:bold;'
                        f'font-size:13px">{name}</div>', unsafe_allow_html=True)
                    st.caption(type_descriptions.get(name, ''))
                    st.metric('TMA_UI', f"{d.get('TMA_UI', 0):.3f}")
                    st.write(f"D = **{d.get('D_mm', 0):.1f} mm**")
                    st.write(f"Q = **{d.get('Q_slm', 0):.1f} slm**")
                    st.write(f"Re = **{d.get('Re', 0):.0f}**")
                    st.write(f"Nozzles = **{d.get('n_nozzles', 0)}**")

        st.divider()

        # ── Pareto comparison plot ────────────────────────────────────────────
        st.subheader('Pareto Front: Track 2 vs Track 1')
        fig_path = _LOCAL_BASE / 'checkpoints' / 'optimizer' / 'track2_pareto.png'
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)
        else:
            st.info('Pareto plot not found — re-run track2_optimizer.py.')

        # ── Pareto design table ───────────────────────────────────────────────
        st.subheader('Top Track 2 Pareto Designs')
        pareto = t2.get('track2_pareto', [])
        if pareto:
            rows = []
            for i, d in enumerate(pareto[:10]):
                rows.append({
                    'Rank':       i + 1,
                    'Type':       d.get('geom_name', '?'),
                    'TMA_UI':     round(d.get('TMA_UI', 0), 3),
                    'D [mm]':     round(d.get('D_mm', 0), 1),
                    'Q [slm]':    round(d.get('Q_slm', 0), 2),
                    'Re':         int(d.get('Re', 0)),
                    'n_nozzles':  int(d.get('n_nozzles', 0)),
                    'extra_param': round(d.get('extra_param', 0), 3),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()

        # ── CSG topology explanation ───────────────────────────────────────────
        with st.expander('About Track 2 VICES Geometry Synthesis'):
            st.markdown("""
**Track 2 — VICES (Voxel-Implicit Computational Engineering Synthesis)**

Geometry is synthesised using a CSG (Constructive Solid Geometry) pipeline:
1. **SDF primitives** — Cylinder, Box, Cone, Sphere defined as Signed Distance Fields
2. **Boolean operations** — Union ∪, Subtract −, Intersect ∩ applied to SDF primitives
3. **Marching Cubes** — SDF voxel grid → triangle mesh (15-case lookup table, 256 voxel configurations)
4. **Point cloud** — mesh surface sampled → 80,000 points → GNN surrogate inference

**Catalan number design space:** For n primitives, the number of distinct CSG tree topologies = C(n−1).
With 6 primitives: C(5) = 42 distinct topologies before varying any dimensions.

| Type | CSG Tree | Key feature |
|---|---|---|
| A — Baffled | Cylinder − BaffleAnnulus − Nozzles | Flow redistribution before nozzle plate |
| B — Conical | Cylinder − Cone − Nozzles | Radial jet deflection |
| C — Annular | Cylinder − RingNozzles | Concentric ring pattern |
| D — Two-zone | Cylinder − DividerRing − Nozzles | Split inner/outer flow paths |

**Why Track 2 outperforms Track 1:** Track 1 is constrained to hex nozzle topology —
it can only vary 5 continuous parameters within a fixed geometry family.
Track 2 searches across fundamentally different topologies, unlocking designs
that the parametric approach cannot represent.
            """)

        # ── Surrogate info ────────────────────────────────────────────────────
        st.caption(
            f"Combined KPI surrogate: {surr.get('n_cases','?')} cases · "
            f"{surr.get('in_dim','?')} features · "
            f"5-fold CV OOF R²={surr.get('oof_r2',0):.3f} ρ={surr.get('oof_rho',0):.3f}"
        )
