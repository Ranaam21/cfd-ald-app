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
DRIVE_BASE  = Path(os.environ.get('CFD_BASE', _DEFAULT_BASE))
CKPT        = DRIVE_BASE / 'checkpoints' / 'multihead' / 'multihead_final.pt'
OPT_JSON    = DRIVE_BASE / 'checkpoints' / 'optimizer' / 'optimizer_results.json'
GR_JSON     = DRIVE_BASE / 'checkpoints' / 'guardrail' / 'guardrail_results.json'
GL_JSON     = DRIVE_BASE / 'checkpoints' / 'geometry_loop' / 'geometry_loop_results.json'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RHO_N2      = 1.145
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
    m.eval()
    return m, norm, cfg



@st.cache_data
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

    x  = torch.from_numpy((xi - node_mean) / node_std).to(DEVICE)
    ei = torch.tensor(np.stack([src, dst]), dtype=torch.long).to(DEVICE)
    ea = torch.from_numpy(ef).to(DEVICE)

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
    Two-panel engineering figure:
    Left  — axisymmetric cross-section with dimension annotations
    Right — top-view nozzle hole pattern

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

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=['Cross-section (axisymmetric right half)', 'Top view — nozzle hole pattern'],
        horizontal_spacing=0.14,
    )

    # ── Cross-section ──────────────────────────────────────────────────────
    wafer_h     = max(3.0, so * 0.10)
    z_face_bot  = so
    z_face_top  = so + tf
    z_plen_top  = so + tf + H
    ann_x_ext   = Dp / 2 + Dp * 0.06  # x for dimension leader lines

    # Wafer
    fig.add_shape(type='rect', x0=0, x1=Dp / 2, y0=-wafer_h, y1=0,
                  fillcolor='#cccccc', line=dict(color='#555', width=1), row=1, col=1)
    fig.add_annotation(x=Dp / 4, y=-wafer_h / 2, text='Wafer',
                       showarrow=False, font=dict(size=10), row=1, col=1)

    # Standoff gap
    fig.add_shape(type='rect', x0=0, x1=Dp / 2, y0=0, y1=z_face_bot,
                  fillcolor='rgba(200,230,255,0.12)',
                  line=dict(color='#aaa', width=1, dash='dot'), row=1, col=1)

    # Faceplate body
    fig.add_shape(type='rect', x0=0, x1=Dp / 2, y0=z_face_bot, y1=z_face_top,
                  fillcolor='#4472C4', opacity=0.80,
                  line=dict(color='#1a3a7a', width=1), row=1, col=1)

    # Holes cut through faceplate
    for r in r_holes:
        x0 = max(0.0, r - r_h)
        x1 = min(Dp / 2, r + r_h)
        if x1 > x0:
            fig.add_shape(type='rect', x0=x0, x1=x1, y0=z_face_bot, y1=z_face_top,
                          fillcolor='white', line=dict(width=0), row=1, col=1)

    # Plenum
    fig.add_shape(type='rect', x0=0, x1=Dp / 2, y0=z_face_top, y1=z_plen_top,
                  fillcolor='rgba(135,206,235,0.35)',
                  line=dict(color='#2288aa', width=1), row=1, col=1)
    fig.add_annotation(x=Dp / 4, y=(z_face_top + z_plen_top) / 2,
                       text='Plenum (N₂ + TMA)', showarrow=False,
                       font=dict(size=10, color='#114466'), row=1, col=1)

    # Gas inlet arrow
    fig.add_annotation(x=Dp / 4, y=z_plen_top + 4, ax=0, ay=-20,
                       text='Gas inlet', showarrow=True,
                       arrowhead=2, arrowcolor='navy',
                       font=dict(size=10), row=1, col=1)

    # ── Dimension leaders ──────────────────────────────────────────────
    def _leader(y0, y1, label):
        for y in [y0, y1]:
            fig.add_shape(type='line', x0=Dp / 2, x1=ann_x_ext + Dp * 0.01,
                          y0=y, y1=y, line=dict(color='black', width=0.8), row=1, col=1)
        fig.add_shape(type='line', x0=ann_x_ext, x1=ann_x_ext,
                      y0=y0, y1=y1, line=dict(color='black', width=0.8), row=1, col=1)
        fig.add_annotation(x=ann_x_ext + Dp * 0.02, y=(y0 + y1) / 2,
                           text=label, showarrow=False,
                           font=dict(size=9), xanchor='left', row=1, col=1)

    _leader(z_face_bot, z_face_top, f't_face<br>{tf:.1f} mm')
    _leader(z_face_top, z_plen_top, f'H_plenum<br>{H:.1f} mm')
    _leader(0, z_face_bot,          f'standoff<br>{so:.1f} mm')

    # D_plate
    fig.add_annotation(x=Dp / 4, y=-wafer_h - 5,
                       text=f'D_plate = {Dp:.0f} mm', showarrow=False,
                       font=dict(size=10), row=1, col=1)

    # Label nozzle diameter on first hole
    if len(r_holes):
        fig.add_annotation(x=r_holes[0], y=z_face_bot - 3,
                           text=f'D = {D:.2f} mm', showarrow=True,
                           ay=0, ax=-28, font=dict(size=9), row=1, col=1)

    fig.update_xaxes(title_text='Radius [mm]',
                     range=[-4, Dp / 2 + Dp * 0.22],
                     showgrid=True, gridcolor='rgba(0,0,0,0.07)',
                     row=1, col=1)
    fig.update_yaxes(title_text='Height from wafer [mm]',
                     range=[-wafer_h - 9, z_plen_top + 14],
                     showgrid=True, gridcolor='rgba(0,0,0,0.07)',
                     row=1, col=1)

    # ── Top-view hole pattern ──────────────────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 300)
    fig.add_trace(go.Scatter(
        x=np.cos(theta) * Dp / 2, y=np.sin(theta) * Dp / 2,
        mode='lines', line=dict(color='#1a3a7a', width=2),
        name='Faceplate edge', showlegend=True,
    ), row=1, col=2)

    marker_px = max(3, min(14, int(D * 3.5)))
    fig.add_trace(go.Scatter(
        x=nozzle_xy[:, 0] * 1e3, y=nozzle_xy[:, 1] * 1e3,
        mode='markers',
        marker=dict(size=marker_px, color='#4472C4', opacity=0.70,
                    line=dict(width=0.5, color='#1a3a7a')),
        name=f"{len(nozzle_xy)} nozzles  D={D:.2f}mm",
        showlegend=True,
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers',
        marker=dict(size=7, color='red', symbol='cross-thin',
                    line=dict(width=2, color='red')),
        name='Centre', showlegend=False,
    ), row=1, col=2)

    # Pitch indicator between two adjacent holes
    if len(nozzle_xy) >= 2:
        p1, p2 = nozzle_xy[0] * 1e3, nozzle_xy[1] * 1e3
        pitch_mm = float(np.linalg.norm(p1 - p2))
        fig.add_trace(go.Scatter(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]],
            mode='lines+markers',
            line=dict(color='orange', width=1.5, dash='dash'),
            marker=dict(size=4, color='orange'),
            name=f'pitch = {pitch_mm:.2f}mm',
            showlegend=True,
        ), row=1, col=2)

    fig.update_xaxes(title_text='X [mm]', scaleanchor='y2', scaleratio=1,
                     showgrid=True, gridcolor='rgba(0,0,0,0.07)', row=1, col=2)
    fig.update_yaxes(title_text='Y [mm]',
                     showgrid=True, gridcolor='rgba(0,0,0,0.07)', row=1, col=2)

    fig.update_layout(
        height=540,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(x=0.52, y=0.02, bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='#ccc', borderwidth=1),
    )
    return fig


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
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=35, b=10))
    return fig


# ── Page layout ────────────────────────────────────────────────────────────
st.set_page_config(page_title='ALD Showerhead CFD', layout='wide', page_icon='CFD')
st.title('ALD Showerhead CFD Surrogate Dashboard')
st.caption(f'MultiHeadMGN  |  Device: {DEVICE}  |  BoTorch Pareto optimiser')

with st.sidebar:
    st.header('Design Parameters')
    D_mm    = st.slider('Nozzle diameter D [mm]', 1.0, 3.0,  2.0, 0.25)
    pitch_D = st.slider('Pitch / D',              3.0, 6.0,  4.0, 0.5)
    Q_slm   = st.slider('Flow rate Q [slm]',      1.0, 10.0, 3.0, 0.5)
    with st.expander('Geometry (advanced)'):
        H_plenum_mm = st.slider('Plenum height H [mm]',      10.0, 40.0, 20.0, 1.0)
        t_face_mm   = st.slider('Faceplate thickness t [mm]',  1.0,  6.0,  3.0, 0.5)
        standoff_mm = st.slider('Standoff gap [mm]',           5.0, 40.0, 20.0, 1.0)
    run_btn = st.button('Run Prediction', type='primary', use_container_width=True)
    st.divider()
    with st.expander('Guardrail Bounds'):
        re_max  = st.number_input('Re max',   value=5000.0,  step=500.0)
        da_min  = st.number_input('Da min',   value=0.0001,  format='%g')
        da_max  = st.number_input('Da max',   value=100.0,   step=10.0)
        ma_max  = st.number_input('Ma max',   value=0.3,     step=0.05)
        eu_max  = st.number_input('Eu max',   value=1.0e9,   format='%g')
        st.divider()
        pr_min  = st.number_input('Pr min',   value=0.5,     format='%g')
        pr_max  = st.number_input('Pr max',   value=100.0,   step=10.0)
        sc_min  = st.number_input('Sc min',   value=0.1,     format='%g')
        sc_max  = st.number_input('Sc max',   value=10.0,    step=1.0)
        peh_max = st.number_input('Pe_h max', value=1.0e5,   format='%g')
        pem_max = st.number_input('Pe_m max', value=1.0e5,   format='%g')
        st.caption('Nu / Bi / Sh checked only after CFD run (require heat-transfer coefficient h).')

model, norm, cfg = load_model()
opt_data = load_json(str(OPT_JSON))
gr_data  = load_json(str(GR_JSON))

tab_pred, tab_opt, tab_gr, tab_geo = st.tabs(
    ['Predictions', 'Optimizer / Pareto', 'Guardrail Report', 'Pareto Designs']
)

# ── Tab 1: Predictions ─────────────────────────────────────────────────────
with tab_pred:
    if run_btn:
        # Build guardrail bounds from sidebar inputs
        pred_bounds = GuardrailBounds(
            Re=(1.0,    re_max),
            Ma=(0.0,    ma_max),
            Da=(da_min, da_max),
            Eu=(0.5,    eu_max),
            Pr=(pr_min, pr_max),
            Sc=(sc_min, sc_max),
            Pe_h=(0.1,  peh_max),
            Pe_m=(0.1,  pem_max),
        )

        with st.spinner('Generating PCGM point cloud + running surrogate...'):
            coords, preds, gd, err, res = run_inference_pcgm(
                D_mm, pitch_D, Q_slm, model, norm, cfg,
                H_plenum=H_plenum_mm / 1000.0,
                t_face=t_face_mm   / 1000.0,
                standoff=standoff_mm / 1000.0,
                bounds=pred_bounds,
            )

        if coords is None:
            st.error(f'Guardrail engine rejected this design: {err}')
            st.stop()

        # ── Key metrics ────────────────────────────────────────────────────
        Q_m3s = float(gd['flow_rate_slm']) * 1.667e-5
        D_m   = float(gd['D'])
        n_h   = max(int(round(float(gd['n_holes']))), 1)
        V_noz = Q_m3s / (n_h * 3.14159 * (D_m / 2) ** 2)
        dp    = float(abs(preds[:, 3].max() - preds[:, 3].min()))
        Eu    = euler(max(dp, 1e-3), RHO_N2, max(V_noz, 1e-3))

        # Uniformity index: 1 − CV in near-wafer band
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

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric('T mean [K]',  f"{preds[:, 4].mean():.1f}")
        m2.metric('T range [K]', f"{preds[:, 4].max() - preds[:, 4].min():.2f}")
        m3.metric('TMA max',     f"{preds[:, 5].max():.3e}")
        m4.metric('Eu',          f'{Eu:.2e}')
        m5.metric('T_UI',        f'{T_UI:.3f}' if not np.isnan(T_UI) else 'n/a')
        m6.metric('TMA_UI',      f'{TMA_UI:.3f}' if not np.isnan(TMA_UI) else 'n/a')

        # ── Field slices ───────────────────────────────────────────────────
        st.subheader('Field slices — bottom 15% of plenum')
        c1, c2 = st.columns(2)
        c1.plotly_chart(scatter_slice(coords, preds[:, 4], 'T [K]'),       use_container_width=True)
        c2.plotly_chart(scatter_slice(coords, preds[:, 5], 'TMA'),          use_container_width=True)
        c3, c4 = st.columns(2)
        c3.plotly_chart(scatter_slice(coords, preds[:, 3], 'p [m²/s²]'),   use_container_width=True)
        U_mag = np.linalg.norm(preds[:, :3], axis=1)
        c4.plotly_chart(scatter_slice(coords, U_mag, '|U| [m/s]'),          use_container_width=True)

        # ── Geometry + field viewer ────────────────────────────────────────
        st.subheader('Geometry + Field Viewer')
        vis_col, ann_col = st.columns([3, 2])

        with vis_col:
            view_mode = st.radio(
                'View', ['2D Engineering Drawing', '3D Fields'],
                horizontal=True, key='pred_mode',
            )
            if view_mode == '2D Engineering Drawing':
                st.plotly_chart(
                    plot_2d_schematic(res.geometry.params, res.geometry.nozzle_xy,
                                      res.dim_nums, {}),
                    use_container_width=True,
                )
            else:
                field_opts = ['T [K]', 'TMA', 'p [Pa]', '|U| [m/s]', 'Ux', 'Uy', 'Uz']
                field_map  = {
                    'T [K]': 4, 'TMA': 5, 'p [Pa]': 3,
                    '|U| [m/s]': None, 'Ux': 0, 'Uy': 1, 'Uz': 2,
                }
                sel_field = st.selectbox('Field to display', field_opts, key='pred_field')
                st.plotly_chart(
                    plot_3d_field(coords, preds, field_map[sel_field], sel_field,
                                  res.geometry.nozzle_xy),
                    use_container_width=True,
                )

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
                    f"{D_mm:.2f} mm",
                    f"{pitch_D:.2f}",
                    f"{Q_slm:.1f} slm",
                    str(n_h),
                    f"{float(gd.get('open_area_frac', 0)) * 100:.1f}%",
                    f"{H_plenum_mm:.1f} mm",
                    f"{t_face_mm:.1f} mm",
                    f"{standoff_mm:.1f} mm",
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

        # ── Guardrail check ────────────────────────────────────────────────
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
    else:
        st.info('Set parameters in the sidebar and click **Run Prediction**.')

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

        c1, c2, c3 = st.columns(3)
        c1.metric('Passed (default bounds)', f'{n_pass} / {len(df_gr)}')
        c2.metric('Mean confidence',          f'{mean_conf:.3f}')
        c3.metric('Need CFD refinement',      str(n_refine))

        if 'confidence' in df_gr.columns:
            fig = px.histogram(
                df_gr, x='confidence', nbins=25,
                title='Guardrail confidence distribution',
                color_discrete_sequence=['steelblue'],
            )
            fig.add_vline(x=0.5, line_dash='dash', line_color='red',
                          annotation_text='CFD refinement threshold')
            st.plotly_chart(fig, use_container_width=True)

        if 'violations' in df_gr.columns:
            from collections import Counter
            all_viols = [v for lst in df_gr['violations'] for v in (lst or [])]
            if all_viols:
                vc = Counter(all_viols).most_common()
                df_vc = pd.DataFrame(vc, columns=['Reason code', 'Count'])
                fig_v = px.bar(df_vc, x='Count', y='Reason code', orientation='h',
                               color='Count', color_continuous_scale='Reds',
                               title='Violations per reason code')
                st.plotly_chart(fig_v, use_container_width=True)
            else:
                st.success('No violations across all 80 cases.')

        show = [c for c in ['name', 'passed', 'confidence', 'n_violations'] if c in df_gr.columns]
        st.dataframe(df_gr[show], use_container_width=True)

# ── Tab 4: Pareto Designs ─────────────────────────────────────────────────
with tab_geo:
    designs, gl_meta = load_geom_loop()

    if not designs:
        st.warning(
            'Run **09_geometry_loop.ipynb** first to generate '
            '`geometry_loop_results.json`, then refresh this page.'
        )
        st.stop()

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
                    r1c1.metric('Score',  f"{d['score']:.3f}")
                    r1c2.metric('T_UI',   f"{d['T_UI']:.3f}")
                    tma_ui = d['TMA_UI']
                    r2c1.metric('TMA_UI', f"{tma_ui:.3f}", help='Negative = non-uniform TMA (std > mean)')
                    eu = d['Eu']
                    eu_str = f"{eu/1e3:.1f}k" if eu >= 1000 else f"{eu:.2f}"
                    r2c2.metric('Eu', eu_str, help='Euler number Δp/(½ρV²)')
                    conf = d.get('confidence_post', d.get('confidence_pre', float('nan')))
                    st.caption(
                        f"Re = {d['Re']:.1f}  |  Da = {d['Da']:.2f}  |  conf = {conf:.2f}"
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
            nxy       = geo.nozzle_xy          # [N_holes, 2]  metres
            p_geo     = geo.params             # D, H_plenum, t_face, standoff, D_plate
            gr        = res_d.guardrail_result
            conf      = dd.get('confidence_post', dd.get('confidence_pre', float('nan')))
            gr_passed = gr.passed if gr is not None else None

            vis_col, ann_col = st.columns([3, 2])

            # ── Visualisation panel ────────────────────────────────────
            with vis_col:
                if view_mode == '2D Engineering Drawing':
                    st.plotly_chart(
                        plot_2d_schematic(p_geo, nxy, dim, dd),
                        use_container_width=True,
                    )
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

            # ── Annotation panel ───────────────────────────────────────
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
