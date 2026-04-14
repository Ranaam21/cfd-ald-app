import sys
import os
sys.path.insert(0, '/content/cfd-ald-app')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import streamlit as st
import numpy as np
import h5py
import json
import torch
import torch.nn as nn
from pathlib import Path
from scipy.spatial import cKDTree
import plotly.express as px
import pandas as pd

from physics.guardrails import GuardrailEngine, GuardrailBounds
from physics.calculator import euler

# ── Constants ──────────────────────────────────────────────────────────────
DRIVE_BASE  = Path('/content/drive/MyDrive/cfd-ald-app')
HDF5_DIR    = DRIVE_BASE / 'showerhead_openfoam'
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


@st.cache_data(show_spinner='Loading case catalogue...')
def load_cases():
    rows = []
    for h5 in sorted(HDF5_DIR.glob('case_*.h5')):
        with h5py.File(h5, 'r') as f:
            gf = f['inputs/global'][:]
        gd = dict(zip(GLOBAL_KEYS, gf))
        rows.append({
            'path':         str(h5),
            'name':         h5.stem,
            'D_mm':         float(gd['D']) * 1e3,
            'pitch_over_D': float(gd['pitch_over_D']),
            'Q_slm':        float(gd['flow_rate_slm']),
            'Re':           float(gd['Re']),
            'Da':           float(gd['Da']),
            'Ma':           float(gd['Ma']),
        })
    return rows


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


# ── Helpers ────────────────────────────────────────────────────────────────
def nearest_case(cases, D_mm, pitch_D, Q_slm):
    pts = np.array([[c['D_mm'], c['pitch_over_D'], c['Q_slm']] for c in cases])
    q   = np.array([D_mm, pitch_D, Q_slm])
    lo, hi = pts.min(0), pts.max(0)
    rng = np.clip(hi - lo, 1e-8, None)
    dists = np.linalg.norm((pts - lo) / rng - (q - lo) / rng, axis=1)
    return cases[int(np.argmin(dists))]


def run_inference(h5_path, model, norm, cfg):
    node_mean = np.array(norm['node_mean'], dtype=np.float32)
    node_std  = np.array(norm['node_std'],  dtype=np.float32)
    out_mean  = np.array(norm['out_mean'],  dtype=np.float32)
    out_std   = np.array(norm['out_std'],   dtype=np.float32)

    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:].astype(np.float32)
        nf     = f['inputs/node_features'][:].astype(np.float32)
        gf     = f['inputs/global'][:].astype(np.float32)

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

    return coords, preds, dict(zip(GLOBAL_KEYS, gf))


def run_inference_pcgm(d_mm, pitch_d, q_slm, model, norm, cfg):
    """Run surrogate inference on a PCGM (Physics-Constrained Geometric Morphogenesis)
    point cloud generated on-the-fly for the given design parameters.

    Returns (coords [N,3], preds [N,6], global_dict, error_str).
    On guardrail rejection returns (None, None, None, reason).
    """
    from geometry.pcgm import generate
    from geometry.grammar import NozzlePattern

    bounds = GuardrailBounds(Re=(0.1, 5000.0), Ma=(0.0, 0.3), Da=(1e-4, 200.0), Eu=(0.5, 1e9))
    result = generate(
        geo_params={'D': d_mm / 1000.0, 'pitch_over_D': pitch_d},
        process_params={'flow_rate_slm': q_slm},
        bounds=bounds,
        pattern=NozzlePattern.HEX,
        n_points=80_000,
    )
    if not result.accepted:
        return None, None, None, result.reason

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

    return coords, preds, dict(zip(GLOBAL_KEYS, gf)), ''


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
    run_btn = st.button('Run Prediction', type='primary', use_container_width=True)
    st.divider()
    with st.expander('Guardrail Bounds'):
        re_max = st.number_input('Re max', value=5000.0,  step=500.0)
        da_min = st.number_input('Da min', value=0.0001,  format='%g')
        da_max = st.number_input('Da max', value=100.0,   step=10.0)
        ma_max = st.number_input('Ma max', value=0.3,     step=0.05)
        eu_max = st.number_input('Eu max', value=1.0e9,   format='%g')

model, norm, cfg = load_model()
cases    = load_cases()
opt_data = load_json(str(OPT_JSON))
gr_data  = load_json(str(GR_JSON))

tab_pred, tab_opt, tab_gr, tab_geo = st.tabs(
    ['Predictions', 'Optimizer / Pareto', 'Guardrail Report', 'Pareto Designs']
)

# ── Tab 1: Predictions ─────────────────────────────────────────────────────
with tab_pred:
    if run_btn:
        case = nearest_case(cases, D_mm, pitch_D, Q_slm)
        st.info(
            f"Nearest: **{case['name']}** | "
            f"D={case['D_mm']:.1f}mm  pitch/D={case['pitch_over_D']:.1f}  "
            f"Q={case['Q_slm']:.1f}slm  Re={case['Re']:.0f}"
        )
        with st.spinner('Running surrogate (~30 s on GPU)...'):
            coords, preds, gd = run_inference(case['path'], model, norm, cfg)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric('T mean [K]',  f"{preds[:, 4].mean():.1f}")
        m2.metric('T range [K]', f"{preds[:, 4].max() - preds[:, 4].min():.2f}")
        m3.metric('TMA max',     f"{preds[:, 5].max():.3e}")

        Q_m3s = float(gd['flow_rate_slm']) * 1.667e-5
        D     = float(gd['D'])
        n_h   = max(int(round(float(gd['n_holes']))), 1)
        V_noz = Q_m3s / (n_h * 3.14159 * (D / 2) ** 2)
        dp    = float(abs(preds[:, 3].max() - preds[:, 3].min()))
        Eu    = euler(max(dp, 1e-3), RHO_N2, max(V_noz, 1e-3))
        m4.metric('Eu', f'{Eu:.2e}')

        with st.expander('Dimensionless numbers'):
            dim_keys = ['Re', 'Pr', 'Sc', 'Ma', 'Da', 'Pe_h', 'Pe_m']
            st.dataframe(
                pd.DataFrame({'Value': {k: float(gd[k]) for k in dim_keys}}).style.format('{:.4g}')
            )

        st.subheader('Field slices — bottom 15% of plenum')
        c1, c2 = st.columns(2)
        c1.plotly_chart(scatter_slice(coords, preds[:, 4], 'T [K]'),  use_container_width=True)
        c2.plotly_chart(scatter_slice(coords, preds[:, 5], 'TMA'),    use_container_width=True)
        c3, c4 = st.columns(2)
        c3.plotly_chart(scatter_slice(coords, preds[:, 3], 'p [m2/s2]'), use_container_width=True)
        U_mag = np.linalg.norm(preds[:, :3], axis=1)
        c4.plotly_chart(scatter_slice(coords, U_mag, '|U| [m/s]'),    use_container_width=True)

        st.subheader('Guardrail check')
        bounds = GuardrailBounds(
            Re=(1.0, re_max), Ma=(0.0, ma_max), Da=(da_min, da_max), Eu=(0.5, eu_max),
        )
        engine   = GuardrailEngine(bounds)
        dim_vals = {k: float(gd[k]) for k in ['Re', 'Pr', 'Sc', 'Ma', 'Da', 'Pe_h', 'Pe_m']}
        dim_vals['Eu'] = Eu
        result = engine.check(dim_vals)

        if result.passed:
            st.success(f'PASS  confidence {result.confidence:.3f}')
        else:
            st.error(f'FAIL  confidence {result.confidence:.3f}')
            for v in result.violations:
                st.warning(
                    f'**{v.symbol}** = {v.value:.4g}  '
                    f'(allowed [{v.lo:.4g}, {v.hi:.4g}])  {v.message}'
                )
        for flag in result.special_flags:
            st.info(f'Flag: {flag}')
        for rec in result.recommendations:
            st.caption(f'-> {rec}')
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
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric('Score',  f"{d['score']:.3f}")
                    c2.metric('T_UI',   f"{d['T_UI']:.3f}")
                    c3.metric('TMA_UI', f"{d['TMA_UI']:.3f}")
                    c4.metric('Eu',     f"{d['Eu']:.2f}")
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
            coords_a, preds_a, gd_a, err_a = run_inference_pcgm(
                da['D_mm'], da['pitch_over_D'], da['Q_slm'], model, norm, cfg,
            )
        with st.spinner(f'PCGM + surrogate — {sel_b} …'):
            coords_b, preds_b, gd_b, err_b = run_inference_pcgm(
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
