"""
optimization/track2_ga_optimizer.py

Genetic Algorithm (GA) optimizer for Track 2 VICES topology search.

Chromosome: [type_int, D_mm, Q_slm, H_plenum_mm, pitch_D, extra_param]
  type_int  : integer 1-4 (A=1 baffled, B=2 conical, C=3 annular, D=4 twozone)
  D_mm      : nozzle diameter [1.0, 3.0] mm
  Q_slm     : flow rate [0.5, 8.0] slm
  H_plenum_mm: plenum height [15.0, 30.0] mm
  pitch_D   : pitch/D ratio [3.0, 6.0]
  extra_param: type-specific float [0.0, 1.0] (normalised)

Fitness: TMA-UI predicted by the combined KPI surrogate ensemble.

GA steps per generation:
  1. Evaluate fitness of all individuals
  2. Tournament selection (k=3) to pick parents
  3. Uniform crossover on continuous genes (prob 0.6)
  4. Topology swap mutation (prob 0.15 per individual)
  5. Gaussian mutation on continuous genes (sigma=0.1, prob 0.3 per gene)
  6. Elitism: always keep top 2 individuals unchanged

Usage:
    python3 optimization/track2_ga_optimizer.py
"""

import sys, os, json, math, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HDF5_T1  = Path('data/processed/ald_hdf5')
HDF5_T2  = Path('data/processed/track2_hdf5')
META_T2  = Path('openfoam/track2_cases/cases')
OUT_PATH = Path('checkpoints/optimizer/track2_ga_results.json')
FIG_PATH = Path('checkpoints/optimizer/track2_ga_comparison.png')
DEVICE   = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# GA hyperparameters
N_POP       = 40     # population size
N_GEN       = 50     # generations
P_CROSS     = 0.6    # crossover probability
P_MUT_TYPE  = 0.15   # topology type mutation probability
P_MUT_CONT  = 0.3    # continuous gene mutation probability (per gene)
SIGMA_MUT   = 0.08   # Gaussian mutation standard deviation (normalised)
K_TOURN     = 3      # tournament size
N_ELITE     = 2      # elitism: keep top N unchanged
RNG_SEED    = 42

# Gene bounds [min, max] for continuous genes
BOUNDS = {
    'D_mm':        (1.0,  3.0),
    'Q_slm':       (0.5,  8.0),
    'H_plenum_mm': (15.0, 30.0),
    'pitch_D':     (3.0,  6.0),
    'extra_param': (0.0,  1.0),
}
TYPE_NAMES = {1:'A_baffled', 2:'B_conical', 3:'C_annular', 4:'D_twozone'}
RHO_N2 = 1.12; MU_N2 = 2.0e-5


# ── Feature extraction (same as track2_optimizer.py) ─────────────────────────
def compute_re(D_m, Q_slm, n_nozzles):
    Q = Q_slm * 1e-3 / 60.0
    A = max(n_nozzles, 1) * math.pi * (D_m / 2) ** 2
    V = Q / (A + 1e-12)
    return RHO_N2 * V * D_m / MU_N2

def extract_features(D_mm, Q_slm, n_nozzles, H_plenum_mm, pitch_D,
                     geom_type_int, extra_param):
    D_m = D_mm / 1000.0
    Re  = compute_re(D_m, Q_slm, n_nozzles)
    oh  = [0.0] * 5; oh[geom_type_int] = 1.0
    return [D_mm, Q_slm, n_nozzles, H_plenum_mm, pitch_D, Re, extra_param] + oh

def n_holes_estimate(D_mm, pitch_D):
    from geometry.parametric import build_showerhead
    from geometry.grammar import NozzlePattern
    try:
        geo = build_showerhead(
            {'D': D_mm/1000, 'pitch_over_D': pitch_D,
             'H_plenum': 0.020, 't_face': 0.003, 'standoff': 0.020},
            pattern=NozzlePattern.HEX)
        return max(1, geo.n_holes)
    except Exception:
        return max(1, int(math.pi * (0.075 / (D_mm/1000 * pitch_D))**2))

def extra_to_physical(type_int, extra_norm):
    """Convert normalised extra_param [0,1] to physical range per type."""
    ranges = {1:(0.2,0.8), 2:(0.2,0.6), 3:(2.0,5.0), 4:(0.3,0.65)}
    lo, hi = ranges[type_int]
    return lo + extra_norm * (hi - lo)


# ── KPI surrogate (same architecture as track2_optimizer.py) ─────────────────
import h5py

def load_all_cases():
    X, Y = [], []
    for f in sorted(HDF5_T1.glob('case_*.h5')):
        with h5py.File(f) as h:
            gf = h['inputs/global'][:]; ui = float(h['uniformity'].attrs['uniformity_index'])
        D_mm = float(gf[7])*1000; pitch_D = float(gf[8])
        H_mm = float(gf[9])*1000
        Q_slm = max(0.1, float(gf[0]) * MU_N2 / (RHO_N2 * max(float(gf[7]),1e-4)))
        n_h = max(1, int(round(float(gf[12]))))
        X.append(extract_features(D_mm, Q_slm, n_h, H_mm, pitch_D, 0, 0.0)); Y.append(ui)
    for f in sorted(HDF5_T2.glob('track2_*.h5')):
        with h5py.File(f) as h:
            ui = float(h['uniformity'].attrs['uniformity_index'])
        meta_path = META_T2 / f.stem / 'case_meta.json'
        if not meta_path.exists(): continue
        meta = json.load(open(meta_path)); geo = meta.get('geometry', {})
        gt = {'A':1,'B':2,'C':3,'D':4}.get(geo.get('type','A')[0], 1)
        D_mm = float(geo.get('D_mm',2.0)); Q_slm = float(meta.get('process',{}).get('flow_rate_slm',3.0))
        n_h = int(geo.get('n_nozzles',19)); H_mm = float(geo.get('H_plenum_mm',20.0))
        pitch_D = float(geo.get('pitch_D',4.0))
        raw_extra = {1:geo.get('baffle_frac',0.5),2:geo.get('cone_r_frac',0.35),
                     3:geo.get('n_rings',3),4:geo.get('divider_r_frac',0.45)}.get(gt,0.5)
        lo,hi = {1:(0.2,0.8),2:(0.2,0.6),3:(2.0,5.0),4:(0.3,0.65)}[gt]
        extra_norm = (float(raw_extra) - lo) / (hi - lo + 1e-8)
        X.append(extract_features(D_mm, Q_slm, n_h, H_mm, pitch_D, gt, extra_norm)); Y.append(ui)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

class KpiMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, n_layers=3):
        super().__init__()
        dims = [in_dim]+[hidden]*n_layers+[1]; layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2: layers += [nn.SiLU(), nn.LayerNorm(dims[i+1])]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

def train_fold(X_tr, Y_tr, X_va, Y_va, in_dim, epochs=2000):
    sc = StandardScaler().fit(X_tr)
    xt = torch.from_numpy(sc.transform(X_tr)).to(DEVICE)
    yt = torch.from_numpy(Y_tr).to(DEVICE)
    xv = torch.from_numpy(sc.transform(X_va)).to(DEVICE)
    yv = torch.from_numpy(Y_va).to(DEVICE)
    m = KpiMLP(in_dim).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_state = 1e9, None
    for ep in range(epochs):
        m.train(); loss = nn.functional.mse_loss(m(xt), yt)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad(): vl = nn.functional.mse_loss(m(xv), yv).item()
        if vl < best_val: best_val = vl; best_state = {k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(best_state); return m, sc


# ── GA implementation ─────────────────────────────────────────────────────────

def random_individual(rng):
    """Sample a random chromosome."""
    return {
        'type':   rng.integers(1, 5),        # 1-4
        'D_mm':   rng.uniform(1.0, 3.0),
        'Q_slm':  rng.uniform(0.5, 8.0),
        'H_mm':   rng.uniform(15.0, 30.0),
        'pitch_D':rng.uniform(3.0, 6.0),
        'extra':  rng.uniform(0.0, 1.0),      # normalised [0,1]
    }

def clip(ind):
    """Clip continuous genes to valid ranges."""
    ind['D_mm']    = np.clip(ind['D_mm'],    1.0,  3.0)
    ind['Q_slm']   = np.clip(ind['Q_slm'],   0.5,  8.0)
    ind['H_mm']    = np.clip(ind['H_mm'],    15.0, 30.0)
    ind['pitch_D'] = np.clip(ind['pitch_D'], 3.0,  6.0)
    ind['extra']   = np.clip(ind['extra'],   0.0,  1.0)
    ind['type']    = int(np.clip(ind['type'], 1, 4))
    return ind

def individual_to_features(ind):
    n_h = n_holes_estimate(ind['D_mm'], ind['pitch_D'])
    extra_phys = extra_to_physical(ind['type'], ind['extra'])
    return extract_features(
        ind['D_mm'], ind['Q_slm'], n_h, ind['H_mm'],
        ind['pitch_D'], ind['type'], ind['extra']
    ), n_h, extra_phys

def evaluate_population(pop, models, scalers):
    """Predict TMA-UI for all individuals using ensemble."""
    feats = []
    for ind in pop:
        f, _, _ = individual_to_features(ind)
        feats.append(f)
    X = np.array(feats, dtype=np.float32)
    preds = np.zeros(len(X))
    for m, sc in zip(models, scalers):
        m.eval()
        with torch.no_grad():
            xp = torch.from_numpy(sc.transform(X)).to(DEVICE)
            preds += m(xp).cpu().numpy()
    return preds / len(models)

def tournament_select(pop, fitness, k, rng):
    """Tournament selection — return index of winner."""
    candidates = rng.choice(len(pop), k, replace=False)
    return candidates[np.argmax(fitness[candidates])]

def crossover(p1, p2, rng):
    """Uniform crossover on continuous genes; keep one parent's topology."""
    child = copy.deepcopy(p1)
    # Topology: inherit from the fitter parent (p1 already selected as fitter)
    for gene in ('D_mm', 'Q_slm', 'H_mm', 'pitch_D', 'extra'):
        if rng.random() < 0.5:
            child[gene] = p2[gene]
    return child

def mutate(ind, rng):
    """Mutate: topology swap + Gaussian on continuous genes."""
    ind = copy.deepcopy(ind)
    if rng.random() < P_MUT_TYPE:
        ind['type'] = rng.integers(1, 5)
    for gene in ('D_mm', 'Q_slm', 'H_mm', 'pitch_D', 'extra'):
        if rng.random() < P_MUT_CONT:
            ind[gene] += rng.normal(0, SIGMA_MUT * (BOUNDS.get(gene, (0,1))[1] - BOUNDS.get(gene, (0,1))[0]))
    return clip(ind)

def run_ga(models, scalers, rng):
    """Run the full genetic algorithm. Returns best designs + convergence history."""
    pop = [random_individual(rng) for _ in range(N_POP)]
    fitness = evaluate_population(pop, models, scalers)
    best_per_gen = []

    print(f'  Gen 00/{N_GEN}  best TMA-UI={fitness.max():.4f}  '
          f'mean={fitness.mean():.4f}')

    for gen in range(N_GEN):
        # Elitism: carry top N_ELITE unchanged
        elite_idx = np.argsort(fitness)[-N_ELITE:]
        new_pop = [copy.deepcopy(pop[i]) for i in elite_idx]

        # Fill rest via selection + crossover + mutation
        while len(new_pop) < N_POP:
            i1 = tournament_select(pop, fitness, K_TOURN, rng)
            i2 = tournament_select(pop, fitness, K_TOURN, rng)
            child = crossover(pop[i1], pop[i2], rng) if rng.random() < P_CROSS \
                    else copy.deepcopy(pop[i1])
            child = mutate(child, rng)
            new_pop.append(child)

        pop = new_pop
        fitness = evaluate_population(pop, models, scalers)
        best_per_gen.append(float(fitness.max()))

        if (gen + 1) % 10 == 0:
            best_type = TYPE_NAMES[pop[int(np.argmax(fitness))]['type']]
            print(f'  Gen {gen+1:02d}/{N_GEN}  best={fitness.max():.4f}  '
                  f'mean={fitness.mean():.4f}  best_type={best_type}')

    # Collect final population with metadata
    results = []
    for ind, fit in zip(pop, fitness):
        n_h = n_holes_estimate(ind['D_mm'], ind['pitch_D'])
        extra_phys = extra_to_physical(ind['type'], ind['extra'])
        Re = compute_re(ind['D_mm']/1000, ind['Q_slm'], n_h)
        results.append({
            'TMA_UI':   float(fit),
            'neg_Re':   -Re,
            'Re':       Re,
            'geom_type': ind['type'],
            'geom_name': TYPE_NAMES[ind['type']],
            'D_mm':     float(ind['D_mm']),
            'Q_slm':    float(ind['Q_slm']),
            'H_plenum_mm': float(ind['H_mm']),
            'pitch_D':  float(ind['pitch_D']),
            'extra_param': float(extra_phys),
            'n_nozzles': n_h,
        })

    return results, best_per_gen


def pareto_front(designs):
    front = []
    for d in designs:
        dominated = False
        for b in designs:
            if (b['TMA_UI'] >= d['TMA_UI'] and b['neg_Re'] >= d['neg_Re'] and
                    (b['TMA_UI'] > d['TMA_UI'] or b['neg_Re'] > d['neg_Re'])):
                dominated = True; break
        if not dominated:
            front.append(d)
    return sorted(front, key=lambda x: -x['TMA_UI'])


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('Loading cases...')
    X, Y = load_all_cases()
    in_dim = X.shape[1]
    print(f'  {len(Y)} cases  features={in_dim}  '
          f'TMA-UI mean={Y.mean():.3f} std={Y.std():.3f}')

    print('\nTraining KPI surrogate (5-fold)...')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models, scalers, oof_pred = [], [], np.zeros(len(Y))
    for fold, (tr, va) in enumerate(kf.split(X)):
        m, sc = train_fold(X[tr], Y[tr], X[va], Y[va], in_dim)
        m.eval()
        with torch.no_grad():
            oof_pred[va] = m(torch.from_numpy(sc.transform(X[va])).to(DEVICE)).cpu().numpy()
        models.append(m); scalers.append(sc)
        rho, _ = spearmanr(Y[va], oof_pred[va])
        print(f'  Fold {fold+1}  ρ={rho:.3f}')

    rho_oof, _ = spearmanr(Y, oof_pred)
    print(f'  OOF ρ = {rho_oof:.3f}')

    # ── Random search baseline (same as track2_optimizer.py) ─────────────────
    print('\nRunning random search baseline (1,200 candidates)...')
    rng_rand = np.random.default_rng(42)
    rand_pop = [random_individual(rng_rand) for _ in range(1200)]
    rand_fit = evaluate_population(rand_pop, models, scalers)
    rand_results = []
    for ind, fit in zip(rand_pop, rand_fit):
        n_h = n_holes_estimate(ind['D_mm'], ind['pitch_D'])
        Re  = compute_re(ind['D_mm']/1000, ind['Q_slm'], n_h)
        rand_results.append({
            'TMA_UI': float(fit), 'neg_Re': -Re, 'Re': Re,
            'geom_name': TYPE_NAMES[ind['type']],
            'D_mm': float(ind['D_mm']), 'Q_slm': float(ind['Q_slm']),
            'n_nozzles': n_h,
        })
    best_rand = max(rand_results, key=lambda x: x['TMA_UI'])
    print(f'  Random search best TMA-UI = {best_rand["TMA_UI"]:.4f}  '
          f'({best_rand["geom_name"]})')

    # ── Genetic algorithm ──────────────────────────────────────────────────────
    print(f'\nRunning GA ({N_POP} pop × {N_GEN} gen = '
          f'{N_POP*N_GEN} evaluations)...')
    rng_ga = np.random.default_rng(42)
    ga_results, convergence = run_ga(models, scalers, rng_ga)
    best_ga = max(ga_results, key=lambda x: x['TMA_UI'])
    print(f'\n  GA best TMA-UI = {best_ga["TMA_UI"]:.4f}  '
          f'({best_ga["geom_name"]}  D={best_ga["D_mm"]:.1f}mm  '
          f'Q={best_ga["Q_slm"]:.1f}slm  Re={best_ga["Re"]:.0f})')

    # ── Pareto fronts ─────────────────────────────────────────────────────────
    ga_pareto   = pareto_front(ga_results)
    rand_pareto = pareto_front(rand_results)

    # ── Improvement summary ───────────────────────────────────────────────────
    improvement = best_ga['TMA_UI'] - best_rand['TMA_UI']
    print(f'\n  GA improvement over random search: {improvement:+.4f} '
          f'({improvement/best_rand["TMA_UI"]*100:+.1f}%)')

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        'ga_best_tma_ui':     best_ga['TMA_UI'],
        'rand_best_tma_ui':   best_rand['TMA_UI'],
        'improvement_abs':    improvement,
        'improvement_pct':    improvement / max(best_rand['TMA_UI'], 1e-6) * 100,
        'oof_rho':            float(rho_oof),
        'n_pop': N_POP, 'n_gen': N_GEN,
        'n_rand': 1200,
        'convergence':        convergence,
        'ga_pareto':  [{k:v for k,v in d.items()} for d in ga_pareto[:15]],
        'rand_pareto':[{k:v for k,v in d.items()} for d in rand_pareto[:15]],
        'ga_best':   best_ga,
        'rand_best': best_rand,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults saved → {OUT_PATH}')

    # ── Comparison plot ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Convergence curve
    ax = axes[0]
    ax.plot(range(1, N_GEN+1), convergence, 'b-o', ms=3, lw=1.5,
            label='GA best per generation')
    ax.axhline(best_rand['TMA_UI'], color='r', linestyle='--', lw=1.5,
               label=f'Random search best ({best_rand["TMA_UI"]:.3f})')
    ax.set_xlabel('Generation'); ax.set_ylabel('Best TMA-UI')
    ax.set_title('GA Convergence')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2: Pareto front comparison
    ax = axes[1]
    colors = {'A_baffled':'#e74c3c','B_conical':'#3498db',
              'C_annular':'#2ecc71','D_twozone':'#f39c12'}
    for d in ga_results:
        ax.scatter(-d['neg_Re'], d['TMA_UI'], s=6, alpha=0.25,
                   color=colors.get(d['geom_name'],'grey'))
    gp_s = sorted(ga_pareto, key=lambda x: x['neg_Re'])
    rp_s = sorted(rand_pareto, key=lambda x: x['neg_Re'])
    ax.plot([-d['neg_Re'] for d in gp_s], [d['TMA_UI'] for d in gp_s],
            'b-o', ms=5, lw=2, label='GA Pareto', zorder=5)
    ax.plot([-d['neg_Re'] for d in rp_s], [d['TMA_UI'] for d in rp_s],
            'r--s', ms=5, lw=2, label='Random Pareto', zorder=5)
    ax.set_xlabel('Reynolds number Re'); ax.set_ylabel('TMA-UI (predicted)')
    ax.set_title('Pareto Front: GA vs Random Search')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 3: Best design per type comparison
    ax = axes[2]
    type_names_short = ['A\nBaffled','B\nConical','C\nAnnular','D\nTwozone']
    types = [1, 2, 3, 4]
    ga_best_per   = [max((d for d in ga_results if d['geom_type']==t),
                         key=lambda x: x['TMA_UI'], default={'TMA_UI':0})['TMA_UI']
                     for t in types]
    rand_best_per = [max((d for d in rand_results if d['geom_name']==TYPE_NAMES[t]),
                         key=lambda x: x['TMA_UI'], default={'TMA_UI':0})['TMA_UI']
                     for t in types]
    x = np.arange(4)
    ax.bar(x - 0.2, rand_best_per, 0.35, label='Random Search',
           color='#E8DAEF', edgecolor='black', lw=0.8)
    ax.bar(x + 0.2, ga_best_per,   0.35, label='Genetic Algorithm',
           color='#2874A6', edgecolor='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(type_names_short)
    ax.set_ylabel('Best TMA-UI'); ax.set_title('Best Per Topology Type')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    for i, (rv, gv) in enumerate(zip(rand_best_per, ga_best_per)):
        if gv > rv:
            ax.text(i+0.2, gv+0.005, f'+{gv-rv:.3f}', ha='center',
                    fontsize=7, color='#1A5276', fontweight='bold')

    plt.suptitle(
        f'Track 2 GA vs Random Search  ·  '
        f'GA best: {best_ga["TMA_UI"]:.3f}  ·  '
        f'Random best: {best_rand["TMA_UI"]:.3f}  ·  '
        f'Improvement: {improvement:+.3f}',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(str(FIG_PATH), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved → {FIG_PATH}')

    print('\n' + '='*60)
    print('GA OPTIMIZER SUMMARY')
    print('='*60)
    print(f'Random search best TMA-UI : {best_rand["TMA_UI"]:.4f}')
    print(f'GA best TMA-UI            : {best_ga["TMA_UI"]:.4f}')
    print(f'Improvement               : {improvement:+.4f}  '
          f'({improvement/max(best_rand["TMA_UI"],1e-6)*100:+.1f}%)')
    print(f'GA Pareto designs         : {len(ga_pareto)}')
    print()
    print('GA best design:')
    for k in ('geom_name','D_mm','Q_slm','H_plenum_mm','pitch_D',
              'extra_param','n_nozzles','Re','TMA_UI'):
        print(f'  {k:<15} = {best_ga.get(k, "?")}')


if __name__ == '__main__':
    main()
