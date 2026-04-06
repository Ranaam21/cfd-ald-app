"""
physics/calculator.py

Computes the 11 dimensionless numbers used throughout the CFD-ALD surrogate app.

All functions accept plain floats or numpy arrays.  Every quantity is in SI units
unless stated otherwise.  Each function returns a float / ndarray and raises
ValueError when a physically meaningless input is supplied.

Dimensionless numbers implemented
──────────────────────────────────
Momentum / flow
  1.  Re  — Reynolds number
  2.  Ma  — Mach number
  3.  Eu  — Euler number  (pressure-drop scaling)

Heat transfer
  4.  Pr  — Prandtl number
  5.  Nu  — Nusselt number
  6.  Bi  — Biot number

Mass transfer + ALD kinetics
  7.  Sc  — Schmidt number
  8.  Sh  — Sherwood number
  9.  Pe_h — Péclet number (heat)
  10. Pe_m — Péclet number (mass)
  11. Da  — Damköhler number
"""

import numpy as np


# ── helpers ────────────────────────────────────────────────────────────────

def _pos(val, name: str):
    """Raise if val <= 0."""
    if np.any(np.asarray(val) <= 0):
        raise ValueError(f"{name} must be > 0, got {val}")


# ══════════════════════════════════════════════════════════════════════════
# 1. Reynolds number
# ══════════════════════════════════════════════════════════════════════════

def reynolds(rho: float, V: float, L: float, mu: float) -> float:
    """
    Re = ρ·V·L / μ

    Parameters
    ----------
    rho : fluid density            [kg/m³]
    V   : characteristic velocity  [m/s]
    L   : characteristic length    [m]  (nozzle diameter, hydraulic diameter, …)
    mu  : dynamic viscosity        [Pa·s]

    Returns
    -------
    Re  : dimensionless Reynolds number
    """
    _pos(rho, "rho"); _pos(V, "V"); _pos(L, "L"); _pos(mu, "mu")
    return float(rho * V * L / mu)


# ══════════════════════════════════════════════════════════════════════════
# 2. Mach number
# ══════════════════════════════════════════════════════════════════════════

def mach(V: float, a: float) -> float:
    """
    Ma = V / a

    Parameters
    ----------
    V : flow velocity      [m/s]
    a : speed of sound     [m/s]

    Returns
    -------
    Ma : dimensionless Mach number

    Notes
    -----
    Ma > 0.3  → compressibility effects become non-negligible;
    the guardrail engine will flag this and recommend switching
    to a compressible flow model branch.
    """
    _pos(V, "V"); _pos(a, "a")
    return float(V / a)


# ══════════════════════════════════════════════════════════════════════════
# 3. Euler number
# ══════════════════════════════════════════════════════════════════════════

def euler(delta_p: float, rho: float, V: float) -> float:
    """
    Eu = Δp / (½·ρ·V²)

    Parameters
    ----------
    delta_p : pressure drop across the distributor  [Pa]
    rho     : fluid density                          [kg/m³]
    V       : characteristic velocity                [m/s]

    Returns
    -------
    Eu : dimensionless Euler number (pressure-drop scaling)
    """
    _pos(rho, "rho"); _pos(V, "V")
    if np.any(np.asarray(delta_p) < 0):
        raise ValueError("delta_p must be >= 0")
    return float(delta_p / (0.5 * rho * V ** 2))


# ══════════════════════════════════════════════════════════════════════════
# 4. Prandtl number
# ══════════════════════════════════════════════════════════════════════════

def prandtl(cp: float, mu: float, k_fluid: float) -> float:
    """
    Pr = cp·μ / k

    Parameters
    ----------
    cp      : specific heat capacity  [J/(kg·K)]
    mu      : dynamic viscosity       [Pa·s]
    k_fluid : thermal conductivity    [W/(m·K)]

    Returns
    -------
    Pr : dimensionless Prandtl number
    """
    _pos(cp, "cp"); _pos(mu, "mu"); _pos(k_fluid, "k_fluid")
    return float(cp * mu / k_fluid)


# ══════════════════════════════════════════════════════════════════════════
# 5. Nusselt number
# ══════════════════════════════════════════════════════════════════════════

def nusselt(h: float, L: float, k_fluid: float) -> float:
    """
    Nu = h·L / k

    Parameters
    ----------
    h       : convective heat transfer coefficient  [W/(m²·K)]
    L       : characteristic length                  [m]
    k_fluid : fluid thermal conductivity             [W/(m·K)]

    Returns
    -------
    Nu : dimensionless Nusselt number
    """
    _pos(h, "h"); _pos(L, "L"); _pos(k_fluid, "k_fluid")
    return float(h * L / k_fluid)


# ══════════════════════════════════════════════════════════════════════════
# 6. Biot number
# ══════════════════════════════════════════════════════════════════════════

def biot(h: float, L: float, k_solid: float) -> float:
    """
    Bi = h·L / k_s

    Parameters
    ----------
    h       : convective heat transfer coefficient  [W/(m²·K)]
    L       : characteristic solid thickness         [m]  (faceplate thickness)
    k_solid : solid thermal conductivity             [W/(m·K)]

    Returns
    -------
    Bi : dimensionless Biot number

    Notes
    -----
    Bi < 0.1  → lumped-capacitance assumption valid (faceplate is thermally thin).
    Bi > 0.1  → internal temperature gradients in faceplate matter; flag in UI.
    """
    _pos(h, "h"); _pos(L, "L"); _pos(k_solid, "k_solid")
    return float(h * L / k_solid)


# ══════════════════════════════════════════════════════════════════════════
# 7. Schmidt number
# ══════════════════════════════════════════════════════════════════════════

def schmidt(mu: float, rho: float, D_m: float) -> float:
    """
    Sc = μ / (ρ·D_m)

    Parameters
    ----------
    mu  : dynamic viscosity          [Pa·s]
    rho : fluid density              [kg/m³]
    D_m : mass diffusivity           [m²/s]

    Returns
    -------
    Sc : dimensionless Schmidt number
    """
    _pos(mu, "mu"); _pos(rho, "rho"); _pos(D_m, "D_m")
    return float(mu / (rho * D_m))


# ══════════════════════════════════════════════════════════════════════════
# 8. Sherwood number
# ══════════════════════════════════════════════════════════════════════════

def sherwood(k_m: float, L: float, D_m: float) -> float:
    """
    Sh = k_m·L / D_m

    Parameters
    ----------
    k_m : mass transfer coefficient  [m/s]
    L   : characteristic length      [m]
    D_m : mass diffusivity           [m²/s]

    Returns
    -------
    Sh : dimensionless Sherwood number
    """
    _pos(k_m, "k_m"); _pos(L, "L"); _pos(D_m, "D_m")
    return float(k_m * L / D_m)


# ══════════════════════════════════════════════════════════════════════════
# 9. Péclet number — heat
# ══════════════════════════════════════════════════════════════════════════

def peclet_heat(Re: float, Pr: float) -> float:
    """
    Pe_h = Re · Pr

    Parameters
    ----------
    Re : Reynolds number   [-]
    Pr : Prandtl number    [-]

    Returns
    -------
    Pe_h : thermal Péclet number (advection vs. thermal diffusion)
    """
    _pos(Re, "Re"); _pos(Pr, "Pr")
    return float(Re * Pr)


# ══════════════════════════════════════════════════════════════════════════
# 10. Péclet number — mass
# ══════════════════════════════════════════════════════════════════════════

def peclet_mass(Re: float, Sc: float) -> float:
    """
    Pe_m = Re · Sc

    Parameters
    ----------
    Re : Reynolds number   [-]
    Sc : Schmidt number    [-]

    Returns
    -------
    Pe_m : mass Péclet number (advection vs. mass diffusion)
    """
    _pos(Re, "Re"); _pos(Sc, "Sc")
    return float(Re * Sc)


# ══════════════════════════════════════════════════════════════════════════
# 11. Damköhler number
# ══════════════════════════════════════════════════════════════════════════

def damkohler(k_rxn: float, L: float, V: float) -> float:
    """
    Da = k_rxn · L / V

    Parameters
    ----------
    k_rxn : first-order surface reaction rate constant  [m/s]
            (= sticking_coeff × thermal_velocity / 4   for ALD)
    L     : characteristic length                        [m]
    V     : characteristic velocity                      [m/s]

    Returns
    -------
    Da : dimensionless Damköhler number

    Notes
    -----
    Da ≪ 1  → transport-limited regime (precursor delivered faster than it reacts).
    Da ≫ 1  → reaction-limited regime  (precursor reacts before it spreads uniformly).
    ALD ideal operation targets Da ~ O(1) or slightly transport-limited near wafer.
    """
    _pos(k_rxn, "k_rxn"); _pos(L, "L"); _pos(V, "V")
    return float(k_rxn * L / V)


# ══════════════════════════════════════════════════════════════════════════
# Convenience: compute all numbers from a single fluid/geometry dict
# ══════════════════════════════════════════════════════════════════════════

def compute_all(params: dict) -> dict:
    """
    Compute all applicable dimensionless numbers from a flat parameter dict.

    Required keys
    -------------
    rho      [kg/m³]   fluid density
    V        [m/s]     characteristic velocity (mean nozzle exit velocity)
    L        [m]       characteristic length   (nozzle diameter D)
    mu       [Pa·s]    dynamic viscosity
    cp       [J/kg·K]  specific heat
    k_fluid  [W/m·K]   fluid thermal conductivity
    D_m      [m²/s]    precursor mass diffusivity

    Optional keys (computed only when present)
    ------------------------------------------
    a        [m/s]     speed of sound          → Ma
    delta_p  [Pa]      pressure drop           → Eu
    h        [W/m²K]   heat transfer coeff     → Nu, Bi
    k_solid  [W/m·K]   solid (faceplate) k     → Bi
    k_m      [m/s]     mass transfer coeff     → Sh
    k_rxn    [m/s]     surface reaction rate   → Da

    Returns
    -------
    dict of {symbol: value}
    """
    p = params
    out = {}

    # Always computable
    out["Re"]   = reynolds(p["rho"], p["V"], p["L"], p["mu"])
    out["Pr"]   = prandtl(p["cp"], p["mu"], p["k_fluid"])
    out["Sc"]   = schmidt(p["mu"], p["rho"], p["D_m"])
    out["Pe_h"] = peclet_heat(out["Re"], out["Pr"])
    out["Pe_m"] = peclet_mass(out["Re"], out["Sc"])

    # Optional
    if "a" in p:
        out["Ma"] = mach(p["V"], p["a"])
    if "delta_p" in p:
        out["Eu"] = euler(p["delta_p"], p["rho"], p["V"])
    if "h" in p:
        out["Nu"] = nusselt(p["h"], p["L"], p["k_fluid"])
        if "k_solid" in p:
            out["Bi"] = biot(p["h"], p["L"], p["k_solid"])
    if "k_m" in p:
        out["Sh"] = sherwood(p["k_m"], p["L"], p["D_m"])
    if "k_rxn" in p:
        out["Da"] = damkohler(p["k_rxn"], p["L"], p["V"])

    return out


# ══════════════════════════════════════════════════════════════════════════
# ALD-specific helper: k_rxn from sticking coefficient + thermal velocity
# ══════════════════════════════════════════════════════════════════════════

def k_rxn_from_sticking(beta: float, v_th: float) -> float:
    """
    First-order surface rate constant from kinetic-theory sticking model.

        k_rxn = beta · v_th / 4

    Parameters
    ----------
    beta : sticking coefficient  [0–1, dimensionless]
    v_th : mean thermal velocity [m/s]  = sqrt(8·k_B·T / (π·m))

    Returns
    -------
    k_rxn [m/s]
    """
    if not (0 < beta <= 1):
        raise ValueError(f"beta must be in (0, 1], got {beta}")
    _pos(v_th, "v_th")
    return float(beta * v_th / 4.0)


# ══════════════════════════════════════════════════════════════════════════
# Tests  (run with:  python -m pytest physics/calculator.py  or  python physics/calculator.py)
# ══════════════════════════════════════════════════════════════════════════

def _run_tests():
    import math

    # Re
    assert math.isclose(reynolds(1.2, 10.0, 0.002, 1.8e-5), 1333.33, rel_tol=1e-4)

    # Ma
    assert math.isclose(mach(34.3, 343.0), 0.1, rel_tol=1e-9)

    # Eu
    assert math.isclose(euler(100.0, 1.2, 10.0), 100.0 / (0.5 * 1.2 * 100), rel_tol=1e-9)

    # Pr (air ≈ 0.71)
    assert math.isclose(prandtl(1005.0, 1.81e-5, 0.0257), 0.708, rel_tol=1e-2)

    # Nu
    assert math.isclose(nusselt(50.0, 0.01, 0.025), 20.0, rel_tol=1e-9)

    # Bi
    assert math.isclose(biot(50.0, 0.003, 200.0), 0.00075, rel_tol=1e-9)

    # Sc
    assert math.isclose(schmidt(1.81e-5, 1.2, 2.0e-5), 1.81e-5 / (1.2 * 2e-5), rel_tol=1e-9)

    # Sh
    assert math.isclose(sherwood(0.05, 0.002, 2e-5), 5.0, rel_tol=1e-9)

    # Pe_h
    assert math.isclose(peclet_heat(100.0, 0.71), 71.0, rel_tol=1e-9)

    # Pe_m
    assert math.isclose(peclet_mass(100.0, 1.5), 150.0, rel_tol=1e-9)

    # Da
    assert math.isclose(damkohler(0.1, 0.002, 10.0), 2e-5, rel_tol=1e-9)

    # k_rxn helper (aldFoam example: beta=0.05, v_th=144 m/s → k_rxn=1.8 m/s)
    assert math.isclose(k_rxn_from_sticking(0.05, 144.0), 1.8, rel_tol=1e-9)

    # compute_all round-trip
    params = {
        "rho": 1.2, "V": 10.0, "L": 0.002, "mu": 1.81e-5,
        "cp": 1005.0, "k_fluid": 0.0257, "D_m": 2e-5,
        "a": 343.0, "delta_p": 50.0,
        "h": 50.0, "k_solid": 200.0,
        "k_m": 0.05, "k_rxn": 0.1,
    }
    result = compute_all(params)
    expected_keys = {"Re", "Pr", "Sc", "Pe_h", "Pe_m", "Ma", "Eu", "Nu", "Bi", "Sh", "Da"}
    assert expected_keys == set(result.keys()), f"Missing keys: {expected_keys - set(result.keys())}"

    print("All tests passed.")


if __name__ == "__main__":
    _run_tests()
