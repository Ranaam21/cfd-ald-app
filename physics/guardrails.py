"""
physics/guardrails.py

Regime checks and confidence scoring for the CFD-ALD surrogate app.

Usage
-----
    from physics.guardrails import GuardrailEngine, GuardrailBounds

    bounds = GuardrailBounds(Re=(10, 2000), Ma=(0.0, 0.3), Da=(0.01, 10.0))
    engine = GuardrailEngine(bounds)
    result = engine.check({"Re": 150, "Ma": 0.05, "Da": 0.8, "Bi": 0.05})
    print(result.confidence)   # 1.0  (all pass)
    print(result.summary())

Structure
---------
GuardrailBounds  — dataclass holding user-set (min, max) per number
ViolationRecord  — one violation: which number, value, bounds, severity
CheckResult      — list of violations + confidence score + recommendations
GuardrailEngine  — runs checks, computes confidence, emits reason codes
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple


# ── Default guardrail bounds ───────────────────────────────────────────────
# Derived from jet-impingement literature, ALD CFD papers, and Da design charts.
# Users override these via the Streamlit physics panel.

DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Re":   (1.0,    5000.0),   # laminar-to-transitional for showerhead flows
    "Ma":   (0.0,    0.3),      # incompressible regime upper limit
    "Eu":   (0.5,    50.0),     # sanity band for distributor pressure drop
    "Pr":   (0.5,    100.0),    # common gas/liquid range
    "Nu":   (1.0,    500.0),    # jet impingement correlation range
    "Bi":   (0.0,    0.1),      # lumped-capacitance valid below 0.1
    "Sc":   (0.1,    10.0),     # light gases to heavy precursors
    "Sh":   (1.0,    200.0),    # mass-transfer coefficient sanity
    "Pe_h": (0.1,    1e5),
    "Pe_m": (0.1,    1e5),
    "Da":   (1e-4,   100.0),    # reaction–transport spectrum for ALD
}

# Severity weights: how much each violation decrements confidence [0–1]
SEVERITY: Dict[str, float] = {
    "Re":   0.10,
    "Ma":   0.30,   # compressibility is a hard model-switch trigger
    "Eu":   0.15,
    "Pr":   0.10,
    "Nu":   0.15,
    "Bi":   0.10,
    "Sc":   0.10,
    "Sh":   0.10,
    "Pe_h": 0.05,
    "Pe_m": 0.05,
    "Da":   0.25,   # Da violations change the entire physics regime
}


# ══════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class GuardrailBounds:
    """User-controllable bounds for each dimensionless number."""
    Re:   Tuple[float, float] = dataclasses.field(default_factory=lambda: DEFAULT_BOUNDS["Re"])
    Ma:   Tuple[float, float] = dataclasses.field(default_factory=lambda: DEFAULT_BOUNDS["Ma"])
    Eu:   Tuple[float, float] = dataclasses.field(default_factory=lambda: DEFAULT_BOUNDS["Eu"])
    Pr:   Tuple[float, float] = dataclasses.field(default_factory=lambda: DEFAULT_BOUNDS["Pr"])
    Nu:   Tuple[float, float] = dataclasses.field(default_factory=lambda: DEFAULT_BOUNDS["Nu"])
    Bi:   Tuple[float, float] = dataclasses.field(default_factory=lambda: DEFAULT_BOUNDS["Bi"])
    Sc:   Tuple[float, float] = dataclasses.field(default_factory=lambda: DEFAULT_BOUNDS["Sc"])
    Sh:   Tuple[float, float] = dataclasses.field(default_factory=lambda: DEFAULT_BOUNDS["Sh"])
    Pe_h: Tuple[float, float] = dataclasses.field(default_factory=lambda: DEFAULT_BOUNDS["Pe_h"])
    Pe_m: Tuple[float, float] = dataclasses.field(default_factory=lambda: DEFAULT_BOUNDS["Pe_m"])
    Da:   Tuple[float, float] = dataclasses.field(default_factory=lambda: DEFAULT_BOUNDS["Da"])

    def as_dict(self) -> Dict[str, Tuple[float, float]]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GuardrailBounds":
        return cls(**{k: tuple(v) for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclasses.dataclass
class ViolationRecord:
    symbol:      str
    value:       float
    lo:          float
    hi:          float
    severity:    float
    reason_code: str       # machine-readable tag for the UI
    message:     str       # human-readable explanation

    @property
    def direction(self) -> str:
        return "above_max" if self.value > self.hi else "below_min"


@dataclasses.dataclass
class CheckResult:
    violations:      List[ViolationRecord]
    confidence:      float          # 1.0 → fully trusted; < 0.5 → recommend CFD refinement
    special_flags:   List[str]      # e.g. ["COMPRESSIBILITY_SWITCH", "DA_REGIME_MISMATCH"]
    recommendations: List[str]      # plain-English action items for the UI

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0

    def summary(self) -> str:
        lines = [f"Confidence: {self.confidence:.2f}  |  Violations: {len(self.violations)}"]
        for v in self.violations:
            lines.append(f"  [{v.reason_code}] {v.symbol}={v.value:.4g}  "
                         f"(allowed [{v.lo:.4g}, {v.hi:.4g}])  — {v.message}")
        for flag in self.special_flags:
            lines.append(f"  ⚑  {flag}")
        for rec in self.recommendations:
            lines.append(f"  → {rec}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════════════

class GuardrailEngine:
    """
    Checks a dict of computed dimensionless numbers against user-set bounds,
    returns a CheckResult with confidence score and reason codes.

    Parameters
    ----------
    bounds : GuardrailBounds
        Min/max for each number.  Numbers not present in `values` are skipped.
    cfr_threshold : float
        Confidence below this value triggers a "Run CFD refinement" recommendation.
    """

    CFD_REFINEMENT_THRESHOLD = 0.50

    def __init__(self, bounds: Optional[GuardrailBounds] = None):
        self.bounds = bounds or GuardrailBounds()

    # ── public API ─────────────────────────────────────────────────────────

    def check(self, values: Dict[str, float]) -> CheckResult:
        """
        Parameters
        ----------
        values : dict  {symbol: computed_value}
            Output of physics.calculator.compute_all()

        Returns
        -------
        CheckResult
        """
        violations: List[ViolationRecord] = []
        special_flags: List[str] = []
        recommendations: List[str] = []

        bounds_dict = self.bounds.as_dict()

        for symbol, value in values.items():
            if symbol not in bounds_dict:
                continue
            lo, hi = bounds_dict[symbol]
            if value < lo or value > hi:
                rec = self._make_violation(symbol, value, lo, hi)
                violations.append(rec)

        # ── special physics logic ─────────────────────────────────────────

        # Ma > 0.3 → compressibility must switch model branch
        if "Ma" in values and values["Ma"] > 0.3:
            special_flags.append("COMPRESSIBILITY_SWITCH")
            recommendations.append(
                f"Ma = {values['Ma']:.3f} > 0.3: switch to compressible flow model "
                f"(rhoPimpleFoam / compressible surrogate branch)."
            )

        # Da inconsistency: Da >> 1 but surrogate operating in transport-limited mode
        # (We detect this by checking if Pe_m is large AND Da is large simultaneously,
        #  which signals that precursor is advected quickly but reaction is also fast —
        #  the model must decide which dominates.)
        if "Da" in values and "Pe_m" in values:
            da, pe_m = values["Da"], values["Pe_m"]
            if da > 10.0 and pe_m < 1.0:
                special_flags.append("DA_REGIME_MISMATCH")
                recommendations.append(
                    f"Da={da:.2g} (reaction-dominated) but Pe_m={pe_m:.2g} (diffusion-dominated): "
                    f"check consistency of k_rxn, D_m, and flow rate inputs."
                )

        # Bi > 0.1 → internal faceplate gradients cannot be ignored
        if "Bi" in values and values["Bi"] > 0.1:
            special_flags.append("FACEPLATE_INTERNAL_GRADIENTS")
            recommendations.append(
                f"Bi = {values['Bi']:.3f} > 0.1: faceplate is thermally thick; "
                f"include conjugate heat transfer (CHT) or increase k_solid / reduce thickness."
            )

        # ── confidence ────────────────────────────────────────────────────
        confidence = 1.0
        for v in violations:
            confidence -= v.severity
        confidence = max(0.0, confidence)

        if confidence < self.CFD_REFINEMENT_THRESHOLD:
            recommendations.append(
                f"Confidence {confidence:.2f} < {self.CFD_REFINEMENT_THRESHOLD}: "
                f"run a targeted CFD refinement case before accepting this design."
            )

        return CheckResult(
            violations=violations,
            confidence=round(confidence, 4),
            special_flags=special_flags,
            recommendations=recommendations,
        )

    def update_bounds(self, **kwargs):
        """Update individual bounds in-place. e.g. engine.update_bounds(Re=(10, 500))"""
        for key, val in kwargs.items():
            if hasattr(self.bounds, key):
                setattr(self.bounds, key, tuple(val))
            else:
                raise KeyError(f"Unknown guardrail: {key}")

    # ── internal ───────────────────────────────────────────────────────────

    @staticmethod
    def _make_violation(symbol: str, value: float, lo: float, hi: float) -> ViolationRecord:
        direction = "above_max" if value > hi else "below_min"
        severity = SEVERITY.get(symbol, 0.10)

        messages = {
            ("Re",   "above_max"): "Flow may be turbulent; RANS surrogate trained on laminar data may not generalise.",
            ("Re",   "below_min"): "Re very low; check velocity and nozzle diameter inputs.",
            ("Ma",   "above_max"): "Compressibility effects non-negligible; incompressible surrogate invalid.",
            ("Eu",   "above_max"): "Pressure drop unusually high; check geometry or flow rate.",
            ("Eu",   "below_min"): "Pressure drop near zero; distributor may not be limiting flow.",
            ("Pr",   "above_max"): "Unusually high Pr; verify fluid properties.",
            ("Bi",   "above_max"): "Faceplate is thermally thick; lumped assumption breaks down.",
            ("Sc",   "below_min"): "Very low Sc; precursor diffuses rapidly — check D_m.",
            ("Sc",   "above_max"): "Very high Sc; mass BL much thinner than momentum BL.",
            ("Da",   "above_max"): "Reaction-limited regime; precursor may deplete before uniform coverage.",
            ("Da",   "below_min"): "Deep transport-limited; verify k_rxn is not unrealistically low.",
            ("Nu",   "below_min"): "Nu < 1 is unphysical; check h and k_fluid.",
            ("Sh",   "below_min"): "Sh < 1 is unphysical; check k_m and D_m.",
        }
        msg = messages.get((symbol, direction), f"{symbol} = {value:.4g} outside [{lo:.4g}, {hi:.4g}].")
        code = f"{symbol}_{direction.upper()}"

        return ViolationRecord(
            symbol=symbol, value=value, lo=lo, hi=hi,
            severity=severity, reason_code=code, message=msg,
        )


# ══════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════

def _run_tests():
    engine = GuardrailEngine()

    # All-pass case
    ok = engine.check({"Re": 200, "Ma": 0.05, "Da": 1.0, "Bi": 0.05, "Pr": 0.71, "Sc": 1.0})
    assert ok.passed, f"Expected pass, got: {ok.summary()}"
    assert ok.confidence == 1.0

    # Ma compressibility flag
    r = engine.check({"Ma": 0.5})
    assert "COMPRESSIBILITY_SWITCH" in r.special_flags
    assert r.confidence < 1.0

    # Da out-of-range
    r = engine.check({"Da": 500.0})
    assert any(v.symbol == "Da" for v in r.violations)

    # Low confidence triggers CFD refinement recommendation
    r = engine.check({"Re": 1e6, "Ma": 0.9, "Da": 500.0, "Bi": 5.0})
    assert r.confidence < GuardrailEngine.CFD_REFINEMENT_THRESHOLD
    assert any("CFD refinement" in rec for rec in r.recommendations)

    # Bounds override
    engine.update_bounds(Re=(100, 500))
    r = engine.check({"Re": 50})
    assert any(v.symbol == "Re" for v in r.violations)

    print("All guardrail tests passed.")


if __name__ == "__main__":
    _run_tests()
