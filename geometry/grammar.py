"""
geometry/grammar.py

Defines the fixed topology of an ALD showerhead.
The grammar never changes — only the dimensions (parameters) do.

Topology (top → bottom, in flow direction):
    PLENUM  →  DISTRIBUTOR_PLATE  →  NOZZLE_ARRAY  →  STANDOFF  →  WAFER_PLANE

Each component is a named token with a role, allowed connections,
and a description of what it represents physically.
"""

from __future__ import annotations
import dataclasses
from enum import Enum, auto
from typing import List, Tuple


# ══════════════════════════════════════════════════════════════════════════
# Component tokens
# ══════════════════════════════════════════════════════════════════════════

class Component(Enum):
    PLENUM             = auto()   # gas mixing / distribution chamber above plate
    DISTRIBUTOR_PLATE  = auto()   # faceplate with nozzle holes
    NOZZLE_ARRAY       = auto()   # the set of nozzle holes through the plate
    STANDOFF           = auto()   # gap between plate exit and wafer surface
    WAFER_PLANE        = auto()   # target deposition surface


# Fixed connection order (flow direction)
TOPOLOGY_ORDER: List[Component] = [
    Component.PLENUM,
    Component.DISTRIBUTOR_PLATE,
    Component.NOZZLE_ARRAY,
    Component.STANDOFF,
    Component.WAFER_PLANE,
]

# What each component contributes to the geometry/physics
COMPONENT_ROLES = {
    Component.PLENUM: (
        "Mixing chamber. Gas enters at top, distributes laterally before "
        "reaching nozzle array. Height controls residence time and pressure drop."
    ),
    Component.DISTRIBUTOR_PLATE: (
        "Solid faceplate of thickness t. Contains N nozzle holes of diameter D "
        "arranged in a pattern. Structural + flow-distribution component."
    ),
    Component.NOZZLE_ARRAY: (
        "Array of cylindrical through-holes. Pattern (ring/hex/square) and pitch "
        "control flow uniformity above wafer. Key geometry for surrogate training."
    ),
    Component.STANDOFF: (
        "Free-stream gap between nozzle exit and wafer surface. "
        "H/D ratio governs jet impingement regime (Nu, uniformity)."
    ),
    Component.WAFER_PLANE: (
        "Target substrate surface. Uniformity of velocity, temperature, and "
        "precursor concentration here are the primary design KPIs."
    ),
}


# ══════════════════════════════════════════════════════════════════════════
# Nozzle pattern tokens
# ══════════════════════════════════════════════════════════════════════════

class NozzlePattern(Enum):
    RING    = "ring"    # concentric rings of holes (common in ALD showerheads)
    HEX     = "hex"     # hexagonal close-packed grid (best area coverage)
    SQUARE  = "square"  # square grid (simple, easy to manufacture)


# ══════════════════════════════════════════════════════════════════════════
# Showerhead topology descriptor
# ══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass(frozen=True)
class ShowerheadTopology:
    """
    Immutable description of the showerhead topology.
    Passed to the parametric generator which fills in the actual dimensions.

    Attributes
    ----------
    components   : ordered list of topology components (fixed)
    pattern      : nozzle hole pattern
    n_inlets     : number of gas inlets into the plenum (1 = centre, >1 = ring)
    has_diffuser : whether nozzle exits have a diverging diffuser cone
    has_restrictor : whether a flow restrictor (orifice) sits above the plate
    """
    components:      Tuple[Component, ...]  = dataclasses.field(
                         default_factory=lambda: tuple(TOPOLOGY_ORDER))
    pattern:         NozzlePattern          = NozzlePattern.HEX
    n_inlets:        int                    = 1
    has_diffuser:    bool                   = False
    has_restrictor:  bool                   = False

    def describe(self) -> str:
        lines = ["ShowerheadTopology"]
        lines.append(f"  Pattern      : {self.pattern.value}")
        lines.append(f"  Inlets       : {self.n_inlets}")
        lines.append(f"  Diffuser     : {self.has_diffuser}")
        lines.append(f"  Restrictor   : {self.has_restrictor}")
        lines.append("  Flow path    :")
        for c in self.components:
            lines.append(f"    [{c.name}]  {COMPONENT_ROLES[c][:60]}…")
        return "\n".join(lines)


# ── Default topology (most common ALD showerhead configuration) ────────────
DEFAULT_TOPOLOGY = ShowerheadTopology(
    pattern=NozzlePattern.HEX,
    n_inlets=1,
    has_diffuser=False,
    has_restrictor=False,
)


# ══════════════════════════════════════════════════════════════════════════
# Parameter schema — defines names, units, and default ranges
# Used by Streamlit sliders and OpenFOAM sweep scripts
# ══════════════════════════════════════════════════════════════════════════

PARAM_SCHEMA = {
    #  key          (default,  min,   max,   unit,       description)
    "D":           (0.002,   0.0005, 0.005, "m",        "Nozzle hole diameter"),
    "pitch_over_D":(4.0,     2.0,   10.0,  "-",        "Pitch / D ratio (centre-to-centre / D)"),
    "H_plenum":    (0.020,   0.005,  0.050, "m",        "Plenum height"),
    "t_face":      (0.003,   0.001,  0.010, "m",        "Faceplate thickness"),
    "theta_deg":   (0.0,     0.0,   30.0,  "deg",      "Diffuser half-angle (0 = straight nozzle)"),
    "standoff":    (0.020,   0.005,  0.100, "m",        "Nozzle-exit to wafer distance"),
    "D_plate":     (0.300,   0.050,  0.500, "m",        "Plate (and wafer) diameter"),
    "Cd":          (0.8,     0.6,    1.0,   "-",        "Restrictor discharge coefficient"),
}


def default_params() -> dict:
    """Return a dict of {key: default_value} for all geometry parameters."""
    return {k: v[0] for k, v in PARAM_SCHEMA.items()}
