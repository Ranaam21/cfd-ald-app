"""
geometry/vices/csg.py

CSG (Constructive Solid Geometry) tree using SDF boolean operations.

Boolean ops on SDFs:
    Union(A, B)     → min(sdf_A, sdf_B)   — everything inside either
    Subtract(A, B)  → max(sdf_A, -sdf_B)  — A minus B
    Intersect(A, B) → max(sdf_A, sdf_B)   — only what's inside both

The number of distinct binary tree topologies for n primitives = Catalan(n-1):
    n=2 → 1,  n=3 → 2,  n=4 → 5,  n=5 → 14,  n=6 → 42

Each node is either a Primitive (leaf) or an Op (internal node).
"""

import numpy as np
from enum import Enum
from geometry.vices.primitives import SDFPrimitive


class Op(Enum):
    UNION     = 'union'
    SUBTRACT  = 'subtract'
    INTERSECT = 'intersect'


class CSGNode:
    """Internal node: applies boolean op to two child nodes."""
    def __init__(self, op: Op, left, right):
        self.op    = op
        self.left  = left
        self.right = right

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        l = self.left.sdf(pts)
        r = self.right.sdf(pts)
        if self.op == Op.UNION:
            return np.minimum(l, r)
        if self.op == Op.SUBTRACT:
            return np.maximum(l, -r)
        if self.op == Op.INTERSECT:
            return np.maximum(l, r)
        raise ValueError(f'Unknown op: {self.op}')


class CSGLeaf:
    """Leaf node wrapping an SDF primitive."""
    def __init__(self, primitive: SDFPrimitive):
        self.primitive = primitive

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        return self.primitive.sdf(pts)


class CSGTree:
    """
    Full CSG tree with a root node.
    Call evaluate(pts) to get signed distances at arbitrary points.
    """
    def __init__(self, root):
        self.root = root

    def evaluate(self, pts: np.ndarray) -> np.ndarray:
        return self.root.sdf(pts)

    @staticmethod
    def union(a, b):
        return CSGNode(Op.UNION, a, b)

    @staticmethod
    def subtract(a, b):
        return CSGNode(Op.SUBTRACT, a, b)

    @staticmethod
    def intersect(a, b):
        return CSGNode(Op.INTERSECT, a, b)

    @staticmethod
    def leaf(primitive: SDFPrimitive):
        return CSGLeaf(primitive)


def catalan(n: int) -> int:
    """Catalan number C(n) — number of distinct binary tree topologies for n+1 primitives."""
    from math import comb
    return comb(2 * n, n) // (n + 1)
