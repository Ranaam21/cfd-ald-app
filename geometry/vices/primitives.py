"""
geometry/vices/primitives.py

SDF (Signed Distance Field) primitive shapes.
Negative inside, positive outside, zero on surface — same convention as PicoGK.

Each primitive implements:
    sdf(pts) -> np.ndarray [N]   signed distance at N query points [N, 3]
"""

import numpy as np


class SDFPrimitive:
    def sdf(self, pts: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Cylinder(SDFPrimitive):
    """Upright cylinder along Z axis."""
    def __init__(self, center, radius: float, height: float):
        self.cx, self.cy, self.cz = center
        self.r = radius
        self.h = height

    def sdf(self, pts):
        dx = pts[:, 0] - self.cx
        dy = pts[:, 1] - self.cy
        dz = pts[:, 2] - self.cz
        r_dist = np.sqrt(dx**2 + dy**2) - self.r
        z_dist = np.abs(dz) - self.h / 2.0
        return np.maximum(r_dist, z_dist)


class Box(SDFPrimitive):
    """Axis-aligned box."""
    def __init__(self, center, size):
        self.cx, self.cy, self.cz = center
        self.sx, self.sy, self.sz = size

    def sdf(self, pts):
        dx = np.abs(pts[:, 0] - self.cx) - self.sx / 2.0
        dy = np.abs(pts[:, 1] - self.cy) - self.sy / 2.0
        dz = np.abs(pts[:, 2] - self.cz) - self.sz / 2.0
        outside = np.sqrt(np.maximum(dx, 0)**2 +
                          np.maximum(dy, 0)**2 +
                          np.maximum(dz, 0)**2)
        inside  = np.minimum(np.maximum(dx, np.maximum(dy, dz)), 0.0)
        return outside + inside


class Sphere(SDFPrimitive):
    def __init__(self, center, radius: float):
        self.cx, self.cy, self.cz = center
        self.r = radius

    def sdf(self, pts):
        dx = pts[:, 0] - self.cx
        dy = pts[:, 1] - self.cy
        dz = pts[:, 2] - self.cz
        return np.sqrt(dx**2 + dy**2 + dz**2) - self.r


class Cone(SDFPrimitive):
    """Upward-pointing cone along Z, apex at top."""
    def __init__(self, center, radius: float, height: float):
        self.cx, self.cy, self.cz = center
        self.r = radius
        self.h = height

    def sdf(self, pts):
        dx = pts[:, 0] - self.cx
        dy = pts[:, 1] - self.cy
        dz = pts[:, 2] - (self.cz - self.h / 2.0)   # dz from base
        r_at_z = self.r * (1.0 - np.clip(dz / self.h, 0, 1))
        r_dist = np.sqrt(dx**2 + dy**2) - r_at_z
        z_dist = np.maximum(-dz, dz - self.h)
        return np.maximum(r_dist, z_dist)
