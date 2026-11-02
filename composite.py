"""
composite.py - parts built from several simple bodies (boxes and cylinders).

The LLM describes a part as an ordered list of bodies:
    add      -> join this body onto the part      (stacking, L-shapes, bosses)
    subtract -> cut this body out of the part     (holes, slots, notches)

Nothing here trusts the LLM. Checks:
  1. bodies are applied in order; the first one must be "add"
  2. every body must actually change the part (an "add" fully buried inside,
     or a hole floating in empty air, is a mistake)
  3. the result must be ONE connected, closed solid (no floating pieces)
  4. the overall size must equal the "envelope" the model said it intended
  5. the mesh volume must agree with an independent dart-throwing
     (Monte Carlo) estimate that never touches the mesh code
"""
from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
import trimesh
from pydantic import BaseModel, Field

SECTIONS = 64
ENVELOPE_TOL = 0.5      # mm
MC_POINTS = 200_000     # darts for the volume cross-check
MC_TOL = 0.02           # mesh volume must agree with the dart estimate within 2 %
BODY_POINTS = 5_000     # darts thrown inside each body for the "does it change anything" check


class Body(BaseModel):
    shape: Literal["box", "cylinder"]
    op: Literal["add", "subtract"] = "add"
    position: List[float] = Field(min_length=3, max_length=3)    # centre of the body (x, y, z)
    size: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)  # box: [x, y, z]
    diameter: Optional[float] = Field(default=None, gt=0)          # cylinder
    height: Optional[float] = Field(default=None, gt=0)            # cylinder, along its axis
    axis: Literal["x", "y", "z"] = "z"                             # cylinder direction


class CompositeSpec(BaseModel):
    part_type: Literal["composite"]
    bodies: List[Body] = Field(min_length=1, max_length=20)
    envelope: List[float] = Field(min_length=3, max_length=3)  # intended overall size [x, y, z]


# ---------------------------------------------------------------- shape maths (no meshes)
def _check_fields(spec: CompositeSpec) -> List[str]:
    errors = []
    if spec.bodies[0].op != "add":
        errors.append("Body 1 must be 'add' (you cannot cut from nothing).")
    for i, b in enumerate(spec.bodies, start=1):
        if b.shape == "box" and (b.size is None or min(b.size) <= 0):
            errors.append(f"Body {i}: box needs 'size' [x, y, z] with positive values.")
        if b.shape == "cylinder" and (b.diameter is None or b.height is None):
            errors.append(f"Body {i}: cylinder needs 'diameter' and 'height'.")
    if min(spec.envelope) <= 0:
        errors.append("'envelope' values must be positive.")
    return errors


def _half_extents(b: Body) -> np.ndarray:
    """Half-size of the body's bounding box along x, y, z."""
    if b.shape == "box":
        return np.array(b.size) / 2
    r, h = b.diameter / 2, b.height / 2
    return {"x": np.array([h, r, r]), "y": np.array([r, h, r]), "z": np.array([r, r, h])}[b.axis]


def _inside(b: Body, pts: np.ndarray) -> np.ndarray:
    """True for each point that lies inside body b (exact maths, not the mesh)."""
    d = pts - np.array(b.position)
    if b.shape == "box":
        return np.all(np.abs(d) <= np.array(b.size) / 2, axis=1)
    k = "xyz".index(b.axis)
    radial = np.delete(d, k, axis=1)
    return (np.abs(d[:, k]) <= b.height / 2) & (np.sum(radial ** 2, axis=1) <= (b.diameter / 2) ** 2)


def _inside_part(bodies: List[Body], pts: np.ndarray) -> np.ndarray:
    """Apply add/subtract in order, exactly like the real build does."""
    inside = np.zeros(len(pts), dtype=bool)
    for b in bodies:
        hit = _inside(b, pts)
        inside = inside | hit if b.op == "add" else inside & ~hit
    return inside


def _sample_in_body(b: Body, n: int, rng) -> np.ndarray:
    he = _half_extents(b)
    pts = np.array(b.position) + rng.uniform(-1, 1, size=(n * 2, 3)) * he
    return pts[_inside(b, pts)][:n]


def _check_each_body_matters(spec: CompositeSpec) -> List[str]:
    rng = np.random.default_rng(1)
    errors = []
    for i, b in enumerate(spec.bodies[1:], start=2):
        pts = _sample_in_body(b, BODY_POINTS, rng)
        before = _inside_part(spec.bodies[: i - 1], pts)
        if b.op == "add" and before.all():
            errors.append(f"Body {i} (add) is completely inside existing material, so it adds nothing.")
        if b.op == "subtract" and not before.any():
            errors.append(f"Body {i} (subtract) misses the part entirely, so it cuts nothing. Check its position.")
    return errors


def monte_carlo_volume(spec: CompositeSpec) -> float:
    """Throw random darts into a box around the part; volume = box volume x fraction that hit."""
    adds = [b for b in spec.bodies if b.op == "add"]
    lo = np.min([np.array(b.position) - _half_extents(b) for b in adds], axis=0)
    hi = np.max([np.array(b.position) + _half_extents(b) for b in adds], axis=0)
    pts = np.random.default_rng(0).uniform(lo, hi, size=(MC_POINTS, 3))
    return float(np.prod(hi - lo) * _inside_part(spec.bodies, pts).mean())


# ---------------------------------------------------------------- mesh build + checks
def _body_mesh(b: Body) -> trimesh.Trimesh:
    if b.shape == "box":
        m = trimesh.creation.box(extents=b.size)
    else:
        m = trimesh.creation.cylinder(radius=b.diameter / 2, height=b.height, sections=SECTIONS)
        if b.axis == "x":
            m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        elif b.axis == "y":
            m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation(b.position)
    return m


def build_composite(spec: CompositeSpec) -> trimesh.Trimesh:
    solid = _body_mesh(spec.bodies[0])
    for b in spec.bodies[1:]:
        op = trimesh.boolean.union if b.op == "add" else trimesh.boolean.difference
        solid = op([solid, _body_mesh(b)], engine="manifold")
    return solid


def run_composite(spec: CompositeSpec) -> dict:
    errors = _check_fields(spec)
    if not errors:
        errors = _check_each_body_matters(spec)
    if errors:
        return {"errors": errors, "holes": [], "mesh": None, "report": None}
    try:
        mesh = build_composite(spec)
    except Exception as e:
        return {"errors": [f"Geometry build failed: {e}"], "holes": [], "mesh": None, "report": None}

    errors = []
    if len(mesh.faces) == 0 or not mesh.is_volume:
        errors.append("Result is not a closed, valid solid.")
    pieces = mesh.body_count if len(mesh.faces) else 0
    if pieces != 1:
        errors.append(f"Result is {pieces} separate pieces; every 'add' body must touch or overlap the part.")
    ext = [float(v) for v in mesh.extents] if len(mesh.faces) else [0.0, 0.0, 0.0]
    for axis, want, got in zip("XYZ", spec.envelope, ext):
        if abs(want - got) > ENVELOPE_TOL:
            errors.append(f"Overall {axis} size is {got:.2f} mm but envelope says {want} mm. "
                          f"A body is the wrong size or in the wrong position.")
    vol = float(mesh.volume) if len(mesh.faces) else 0.0
    mc = monte_carlo_volume(spec)
    rel = abs(vol - mc) / mc if mc > 0 else 1.0
    if rel > MC_TOL:
        errors.append(f"Mesh volume {vol:.0f} mm^3 disagrees with independent estimate {mc:.0f} mm^3 ({rel:.1%}).")

    report = {
        "volume_mm3": round(vol, 2),
        "expected_volume_mm3": round(mc, 2),
        "volume_error_pct": round(rel * 100, 3),
        "bounding_box_mm": [round(v, 2) for v in ext],
        "bodies": len(spec.bodies),
        "pieces": int(pieces),
        "faces": int(len(mesh.faces)),
        "valid_solid": bool(mesh.is_volume),
    }
    return {"errors": errors, "holes": [], "mesh": mesh, "report": report}
