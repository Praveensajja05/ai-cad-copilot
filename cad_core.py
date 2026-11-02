"""
cad_core.py - the deterministic half of AI CAD Copilot.

Nothing in this file talks to an LLM. It takes a JSON spec and:
  1. validates its shape (Pydantic)
  2. expands hole patterns into exact hole positions
  3. checks geometry rules (holes inside the part, holes not touching)
  4. builds the 3D mesh (trimesh + manifold3d booleans)
  5. checks the built mesh against what the spec asked for
"""
from __future__ import annotations

import math
from typing import List, Literal, Optional

import trimesh
from typing import Annotated, Union

from pydantic import BaseModel, Field, TypeAdapter

# ISO 273 medium-fit clearance hole diameters (mm)
CLEARANCE_MM = {"M3": 3.4, "M4": 4.5, "M5": 5.5, "M6": 6.6,
                "M8": 9.0, "M10": 11.0, "M12": 13.5}

SECTIONS = 64          # every circle is drawn as a 64-sided polygon
MIN_EDGE_MARGIN = 2.0  # mm of material required between a hole and the outer edge
MIN_WEB = 2.0          # mm of material required between two holes
VOLUME_TOL = 1e-3      # built volume must match expected volume within 0.1 %
DIM_TOL = 0.01         # bounding box must match requested size within 0.01 mm


# ---------------------------------------------------------------- schema
class HolePattern(BaseModel):
    pattern: Literal["single", "corners", "bolt_circle"]
    diameter: Optional[float] = Field(default=None, gt=0)  # plain hole size
    bolt: Optional[str] = None                             # or a bolt name like "M5"
    x: Optional[float] = None                              # single: centre / bolt_circle: circle centre
    y: Optional[float] = None
    edge_offset: Optional[float] = Field(default=None, gt=0)  # corners: edge -> hole centre
    count: Optional[int] = Field(default=None, ge=1, le=36)   # bolt_circle
    pcd: Optional[float] = Field(default=None, gt=0)          # bolt_circle diameter


class PartSpec(BaseModel):
    part_type: Literal["rect_plate", "round_plate"]
    length: Optional[float] = Field(default=None, gt=0)    # rect_plate
    width: Optional[float] = Field(default=None, gt=0)     # rect_plate
    diameter: Optional[float] = Field(default=None, gt=0)  # round_plate
    thickness: float = Field(gt=0)
    holes: List[HolePattern] = Field(default_factory=list)


class Hole(BaseModel):
    x: float
    y: float
    d: float


# ---------------------------------------------------------------- rule checks
def _hole_diameter(p: HolePattern, i: int, errors: list) -> Optional[float]:
    if p.bolt:
        key = p.bolt.upper().replace(" ", "")
        if key not in CLEARANCE_MM:
            errors.append(f"Hole pattern {i}: unknown bolt size '{p.bolt}'. "
                          f"Supported: {', '.join(CLEARANCE_MM)}.")
            return None
        return CLEARANCE_MM[key]
    if p.diameter:
        return p.diameter
    errors.append(f"Hole pattern {i}: give either 'bolt' or 'diameter'.")
    return None


def _expand(spec: PartSpec, errors: list) -> List[Hole]:
    """Turn patterns ('4 holes in the corners') into exact (x, y, d) holes."""
    holes: List[Hole] = []
    for i, p in enumerate(spec.holes, start=1):
        d = _hole_diameter(p, i, errors)
        if d is None:
            continue
        cx, cy = p.x or 0.0, p.y or 0.0
        if p.pattern == "single":
            holes.append(Hole(x=cx, y=cy, d=d))
        elif p.pattern == "corners":
            if spec.part_type != "rect_plate":
                errors.append(f"Hole pattern {i}: 'corners' only works on rect_plate.")
            elif p.edge_offset is None:
                errors.append(f"Hole pattern {i}: 'corners' needs 'edge_offset'.")
            else:
                ox = spec.length / 2 - p.edge_offset
                oy = spec.width / 2 - p.edge_offset
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        holes.append(Hole(x=sx * ox, y=sy * oy, d=d))
        elif p.pattern == "bolt_circle":
            if p.count is None or p.pcd is None:
                errors.append(f"Hole pattern {i}: 'bolt_circle' needs 'count' and 'pcd'.")
            else:
                r = p.pcd / 2
                for k in range(p.count):
                    a = 2 * math.pi * k / p.count
                    holes.append(Hole(x=cx + r * math.cos(a), y=cy + r * math.sin(a), d=d))
    return holes


def check_spec(spec: PartSpec):
    """Return (holes, errors). An empty error list means the design is buildable."""
    errors: list = []
    if spec.part_type == "rect_plate" and (spec.length is None or spec.width is None):
        return [], ["rect_plate needs 'length' and 'width'."]
    if spec.part_type == "round_plate" and spec.diameter is None:
        return [], ["round_plate needs 'diameter'."]

    holes = _expand(spec, errors)

    # Rule 1: every hole sits inside the part with MIN_EDGE_MARGIN of material left
    for n, h in enumerate(holes, start=1):
        r = h.d / 2
        if spec.part_type == "rect_plate":
            lim_x = spec.length / 2 - MIN_EDGE_MARGIN
            lim_y = spec.width / 2 - MIN_EDGE_MARGIN
            if abs(h.x) + r > lim_x + 1e-9 or abs(h.y) + r > lim_y + 1e-9:
                errors.append(f"Hole {n} at ({h.x:.2f}, {h.y:.2f}), dia {h.d}: too close to the edge. "
                              f"Need |x| <= {lim_x - r:.2f} and |y| <= {lim_y - r:.2f}.")
        else:
            lim = spec.diameter / 2 - MIN_EDGE_MARGIN
            if math.hypot(h.x, h.y) + r > lim + 1e-9:
                errors.append(f"Hole {n} at ({h.x:.2f}, {h.y:.2f}), dia {h.d}: too close to the edge. "
                              f"Its centre must be within {lim - r:.2f} mm of the part centre.")

    # Rule 2: no two holes touch (MIN_WEB of material between them)
    for a in range(len(holes)):
        for b in range(a + 1, len(holes)):
            ha, hb = holes[a], holes[b]
            need = ha.d / 2 + hb.d / 2 + MIN_WEB
            gap = math.hypot(ha.x - hb.x, ha.y - hb.y)
            if gap < need - 1e-9:
                errors.append(f"Holes {a + 1} and {b + 1} are {gap:.2f} mm apart; need at least {need:.2f} mm.")
    return holes, errors


# ---------------------------------------------------------------- build + verify
def build_mesh(spec: PartSpec, holes: List[Hole]) -> trimesh.Trimesh:
    t = spec.thickness
    if spec.part_type == "rect_plate":
        body = trimesh.creation.box(extents=[spec.length, spec.width, t])
    else:
        body = trimesh.creation.cylinder(radius=spec.diameter / 2, height=t, sections=SECTIONS)
    if not holes:
        return body
    cutters = []
    for h in holes:
        c = trimesh.creation.cylinder(radius=h.d / 2, height=t * 3, sections=SECTIONS)
        c.apply_translation([h.x, h.y, 0])
        cutters.append(c)
    # body minus every cutter, using the manifold3d boolean engine
    return trimesh.boolean.difference([body] + cutters, engine="manifold")


def _polygon_area(r: float, n: int = SECTIONS) -> float:
    """Area of the n-sided polygon trimesh draws for a circle of radius r."""
    return 0.5 * n * r * r * math.sin(2 * math.pi / n)


def expected_volume(spec: PartSpec, holes: List[Hole]) -> float:
    outer = spec.length * spec.width if spec.part_type == "rect_plate" else _polygon_area(spec.diameter / 2)
    return (outer - sum(_polygon_area(h.d / 2) for h in holes)) * spec.thickness


def check_mesh(spec: PartSpec, holes: List[Hole], mesh: trimesh.Trimesh):
    errors = []
    if not mesh.is_volume:
        errors.append("Built mesh is not a closed, valid solid.")
    exp = expected_volume(spec, holes)
    got = float(mesh.volume)
    rel = abs(got - exp) / exp
    if rel > VOLUME_TOL:
        errors.append(f"Volume {got:.1f} mm^3 differs from expected {exp:.1f} mm^3 ({rel:.2%}).")
    if spec.part_type == "rect_plate":
        want = [spec.length, spec.width, spec.thickness]
    else:
        want = [spec.diameter, spec.diameter, spec.thickness]
    got_ext = [float(v) for v in mesh.extents]
    for axis, w, g in zip("XYZ", want, got_ext):
        if abs(w - g) > DIM_TOL:
            errors.append(f"{axis} size is {g:.2f} mm but {w} mm was requested.")
    report = {
        "volume_mm3": round(got, 2),
        "expected_volume_mm3": round(exp, 2),
        "volume_error_pct": round(rel * 100, 4),
        "bounding_box_mm": [round(v, 2) for v in got_ext],
        "hole_count": len(holes),
        "faces": int(len(mesh.faces)),
        "valid_solid": bool(mesh.is_volume),
    }
    return report, errors


def run_pipeline(spec: PartSpec) -> dict:
    """Spec in -> rule check -> build -> verify. Never raises; errors go in the list."""
    holes, errors = check_spec(spec)
    if errors:
        return {"errors": errors, "holes": holes, "mesh": None, "report": None}
    try:
        mesh = build_mesh(spec, holes)
    except Exception as e:  # boolean engine failure
        return {"errors": [f"Geometry build failed: {e}"], "holes": holes, "mesh": None, "report": None}
    report, errors = check_mesh(spec, holes, mesh)
    return {"errors": errors, "holes": holes, "mesh": mesh, "report": report}


# ---------------------------------------------------------------- single entry point
from composite import CompositeSpec, run_composite  # noqa: E402

AnySpec = Annotated[Union[PartSpec, CompositeSpec], Field(discriminator="part_type")]
_spec_adapter = TypeAdapter(AnySpec)


def parse_spec(data: dict):
    """Validate JSON into PartSpec (plates) or CompositeSpec (multi-body). Raises ValidationError."""
    return _spec_adapter.validate_python(data)


def run_any(spec) -> dict:
    return run_composite(spec) if isinstance(spec, CompositeSpec) else run_pipeline(spec)
