import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad_core import PartSpec, check_spec, run_pipeline  # noqa: E402
from llm import generate_part  # noqa: E402


def spec(**kw):
    return PartSpec.model_validate(kw)


def test_plate_with_m5_corner_holes_is_valid_solid():
    s = spec(part_type="rect_plate", length=120, width=80, thickness=8,
             holes=[{"pattern": "corners", "edge_offset": 12, "bolt": "M5"}])
    r = run_pipeline(s)
    assert r["errors"] == []
    assert r["mesh"].is_volume
    assert len(r["holes"]) == 4
    assert all(h.d == 5.5 for h in r["holes"])           # ISO 273 medium fit
    assert {(abs(h.x), abs(h.y)) for h in r["holes"]} == {(48.0, 28.0)}


def test_round_plate_with_bore_and_bolt_circle():
    s = spec(part_type="round_plate", diameter=120, thickness=10,
             holes=[{"pattern": "single", "diameter": 30},
                    {"pattern": "bolt_circle", "count": 6, "pcd": 90, "bolt": "M8"}])
    r = run_pipeline(s)
    assert r["errors"] == []
    ring = r["holes"][1:]
    assert len(ring) == 6
    assert all(math.isclose(math.hypot(h.x, h.y), 45.0) for h in ring)


def test_hole_too_close_to_edge_is_caught():
    s = spec(part_type="rect_plate", length=50, width=50, thickness=5,
             holes=[{"pattern": "single", "x": 24, "y": 0, "diameter": 6}])
    _, errors = check_spec(s)
    assert any("edge" in e for e in errors)


def test_overlapping_holes_are_caught():
    s = spec(part_type="rect_plate", length=50, width=50, thickness=5,
             holes=[{"pattern": "single", "x": 0, "y": 0, "diameter": 6},
                    {"pattern": "single", "x": 5, "y": 0, "diameter": 6}])
    _, errors = check_spec(s)
    assert any("apart" in e for e in errors)


def test_unknown_bolt_is_caught():
    s = spec(part_type="rect_plate", length=50, width=50, thickness=5,
             holes=[{"pattern": "single", "bolt": "M7"}])
    _, errors = check_spec(s)
    assert any("unknown bolt" in e for e in errors)


def test_volume_matches_hand_calculation():
    s = spec(part_type="rect_plate", length=100, width=50, thickness=10)
    r = run_pipeline(s)
    assert math.isclose(r["report"]["volume_mm3"], 50000.0)


# ---- self-correction loop, tested with a fake LLM (no API needed)
BAD = json.dumps({"part_type": "rect_plate", "length": 50, "width": 50, "thickness": 5,
                  "holes": [{"pattern": "single", "x": 24, "y": 0, "diameter": 6}]})
GOOD = json.dumps({"part_type": "rect_plate", "length": 50, "width": 50, "thickness": 5,
                   "holes": [{"pattern": "single", "x": 0, "y": 0, "diameter": 6}]})


def fake_llm(answers):
    it, seen = iter(answers), []

    def call(messages):
        seen.append(len(messages))
        return next(it)
    return call, seen


def test_loop_fixes_bad_first_answer():
    call, seen = fake_llm([BAD, GOOD])
    res = generate_part("anything", call, max_attempts=3)
    assert res["status"] == "ok" and res["attempts"] == 2
    assert seen == [2, 4]   # 2nd call carried the error feedback


def test_loop_recovers_from_non_json():
    call, _ = fake_llm(["sure, here you go", GOOD])
    assert generate_part("x", call)["attempts"] == 2


def test_model_refusal_is_reported():
    call, _ = fake_llm([json.dumps({"error": "not a flat plate"})])
    res = generate_part("design a gearbox", call)
    assert res["status"] == "rejected"


def test_loop_gives_up_after_max_attempts():
    call, _ = fake_llm([BAD, BAD, BAD])
    res = generate_part("x", call, max_attempts=3)
    assert res["status"] == "failed" and res["attempts"] == 3


def test_json_in_code_fence_is_accepted():
    call, _ = fake_llm(["```json\n" + GOOD + "\n```"])
    assert generate_part("x", call)["status"] == "ok"
