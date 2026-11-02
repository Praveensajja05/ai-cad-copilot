import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad_core import parse_spec, run_any  # noqa: E402
from llm import SYSTEM_PROMPT, generate_part  # noqa: E402


def run(d):
    return run_any(parse_spec(d))


def box(size, pos, op="add"):
    return {"shape": "box", "op": op, "size": size, "position": pos}


def cyl(d, h, pos, axis="z", op="add"):
    return {"shape": "cylinder", "op": op, "diameter": d, "height": h, "axis": axis, "position": pos}


L_BRACKET = {"part_type": "composite", "envelope": [100, 40, 68],
             "bodies": [box([100, 40, 8], [0, 0, 4]), box([8, 40, 60], [-46, 0, 38])]}


def test_l_bracket_volume_matches_hand_calculation():
    r = run(L_BRACKET)
    assert r["errors"] == []
    assert math.isclose(r["report"]["volume_mm3"], 100 * 40 * 8 + 8 * 40 * 60)  # 51,200
    assert r["report"]["pieces"] == 1


def test_cylinder_stacked_on_box_is_one_piece():
    r = run({"part_type": "composite", "envelope": [60, 60, 50],
             "bodies": [box([60, 60, 10], [0, 0, 5]), cyl(30, 40, [0, 0, 30])]})
    assert r["errors"] == [] and r["report"]["pieces"] == 1


def test_side_hole_through_wall_along_x():
    d = json.loads(json.dumps(L_BRACKET))
    d["bodies"].append(cyl(10, 10, [-46, 0, 40], axis="x", op="subtract"))
    r = run(d)
    assert r["errors"] == []
    hole = 0.5 * 64 * 5 ** 2 * math.sin(2 * math.pi / 64) * 8   # 64-sided hole, 8 mm deep
    assert math.isclose(r["report"]["volume_mm3"], 51200 - hole, rel_tol=1e-4)


def test_floating_body_is_caught():
    r = run({"part_type": "composite", "envelope": [100, 40, 100],
             "bodies": [box([100, 40, 8], [0, 0, 4]), box([10, 10, 10], [0, 0, 95])]})
    assert any("separate pieces" in e for e in r["errors"])


def test_hole_that_misses_the_part_is_caught():
    d = json.loads(json.dumps(L_BRACKET))
    d["bodies"].append(cyl(10, 10, [0, 0, 200], op="subtract"))
    assert any("misses the part" in e for e in run(d)["errors"])


def test_buried_add_body_is_caught():
    d = json.loads(json.dumps(L_BRACKET))
    d["bodies"].append(box([10, 10, 4], [20, 0, 4]))
    assert any("adds nothing" in e for e in run(d)["errors"])


def test_wrong_envelope_is_caught():
    d = json.loads(json.dumps(L_BRACKET))
    d["envelope"] = [100, 40, 60]       # model forgot the base thickness
    assert any("envelope" in e for e in run(d)["errors"])


def test_first_body_must_be_add():
    d = {"part_type": "composite", "envelope": [10, 10, 10],
         "bodies": [box([10, 10, 10], [0, 0, 5], op="subtract")]}
    assert any("must be 'add'" in e for e in run(d)["errors"])


def test_example_in_system_prompt_is_itself_valid():
    start = SYSTEM_PROMPT.index('{"part_type": "composite", "envelope": [70')
    block = SYSTEM_PROMPT[start:SYSTEM_PROMPT.index("}]}", start) + 3]
    assert run(json.loads(block))["errors"] == []


def test_loop_fixes_wrong_envelope_on_retry():
    bad = json.loads(json.dumps(L_BRACKET))
    bad["envelope"] = [100, 40, 60]
    answers = iter([json.dumps(bad), json.dumps(L_BRACKET)])
    res = generate_part("L-bracket", lambda m: next(answers))
    assert res["status"] == "ok" and res["attempts"] == 2
