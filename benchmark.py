"""
benchmark.py - compare LLMs on the same fixed test prompts.

Usage:
    export GROQ_API_KEY=...
    python benchmark.py openai/gpt-oss-120b openai/gpt-oss-20b
Writes results.csv and prints a summary per model.
"""
import csv
import os
import sys
import time

from llm import generate_part, make_groq_caller

REJECT = "reject"
# expected: overall size (x, y, z) in mm, plus hole count and/or hand-calculated volume
# volume uses true circles, so 1.5 % tolerance covers the 64-sided polygon approximation
CASES = [
    ("rect_4_m5", "A 120 x 80 x 8 mm plate with four M5 clearance holes, 12 mm from each edge.",
     {"size": (120, 80, 8), "holes": 4}),
    ("rect_centre", "Plate 100 by 60 mm, 10 mm thick, one 20 mm hole in the middle.",
     {"size": (100, 60, 10), "holes": 1}),
    ("rect_mix", "150x100x6 plate: M6 holes in all four corners 10 mm in from the edges and a 40 mm hole at the centre.",
     {"size": (150, 100, 6), "holes": 5}),
    ("rect_row", "A 90 mm x 90 mm x 12 mm plate with three 10 mm holes in a row along the centre line, 25 mm apart.",
     {"size": (90, 90, 12), "holes": 3}),
    ("rect_units", "Make a 0.1 m by 0.05 m plate that is 4 mm thick with an M4 hole in each corner, 6 mm from the edges.",
     {"size": (100, 50, 4), "holes": 4}),
    ("round_flange", "Round plate 120 mm diameter, 10 mm thick, 30 mm centre bore, six M8 holes on a 90 mm PCD.",
     {"size": (120, 120, 10), "holes": 7}),
    ("round_4", "Disc of diameter 80 mm and thickness 6 mm with 4 holes of 6 mm on a 60 mm bolt circle.",
     {"size": (80, 80, 6), "holes": 4}),
    ("tube", "A hollow cylinder, outer diameter 12 cm, inner diameter 9 cm, height 10 cm.",
     {"size": (120, 120, 100), "volume": 3.14159265 / 4 * (120**2 - 90**2) * 100}),
    ("l_bracket", "L-bracket: base 100 x 40 x 8 mm with a vertical wall 8 mm thick at one end, 68 mm tall overall.",
     {"size": (100, 40, 68), "volume": 100 * 40 * 8 + 8 * 40 * 60}),
    ("stack", "A 60 x 60 x 10 mm base with a 30 mm diameter, 40 mm tall cylinder standing on its centre.",
     {"size": (60, 60, 50), "volume": 60 * 60 * 10 + 3.14159265 * 15**2 * 40}),
    ("stepped_shaft", "Stepped shaft along z: a 40 mm diameter section 30 mm long, then a 25 mm diameter section 50 mm long on top.",
     {"size": (40, 40, 80), "volume": 3.14159265 / 4 * (40**2 * 30 + 25**2 * 50)}),
    ("t_shape", "T-shaped block: a 120 x 30 x 10 mm top bar resting on a 30 x 30 x 50 mm stem centred under it.",
     {"size": (120, 30, 60), "volume": 120 * 30 * 10 + 30 * 30 * 50}),
    ("reject_gearbox", "Design a two-stage helical gearbox.", REJECT),
    ("reject_sphere", "A solid sphere of 50 mm diameter.", REJECT),
    ("reject_impossible", "A 20 x 20 x 3 mm plate with four M12 holes 4 mm from each edge.", REJECT),
]


def score(expected, res) -> bool:
    """Judge the finished geometry, not the JSON wording."""
    if expected == REJECT:
        return res["status"] != "ok"          # correct = no part produced
    if res["status"] != "ok":
        return False
    rep = res["report"]
    if any(abs(a - b) > 0.5 for a, b in zip(sorted(rep["bounding_box_mm"]), sorted(expected["size"]))):
        return False
    if "holes" in expected and rep.get("hole_count") != expected["holes"]:
        return False
    if "volume" in expected and abs(rep["volume_mm3"] - expected["volume"]) / expected["volume"] > 0.015:
        return False
    return True


def run(models, make_caller):
    rows = []
    for model in models:
        call = make_caller(model)
        for case_id, prompt, expected in CASES:
            try:
                res = generate_part(prompt, call, max_attempts=3)
                ok = score(expected, res)
                rows.append({"model": model, "case": case_id, "status": res["status"],
                             "correct": ok, "first_try": ok and res["attempts"] == 1,
                             "attempts": res["attempts"],
                             "latency_s": round(sum(e["latency_s"] for e in res["log"]), 2)})
            except Exception as e:  # API error: record it, don't hide it
                rows.append({"model": model, "case": case_id, "status": f"api_error: {e}",
                             "correct": False, "first_try": False, "attempts": 0, "latency_s": 0})
            time.sleep(1)  # stay under free-tier rate limits
    return rows


def summarise(rows):
    for model in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == model]
        n = len(mine)
        print(f"\n{model}")
        print(f"  correct          : {sum(r['correct'] for r in mine)}/{n}")
        print(f"  correct 1st try  : {sum(r['first_try'] for r in mine)}/{n}")
        print(f"  avg attempts     : {sum(r['attempts'] for r in mine) / n:.2f}")
        print(f"  avg latency (s)  : {sum(r['latency_s'] for r in mine) / n:.2f}")


if __name__ == "__main__":
    models = sys.argv[1:] or ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    key = os.environ["GROQ_API_KEY"]
    rows = run(models, lambda m: make_groq_caller(key, m))
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarise(rows)
    print("\nSaved results.csv")
