"""
llm.py - the AI half of AI CAD Copilot.

The LLM only translates English into a JSON spec. It never does geometry.
If the spec fails validation, the exact errors are sent back and the model
gets another try (up to max_attempts).
"""
from __future__ import annotations

import json
import time
from typing import Callable, Dict, List

from pydantic import ValidationError

from cad_core import parse_spec, run_any

SYSTEM_PROMPT = """You convert a plain-English description of a part into JSON.
All units are millimetres (convert cm, m, inches). Axes: x = length, y = width, z = up.

Choose ONE of two formats.

FORMAT A - simple plates, discs, rings, tubes (preferred when it fits):
- rect_plate: length (x), width (y), thickness (z)
- round_plate: diameter, thickness. Use it for any disc, cylinder, ring or tube;
  thickness = height. A tube is a round_plate with one centre hole = inner diameter.
- "holes": list of patterns, each ONE of:
  {"pattern": "single", "x": .., "y": .., "diameter" or "bolt": ..}
  {"pattern": "corners", "edge_offset": .., "diameter" or "bolt": ..}  (rect_plate only; 4 holes)
  {"pattern": "bolt_circle", "count": .., "pcd": .., "diameter" or "bolt": ..}
  Use "bolt": "M5" etc. when a bolt size is named. Holes go straight through along z.
  Example: {"part_type": "rect_plate", "length": 90, "width": 50, "thickness": 6,
            "holes": [{"pattern": "corners", "edge_offset": 8, "bolt": "M4"}]}

FORMAT B - composite: L-shapes, T-shapes, steps, one object on top of another,
holes that are not vertical. Build the part from boxes and cylinders, applied in order:
  {"part_type": "composite",
   "envelope": [x, y, z],            <- overall size of the finished part
   "bodies": [
     {"shape": "box", "op": "add", "size": [x, y, z], "position": [cx, cy, cz]},
     {"shape": "cylinder", "op": "add" or "subtract", "diameter": .., "height": ..,
      "axis": "x" | "y" | "z", "position": [cx, cy, cz]}
   ]}
Rules for FORMAT B:
- "position" is the CENTRE of the body. Put the bottom of the part at z = 0.
- "add" joins material; "subtract" cuts it (holes, slots). The first body must be "add".
- Bodies that are joined must touch or overlap. Stacked bodies sit on the top face of the one below.
- A hole must be slightly longer than the material it goes through.
  Example: 70 x 30 x 5 base with a 5 mm thick, 40 mm tall wall at the left end and a 6 mm hole through the wall:
  {"part_type": "composite", "envelope": [70, 30, 45], "bodies": [
    {"shape": "box", "op": "add", "size": [70, 30, 5], "position": [0, 0, 2.5]},
    {"shape": "box", "op": "add", "size": [5, 30, 40], "position": [-32.5, 0, 25]},
    {"shape": "cylinder", "op": "subtract", "diameter": 6, "height": 7, "axis": "x", "position": [-32.5, 0, 25]}]}

Never change a dimension the user stated. Leave out fields that do not apply.
If the part cannot be made from boxes and cylinders (gears, threads, springs, spheres,
curved or freeform surfaces) or is impossible as stated, return {"error": "<short reason>"}.
Return ONLY one JSON object."""

LLMCaller = Callable[[List[Dict[str, str]]], str]


def make_groq_caller(api_key: str, model: str) -> LLMCaller:
    """Return a function messages -> raw text, backed by the Groq API."""
    from groq import Groq
    client = Groq(api_key=api_key)

    extra = {}
    if model.startswith("openai/gpt-oss"):
        # GPT-OSS models "think" first; Groq requires the reasoning to be hidden in JSON mode
        extra["reasoning_format"] = "hidden"

    def call(messages):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,                            # same prompt -> same answer
            response_format={"type": "json_object"},  # JSON mode
            extra_body=extra or None,
        )
        return resp.choices[0].message.content or ""

    return call


def _parse_json(raw: str):
    """json.loads, but tolerate ```json fences around the object."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text)


def _feedback(errors: List[str]) -> str:
    return ("Your JSON has these problems:\n- " + "\n- ".join(errors) +
            "\nFix them. Do not change any dimension the user stated. "
            'If the user\'s request itself is impossible, return {"error": "<reason>"}. '
            "Return JSON only.")


def generate_part(prompt: str, call_llm: LLMCaller, max_attempts: int = 3) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    log, errors = [], []

    for attempt in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        raw = call_llm(messages)
        latency = round(time.perf_counter() - t0, 2)

        # Step 1: is it JSON at all?
        try:
            data = _parse_json(raw)
        except json.JSONDecodeError as e:
            data, errors = None, [f"Output was not valid JSON: {e}"]
        if data is not None and not isinstance(data, dict):
            data, errors = None, ["Top-level JSON must be an object."]

        # Step 2: did the model refuse?
        if isinstance(data, dict) and "error" in data:
            log.append({"attempt": attempt, "latency_s": latency, "raw": raw, "errors": []})
            return {"status": "rejected", "reason": str(data["error"]), "attempts": attempt, "log": log}

        # Step 3: schema check (Pydantic), then geometry pipeline
        result = None
        if data is not None:
            try:
                spec = parse_spec(data)
                result = run_any(spec)
                errors = result["errors"]
            except ValidationError as e:
                errors = [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]

        log.append({"attempt": attempt, "latency_s": latency, "raw": raw, "errors": errors})
        if not errors:
            return {"status": "ok", "attempts": attempt, "spec": spec, "log": log, **result}

        # Step 4: send the exact errors back and try again
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": _feedback(errors)})

    return {"status": "failed", "attempts": max_attempts, "errors": errors, "log": log}
