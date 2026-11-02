# AI CAD Copilot

Describe a flat part in plain English → get a validated 3D model (STL).

> "A 120 x 80 x 8 mm plate with four M5 clearance holes, 12 mm from each edge."

## How it works

```
English prompt
   │
   ▼
LLM (Groq GPT-OSS, JSON mode, temperature 0) ── extracts a JSON spec, does NO geometry
   │
   ▼
Pydantic schema check ── wrong types / missing fields → error list
   │
   ▼
Rule checks ── holes inside the part (2 mm margin), holes not touching (2 mm web),
   │            known bolt sizes (ISO 273 medium-fit clearance)
   ▼
Build mesh ── trimesh box/cylinder minus hole cylinders (manifold3d boolean)
   │
   ▼
Mesh checks ── closed valid solid, volume = expected (0.1 %), bounding box = requested
   │
   ├── errors? → exact error text sent back to the LLM → retry (max 3)
   ▼
Streamlit: 3D view, metrics, attempt log, STL download
```

Two spec formats:

- **Plates** (`rect_plate`, `round_plate`): plates, discs, rings, tubes, with hole patterns
  `single`, `corners`, `bolt_circle`. Volume is checked against an exact hand formula.
- **Composite**: an ordered list of boxes and cylinders, each `add` (join) or `subtract` (cut).
  Covers L-shapes, T-shapes, stepped shafts, parts stacked on parts, and holes along x/y/z.

Gears, threads, springs, spheres and freeform surfaces are refused (`{"error": ...}`).

### Composite checks (none of them trust the LLM)
1. First body must be `add`; every later body must change the part
   (an `add` buried inside material or a hole in empty air is an error).
2. The result must be **one** connected, closed solid.
3. Overall size must equal the `envelope` the model declared (catches mis-placed bodies).
4. Mesh volume must agree within 2 % with an independent Monte Carlo estimate:
   200,000 random points tested against the exact box/cylinder maths, applied in the
   same add/subtract order, without using the mesh code at all.

## Design decisions

- **LLM extracts, code computes.** LLMs make arithmetic mistakes, so hole positions,
  clearance sizes and all checks are deterministic Python.
- **JSON spec, not generated code.** Running LLM-written code with `exec()` is a security
  risk and cannot be validated against the request. A fixed schema can.
- **Patterns instead of raw coordinates.** "Four corners, 12 mm from the edge" is expanded
  by code, so the model never has to calculate positions.
- **Volume check uses polygon area.** Circles are 64-sided polygons in the mesh, so the
  expected volume uses the polygon area formula, not πr².
- **Retries are capped and must not change user dimensions.** If the request itself is
  impossible, the model must refuse instead of silently "fixing" the design.

## Run (Python 3.11 or 3.12)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pytest -q                       # validator + retry-loop tests, no API key needed
```

Put your key in `.streamlit/secrets.toml` (see `secrets.toml.example`), then:

```bash
streamlit run app.py
python benchmark.py openai/gpt-oss-120b openai/gpt-oss-20b
```

The API key is read on the server only; it is never sent to the browser.

## Benchmark (12 prompts: 9 buildable, 3 that must be refused)

| Model | Correct | Correct 1st try | Avg attempts | Avg latency (s) |
|---|---|---|---|---|
| _fill from results.csv_ | | | | |

## Limitations

- Only boxes and cylinders, axis-aligned; no fillets, chamfers, threads or curved surfaces.
- Composite volume is cross-checked statistically (2 % tolerance), not exactly like plates.
- Output is a triangle mesh (STL), not an exact B-rep/STEP solid.
- Geometry checks only; no structural or stress analysis.
- The model can still misread a prompt; the benchmark measures how often.
