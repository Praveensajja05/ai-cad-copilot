"""
app.py - Streamlit front end.  Run:  streamlit run app.py
"""
import os

import plotly.graph_objects as go
import streamlit as st

from llm import generate_part, make_groq_caller

st.set_page_config(page_title="AI CAD Copilot", page_icon="⚙️", layout="wide")
st.title("AI CAD Copilot")
st.caption("Plain English → validated 3D part. Supports plates, discs, rings, tubes, and parts built "
           "from boxes and cylinders (L-shapes, stacked parts, steps, side holes). The LLM only describes "
           "the design; Python builds and checks it. Gears, threads and curved surfaces are refused.")

def server_key() -> str:
    """Key from the environment or Streamlit secrets. It stays on the server, never sent to the browser."""
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:  # no secrets file
        return ""


with st.sidebar:
    api_key = server_key()
    if api_key:
        st.caption("Using the server's API key.")
    else:
        api_key = st.text_input("Groq API key", type="password")
    model = st.selectbox("Model", ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
    max_attempts = st.slider("Max attempts", 1, 5, 3)

EXAMPLES = [
    "A 120 x 80 x 8 mm plate with four M5 clearance holes, 12 mm from each edge.",
    "L-bracket: base 100 x 40 x 8 mm with a vertical wall 8 mm thick at one end, 68 mm tall overall.",
    "A 60 x 60 x 10 mm base with a 30 mm diameter, 40 mm tall cylinder standing on its centre.",
    "A hollow cylinder, outer diameter 12 cm, inner diameter 9 cm, height 10 cm.",
    "Round plate 120 mm diameter, 10 mm thick, 30 mm centre bore, six M8 holes on a 90 mm PCD.",
    "Plate 100 by 60 mm, 10 mm thick, one 20 mm hole in the middle.",
]
choice = st.selectbox("Example prompts", EXAMPLES)
prompt = st.text_area("Describe the part", value=choice, height=90)

if st.button("Generate", type="primary"):
    if not api_key:
        st.error("Add your Groq API key in the sidebar.")
        st.stop()
    with st.spinner("Asking the model and checking the geometry..."):
        try:
            st.session_state.result = generate_part(prompt, make_groq_caller(api_key, model), max_attempts)
        except Exception as e:  # network / auth / rate-limit
            st.error(f"API call failed: {e}")
            st.stop()

res = st.session_state.get("result")
if res:
    if res["status"] == "ok":
        st.success(f"Valid part after {res['attempts']} attempt(s).")
        mesh, rep = res["mesh"], res["report"]
        left, right = st.columns([2, 1])
        with left:
            v, f = mesh.vertices, mesh.faces
            fig = go.Figure(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                                      i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                      color="#4a7bd0", flatshading=True))
            fig.update_layout(scene=dict(aspectmode="data"), height=480,
                              margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig)
        with right:
            if "hole_count" in rep:
                st.metric("Holes", rep["hole_count"])
            else:
                st.metric("Bodies / pieces", f"{rep['bodies']} / {rep['pieces']}")
            st.metric("Volume (mm³)", f"{rep['volume_mm3']:,}")
            st.metric("Volume check error", f"{rep['volume_error_pct']} %")
            st.write("Bounding box (mm):", rep["bounding_box_mm"])
            st.download_button("Download STL", data=mesh.export(file_type="stl"),
                               file_name="part.stl")
            st.json(res["spec"].model_dump(exclude_none=True))
    elif res["status"] == "rejected":
        st.warning(f"Model declined the request: {res['reason']}")
    else:
        st.error("No valid part within the attempt limit. Last problems:")
        for e in res["errors"]:
            st.write("- " + e)

    with st.expander("Attempt log (what the model said, what the validator found)"):
        for entry in res["log"]:
            st.markdown(f"**Attempt {entry['attempt']}** · {entry['latency_s']} s")
            st.code(entry["raw"], language="json")
            for e in entry["errors"]:
                st.write("- " + e)
