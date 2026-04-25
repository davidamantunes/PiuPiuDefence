from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


ROOT_DIR = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT_DIR / "Engine"
ENGINE_MAIN = ENGINE_DIR / "main.py"
OUT_CSV = ENGINE_DIR / "out.csv"


if __name__ == "__main__" and get_script_run_ctx() is None:
    print("Run this app with: streamlit run GUI/app.py")
    raise SystemExit(0)


st.set_page_config(page_title="Threat Decision Viewer", layout="wide")
st.title("Threat Decision Viewer")

st.write("Run the engine and inspect the latest output CSV.")

col1, col2 = st.columns([1, 3])

with col1:
    run_clicked = st.button("Run Engine", type="primary")

if run_clicked:
    try:
        result = subprocess.run(
            [sys.executable, str(ENGINE_MAIN)],
            cwd=str(ENGINE_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        st.success("Engine completed successfully.")
        if result.stdout.strip():
            st.text_area("Engine stdout", result.stdout, height=180)
    except subprocess.CalledProcessError as exc:
        st.error("Engine failed.")
        if exc.stdout:
            st.text_area("Engine stdout", exc.stdout, height=180)
        if exc.stderr:
            st.text_area("Engine stderr", exc.stderr, height=180)

if OUT_CSV.exists():
    try:
        df = pd.read_csv(OUT_CSV)
        st.subheader("Current Output")
        st.dataframe(df, width="stretch")
    except Exception as exc:  # pragma: no cover - UI fallback
        st.error(f"Could not read out.csv: {exc}")
else:
    st.info("No output file yet. Click 'Run Engine' to generate Engine/out.csv.")
