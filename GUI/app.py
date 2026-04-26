from pathlib import Path
import subprocess
import sys
import time
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


ROOT_DIR = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT_DIR / "Engine"
ENGINE_MAIN = ENGINE_DIR / "main.py"
SIMGUI_DIR = ROOT_DIR / "SimGui"
SIMOUT_CSV = SIMGUI_DIR / "SimOut.csv"

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from SimGui.animate import create_map_figure, load_sim_data


if __name__ == "__main__" and get_script_run_ctx() is None:
    print("Run this app with: streamlit run GUI/app.py")
    raise SystemExit(0)


st.set_page_config(page_title="Threat Decision Viewer", layout="wide")
st.title("Threat Decision Viewer")

st.write("Run the engine and print the output table from Engine/main.py.")


def parse_engine_output(stdout_text: str) -> pd.DataFrame | None:
    """Parse the dataframe printed by Engine/main.py."""
    text = stdout_text.strip()
    if not text:
        return None

    try:
        parsed_df = pd.read_csv(StringIO(text))
    except Exception:
        return None

    if parsed_df.empty:
        return None

    return parsed_df


def render_engine_output_table(output_df: pd.DataFrame) -> None:
    """Render the engine output with a Detonate button on each row."""
    visible_df = output_df.drop(index=st.session_state.get("detonated_rows", set()), errors="ignore")

    if visible_df.empty:
        st.info("All rows were detonated.")
        return

    header_cols = st.columns([1, 2, 1])
    header_cols[0].markdown("**UAV_id**")
    header_cols[1].markdown("**Predicted_Threat_x_Damage**")
    header_cols[2].markdown("**Action**")

    for row_idx, row in visible_df.iterrows():
        row_cols = st.columns([1, 2, 1])
        row_cols[0].write(row["UAV_id"])
        row_cols[1].write(row["Predicted_Threat_x_Damage"])
        if row_cols[2].button("Detonate", key=f"detonate_{row_idx}", type="primary"):
            st.session_state.setdefault("detonated_rows", set()).add(row_idx)
            st.rerun()

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
        parsed_output_df = parse_engine_output(result.stdout)
        if parsed_output_df is not None:
            st.session_state["engine_output_df"] = parsed_output_df
            st.session_state["detonated_rows"] = set()
        else:
            st.error("Could not parse the engine output table.")
    except subprocess.CalledProcessError as exc:
        st.error("Engine failed.")
        if exc.stdout:
            st.text_area("Engine stdout", exc.stdout, height=180)
        if exc.stderr:
            st.text_area("Engine stderr", exc.stderr, height=180)

if "engine_output_df" in st.session_state:
    st.subheader("Current Output")
    render_engine_output_table(st.session_state["engine_output_df"])
else:
    st.info("No engine output yet. Click 'Run Engine' to print the table.")


st.subheader("Simulation Map")

if SIMOUT_CSV.exists():
    try:
        sim_df = load_sim_data(SIMOUT_CSV)
        times = sorted(sim_df["Time"].unique())

        if not times:
            st.info("SimOut.csv has no valid rows to plot.")
        else:
            run_simulation = st.button("Run Map Simulation", type="primary")
            _, map_col, _ = st.columns([1, 2, 1])
            map_placeholder = map_col.empty()

            # Show the latest frame when idle.
            latest_time = float(times[-1])
            fig = create_map_figure(sim_df, latest_time)
            map_placeholder.pyplot(fig, clear_figure=True, width="content")

            if run_simulation:
                time_step_seconds = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
                frame_delay = max(0.02, time_step_seconds)

                for current_time in times:
                    fig = create_map_figure(sim_df, float(current_time))
                    map_placeholder.pyplot(fig, clear_figure=True, width="content")
                    time.sleep(frame_delay)
    except Exception as exc:  # pragma: no cover - UI fallback
        st.error(f"Could not render simulation map: {exc}")
else:
    st.info("No SimGui/SimOut.csv found. Add the file to display the simulation map.")
