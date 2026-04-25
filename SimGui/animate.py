import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
from pathlib import Path
import numpy as np
from matplotlib.colors import to_rgba
from collections import defaultdict

colors = {
    "Kalibr": "red",
    "Kinzhal": "purple",
    "Geran2": "orange"
}


def load_sim_data(csv_path: Path) -> pd.DataFrame:
    """Load and normalize simulation CSV data used by both animation and GUI map."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["EnemyWeaponType"] = df["EnemyWeaponType"].astype(str).str.strip()
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")

    x_col = "TrueX"
    y_col = "TrueY"
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    df = df.dropna(subset=["Time", x_col, y_col])
    df = df.dropna(subset=["ID"])
    df["ID"] = df["ID"].astype(int)
    df["WeaponKey"] = df["EnemyWeaponType"] + "_ID_" + df["ID"].astype(str)
    return df


def create_map_figure(sim_df: pd.DataFrame, selected_time: float) -> plt.Figure:
    """Create a static map for a selected time with trails up to that timestamp."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor("#cdeccf")
    ax.set_xlim(0, 10000)
    ax.set_ylim(0, 10000)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.grid(True, alpha=0.35)

    for (weapon_type, weapon_id), group in sim_df.groupby(["EnemyWeaponType", "ID"]):
        group = group.sort_values("Time")
        trail = group[group["Time"] <= selected_time]
        if trail.empty:
            continue

        color = colors.get(weapon_type, "blue")
        ax.plot(trail["TrueX"], trail["TrueY"], color=color, alpha=0.45, linewidth=2)
        latest = trail.iloc[-1]
        ax.scatter(
            [latest["TrueX"]],
            [latest["TrueY"]],
            s=70,
            color=color,
            label=f"{weapon_type} ID {weapon_id}",
        )

    ax.scatter([7500], [2000], s=200, color="black", marker="s", label="Civilians")
    ax.scatter([9000], [5000], s=200, color="blue", marker="s", label="Power Plants")
    ax.set_title(f"Battlefield Simulation   t = {selected_time:.1f}s")

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    dedup_handles = []
    dedup_labels = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            dedup_handles.append(handle)
            dedup_labels.append(label)
    ax.legend(dedup_handles, dedup_labels, loc="upper right")
    return fig


def run_animation(csv_path: Path | None = None, speed_multiplier: float = 1.0) -> None:
    """Run the matplotlib animation window."""
    csv_path = csv_path or Path(__file__).with_name("SimOut.csv")
    df = load_sim_data(csv_path)

    times = sorted(df["Time"].unique())
    time_step_seconds = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
    frame_interval_ms = max(10, int((time_step_seconds / speed_multiplier) * 1000))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#cdeccf")
    ax.set_xlim(0, 10000)
    ax.set_ylim(0, 10000)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.grid(True)

    scatters = {}
    trail_lines = {}
    history_x = defaultdict(list)
    history_y = defaultdict(list)
    max_trail_points = 40

    for weapon_key, weapon_df in df.groupby("WeaponKey"):
        weapon_type = weapon_df["EnemyWeaponType"].iloc[0]
        weapon_id = weapon_df["ID"].iloc[0]
        label = f"{weapon_type} ID {weapon_id}"
        scatters[weapon_key] = ax.scatter([], [], s=80, label=label, color=colors.get(weapon_type, "blue"))
        trail_lines[weapon_key] = LineCollection([], linewidths=[], colors=[])
        ax.add_collection(trail_lines[weapon_key])

    def build_trail(points_x, points_y, base_color):
        points = list(zip(points_x, points_y))
        if len(points) < 2:
            return [], [], []

        segments = [np.array([points[i], points[i + 1]]) for i in range(len(points) - 1)]
        segment_count = len(segments)
        linewidths = np.linspace(1.0, 4.0, segment_count)

        r, g, b, _ = to_rgba(base_color)
        colors_for_segments = []
        for index in range(segment_count):
            alpha = 0.08 + 0.92 * ((index + 1) / segment_count)
            colors_for_segments.append((r, g, b, alpha))

        return segments, linewidths, colors_for_segments

    ax.scatter([7500], [2000], s=200, color="black", marker="s", label="Civilians")
    ax.scatter([9000], [5000], s=200, color="blue", marker="s", label="Power Plants")
    ax.legend()

    def update(frame):
        t = times[frame]
        current = df[df["Time"] == t]

        for weapon_key in scatters:
            sub = current[current["WeaponKey"] == weapon_key]

            if len(sub) > 0:
                xy = sub[["TrueX", "TrueY"]].to_numpy()
                current_x = float(xy[0][0])
                current_y = float(xy[0][1])

                history_x[weapon_key].append(current_x)
                history_y[weapon_key].append(current_y)

                if len(history_x[weapon_key]) > max_trail_points:
                    history_x[weapon_key] = history_x[weapon_key][-max_trail_points:]
                    history_y[weapon_key] = history_y[weapon_key][-max_trail_points:]

                scatters[weapon_key].set_offsets(xy)

                segments, linewidths, segment_colors = build_trail(
                    history_x[weapon_key],
                    history_y[weapon_key],
                    colors.get(sub["EnemyWeaponType"].iloc[0], "blue"),
                )
                trail_lines[weapon_key].set_segments(segments)
                trail_lines[weapon_key].set_linewidths(linewidths)
                trail_lines[weapon_key].set_colors(segment_colors)
            else:
                scatters[weapon_key].set_offsets(np.empty((0, 2)))

        ax.set_title(f"Battlefield Simulation   t = {t:.1f}s")
        return list(scatters.values()) + list(trail_lines.values())

    # Keep a reference alive; otherwise Matplotlib may garbage-collect the animation.
    ani = FuncAnimation(fig, update, frames=len(times), interval=frame_interval_ms, blit=False)
    fig._ani = ani
    plt.show()


if __name__ == "__main__":
    run_animation()