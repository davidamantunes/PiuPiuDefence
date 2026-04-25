import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator
from pathlib import Path
import numpy as np
from collections import defaultdict

csv_path = Path(__file__).with_name("SimOutBigBoy.csv")
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df["EnemyWeaponType"] = df["EnemyWeaponType"].astype(str).str.strip()
df["Time"] = pd.to_numeric(df["Time"], errors="coerce")

# Use true positions for visible motion; fallback to measured positions if needed.
x_col = "TrueX" if "TrueX" in df.columns else "MeasuredX"
y_col = "TrueY" if "TrueY" in df.columns else "MeasuredY"
df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
df = df.dropna(subset=["Time", x_col, y_col])

times = sorted(df["Time"].unique())
time_step_seconds = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
speed_multiplier = 1.0
frame_interval_ms = max(10, int((time_step_seconds / speed_multiplier) * 1000))

x_min = 0
x_max = 10000
y_min = 0
y_max = 10000

fig, ax = plt.subplots(figsize=(10,7))
ax.set_facecolor("#cdeccf")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect("equal", adjustable="box")
ax.xaxis.set_major_locator(MultipleLocator(1000))
ax.yaxis.set_major_locator(MultipleLocator(1000))
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.grid(True)

colors = {
    "Kalibr": "red",
    "Kinzhal": "purple",
    "Geran2": "orange"
}

scatters = {}
trail_lines = {}
history_x = defaultdict(list)
history_y = defaultdict(list)

for weapon in df["EnemyWeaponType"].unique():
    scatters[weapon] = ax.scatter([], [], s=80, label=weapon, color=colors.get(weapon, "blue"))
    trail_lines[weapon], = ax.plot([], [], color=colors.get(weapon, "blue"), alpha=0.35, linewidth=2)

ax.scatter([100], [80], s=200, color="green", marker="X", label="Target")
ax.scatter([7500], [2000], s=200, color="black", marker="s", label="Civilians")
ax.scatter([9000], [5000], s=200, color="blue", marker="s", label="Power Plants")

ax.legend()

def update(frame):
    t = times[frame]

    current = df[df["Time"] == t]

    for weapon in scatters:
        sub = current[current["EnemyWeaponType"] == weapon]

        if len(sub) > 0:
            xy = sub[[x_col, y_col]].to_numpy()
            current_x = float(xy[0][0])
            current_y = float(xy[0][1])

            history_x[weapon].append(current_x)
            history_y[weapon].append(current_y)

            scatters[weapon].set_offsets(xy)
            trail_lines[weapon].set_data(history_x[weapon], history_y[weapon])
        else:
            scatters[weapon].set_offsets(np.empty((0, 2)))
            trail_lines[weapon].set_data(history_x[weapon], history_y[weapon])

    ax.set_title(f"Battlefield Simulation   t = {t:.1f}s")
    return list(scatters.values()) + list(trail_lines.values())

ani = FuncAnimation(fig, update, frames=len(times), interval=frame_interval_ms, blit=False)

plt.show()