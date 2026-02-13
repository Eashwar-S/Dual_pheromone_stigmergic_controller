import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === CONFIG ===
DIR = Path("output_metrics/")
FILES = [
    "centralized_appraoch_without_failure.npy",
    "centralized_appraoch_with_failure.npy",
    "stigmergy_without_failure.npy",
    "stigmergy_without_failure_with_evaporating_pheromones.npy",
    "stigmergy_with_failure.npy",
    "stigmergy_with_failure_with_evaporating_pheromones.npy",
]
OUT_PATH = DIR / "targets_over_time_comparison.png"

def pretty_label(fname: str) -> str:
    base = fname.rsplit(".", 1)[0].replace("_", " ").strip()
    base = base.replace("appraoch", "approach")  # fix typo
    # Compose a nice short label, with variants
    name = base.lower()
    if "centralized" in name:
        core = "Centralized"
    elif "stigmergy" in name:
        core = "Stigmergy"
    else:
        core = base.title()

    parts = []
    if "without failure" in name:
        parts.append("no failure")
    if "with failure" in name and "without failure" not in name:
        parts.append("failure")
    if "evaporating pheromones" in name:
        parts.append("evap. pheromones")

    suffix = ""
    if parts:
        suffix = " (" + ", ".join(parts) + ")"
    return core + suffix

plt.figure(figsize=(9, 5))
max_y = 0

for fn in FILES:
    path = DIR / fn
    if not path.exists():
        print(f"[warn] missing file: {path}")
        continue

    y = np.load(path)
    y = np.asarray(y).ravel()
    x = np.arange(len(y))
    max_y = max(max_y, float(y.max()) if y.size else 0.0)

    # "Completion step": first time the series hits its final max (if ever)
    if y.size:
        final_val = y.max()
        idxs = np.where(y == final_val)[0]
        t_complete = int(idxs[0]) if idxs.size else len(y) - 1
    else:
        final_val = 0
        t_complete = 0

    label = f"{pretty_label(fn)}  (T={t_complete}, final={int(final_val)})"
    plt.plot(x, y, linestyle=":", linewidth=2, label=label)         # dotted line
    plt.scatter([t_complete], [y[t_complete] if y.size else 0], s=25)
    plt.annotate(f"T={t_complete}",
                 xy=(t_complete, y[t_complete] if y.size else 0),
                 xytext=(6, 6), textcoords="offset points", fontsize=8)

plt.xlabel("Time step")
plt.ylabel("Total targets detected")
plt.title("Targets detected over time — centralized vs. stigmergy variants")
plt.grid(True, alpha=0.3)
plt.legend(loc="best", fontsize=9)
plt.tight_layout()

# Nice y-limit padding
if max_y > 0:
    plt.ylim(0, max_y * 1.05)

plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"[ok] saved plot to: {OUT_PATH}")
plt.show()  # uncomment if you also want an interactive window
