"""
Paste timing data below in this format:

LABEL 1.5
model_ms, energy_ms, hessian_ms, line_ms, total_ms, fullspace_ms
model_ms, energy_ms, hessian_ms, line_ms, total_ms, fullspace_ms

LABEL 2.0
...

Multiple entries under a label are averaged into one data point.
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

DATA = """
LABEL 2
4.80, 15.11, 30.65, 9.28, 61.71, 99
4.42, 14.70, 25.48, 8.28, 53.14, 63
4.51, 15.02, 31.15, 8.32, 59.29, 60

LABEL 14
26.52, 103.94, 305.05, 66.65, 502.75, 944
27.23, 102.81, 318.73, 59.65, 509.02, 695
27.98, 101.89, 300.13, 60.99, 491.62, 723

LABEL 20
38.19, 143.72, 448.32, 83.81, 714.75, 747
38.18, 153.43, 423.43, 87.51, 703.16, 732
41.24, 222.09, 548.49, 124.08, 936.58, 747

LABEL 28
58.53, 201.50, 586.45, 114.28, 961.48, 1207
57.11, 193.86, 608.53, 114.67, 974.75, 1121
54.26, 199.09, 600.67, 112.79, 967.41, 1207

"""

COLS = ["model", "energy", "hessian", "line", "total", "fullspace"]

# Parse
groups = defaultdict(list)
current_label = None
for line in DATA.strip().splitlines():
    line = line.strip()
    if not line:
        continue
    if line.upper().startswith("LABEL"):
        current_label = float(line.split()[1])
    elif current_label is not None:
        vals = [float(v.strip()) for v in line.split(",")]
        assert len(vals) == len(COLS), f"Expected {len(COLS)} values, got {len(vals)}: {line}"
        groups[current_label].append(vals)

xs = sorted(groups.keys())
averaged = {x: np.mean(groups[x], axis=0) for x in xs}

ys = np.array([averaged[x] for x in xs])  # shape (n_labels, n_cols)

fig, ax = plt.subplots(figsize=(8, 5))
for i, col in enumerate(COLS):
    ax.plot(xs, ys[:, i], marker="o", label=col)

ax.set_xlabel("Number of legs")
ax.set_ylabel("Time (ms)")
ax.set_title("Integrator timings vs. number of legs")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()