"""
Hydrocal Quick Start
====================

Minimal example showing how to load, visualize, and analyze
hydration calorimetry data with Hydrocal.

Before running, ensure your data directory has this structure:

    my_experiment/
    ├── methods/       # .txt method files exported from STARe
    │   ├── Method_A.txt
    │   └── Method_B.txt
    └── results/       # .csv result files exported from STARe
        └── Experiment.csv

Usage:
    python examples/quickstart.py
"""

from utils.hydrocal_preprocess import Preprocess
from utils.hydrocal_analyze import Analyze

# ── 1. Load data ──────────────────────────────────────────────────────────────
# Point Preprocess at the base directory containing results/ and methods/.
# Adjust extension, method_extension, and delimiter for your instrument output.
data = Preprocess(
    "test_data/123_hydration/",
    extension="csv",           # results file extension
    method_extension="txt",    # method file extension
    delimiter=",",             # results file delimiter (',' or '\t')
)

# ── 2. Inspect loaded data ────────────────────────────────────────────────────
# List available data files
data_files = list(data.data.keys())
print("Loaded data files:", data_files)

# List parsed methods
method_names = data.metadata.get("method_file_names", [])
print("Parsed methods:", method_names)

# Show automatic method-to-experiment assignments
print("Method assignments:", data.method_assignments)

# ── 3. Visualize ──────────────────────────────────────────────────────────────
# Plot a method profile (temperature + RH vs time)
data.plot.plot_method(method_names[0])

# Plot all experiments in a results file (overlaid)
name = data_files[0]
data.data_view(name)

# Plot a single experiment by index (0-based)
data.data_view(name, experiment=0)

# Plot a single experiment by its assigned method name
assigned = data.method_assignments.get(name, [])
if assigned:
    data.data_view(name, experiment=assigned[0])

# ── 4. Preprocessing ─────────────────────────────────────────────────────────
# Compute and plot gradients
data.data_gradient(name)

# Compute segmentwise statistics
data.data_stats(name)

# ── 5. Analysis ───────────────────────────────────────────────────────────────
analysis = Analyze(data)
# See hydrocal_v1.ipynb for inverse Laplace transform and curve fitting examples.

print("\nDone! See hydrocal_v1.ipynb for a full worked example.")
