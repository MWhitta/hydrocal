# Hydrocal

A Python toolkit for processing and analyzing hydration calorimetry data from TGA/DSC instruments (e.g., Mettler Toledo).

Hydrocal loads raw instrument output, parses method files describing experimental protocols (temperature ramps, relative humidity profiles, segment timing), and provides preprocessing, visualization, and analysis routines — including gradient computation, PCA, functional PCA, FFT analysis, and inverse Laplace transform fitting.

## Features

- **Data loading** — Reads multi-experiment CSV result files and TGA/DSC method files with automatic encoding handling.
- **Method parsing** — Extracts segment structure (static holds, dynamic ramps) for temperature and relative humidity from instrument method files.
- **Automatic method assignment** — Matches methods to experiments by duration, temperature profile, and RH profile similarity.
- **Preprocessing** — Normalization, gradient computation, segmentwise statistics, and FFT-based smoothing.
- **Dimensionality reduction** — PCA and functional PCA (via scikit-fda) on experiment data.
- **Analysis** — Multi-exponential fitting, inverse Laplace transforms, and Gaussian mixture modeling.
- **Visualization** — Built-in plotting for methods, raw data, gradients, statistics, PCA components, and FFT spectra.

## Installation

### From source

```bash
git clone https://github.com/mwhittaker/hydrocal.git
cd hydrocal
pip install -e .
```

### Dependencies only

```bash
pip install -r requirements.txt
```

## Quick start

```python
from utils.hydrocal_preprocess import Preprocess
from utils.hydrocal_analyze import Analyze

# Load data — expects a directory with results/ and methods/ subdirectories
data = Preprocess("path/to/experiment/", extension='csv',
                  method_extension='txt', delimiter=',')

# View method assignments (auto-detected on load)
print(data.method_assignments)

# Plot a method profile (temperature + RH over time)
data.plot.plot_method('MW2_123A')

# View all experiments in a results file
data.data_view('MW2_123')

# View a single experiment by method name or index
data.data_view('MW2_123', experiment='MW2_123A')
data.data_view('MW2_123', experiment=0)

# Compute and plot gradients
data.data_gradient('MW2_123')

# Segment statistics
data.data_stats('MW2_123')

# Run analysis (inverse Laplace transform, curve fitting)
analysis = Analyze(data)
```

See [hydrocal_v1.ipynb](hydrocal_v1.ipynb) for a full worked example.

## Project structure

```
hydrocal/
├── hydrocal_v1.ipynb          # Example notebook
├── utils/
│   ├── hydrocal_dataloader.py # Data and method file loading (Data class)
│   ├── hydrocal_preprocess.py # Preprocessing and visualization (Preprocess class)
│   ├── hydrocal_analyze.py    # Analysis routines (Analyze class)
│   └── hydrocal_plot.py       # Plotting utilities (Plot class)
└── test_data/                 # Example datasets (not tracked in git)
    └── 123_hydration/
        ├── methods/           # .txt method files
        └── results/           # .csv result files
```

## Data format

### Results files (CSV)

Comma- or tab-delimited files with two header rows (names and units), followed by data. Multiple experiments appear as repeated column groups:

```
Time,Weight,Weight,Weight,Weight,Temperature,Temperature,Temperature,Temperature,RH,RH,RH,RH
[s],[mg],[mg],[mg],[mg],[°C],[°C],[°C],[°C],[%],[%],[%],[%]
0.000000,23.99,20.58,20.07,20.02,22.21,22.63,22.67,22.43,21.50,32.90,18.60,18.20
...
```

### Method files (TXT)

Mettler Toledo STARe method exports. Each file defines segments with temperature, duration, gas flow, and relative humidity setpoints:

```
Segment 1
  Temperature   : 20.00 °C
  Duration      : 30.00 min
  Rel. humidity : 0.00 %

Segment 2
  Temperature   : 20.00 °C
  Duration      : 60.00 min
  Rel. humidity : 0.00 - 100.00 %
```

Static segments hold a constant value; dynamic segments (with `start - end` syntax) define a linear ramp.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
