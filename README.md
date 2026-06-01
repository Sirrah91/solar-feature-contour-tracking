# Solar Feature Contour Tracking

A Python framework for tracking, segmenting, and analysing evolving solar
features (sunspots and pores) using contour-based methods.

The code is intended for **scientific workflows**, with an emphasis on

- reproducibility,
- clear separation of concerns, and
- publication-quality visualisation.

This repository accompanies ongoing research and is primarily designed for
expert users working with solar image data.

---

## Main features

- Contour-based tracking of evolving solar features
- Phase segmentation (forming / stable / decaying)
- Statistical analysis of physical and geometrical quantities
- Modular plotting pipelines:
  - snapshot figures,
  - PDFs,
  - animations.

---

## Repository structure (overview)

```
src/
├─ geometry/     # contour extraction and geometry utilities
├─ tracks/       # tracked feature containers and temporal linking
├─ pipelines/    # tracking and statistics pipelines
├─ plotting/     # plotting helpers and animations
├─ stats/        # statistical analysis and phase segmentation
├─ io/           # FITS and data I/O
└─ utils/        # small reusable utilities
```

Only the high-level structure is shown here; individual modules are documented
inline in the source code.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sirrah91/solar-feature-contour-tracking.git
cd solar-feature-contour-tracking
```

---

### 2. Create a Python environment (recommended)

Using conda:

```bash
conda create -n contour python=3.11
conda activate contour
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should list **exact package versions** to ensure
reproducibility.

---

## Required data and configuration

This code **does not download data automatically**.

You are expected to provide:

- calibrated solar image data (e.g. FITS files),
- metadata required for tracking and statistics,
- configuration paths inside the scripts or via a user-defined config file.

Please inspect and modify the configuration file in `src/config/paths.py` to match your local data layout.


---

## Quick start


Run contour tracking via the main pipeline script:


```bash
python run_contour_tracking.py --data_dir /path/to/fits --contour_quantity Ic --contour_level 0.9
```

Associate the tracks with their inner structure:


```bash
python run_sunspot_association.py --track_input_path /path/to/tracks/track_file.npz --inner_contour_quantity Ic --component 0.65 --component 0.5
```


Compute statistics:


```bash
python run_calc_stats.py --sunspot_input_path /path/to/sunspots/sunspot_file.npz --quantities B Bhor Br
```


Segment the temporal evolution into distinct phases


```bash
python run_split_to_phases.py --stats_dir /path/to/sunspots_stats
```


The pipeline script is designed to be configured via command-line arguments
and parameter files. It can be edited, wrapped, or extended for specific
datasets and experiments.

---

## Dependencies

Core dependencies include:

- numpy
- pandas
- scipy
- matplotlib
- scikit-image
- astropy
- sunpy
- shapely
- pwlf

See `requirements.txt` for exact versions.

---

## Citation

If this repository contributes to a scientific publication, please cite:

> Korda, D., Jurčák, J., Bello González, N., & Schmassmann, M. (2026).  
> *Equipartition field strength on the sunspot boundary: Statistical study*.  
> *Astronomy & Astrophysics*, **708**, A51.  
> https://doi.org/10.1051/0004-6361/202558637

The code in this repository was developed for the analysis presented in the paper.
