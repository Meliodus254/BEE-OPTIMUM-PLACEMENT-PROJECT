# 🐝 Bee Optimal Placement Project
## Machine-Learning Suitability Analysis – Kenya

---

## Project overview

This project uses a multi-criteria ML pipeline to predict the most
suitable locations for honeybee hive placement across Kenya.

**Data inputs:**
| Layer | Files | Purpose |
|-------|-------|---------|
| Temperature | `climatology-tas-…nc` | ERA5 monthly mean (°C) |
| Wind speed  | `climatology-sfcwind-…nc` | ERA5 surface wind (m/s) |
| Precipitation | `natvar-pr-…nc` | ERA5 seasonal rain (mm/yr) |
| Solar radiation | `natvar-rsds-…nc` | ERA5 shortwave down (W/m²) |
| Water bodies | `gis_osm_water_a_free_1.shp` + `gis_osm_waterways_free_1.shp` | Proximity to water |
| Land cover | `gis_osm_landuse_a_free_1.shp` + `gis_osm_natural_a_free_1.shp` | Vegetation quality |
| Roads | `gis_osm_roads_free_1.shp` | Accessibility |
| Buildings | `gis_osm_buildings_a_free_1.shp` | Urban buffer |

**ML pipeline:**
1. 0.1° (~11 km) grid sampled over Kenya
2. Climate variables interpolated via `scipy.RegularGridInterpolator`
3. Spatial distances computed with `scipy.cKDTree` (fast KD-tree)
4. Physics-based suitability score computed as weighted sum
5. Random Forest (300 trees) + Gradient Boosting (200 iters) trained
6. Ensemble prediction applied to full grid
7. Two interactive Folium HTML maps generated

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Tested on Python 3.10/3.11. If you hit conflicts, use a virtual env:

```bash
python -m venv bee_env
bee_env\Scripts\activate          # Windows
source bee_env/bin/activate        # Linux/Mac
pip install -r requirements.txt
```

### 2. Verify your data directory

Open `config.py` and check:

```python
DATA_DIR = r"C:\Users\ZUPLO\Desktop\BEE PROJECT"
```

Change this if your data is elsewhere.

### 3. Run the pipeline

```bash
cd "C:\Users\ZUPLO\Desktop\BEE PROJECT"
python main.py
```

This will take **15–40 minutes** the first time (the road and building
shapefiles are large – 1.2+ GB combined).  Use `--skip-training` on
subsequent runs to reload the saved model:

```bash
python main.py --skip-training
```

### 4. Open the maps

After the pipeline finishes, open these files in Chrome / Firefox:

```
outputs\main_map.html          ← Full feature dashboard
outputs\optimal_areas_map.html ← Clean optimal-zones map
```

You can also open `preview_map.html` **right now** (no Python needed)
to see a simulated demo of what the maps look like.

---

## File structure

```
BEE PROJECT/
├── config.py              Configuration (paths, thresholds, weights)
├── data_loader.py         Load NetCDF + shapefiles
├── feature_engineering.py Grid → features → suitability score
├── ml_model.py            RF + GB training, evaluation, plots
├── map_visualizer.py      Folium map generation
├── main.py                Entry point (run this)
├── preview_map.html       Standalone demo map (open immediately)
├── requirements.txt       Python dependencies
│
└── outputs/               (created after running main.py)
    ├── kenya_bee_suitability.csv
    ├── top20_locations.csv
    ├── bee_model.pkl
    ├── feature_importance.png
    ├── actual_vs_predicted.png
    ├── main_map.html
    └── optimal_areas_map.html
```

---

## Suitability scoring criteria

| Factor | Optimal range | Weight |
|--------|--------------|--------|
| Temperature | 20–30°C | 22% |
| Precipitation | 800–1200 mm/yr | 15% |
| Water proximity | < 400 m from river/lake | 20% |
| Solar radiation | High (normalized) | 12% |
| Wind speed | < 6.7 m/s | 13% |
| Land cover | Natural vegetation | 12% |
| Road access | Within 1–10 km | 6% |
| Building buffer | Penalty if < 200 m | ×multiplier |

**Score tiers:**
- 🔴 **Prime** (≥ 0.80): Ideal conditions
- 🟠 **Optimal** (0.65–0.79): Very good placement
- 🟡 **Good** (0.50–0.64): Suitable with management
- 🔵 **Fair** (< 0.50): Marginal or unsuitable

---

## Customising the model

Edit `config.py` to adjust:
- `GRID_RESOLUTION` – finer grid = more detail but slower (0.05° ≈ 5 km)
- `WEIGHTS` – change factor importance
- `SCORING` – adjust optimal thresholds
- `ML_CONFIG` – change n_estimators, threshold

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: NetCDF file not found` | Check file names in `config.py` match exactly |
| `MemoryError` on roads/buildings | Reduce `GRID_RESOLUTION` to 0.15 or filter to smaller region |
| Empty map | Run pipeline first; `preview_map.html` works without pipeline |
| Slow runtime | Use `--skip-training` flag after first run |

---

*Built for Kenya beekeeping site selection using ERA5 climate reanalysis
and OpenStreetMap geospatial data.*
