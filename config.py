"""
=============================================================
  BEE OPTIMUM PLACEMENT PROJECT - Configuration
=============================================================
  Adjust DATA_DIR to your local project folder path.
=============================================================
"""

import os

# ── Paths ─────────────────────────────────────────────────
DATA_DIR   = r"C:\Users\ZUPLO\Desktop\BEE PROJECT"
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Kenya bounding box ────────────────────────────────────
KENYA_BOUNDS = dict(lat_min=-4.67, lat_max=4.62,
                    lon_min=33.91, lon_max=41.90)

# Grid resolution in degrees (~0.1° ≈ 11 km)
GRID_RESOLUTION = 0.1

# ── NetCDF climate file paths ─────────────────────────────
NC_FILES = {
    "wind":          os.path.join(DATA_DIR,
        "climatology-sfcwind-monthly-mean_era5-x0.25_era5-x0.25-historical"
        "_climatology_mean_1991-2020.nc"),
    "temperature":   os.path.join(DATA_DIR,
        "climatology-tas-monthly-mean_era5-x0.25_era5-x0.25-historical"
        "_climatology_mean_1991-2020.nc"),
    "precipitation": os.path.join(DATA_DIR,
        "natvar-pr-seasonal-mean_era5-x0.25_era5-x0.25-historical"
        "_climatology_mean_1991-2020.nc"),
    "solar":         os.path.join(DATA_DIR,
        "natvar-rsds-seasonal-mean_era5-x0.25_era5-x0.25-historical"
        "_climatology_mean_1991-2020.nc"),
}

# ── Shapefile paths ───────────────────────────────────────
SHP_FILES = {
    "waterways":      os.path.join(DATA_DIR, "gis_osm_waterways_free_1.shp"),
    "water":          os.path.join(DATA_DIR, "gis_osm_water_a_free_1.shp"),
    "landuse":        os.path.join(DATA_DIR, "gis_osm_landuse_a_free_1.shp"),
    "natural_areas":  os.path.join(DATA_DIR, "gis_osm_natural_a_free_1.shp"),
    "natural_points": os.path.join(DATA_DIR, "gis_osm_natural_free_1.shp"),
    "roads":          os.path.join(DATA_DIR, "gis_osm_roads_free_1.shp"),
    "buildings":      os.path.join(DATA_DIR, "gis_osm_buildings_a_free_1.shp"),
}

# ── Ecological / agronomic thresholds ────────────────────
SCORING = {
    "temp_optimal_min":  20,    # °C – lower bound of optimal range
    "temp_optimal_max":  30,    # °C – upper bound of optimal range
    "temp_penalty_min":  10,    # °C – below this → score = 0
    "temp_penalty_max":  40,    # °C – above this → score = 0
    "wind_max":           6.7,  # m/s – stressful above this
    "solar_good":       200,    # W/m² annual mean (ERA5 rsds)
    "precip_optimal_min": 800,  # mm/yr
    "precip_optimal_max": 1200, # mm/yr
    "precip_hard_min":   300,   # mm/yr (too dry)
    "precip_hard_max":  2500,   # mm/yr (too wet)
    "water_dist_optimal": 400,  # m – hive within 400 m of water
    "water_dist_max":    2000,  # m – score → 0 beyond this
    "road_dist_optimal": 1000,  # m – easy maintenance access
    "road_dist_max":    10000,  # m – too remote beyond 10 km
    "building_buffer":   200,   # m – avoid dense urban within 200 m
    "building_penalty": 1000,   # m – partial penalty zone
}

# ── Feature weights (must sum to 1.0) ────────────────────
WEIGHTS = {
    "temperature":    0.22,
    "wind":           0.13,
    "solar":          0.12,
    "precipitation":  0.15,
    "water_proximity":0.20,
    "land_cover":     0.12,
    "road_access":    0.06,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1"

# ── ML settings ───────────────────────────────────────────
ML_CONFIG = {
    "n_estimators":      300,
    "max_depth":         None,
    "random_state":      42,
    "test_size":         0.2,
    "n_samples":         8000,    # synthetic training samples
    "optimal_threshold": 0.65,    # suitability score → "optimal" label
    "high_threshold":    0.80,    # "highly optimal" label
}

# ── Map settings ──────────────────────────────────────────
MAP_CONFIG = {
    "center":      [-0.5, 37.9],  # Kenya centre (approx)
    "zoom_start":  6,
    "tile_layer":  "CartoDB positron",
}

# Land-cover score lookup (OSM fclass / landuse values)
LANDCOVER_SCORES = {
    # Natural vegetation – ideal
    "forest": 1.0, "wood": 1.0, "scrub": 0.9, "heath": 0.9,
    "grassland": 0.85, "meadow": 0.85, "wetland": 0.7,
    "orchard": 0.9,       # flowering trees, great for bees
    "vineyard": 0.85,
    # Agriculture
    "farmland": 0.6, "farm": 0.6, "allotments": 0.7,
    "greenhouse_horticulture": 0.5,
    # Urban / built-up
    "residential": 0.3, "commercial": 0.2, "industrial": 0.1,
    "retail": 0.2, "construction": 0.05,
    # Default
    "_default": 0.4,
}