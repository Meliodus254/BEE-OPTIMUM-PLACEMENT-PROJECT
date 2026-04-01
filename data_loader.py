"""
=============================================================
  BEE OPTIMUM PLACEMENT PROJECT – Data Loader
=============================================================
  Loads NetCDF climate files and OSM shapefiles, clips them
  to the Kenya bounding box, and returns clean DataFrames /
  GeoDataFrames ready for feature engineering.
=============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box

from config import DATA_DIR, NC_FILES, SHP_FILES, KENYA_BOUNDS

warnings.filterwarnings("ignore")

KENYA_BOX = box(
    KENYA_BOUNDS["lon_min"], KENYA_BOUNDS["lat_min"],
    KENYA_BOUNDS["lon_max"], KENYA_BOUNDS["lat_max"],
)


# ─────────────────────────────────────────────────────────
#  NetCDF helpers
# ─────────────────────────────────────────────────────────

def _detect_coord_names(ds):
    """Return (lat_name, lon_name) from a dataset."""
    lat_candidates = ["lat", "latitude", "y", "nav_lat"]
    lon_candidates = ["lon", "longitude", "x", "nav_lon"]
    lat = next((c for c in ds.coords if c.lower() in lat_candidates), None)
    lon = next((c for c in ds.coords if c.lower() in lon_candidates), None)
    if lat is None or lon is None:
        # Try dims
        lat = next((d for d in ds.dims if d.lower() in lat_candidates), None)
        lon = next((d for d in ds.dims if d.lower() in lon_candidates), None)
    if lat is None or lon is None:
        raise ValueError(f"Cannot find lat/lon in dataset. Coords: {list(ds.coords)}")
    return lat, lon


def load_nc_annual_mean(key: str) -> xr.DataArray:
    """
    Load a NetCDF file and return the annual-mean DataArray
    clipped to the Kenya bounding box.

    Parameters
    ----------
    key : str  One of 'wind', 'temperature', 'precipitation', 'solar'

    Returns
    -------
    xr.DataArray with dims (lat, lon)
    """
    path = NC_FILES[key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"NetCDF file not found: {path}")

    print(f"  Loading {key} from {os.path.basename(path)} …")
    ds = xr.open_dataset(path)

    lat_name, lon_name = _detect_coord_names(ds)

    # Pick the first data variable (exclude coords)
    var_name = [v for v in ds.data_vars][0]
    da = ds[var_name]

    # Rename to standard names for downstream processing
    rename = {lat_name: "lat", lon_name: "lon"}
    da = da.rename(rename)

    # Clip to Kenya
    lat_mask = (da["lat"] >= KENYA_BOUNDS["lat_min"]) & \
               (da["lat"] <= KENYA_BOUNDS["lat_max"])
    lon_mask = (da["lon"] >= KENYA_BOUNDS["lon_min"]) & \
               (da["lon"] <= KENYA_BOUNDS["lon_max"])
    da = da.isel(lat=lat_mask, lon=lon_mask)

    # Collapse time / season dimension → annual mean
    time_dims = [d for d in da.dims if d.lower() in
                 ("time", "month", "season", "valid_time", "t")]
    for td in time_dims:
        da = da.mean(dim=td)

    # Unit conversions
    if key == "temperature":
        # ERA5 tas is in Kelvin → Celsius
        if float(da.mean()) > 100:
            da = da - 273.15
            da.attrs["units"] = "degC"

    if key == "precipitation":
        # ERA5 pr: kg/m²/s or mm/day → mm/yr
        units = da.attrs.get("units", "").lower()
        mean_val = float(da.mean())
        if "kg" in units or mean_val < 10:   # likely kg/m²/s
            da = da * 86_400 * 365            # → mm/yr
        elif mean_val < 100:                  # likely mm/day
            da = da * 365
        da.attrs["units"] = "mm/yr"

    if key == "wind":
        # Already in m/s – confirm
        da.attrs["units"] = "m/s"

    if key == "solar":
        # ERA5 rsds: W/m²
        da.attrs["units"] = "W/m2"

    print(f"    → {key}: min={float(da.min()):.2f}  "
          f"max={float(da.max()):.2f}  "
          f"units={da.attrs.get('units', 'unknown')}")
    return da


# ─────────────────────────────────────────────────────────
#  Shapefile helpers
# ─────────────────────────────────────────────────────────

def _load_shp(key: str, columns: list = None,
              query: str = None) -> gpd.GeoDataFrame:
    """
    Load a shapefile, reproject to EPSG:4326, clip to Kenya,
    and optionally filter by a pandas query string.
    """
    path = SHP_FILES[key]
    if not os.path.exists(path):
        print(f"  ⚠  Shapefile not found (skipping): {path}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    print(f"  Loading {key} shapefile …")
    gdf = gpd.read_file(path, bbox=(
        KENYA_BOUNDS["lon_min"], KENYA_BOUNDS["lat_min"],
        KENYA_BOUNDS["lon_max"], KENYA_BOUNDS["lat_max"],
    ))

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    if query:
        try:
            gdf = gdf.query(query)
        except Exception:
            pass  # column may not exist in every file

    if columns:
        keep = [c for c in columns if c in gdf.columns] + ["geometry"]
        gdf = gdf[keep]

    print(f"    → {len(gdf):,} features")
    return gdf


def load_water_features() -> gpd.GeoDataFrame:
    """Combined water body + waterway layer (lines & polygons)."""
    water    = _load_shp("water",
                         columns=["fclass", "name"])
    waterways = _load_shp("waterways",
                           columns=["fclass", "name"])
    combined = pd.concat([water, waterways], ignore_index=True)
    return gpd.GeoDataFrame(combined, crs="EPSG:4326")


def load_landuse() -> gpd.GeoDataFrame:
    return _load_shp("landuse", columns=["fclass", "name"])


def load_natural() -> gpd.GeoDataFrame:
    nat_a = _load_shp("natural_areas", columns=["fclass", "name"])
    nat_p = _load_shp("natural_points", columns=["fclass", "name"])
    combined = pd.concat([nat_a, nat_p], ignore_index=True)
    return gpd.GeoDataFrame(combined, crs="EPSG:4326")


def load_roads(road_types=None) -> gpd.GeoDataFrame:
    """
    Load roads; optionally filter to specific fclass values.
    Default: only major roads for accessibility analysis.
    """
    if road_types is None:
        road_types = [
            "motorway", "trunk", "primary", "secondary",
            "tertiary", "unclassified", "residential",
            "motorway_link", "trunk_link", "primary_link",
        ]
    gdf = _load_shp("roads", columns=["fclass", "name"])
    if "fclass" in gdf.columns and road_types:
        gdf = gdf[gdf["fclass"].isin(road_types)]
    print(f"    → Roads after filter: {len(gdf):,} features")
    return gdf


def load_buildings() -> gpd.GeoDataFrame:
    return _load_shp("buildings", columns=["fclass", "type"])


# ─────────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────────

def load_all():
    """
    Load all datasets and return a dict of objects.
    Use this for convenience in main pipeline.
    """
    print("\n" + "="*55)
    print("  LOADING ALL DATASETS")
    print("="*55)

    print("\n[Climate – NetCDF]")
    climate = {}
    for key in ("wind", "temperature", "precipitation", "solar"):
        try:
            climate[key] = load_nc_annual_mean(key)
        except FileNotFoundError as e:
            print(f"  ⚠  {e}")
            climate[key] = None

    print("\n[Spatial – Shapefiles]")
    spatial = {
        "water":     load_water_features(),
        "landuse":   load_landuse(),
        "natural":   load_natural(),
        "roads":     load_roads(),
        "buildings": load_buildings(),
    }

    print("\n✓ All datasets loaded.\n")
    return climate, spatial