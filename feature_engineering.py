"""
=============================================================
  BEE OPTIMUM PLACEMENT PROJECT – Feature Engineering
=============================================================
  Builds a regular grid over Kenya, interpolates climate
  variables onto each grid point, computes spatial features
  (proximity to water, roads, buildings, land-cover type),
  and calculates a multi-criteria suitability score.
=============================================================
"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from shapely.geometry import Point

from config import KENYA_BOUNDS, GRID_RESOLUTION, SCORING, WEIGHTS, LANDCOVER_SCORES

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
#  Grid creation
# ─────────────────────────────────────────────────────────

def make_grid() -> gpd.GeoDataFrame:
    """
    Return a GeoDataFrame of grid-point locations covering Kenya.
    Each point is at the centre of a GRID_RESOLUTION° cell.
    """
    lats = np.arange(KENYA_BOUNDS["lat_min"] + GRID_RESOLUTION / 2,
                     KENYA_BOUNDS["lat_max"], GRID_RESOLUTION)
    lons = np.arange(KENYA_BOUNDS["lon_min"] + GRID_RESOLUTION / 2,
                     KENYA_BOUNDS["lon_max"], GRID_RESOLUTION)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    df = pd.DataFrame({
        "lat": lat_grid.ravel(),
        "lon": lon_grid.ravel(),
    })
    df["geometry"] = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")
    print(f"Grid: {len(gdf):,} points "
          f"({len(lats)} rows × {len(lons)} cols)")
    return gdf


# ─────────────────────────────────────────────────────────
#  Climate interpolation
# ─────────────────────────────────────────────────────────

def _build_interpolator(da):
    """
    Build a scipy RegularGridInterpolator from a (lat, lon)
    xarray DataArray.
    """
    lats = da["lat"].values
    lons = da["lon"].values
    vals = da.values

    # Ensure ascending order
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        vals = vals[::-1, :]
    if lons[0] > lons[-1]:
        lons = lons[::-1]
        vals = vals[:, ::-1]

    return RegularGridInterpolator(
        (lats, lons), vals,
        method="linear", bounds_error=False, fill_value=np.nan
    )


def extract_climate_features(gdf: gpd.GeoDataFrame,
                              climate: dict) -> gpd.GeoDataFrame:
    """
    Interpolate climate variables onto each grid point.
    Adds columns: temp_c, wind_ms, precip_mm, solar_wm2
    """
    pts = np.column_stack([gdf["lat"].values, gdf["lon"].values])

    var_map = {
        "temperature":   "temp_c",
        "wind":          "wind_ms",
        "precipitation": "precip_mm",
        "solar":         "solar_wm2",
    }

    for key, col in var_map.items():
        da = climate.get(key)
        if da is None:
            gdf[col] = np.nan
            print(f"  ⚠  {key} not available → {col} = NaN")
            continue
        interp = _build_interpolator(da)
        gdf[col] = interp(pts)
        print(f"  {col}: {gdf[col].min():.2f} – {gdf[col].max():.2f}")

    return gdf


# ─────────────────────────────────────────────────────────
#  Spatial feature helpers
# ─────────────────────────────────────────────────────────

def _projected_coords(gdf: gpd.GeoDataFrame,
                      crs_to: str = "EPSG:32737") -> np.ndarray:
    """Return (N, 2) array of (x, y) in metres for KDTree queries."""
    g = gdf.to_crs(crs_to)
    return np.column_stack([g.geometry.x, g.geometry.y])


def _feature_coords(feature_gdf: gpd.GeoDataFrame,
                    crs_to: str = "EPSG:32737") -> np.ndarray:
    """
    Representative point coordinates for feature geometries.
    Works for both point and polygon/line layers.
    """
    g = feature_gdf.copy().to_crs(crs_to)
    # Use centroid for polygons/lines
    pts = g.geometry.centroid
    return np.column_stack([pts.x, pts.y])


def min_distance_m(grid_gdf: gpd.GeoDataFrame,
                   feature_gdf: gpd.GeoDataFrame,
                   crs: str = "EPSG:32737") -> np.ndarray:
    """
    Fast KDTree nearest-neighbour distance (metres) from each
    grid point to the nearest feature geometry centroid.
    Returns array of shape (N,).
    """
    if feature_gdf is None or len(feature_gdf) == 0:
        return np.full(len(grid_gdf), np.nan)

    grid_pts  = _projected_coords(grid_gdf, crs)
    feat_pts  = _feature_coords(feature_gdf, crs)
    feat_pts  = feat_pts[np.isfinite(feat_pts).all(axis=1)]

    if len(feat_pts) == 0:
        return np.full(len(grid_gdf), np.nan)

    tree = cKDTree(feat_pts)
    dist, _ = tree.query(grid_pts)
    return dist


# ─────────────────────────────────────────────────────────
#  Land-cover scoring
# ─────────────────────────────────────────────────────────

def extract_landcover_score(grid_gdf: gpd.GeoDataFrame,
                            landuse_gdf: gpd.GeoDataFrame,
                            natural_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Spatial join: each grid point → nearest land-cover polygon.
    Returns array of suitability scores [0, 1].
    """
    print("  Computing land-cover scores …")
    combined = pd.concat([landuse_gdf, natural_gdf], ignore_index=True)
    combined = gpd.GeoDataFrame(combined, crs="EPSG:4326")

    if len(combined) == 0:
        print("  ⚠  No land-cover features found → default score 0.4")
        return np.full(len(grid_gdf), 0.4)

    # Spatial join (nearest)
    joined = gpd.sjoin_nearest(
        grid_gdf[["geometry"]],
        combined[["fclass", "geometry"]].rename(
            columns={"fclass": "lc_fclass"}),
        how="left",
        max_distance=0.5,    # degrees – ~55 km
        distance_col="_dist"
    )
    joined = joined[~joined.index.duplicated(keep="first")]

    def score_fclass(fc):
        if pd.isna(fc):
            return LANDCOVER_SCORES["_default"]
        return LANDCOVER_SCORES.get(str(fc).lower(),
                                    LANDCOVER_SCORES["_default"])

    scores = joined["lc_fclass"].apply(score_fclass).values
    print(f"    Land-cover score: {scores.min():.2f} – {scores.max():.2f}")
    return scores


# ─────────────────────────────────────────────────────────
#  Individual suitability scores
# ─────────────────────────────────────────────────────────

def score_temperature(temp_c: np.ndarray) -> np.ndarray:
    s = SCORING
    score = np.where(
        (temp_c >= s["temp_optimal_min"]) & (temp_c <= s["temp_optimal_max"]),
        1.0,
        np.where(temp_c < s["temp_optimal_min"],
                 np.clip((temp_c - s["temp_penalty_min"]) /
                         (s["temp_optimal_min"] - s["temp_penalty_min"]), 0, 1),
                 np.clip(1 - (temp_c - s["temp_optimal_max"]) /
                         (s["temp_penalty_max"] - s["temp_optimal_max"]), 0, 1)
                 )
    )
    return np.where(np.isnan(temp_c), 0.5, score)


def score_wind(wind_ms: np.ndarray) -> np.ndarray:
    score = np.where(wind_ms <= SCORING["wind_max"], 1.0,
                     np.clip(1 - (wind_ms - SCORING["wind_max"]) /
                             SCORING["wind_max"], 0, 1))
    return np.where(np.isnan(wind_ms), 0.5, score)


def score_solar(solar_wm2: np.ndarray) -> np.ndarray:
    mn, mx = float(np.nanmin(solar_wm2)), float(np.nanmax(solar_wm2))
    if mx <= mn:
        return np.ones(len(solar_wm2)) * 0.5
    score = (solar_wm2 - mn) / (mx - mn)
    return np.where(np.isnan(solar_wm2), 0.5, np.clip(score, 0, 1))


def score_precipitation(precip_mm: np.ndarray) -> np.ndarray:
    s = SCORING
    opt_mid = (s["precip_optimal_min"] + s["precip_optimal_max"]) / 2
    score = np.where(
        (precip_mm >= s["precip_optimal_min"]) &
        (precip_mm <= s["precip_optimal_max"]),
        1.0,
        np.where(precip_mm < s["precip_optimal_min"],
                 np.clip((precip_mm - s["precip_hard_min"]) /
                         (s["precip_optimal_min"] - s["precip_hard_min"]), 0, 1),
                 np.clip(1 - (precip_mm - s["precip_optimal_max"]) /
                         (s["precip_hard_max"] - s["precip_optimal_max"]), 0, 1)
                 )
    )
    return np.where(np.isnan(precip_mm), 0.5, score)


def score_water_proximity(dist_m: np.ndarray) -> np.ndarray:
    opt  = SCORING["water_dist_optimal"]
    hard = SCORING["water_dist_max"]
    score = np.where(dist_m <= opt, 1.0,
                     np.clip(1 - (dist_m - opt) / (hard - opt), 0, 1))
    return np.where(np.isnan(dist_m), 0.3, score)


def score_road_access(dist_m: np.ndarray) -> np.ndarray:
    opt  = SCORING["road_dist_optimal"]
    hard = SCORING["road_dist_max"]
    score = np.where(dist_m <= opt, 1.0,
                     np.clip(1 - (dist_m - opt) / (hard - opt), 0, 1))
    return np.where(np.isnan(dist_m), 0.3, score)


def score_building_penalty(dist_m: np.ndarray) -> np.ndarray:
    """
    Penalise locations too close to buildings (urban stress).
    Far from buildings → score = 1.0 (good).
    Very close (<200 m) → score = 0.0.
    """
    buf = SCORING["building_buffer"]
    pen = SCORING["building_penalty"]
    score = np.where(dist_m >= pen, 1.0,
                     np.where(dist_m < buf, 0.0,
                              (dist_m - buf) / (pen - buf)))
    return np.where(np.isnan(dist_m), 0.8, np.clip(score, 0, 1))


# ─────────────────────────────────────────────────────────
#  Master feature-engineering function
# ─────────────────────────────────────────────────────────

def build_feature_matrix(climate: dict, spatial: dict) -> gpd.GeoDataFrame:
    """
    Full pipeline: grid → features → individual scores → composite.
    Returns an enriched GeoDataFrame ready for ML training.
    """
    print("\n" + "="*55)
    print("  FEATURE ENGINEERING")
    print("="*55)

    # 1. Grid
    gdf = make_grid()

    # 2. Climate features
    print("\n[Climate interpolation]")
    gdf = extract_climate_features(gdf, climate)

    # 3. Spatial distances
    print("\n[Spatial distance features]")
    print("  Water proximity …")
    gdf["water_dist_m"] = min_distance_m(gdf, spatial["water"])

    print("  Road proximity …")
    gdf["road_dist_m"]  = min_distance_m(gdf, spatial["roads"])

    print("  Building proximity …")
    gdf["building_dist_m"] = min_distance_m(gdf, spatial["buildings"])

    # 4. Land-cover score
    print("\n[Land-cover scoring]")
    gdf["land_cover_score"] = extract_landcover_score(
        gdf, spatial["landuse"], spatial["natural"])

    # 5. Individual suitability scores
    print("\n[Individual suitability scores]")
    gdf["s_temperature"]    = score_temperature(gdf["temp_c"].values)
    gdf["s_wind"]           = score_wind(gdf["wind_ms"].values)
    gdf["s_solar"]          = score_solar(gdf["solar_wm2"].values)
    gdf["s_precipitation"]  = score_precipitation(gdf["precip_mm"].values)
    gdf["s_water"]          = score_water_proximity(gdf["water_dist_m"].values)
    gdf["s_road"]           = score_road_access(gdf["road_dist_m"].values)
    gdf["s_building"]       = score_building_penalty(gdf["building_dist_m"].values)
    gdf["s_land_cover"]     = gdf["land_cover_score"]

    # 6. Weighted composite suitability score
    W = WEIGHTS
    gdf["suitability"] = (
        W["temperature"]     * gdf["s_temperature"]  +
        W["wind"]            * gdf["s_wind"]          +
        W["solar"]           * gdf["s_solar"]         +
        W["precipitation"]   * gdf["s_precipitation"] +
        W["water_proximity"] * gdf["s_water"]         +
        W["land_cover"]      * gdf["s_land_cover"]    +
        W["road_access"]     * gdf["s_road"]
    )
    # Apply building penalty as a multiplicative factor
    gdf["suitability"] *= gdf["s_building"]
    gdf["suitability"] = gdf["suitability"].clip(0, 1)

    print(f"\n  Composite suitability: "
          f"min={gdf['suitability'].min():.3f}  "
          f"max={gdf['suitability'].max():.3f}  "
          f"mean={gdf['suitability'].mean():.3f}")

    print("\n✓ Feature matrix ready.\n")
    return gdf


# ─────────────────────────────────────────────────────────
#  Feature column list (for ML)
# ─────────────────────────────────────────────────────────

FEATURE_COLS = [
    "temp_c", "wind_ms", "solar_wm2", "precip_mm",
    "water_dist_m", "road_dist_m", "building_dist_m",
    "s_temperature", "s_wind", "s_solar", "s_precipitation",
    "s_water", "s_road", "s_building", "s_land_cover",
]