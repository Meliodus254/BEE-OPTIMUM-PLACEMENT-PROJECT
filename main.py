"""
=============================================================
  BEE OPTIMUM PLACEMENT PROJECT – Main Pipeline
=============================================================
  Run this script to execute the full pipeline:

    python main.py

  Steps:
    1. Load all datasets (NetCDF climate + OSM shapefiles)
    2. Build feature matrix on a Kenya grid
    3. Train ML ensemble (Random Forest + Gradient Boosting)
    4. Predict suitability on full grid
    5. Export results to CSV
    6. Generate two interactive HTML maps

  Outputs (in DATA_DIR/outputs/):
    • bee_model.pkl             – saved model
    • kenya_bee_suitability.csv – full grid with scores
    • main_map.html             – full feature dashboard map
    • optimal_areas_map.html    – clean optimal-zones map
    • feature_importance.png    – feature importance chart
    • actual_vs_predicted.png   – model evaluation plot
=============================================================
"""

import os
import sys
import time
import warnings
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")

# ── Ensure project directory is on path ──────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config   import OUTPUT_DIR, ML_CONFIG
from data_loader       import load_all
from feature_engineering import build_feature_matrix
from ml_model          import (train, predict,
                                plot_feature_importance,
                                plot_actual_vs_predicted,
                                print_metrics)
from map_visualizer    import create_all_maps


# ─────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────

def _banner(msg: str):
    print("\n" + "═"*55)
    print(f"  {msg}")
    print("═"*55)


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{int(s//60)}m {s%60:.1f}s"


# ─────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────

def run_pipeline(skip_training: bool = False):
    """
    Full end-to-end pipeline.

    Parameters
    ----------
    skip_training : bool
        If True and bee_model.pkl exists, load it instead of
        retraining (saves time on repeated runs).
    """
    t_start = time.time()

    _banner("🐝  BEE OPTIMUM PLACEMENT PROJECT")
    print(f"\n  Output directory : {OUTPUT_DIR}")

    # ── Step 1: Load data ─────────────────────────────────
    _banner("Step 1 – Loading datasets")
    t = time.time()
    climate, spatial = load_all()
    print(f"\n  → Done in {_elapsed(t)}")

    # ── Step 2: Feature engineering ───────────────────────
    _banner("Step 2 – Feature engineering")
    t = time.time()
    gdf = build_feature_matrix(climate, spatial)
    print(f"\n  → Done in {_elapsed(t)}")

    # ── Step 3: ML training ───────────────────────────────
    _banner("Step 3 – ML model training")
    t = time.time()
    model_path = os.path.join(OUTPUT_DIR, "bee_model.pkl")

    if skip_training and os.path.exists(model_path):
        import pickle
        print("  Loading saved model …")
        with open(model_path, "rb") as f:
            trained = pickle.load(f)
    else:
        trained = train(gdf, augment=True)

    print_metrics(trained)
    plot_feature_importance(trained)
    plot_actual_vs_predicted(trained)
    print(f"\n  → Done in {_elapsed(t)}")

    # ── Step 4: Prediction on full grid ───────────────────
    _banner("Step 4 – Predicting suitability (full grid)")
    t = time.time()
    gdf["ml_suitability"] = predict(gdf, trained)

    thresh    = ML_CONFIG["optimal_threshold"]
    hi_thresh = ML_CONFIG["high_threshold"]
    n_opt = (gdf["ml_suitability"] >= thresh).sum()
    n_hi  = (gdf["ml_suitability"] >= hi_thresh).sum()
    print(f"  Grid points:    {len(gdf):,}")
    print(f"  Optimal  (≥{thresh}): {n_opt:,} ({100*n_opt/len(gdf):.1f}%)")
    print(f"  Prime    (≥{hi_thresh}): {n_hi:,} ({100*n_hi/len(gdf):.1f}%)")
    print(f"\n  → Done in {_elapsed(t)}")

    # ── Step 5: Export CSV ────────────────────────────────
    _banner("Step 5 – Exporting results")
    export_cols = [
        "lat", "lon",
        "temp_c", "wind_ms", "solar_wm2", "precip_mm",
        "water_dist_m", "road_dist_m", "building_dist_m",
        "s_temperature", "s_wind", "s_solar", "s_precipitation",
        "s_water", "s_road", "s_land_cover", "s_building",
        "suitability", "ml_suitability",
    ]
    export_cols = [c for c in export_cols if c in gdf.columns]
    csv_path = os.path.join(OUTPUT_DIR, "kenya_bee_suitability.csv")
    gdf[export_cols].to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  CSV → {csv_path}")

    # Top 20 recommended locations
    top20 = (gdf.nlargest(20, "ml_suitability")
             [["lat", "lon", "ml_suitability", "suitability",
               "temp_c", "wind_ms", "precip_mm", "water_dist_m"]]
             .reset_index(drop=True))
    top20.index += 1
    top20_path = os.path.join(OUTPUT_DIR, "top20_locations.csv")
    top20.to_csv(top20_path, float_format="%.4f")
    print(f"\n  📍 Top 20 recommended locations:")
    print(top20.to_string())
    print(f"\n  Top 20 CSV → {top20_path}")

    # ── Step 6: Maps ──────────────────────────────────────
    _banner("Step 6 – Generating interactive maps")
    t = time.time()
    p1, p2 = create_all_maps(gdf, score_col="ml_suitability")
    print(f"\n  → Done in {_elapsed(t)}")

    # ── Summary ───────────────────────────────────────────
    _banner("✅  PIPELINE COMPLETE")
    print(f"\n  Total time  : {_elapsed(t_start)}")
    print(f"\n  Outputs in  : {OUTPUT_DIR}")
    print(f"    📄 kenya_bee_suitability.csv")
    print(f"    📄 top20_locations.csv")
    print(f"    🤖 bee_model.pkl")
    print(f"    📊 feature_importance.png")
    print(f"    📊 actual_vs_predicted.png")
    print(f"    🗺  main_map.html          ← Open in browser")
    print(f"    🗺  optimal_areas_map.html ← Open in browser\n")

    return gdf, trained


# ─────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Bee Optimum Placement – ML Pipeline")
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip ML training and load saved model (if it exists)")
    args = parser.parse_args()

    run_pipeline(skip_training=args.skip_training)