"""
Post-hoc validation against held-out observed bee records.
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import roc_auc_score

from config import OUTPUT_DIR, ML_CONFIG

METRIC_CRS = "EPSG:32737"


def _tier(score: float) -> str:
    if score >= ML_CONFIG["high_threshold"]:
        return "Prime"
    if score >= ML_CONFIG["optimal_threshold"]:
        return "Optimal"
    if score >= 0.50:
        return "Good"
    return "Fair/Poor"


def _precision_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return float("nan")
    k = max(1, min(k, len(y_true)))
    idx = np.argsort(score)[::-1][:k]
    return float(np.mean(y_true[idx]))


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def _distance_to_top_zones(eval_obs: gpd.GeoDataFrame,
                           grid_gdf: gpd.GeoDataFrame,
                           score_col: str,
                           thr: float) -> pd.Series:
    top = grid_gdf[grid_gdf[score_col] >= thr][["geometry"]].copy()
    if len(top) == 0 or len(eval_obs) == 0:
        return pd.Series(np.nan, index=eval_obs.index)

    eval_m = eval_obs[["geometry"]].copy().to_crs(METRIC_CRS)
    top_m = top.to_crs(METRIC_CRS)
    joined = gpd.sjoin_nearest(eval_m, top_m, how="left", distance_col="dist_m")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(eval_m.index)
    return joined["dist_m"]


def evaluate_predictions(gdf: gpd.GeoDataFrame,
                         eval_observations: gpd.GeoDataFrame,
                         negative_radius_m: float = 15_000):
    """
    Evaluate physics and ML scores against held-out observations.
    Exports:
      - outputs/validation_report.json
      - outputs/validation_by_county.csv
      - outputs/eval_observations.geojson
    """
    if eval_observations is None or len(eval_observations) == 0:
        report = {"error": "No eval observations available"}
        path = os.path.join(OUTPUT_DIR, "validation_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return report

    # 1) Map eval observations to nearest predicted grid point
    grid_cols = ["geometry", "suitability", "ml_suitability", "lat", "lon"]
    grid_use = gdf[grid_cols].copy()
    eval_use = eval_observations.copy()

    eval_m = eval_use.to_crs(METRIC_CRS)
    grid_m = grid_use.to_crs(METRIC_CRS)
    joined_eval = gpd.sjoin_nearest(
        eval_m,
        grid_m[["geometry", "suitability", "ml_suitability", "lat", "lon"]],
        how="left",
        distance_col="nearest_grid_dist_m",
    )
    joined_eval = joined_eval[~joined_eval.index.duplicated(keep="first")]

    # 2) Build negatives from far-away grid points (for AUC/Precision@K)
    dist_from_eval = gpd.sjoin_nearest(
        grid_m[["geometry", "suitability", "ml_suitability"]],
        eval_m[["geometry"]],
        how="left",
        distance_col="dist_to_eval_m",
    )
    dist_from_eval = dist_from_eval[~dist_from_eval.index.duplicated(keep="first")]
    neg_candidates = dist_from_eval[dist_from_eval["dist_to_eval_m"] > negative_radius_m].copy()

    n_pos = len(joined_eval)
    n_neg = min(max(n_pos, 200), len(neg_candidates))
    if n_neg > 0:
        neg_sample = neg_candidates.sample(n=n_neg, random_state=ML_CONFIG.get("random_state", 42))
    else:
        neg_sample = neg_candidates

    pos_df = pd.DataFrame({
        "y_true": 1,
        "suitability": joined_eval["suitability"].values,
        "ml_suitability": joined_eval["ml_suitability"].values,
    })
    neg_df = pd.DataFrame({
        "y_true": 0,
        "suitability": neg_sample["suitability"].values if len(neg_sample) else np.array([]),
        "ml_suitability": neg_sample["ml_suitability"].values if len(neg_sample) else np.array([]),
    })
    eval_bin = pd.concat([pos_df, neg_df], ignore_index=True)

    y = eval_bin["y_true"].values.astype(int)
    phy = eval_bin["suitability"].values.astype(float)
    ml = eval_bin["ml_suitability"].values.astype(float)

    k = int(max(1, n_pos))
    t_opt = ML_CONFIG["optimal_threshold"]
    t_prime = ML_CONFIG["high_threshold"]

    # 3) Tier calibration on combined eval set
    eval_bin["tier_ml"] = eval_bin["ml_suitability"].apply(_tier)
    tier_cal = (eval_bin.groupby("tier_ml")["y_true"]
                .agg(["mean", "count"]).reset_index())

    # 4) Distances from eval observations to top predicted zones
    d_opt_ml = _distance_to_top_zones(eval_use, gdf, "ml_suitability", t_opt)
    d_prime_ml = _distance_to_top_zones(eval_use, gdf, "ml_suitability", t_prime)
    d_opt_phy = _distance_to_top_zones(eval_use, gdf, "suitability", t_opt)
    d_prime_phy = _distance_to_top_zones(eval_use, gdf, "suitability", t_prime)

    # 5) County-level table
    county_col = "county" if "county" in joined_eval.columns else None
    county_df = joined_eval.copy()
    if county_col is None:
        county_df["county"] = "Unknown"
    county_summary = (county_df.groupby("county")
                      .agg(
                          n_eval=("geometry", "count"),
                          mean_ml=("ml_suitability", "mean"),
                          mean_physics=("suitability", "mean"),
                      )
                      .reset_index())
    county_summary["pct_ml_optimal_plus"] = (
        county_df.assign(hit=(county_df["ml_suitability"] >= t_opt).astype(int))
        .groupby("county")["hit"].mean().values * 100
    )
    county_summary["pct_ml_prime"] = (
        county_df.assign(hit=(county_df["ml_suitability"] >= t_prime).astype(int))
        .groupby("county")["hit"].mean().values * 100
    )

    report = {
        "n_eval_observations": int(n_pos),
        "n_background_negatives": int(n_neg),
        "physics": {
            "auc": _safe_auc(y, phy),
            "precision_at_k": _precision_at_k(y, phy, k),
            "recall_at_optimal_threshold": float(np.mean(joined_eval["suitability"].values >= t_opt)),
            "pct_eval_in_prime": float(100 * np.mean(joined_eval["suitability"].values >= t_prime)),
            "pct_eval_in_optimal_plus": float(100 * np.mean(joined_eval["suitability"].values >= t_opt)),
            "mean_dist_to_optimal_zone_km": float(np.nanmean(d_opt_phy) / 1000),
            "mean_dist_to_prime_zone_km": float(np.nanmean(d_prime_phy) / 1000),
        },
        "ml": {
            "auc": _safe_auc(y, ml),
            "precision_at_k": _precision_at_k(y, ml, k),
            "recall_at_optimal_threshold": float(np.mean(joined_eval["ml_suitability"].values >= t_opt)),
            "pct_eval_in_prime": float(100 * np.mean(joined_eval["ml_suitability"].values >= t_prime)),
            "pct_eval_in_optimal_plus": float(100 * np.mean(joined_eval["ml_suitability"].values >= t_opt)),
            "mean_dist_to_optimal_zone_km": float(np.nanmean(d_opt_ml) / 1000),
            "mean_dist_to_prime_zone_km": float(np.nanmean(d_prime_ml) / 1000),
        },
        "tier_calibration_ml": tier_cal.to_dict(orient="records"),
    }

    # 6) Exports
    report_path = os.path.join(OUTPUT_DIR, "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    county_path = os.path.join(OUTPUT_DIR, "validation_by_county.csv")
    county_summary.to_csv(county_path, index=False, float_format="%.4f")

    eval_export = joined_eval.to_crs("EPSG:4326").copy()
    eval_export["tier_ml"] = eval_export["ml_suitability"].apply(_tier)
    eval_geojson_path = os.path.join(OUTPUT_DIR, "eval_observations.geojson")
    with open(eval_geojson_path, "w", encoding="utf-8") as f:
        f.write(eval_export.to_json(drop_id=True))

    report["paths"] = {
        "validation_report_json": report_path,
        "validation_by_county_csv": county_path,
        "eval_observations_geojson": eval_geojson_path,
    }
    return report
