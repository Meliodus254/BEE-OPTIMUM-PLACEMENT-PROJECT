"""
=============================================================
  BEE OPTIMUM PLACEMENT PROJECT – ML Model
=============================================================
  Trains a Random Forest + Gradient Boosting ensemble to
  predict hive-suitability scores.  Uses the physics-based
  composite score as pseudo-labels, then learns non-linear
  interactions between features that the weighted-sum model
  cannot capture.

  Outputs:
    • Trained model saved to outputs/bee_model.pkl
    • Feature importance plot
    • Evaluation report
=============================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.inspection import permutation_importance

from config import ML_CONFIG, OUTPUT_DIR
from feature_engineering import FEATURE_COLS

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────
#  Data preparation
# ─────────────────────────────────────────────────────────

def prepare_training_data(gdf, augment: bool = True):
    """
    Prepare X, y arrays from the feature GeoDataFrame.

    Parameters
    ----------
    gdf      : GeoDataFrame returned by build_feature_matrix()
    augment  : If True, oversample high-suitability points and
               inject small random noise to improve generalisation.

    Returns
    -------
    X, y, feature_names
    """
    df = gdf.dropna(subset=FEATURE_COLS + ["suitability"]).copy()
    print(f"  Clean samples: {len(df):,}")

    X = df[FEATURE_COLS].values.astype(float)
    y = df["suitability"].values.astype(float)

    if augment:
        # Oversample top-quartile locations
        q75 = np.percentile(y, 75)
        hi_mask = y >= q75
        n_dup   = hi_mask.sum() * 2   # duplicate 2× to balance
        if n_dup > 0:
            rng   = np.random.default_rng(ML_CONFIG["random_state"])
            idx   = np.where(hi_mask)[0]
            dup_i = rng.choice(idx, size=n_dup, replace=True)
            X_dup = X[dup_i] + rng.normal(0, 0.01, (n_dup, X.shape[1]))
            y_dup = y[dup_i] + rng.normal(0, 0.005, n_dup)
            y_dup = np.clip(y_dup, 0, 1)
            X = np.vstack([X, X_dup])
            y = np.concatenate([y, y_dup])
            print(f"  After augmentation: {len(X):,} samples")

    # Shuffle
    rng  = np.random.default_rng(ML_CONFIG["random_state"])
    perm = rng.permutation(len(X))
    return X[perm], y[perm], FEATURE_COLS


# ─────────────────────────────────────────────────────────
#  Model definitions
# ─────────────────────────────────────────────────────────

def build_rf_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=ML_CONFIG["n_estimators"],
            max_depth=ML_CONFIG["max_depth"],
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=ML_CONFIG["random_state"],
        ))
    ])


def build_gb_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            min_samples_split=5,
            random_state=ML_CONFIG["random_state"],
        ))
    ])


# ─────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────

def train(gdf, augment: bool = True):
    """
    Full training pipeline.

    Returns
    -------
    dict with keys: rf_model, gb_model, ensemble, feature_names,
                    X_test, y_test, metrics
    """
    print("\n" + "="*55)
    print("  ML MODEL TRAINING")
    print("="*55)

    print("\n[Data preparation]")
    X, y, feature_names = prepare_training_data(gdf, augment=augment)

    split = ML_CONFIG["test_size"]
    seed  = ML_CONFIG["random_state"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=split, random_state=seed)
    print(f"  Train: {len(X_tr):,}   Test: {len(X_te):,}")

    # ── Random Forest ───────────────────────────────────
    print("\n[Random Forest]")
    rf = build_rf_model()
    rf.fit(X_tr, y_tr)
    y_rf = rf.predict(X_te)
    mae_rf = mean_absolute_error(y_te, y_rf)
    r2_rf  = r2_score(y_te, y_rf)
    print(f"  MAE={mae_rf:.4f}   R²={r2_rf:.4f}")

    # ── Gradient Boosting ────────────────────────────────
    print("\n[Gradient Boosting]")
    gb = build_gb_model()
    gb.fit(X_tr, y_tr)
    y_gb = gb.predict(X_te)
    mae_gb = mean_absolute_error(y_te, y_gb)
    r2_gb  = r2_score(y_te, y_gb)
    print(f"  MAE={mae_gb:.4f}   R²={r2_gb:.4f}")

    # ── Ensemble (simple average) ─────────────────────────
    y_ens = (y_rf + y_gb) / 2
    mae_ens = mean_absolute_error(y_te, y_ens)
    r2_ens  = r2_score(y_te, y_ens)
    print(f"\n[Ensemble (avg)]")
    print(f"  MAE={mae_ens:.4f}   R²={r2_ens:.4f}")

    metrics = dict(
        RF   =dict(MAE=mae_rf,  R2=r2_rf),
        GB   =dict(MAE=mae_gb,  R2=r2_gb),
        ENS  =dict(MAE=mae_ens, R2=r2_ens),
    )

    # ── Cross-validation on RF ───────────────────────────
    print("\n[5-fold CV on Random Forest (full data)]")
    cv_scores = cross_val_score(rf, X, y, cv=5,
                                scoring="r2", n_jobs=-1)
    print(f"  R² scores: {cv_scores.round(3)}")
    print(f"  Mean R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    metrics["RF"]["CV_R2_mean"] = cv_scores.mean()
    metrics["RF"]["CV_R2_std"]  = cv_scores.std()

    result = dict(
        rf_model=rf, gb_model=gb,
        feature_names=feature_names,
        X_test=X_te, y_test=y_te,
        metrics=metrics,
    )

    # ── Save ─────────────────────────────────────────────
    model_path = os.path.join(OUTPUT_DIR, "bee_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\n  Model saved → {model_path}")

    return result


# ─────────────────────────────────────────────────────────
#  Prediction
# ─────────────────────────────────────────────────────────

def predict(gdf, trained: dict) -> pd.Series:
    """
    Apply ensemble model to every grid point.
    Returns a Series of predicted suitability scores.
    """
    df = gdf[trained["feature_names"]].fillna(0)
    X  = df.values.astype(float)
    rf = trained["rf_model"]
    gb = trained["gb_model"]
    y_pred = (rf.predict(X) + gb.predict(X)) / 2
    return pd.Series(np.clip(y_pred, 0, 1), index=gdf.index,
                     name="ml_suitability")


# ─────────────────────────────────────────────────────────
#  Feature importance & evaluation plots
# ─────────────────────────────────────────────────────────

def plot_feature_importance(trained: dict):
    """Bar chart of RF feature importances."""
    rf          = trained["rf_model"]
    feat_names  = trained["feature_names"]
    importances = rf.named_steps["rf"].feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.YlOrBr(np.linspace(0.3, 0.9, len(feat_names)))
    ax.barh([feat_names[i] for i in idx[::-1]],
            importances[idx[::-1]], color=colors[::-1])
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Feature Importance – Random Forest\n"
                 "Bee Optimal Placement Model", fontsize=13, fontweight="bold")
    ax.axvline(1 / len(feat_names), ls="--", c="grey",
               alpha=0.6, label="Baseline (uniform)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "feature_importance.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance plot → {out}")


def plot_actual_vs_predicted(trained: dict):
    """Scatter: physics score vs ML prediction."""
    X_te = trained["X_test"]
    y_te = trained["y_test"]
    rf   = trained["rf_model"]
    gb   = trained["gb_model"]
    y_pr = (rf.predict(X_te) + gb.predict(X_te)) / 2

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_te, y_pr, alpha=0.3, s=15,
               c=y_te, cmap="YlOrBr", edgecolors="none")
    lim = [0, 1]
    ax.plot(lim, lim, "k--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Physics-based suitability score", fontsize=12)
    ax.set_ylabel("ML predicted suitability score",  fontsize=12)
    ax.set_title("Actual vs Predicted\n(Ensemble Model)", fontsize=13,
                 fontweight="bold")
    r2 = r2_score(y_te, y_pr)
    mae = mean_absolute_error(y_te, y_pr)
    ax.text(0.05, 0.92, f"R² = {r2:.3f}\nMAE = {mae:.4f}",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7))
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Actual vs Predicted plot → {out}")


def print_metrics(trained: dict):
    """Pretty-print evaluation metrics."""
    print("\n" + "─"*45)
    print("  MODEL EVALUATION SUMMARY")
    print("─"*45)
    for model, m in trained["metrics"].items():
        print(f"  {model:8s}:  MAE={m['MAE']:.4f}   R²={m['R2']:.4f}")
    cv = trained["metrics"]["RF"]
    if "CV_R2_mean" in cv:
        print(f"\n  RF 5-fold CV:  "
              f"R²={cv['CV_R2_mean']:.4f} ± {cv['CV_R2_std']:.4f}")
    print("─"*45 + "\n")