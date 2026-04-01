"""
=============================================================
  BEE OPTIMUM PLACEMENT PROJECT – Map Visualizer
=============================================================
  Generates two interactive Folium HTML maps:
    1. main_map.html      – full feature-rich dashboard
    2. optimal_areas_map.html – clean optimal-zones map
=============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import (MiniMap, Fullscreen, MousePosition,
                             MarkerCluster, HeatMap)
from branca.colormap import LinearColormap
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from config import OUTPUT_DIR, MAP_CONFIG, ML_CONFIG

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
#  Colour utilities
# ─────────────────────────────────────────────────────────

SCORE_CMAP = LinearColormap(
    colors=["#d73027", "#fc8d59", "#fee090",
            "#e0f3f8", "#91bfdb", "#4575b4"],
    vmin=0.0, vmax=1.0,
    caption="Suitability Score (0 = poor → 1 = excellent)",
)

BEE_CMAP = LinearColormap(
    colors=["#fff7bc", "#fec44f", "#fe9929",
            "#ec7014", "#cc4c02", "#8c2d04"],
    vmin=0.0, vmax=1.0,
    caption="Bee Habitat Suitability",
)

OPTIMAL_CMAP = LinearColormap(
    colors=["#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c"],
    vmin=0.6, vmax=1.0,
    caption="Optimal Zone Score (≥0.65 shown)",
)


def score_to_hex(score: float, cmap=None) -> str:
    """Convert [0,1] score → HTML hex colour."""
    if cmap is None:
        cmap_fn = cm.RdYlGn
        r, g, b, _ = cmap_fn(float(np.clip(score, 0, 1)))
    else:
        rgba = mcolors.to_rgba(cmap(score))
        r, g, b = rgba[:3]
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))


# ─────────────────────────────────────────────────────────
#  Shared base map factory
# ─────────────────────────────────────────────────────────

def _base_map(title: str) -> folium.Map:
    m = folium.Map(
        location=MAP_CONFIG["center"],
        zoom_start=MAP_CONFIG["zoom_start"],
        tiles=None,
    )
    # Tile layers
    folium.TileLayer(
        "CartoDB positron", name="Light (CartoDB Positron)",
        attr="© OpenStreetMap contributors | © CartoDB",
    ).add_to(m)
    folium.TileLayer(
        "CartoDB dark_matter", name="Dark (CartoDB Dark Matter)",
        attr="© OpenStreetMap contributors | © CartoDB",
    ).add_to(m)
    folium.TileLayer(
        "OpenStreetMap", name="OpenStreetMap",
        attr="© OpenStreetMap contributors",
    ).add_to(m)

    # Controls
    Fullscreen().add_to(m)
    MiniMap(toggle_display=True, tile_layer="CartoDB positron").add_to(m)
    MousePosition(
        position="bottomleft",
        separator=" | ",
        prefix="Lat/Lon:",
    ).add_to(m)

    # Title
    title_html = f"""
    <div style="position:fixed;top:12px;left:55px;z-index:9999;
                background:rgba(255,255,255,0.9);padding:8px 14px;
                border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.25);
                font-family:'Segoe UI',sans-serif;font-size:15px;
                font-weight:700;color:#333;">
      🐝 {title}
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))
    return m


# ─────────────────────────────────────────────────────────
#  Grid layer helper
# ─────────────────────────────────────────────────────────

def _score_to_circles(m: folium.Map, gdf: gpd.GeoDataFrame,
                      score_col: str, label: str,
                      min_score: float = 0.0,
                      radius: int = 5000,
                      opacity: float = 0.55) -> folium.FeatureGroup:
    """Add coloured circle markers to a FeatureGroup."""
    fg = folium.FeatureGroup(name=label, show=True)
    sub = gdf[gdf[score_col] >= min_score].copy()
    cmap_fn = cm.RdYlGn

    for _, row in sub.iterrows():
        sc = float(row[score_col])
        r_, g_, b_, _ = cmap_fn(sc)
        hex_c = "#{:02x}{:02x}{:02x}".format(
            int(r_*255), int(g_*255), int(b_*255))

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius / 1000,      # pixels
            color=hex_c,
            fill=True,
            fill_color=hex_c,
            fill_opacity=opacity,
            weight=0,
            popup=folium.Popup(
                f"<b>Score:</b> {sc:.3f}<br>"
                f"<b>Lat:</b> {row['lat']:.3f}  "
                f"<b>Lon:</b> {row['lon']:.3f}<br>"
                + _popup_details(row),
                max_width=280,
            ),
            tooltip=f"Score: {sc:.2f}",
        ).add_to(fg)

    fg.add_to(m)
    return fg


def _popup_details(row) -> str:
    fields = {
        "temp_c":        ("🌡 Temperature", "°C"),
        "wind_ms":       ("💨 Wind speed",  " m/s"),
        "solar_wm2":     ("☀ Solar rad.",   " W/m²"),
        "precip_mm":     ("🌧 Precip.",     " mm/yr"),
        "water_dist_m":  ("💧 Water dist.", " m"),
        "road_dist_m":   ("🛣 Road dist.",  " m"),
    }
    html = ""
    for col, (lbl, unit) in fields.items():
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                html += f"<b>{lbl}:</b> {val:.1f}{unit}<br>"
    return html


# ─────────────────────────────────────────────────────────
#  Map 1 – Full feature dashboard
# ─────────────────────────────────────────────────────────

def create_main_map(gdf: gpd.GeoDataFrame,
                    score_col: str = "ml_suitability"):
    """
    Interactive dashboard with:
      • Physics-based & ML suitability heat maps
      • Individual factor layers (toggle)
      • Legend & colour bar
    """
    print("\n[Map 1] Building main dashboard map …")
    m = _base_map("Bee Optimal Placement – Full Dashboard")

    # ── ML suitability (all points) ──────────────────────
    _score_to_circles(m, gdf, score_col,
                      label="ML Suitability (all grid)",
                      min_score=0.0, radius=7, opacity=0.5)

    # ── Physics score comparison layer ───────────────────
    if "suitability" in gdf.columns:
        _score_to_circles(m, gdf, "suitability",
                          label="Physics Score (comparison)",
                          min_score=0.0, radius=6, opacity=0.4)

    # ── Heat map ─────────────────────────────────────────
    heat_data = gdf[[score_col, "lat", "lon"]].dropna()
    heat_data = heat_data[heat_data[score_col] > 0]
    HeatMap(
        data=heat_data[["lat", "lon", score_col]].values.tolist(),
        name="Suitability Heat Map",
        min_opacity=0.3,
        radius=18,
        blur=15,
        gradient={0.0: "blue", 0.4: "lime",
                  0.7: "yellow", 1.0: "red"},
        show=False,
    ).add_to(m)

    # ── Individual factor layers ──────────────────────────
    factor_map = {
        "s_temperature":   ("🌡 Temperature Score",  False),
        "s_wind":          ("💨 Wind Score",          False),
        "s_solar":         ("☀ Solar Score",         False),
        "s_precipitation": ("🌧 Precipitation Score", False),
        "s_water":         ("💧 Water Proximity",     False),
        "s_road":          ("🛣 Road Accessibility",  False),
        "s_land_cover":    ("🌿 Land Cover",          False),
    }
    for col, (label, show) in factor_map.items():
        if col in gdf.columns:
            fg = folium.FeatureGroup(name=label, show=show)
            sub = gdf.dropna(subset=[col])
            cmap_fn = cm.YlOrBr
            for _, row in sub.iterrows():
                sc = float(row[col])
                r_, g_, b_, _ = cmap_fn(sc)
                hex_c = "#{:02x}{:02x}{:02x}".format(
                    int(r_*255), int(g_*255), int(b_*255))
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=5, color=hex_c, fill=True,
                    fill_color=hex_c, fill_opacity=0.6,
                    weight=0,
                    tooltip=f"{label}: {sc:.2f}",
                ).add_to(fg)
            fg.add_to(m)

    # ── Optimal zone outline ──────────────────────────────
    thresh = ML_CONFIG["optimal_threshold"]
    opt = gdf[gdf[score_col] >= thresh]
    if len(opt) > 0:
        fg_opt = folium.FeatureGroup(name=f"✅ Optimal Zones (≥{thresh})",
                                     show=True)
        for _, row in opt.iterrows():
            sc = float(row[score_col])
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=9,
                color="#ff6600",
                fill=True,
                fill_color="#ff9900",
                fill_opacity=0.75,
                weight=2,
                popup=folium.Popup(
                    f"<b>⭐ OPTIMAL ZONE</b><br>"
                    f"<b>ML Score:</b> {sc:.3f}<br>"
                    + _popup_details(row),
                    max_width=280,
                ),
                tooltip=f"⭐ Optimal | {sc:.2f}",
            ).add_to(fg_opt)
        fg_opt.add_to(m)

    # ── Highly optimal ────────────────────────────────────
    hi_thresh = ML_CONFIG["high_threshold"]
    hi_opt = gdf[gdf[score_col] >= hi_thresh]
    if len(hi_opt) > 0:
        fg_hi = folium.FeatureGroup(
            name=f"🌟 Highly Optimal (≥{hi_thresh})", show=True)
        for _, row in hi_opt.iterrows():
            sc = float(row[score_col])
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=12,
                color="#cc0000",
                fill=True,
                fill_color="#ff0000",
                fill_opacity=0.85,
                weight=2.5,
                popup=folium.Popup(
                    f"<b>🌟 HIGHLY OPTIMAL</b><br>"
                    f"<b>ML Score:</b> {sc:.3f}<br>"
                    + _popup_details(row),
                    max_width=280,
                ),
                tooltip=f"🌟 Highly Optimal | {sc:.2f}",
            ).add_to(fg_hi)
        fg_hi.add_to(m)

    # ── Legend ────────────────────────────────────────────
    BEE_CMAP.add_to(m)
    _add_legend(m)
    folium.LayerControl(collapsed=False).add_to(m)

    path = os.path.join(OUTPUT_DIR, "main_map.html")
    m.save(path)
    print(f"  ✓ Main map saved → {path}")
    return path


# ─────────────────────────────────────────────────────────
#  Map 2 – Optimal areas only
# ─────────────────────────────────────────────────────────

def create_optimal_areas_map(gdf: gpd.GeoDataFrame,
                              score_col: str = "ml_suitability"):
    """
    Clean, focused map showing only the optimal bee placement
    zones.  Color-coded by tier:
      🟡 Good     (0.50–0.64)
      🟠 Optimal  (0.65–0.79)
      🔴 Prime    (≥0.80)
    """
    print("\n[Map 2] Building optimal-areas map …")
    m = _base_map("Bee Placement – Optimal Areas")

    tiers = [
        (0.50, 0.64, "#f0e442", "🟡 Good (0.50–0.64)",     8,  0.65),
        (0.65, 0.79, "#e69f00", "🟠 Optimal (0.65–0.79)",  11, 0.80),
        (0.80, 1.00, "#d55e00", "🔴 Prime (≥0.80)",        14, 0.90),
    ]

    for (lo, hi, colour, label, radius, opacity) in tiers:
        sub = gdf[(gdf[score_col] >= lo) & (gdf[score_col] < hi)]
        if len(sub) == 0:
            continue
        fg = folium.FeatureGroup(name=label, show=True)
        for _, row in sub.iterrows():
            sc = float(row[score_col])
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=radius,
                color=colour,
                fill=True,
                fill_color=colour,
                fill_opacity=opacity,
                weight=1.5,
                popup=folium.Popup(
                    _build_optimal_popup(row, sc),
                    max_width=300,
                ),
                tooltip=f"{label.split()[0]} | Score: {sc:.2f}",
            ).add_to(fg)
        fg.add_to(m)

    # Cluster view for all optimal points
    opt_all = gdf[gdf[score_col] >= 0.50].copy()
    if len(opt_all) > 0:
        cluster = MarkerCluster(name="📍 Clustered view (toggle)",
                                show=False)
        for _, row in opt_all.iterrows():
            sc = float(row[score_col])
            icon_color = (
                "red"    if sc >= 0.80 else
                "orange" if sc >= 0.65 else
                "beige"
            )
            folium.Marker(
                location=[row["lat"], row["lon"]],
                icon=folium.Icon(color=icon_color, icon="home",
                                 prefix="fa"),
                popup=folium.Popup(
                    _build_optimal_popup(row, sc), max_width=300),
                tooltip=f"Score: {sc:.2f}",
            ).add_to(cluster)
        cluster.add_to(m)

    OPTIMAL_CMAP.add_to(m)
    _add_optimal_legend(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # Stats panel
    _add_stats_panel(m, gdf, score_col)

    path = os.path.join(OUTPUT_DIR, "optimal_areas_map.html")
    m.save(path)
    print(f"  ✓ Optimal areas map saved → {path}")
    return path


# ─────────────────────────────────────────────────────────
#  HTML helpers
# ─────────────────────────────────────────────────────────

def _build_optimal_popup(row, sc: float) -> str:
    tier = (
        "🔴 Prime"   if sc >= 0.80 else
        "🟠 Optimal" if sc >= 0.65 else
        "🟡 Good"
    )
    html = f"""
    <div style="font-family:'Segoe UI',sans-serif;font-size:13px;">
      <h4 style="margin:0 0 6px 0;color:#8B4513;">🐝 {tier} Location</h4>
      <b>ML Suitability:</b> {sc:.3f}<br>
      <b>Coordinates:</b> {row['lat']:.4f}°, {row['lon']:.4f}°<br>
      <hr style="margin:5px 0;border-color:#ddd;">
      {_popup_details(row)}
      <hr style="margin:5px 0;border-color:#ddd;">
      <small style="color:#666;">Click map for directions</small>
    </div>"""
    return html


def _add_legend(m: folium.Map):
    legend_html = """
    <div style="position:fixed;bottom:50px;right:10px;z-index:9999;
                background:rgba(255,255,255,0.93);padding:12px 16px;
                border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.3);
                font-family:'Segoe UI',sans-serif;font-size:12px;
                min-width:180px;">
      <b style="font-size:13px;">🐝 Suitability Key</b><br><br>
      <span style="background:#ff0000;display:inline-block;
             width:12px;height:12px;border-radius:50%;"></span>
        &nbsp;🌟 Highly Optimal (≥0.80)<br><br>
      <span style="background:#ff9900;display:inline-block;
             width:12px;height:12px;border-radius:50%;"></span>
        &nbsp;✅ Optimal (0.65–0.79)<br><br>
      <span style="background:#66cc33;display:inline-block;
             width:12px;height:12px;border-radius:50%;"></span>
        &nbsp;🟢 Good (0.50–0.64)<br><br>
      <span style="background:#3399ff;display:inline-block;
             width:12px;height:12px;border-radius:50%;"></span>
        &nbsp;🔵 Fair (0.35–0.49)<br><br>
      <span style="background:#cc0000;display:inline-block;
             width:12px;height:12px;border-radius:50%;"></span>
        &nbsp;🔴 Poor (&lt;0.35)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))


def _add_optimal_legend(m: folium.Map):
    legend_html = """
    <div style="position:fixed;bottom:50px;right:10px;z-index:9999;
                background:rgba(255,255,255,0.95);padding:14px 18px;
                border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,0.3);
                font-family:'Segoe UI',sans-serif;font-size:12px;
                min-width:200px;border-left:4px solid #e69f00;">
      <b style="font-size:14px;">🐝 Optimal Zones</b><br>
      <small style="color:#666;">Kenya Beekeeping Suitability</small><br><br>
      <div style="display:flex;align-items:center;margin:4px 0;">
        <div style="width:14px;height:14px;border-radius:50%;
                    background:#d55e00;margin-right:8px;"></div>
        <b>🔴 Prime</b>&nbsp;(score ≥ 0.80)
      </div>
      <div style="display:flex;align-items:center;margin:4px 0;">
        <div style="width:14px;height:14px;border-radius:50%;
                    background:#e69f00;margin-right:8px;"></div>
        <b>🟠 Optimal</b>&nbsp;(0.65–0.79)
      </div>
      <div style="display:flex;align-items:center;margin:4px 0;">
        <div style="width:14px;height:14px;border-radius:50%;
                    background:#f0e442;margin-right:8px;"></div>
        <b>🟡 Good</b>&nbsp;(0.50–0.64)
      </div>
      <hr style="border-color:#eee;margin:8px 0;">
      <small style="color:#444;">Click any marker for details</small>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))


def _add_stats_panel(m: folium.Map, gdf: gpd.GeoDataFrame,
                     score_col: str):
    thresh    = ML_CONFIG["optimal_threshold"]
    hi_thresh = ML_CONFIG["high_threshold"]
    total     = len(gdf)
    n_opt     = (gdf[score_col] >= thresh).sum()
    n_hi      = (gdf[score_col] >= hi_thresh).sum()
    n_good    = ((gdf[score_col] >= 0.50) & (gdf[score_col] < thresh)).sum()
    pct_opt   = 100 * n_opt / total
    mean_sc   = gdf[score_col].mean()

    stats_html = f"""
    <div style="position:fixed;top:60px;right:10px;z-index:9999;
                background:rgba(255,255,255,0.95);padding:12px 16px;
                border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.3);
                font-family:'Segoe UI',sans-serif;font-size:12px;
                min-width:210px;border-top:4px solid #e69f00;">
      <b style="font-size:13px;">📊 Analysis Summary</b><br><br>
      <b>Total grid cells:</b> {total:,}<br>
      <b>🌟 Prime zones:</b> {n_hi:,}<br>
      <b>✅ Optimal zones:</b> {n_opt:,} ({pct_opt:.1f}%)<br>
      <b>🟡 Good zones:</b> {n_good:,}<br>
      <b>Mean suitability:</b> {mean_sc:.3f}<br>
      <hr style="border-color:#eee;margin:7px 0;">
      <small style="color:#888;">Grid res: {
          __import__('config').GRID_RESOLUTION}° (~11 km)</small>
    </div>"""
    m.get_root().html.add_child(folium.Element(stats_html))


# ─────────────────────────────────────────────────────────
#  Convenience wrapper
# ─────────────────────────────────────────────────────────

def create_all_maps(gdf: gpd.GeoDataFrame,
                    score_col: str = "ml_suitability"):
    print("\n" + "="*55)
    print("  GENERATING MAPS")
    print("="*55)
    p1 = create_main_map(gdf, score_col)
    p2 = create_optimal_areas_map(gdf, score_col)
    print(f"\n✓ Both maps generated.")
    return p1, p2