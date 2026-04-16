/**
 * Bee Suitability Map
 * Heat mode: weighted Gaussian KDE (ML scores) on a grid → canvas ImageOverlay; masked over water.
 * Serve project root: python -m http.server 8765 — open http://localhost:8765/web/index.html
 */
(function () {
  "use strict";

  const qs = new URLSearchParams(window.location.search);
  const GEO_CANDIDATES = [
    qs.get("geo"),
    "data/bee_points.geojson",
    "../outputs/bee_points.geojson",
  ].filter(Boolean);
  const MAN_CANDIDATES = [
    qs.get("manifest"),
    "data/validation_report.json",
    "../outputs/validation_report.json",
    "data/summary_stats.json",
    "../outputs/summary_stats.json",
  ].filter(Boolean);
  const EVAL_OBS_CANDIDATES = [
    qs.get("eval_obs"),
    "data/eval_observations.geojson",
    "../outputs/eval_observations.geojson",
  ].filter(Boolean);
  const WATER_CANDIDATES = [
    qs.get("water"),
    "data/kenya_water.geojson",
    "../outputs/kenya_water.geojson",
  ].filter(Boolean);

  const DEFAULT_BOUNDS = [[-4.67, 33.91], [4.62, 41.9]];

  let manifest = null;
  let allFeatures = [];
  let minScore = 0;
  let maxElev = 5000;
  let mode = "dots";
  let map;
  /** Weighted KDE raster (L.ImageOverlay canvas), not drawn over OSM water polygons. */
  let heatRasterLayer = null;
  let heatBuildToken = 0;
  /** Parsed water polygons: { outer, holes, bb }[] */
  let waterPolygons = [];
  let currentTile = "dark";
  let activeSearchBBox = null;
  let userLocation = null;

  let searchPickIdx = -1;
  let searchList = [];
  let showEvalObs = false;
  let evalObsFeatures = [];
  const dotsLayer = L.layerGroup();
  const clusterGroup = L.markerClusterGroup({ maxClusterRadius: 45, spiderfyOnMaxZoom: true });
  const searchAreaLayer = L.layerGroup();
  const userLayer = L.layerGroup();
  const evalObsLayer = L.layerGroup();
  const tiles = {};

  function toast(msg, ms) {
    const el = document.getElementById("toast");
    el.textContent = msg;
    el.classList.add("show");
    setTimeout(function () { el.classList.remove("show"); }, ms || 2600);
  }

  async function fetchFirstOk(urls) {
    for (let i = 0; i < urls.length; i++) {
      const u = urls[i];
      try {
        const r = await fetch(u, { cache: "no-store" });
        if (r.ok) return { url: u, response: r };
      } catch (_e) {}
    }
    throw new Error("No data URL worked");
  }

  function scoreColor(s) {
    if (s >= 0.8) return "#ef4444";
    if (s >= 0.65) return "#f59e0b";
    if (s >= 0.5) return "#84cc16";
    if (s >= 0.35) return "#06b6d4";
    return "#64748b";
  }

  function tierLabel(s) {
    if (s >= 0.8) return "Prime";
    if (s >= 0.65) return "Optimal";
    if (s >= 0.5) return "Good";
    if (s >= 0.35) return "Fair";
    return "Poor";
  }

  function props(f) { return f.properties || {}; }
  function scoreOf(f) {
    const v = props(f).ml_suitability;
    return typeof v === "number" && !Number.isNaN(v) ? v : 0;
  }
  function elevOf(f) {
    const e = props(f).elevation_m;
    return typeof e === "number" && !Number.isNaN(e) ? e : null;
  }
  function countyOf(f) {
    const c = props(f).county;
    return typeof c === "string" && c.trim() ? c.trim() : "Unknown";
  }

  function featureInBBox(f, bb) {
    if (!bb || !f.geometry || f.geometry.type !== "Point") return false;
    const lat = f.geometry.coordinates[1];
    const lon = f.geometry.coordinates[0];
    return lat >= bb.south && lat <= bb.north && lon >= bb.west && lon <= bb.east;
  }

  function fmt(v, d) {
    if (v === null || v === undefined || (typeof v === "number" && Number.isNaN(v))) return "-";
    if (typeof v === "number" && d !== undefined) return v.toFixed(d);
    return String(v);
  }

  /** Percentile in [0,100] for contrast stretching on the heat raster. */
  function percentile(arr, p) {
    if (!arr.length) return 0;
    const s = arr.slice().sort(function (a, b) { return a - b; });
    const idx = (p / 100) * (s.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return s[lo];
    return s[lo] + (s[hi] - s[lo]) * (idx - lo);
  }

  function ringBBox(ring) {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < ring.length; i++) {
      const x = ring[i][0];
      const y = ring[i][1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    return { minX: minX, maxX: maxX, minY: minY, maxY: maxY };
  }

  function pointInRing(lng, lat, ring) {
    let inside = false;
    const n = ring.length;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const xi = ring[i][0];
      const yi = ring[i][1];
      const xj = ring[j][0];
      const yj = ring[j][1];
      const intersect = (yi > lat) !== (yj > lat) &&
        lng < ((xj - xi) * (lat - yi)) / (yj - yi + 1e-14) + xi;
      if (intersect) inside = !inside;
    }
    return inside;
  }

  function pointInPolygonWithHoles(lng, lat, poly) {
    if (!pointInRing(lng, lat, poly.outer)) return false;
    for (let h = 0; h < poly.holes.length; h++) {
      if (pointInRing(lng, lat, poly.holes[h])) return false;
    }
    return true;
  }

  function mergeWaterGeoJSON(gj) {
    const feats = (gj && gj.features) ? gj.features : [];
    for (let fi = 0; fi < feats.length; fi++) {
      const g = feats[fi].geometry;
      if (!g) continue;
      if (g.type === "Polygon") {
        pushWaterPoly(g.coordinates);
      } else if (g.type === "MultiPolygon") {
        for (let m = 0; m < g.coordinates.length; m++) {
          pushWaterPoly(g.coordinates[m]);
        }
      }
    }
    function pushWaterPoly(coords) {
      const outer = coords[0];
      const holes = coords.slice(1);
      waterPolygons.push({ outer: outer, holes: holes, bb: ringBBox(outer) });
    }
  }

  /**
   * Rough Indian Ocean wedge (Kenya bbox) when kenya_water.geojson is missing.
   * Avoids land overlap by staying east/south; pipeline export replaces this with SRTM+OSM.
   */
  function appendBuiltInOceanMask() {
    const ocean = {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          properties: { name: "Indian Ocean (approx)" },
          geometry: {
            type: "Polygon",
            coordinates: [[
              [39.2, -4.67],
              [41.9, -4.67],
              [41.9, -4.15],
              [40.2, -4.05],
              [39.2, -4.67],
            ]],
          },
        },
      ],
    };
    mergeWaterGeoJSON(ocean);
  }

  function isOverWater(lng, lat) {
    for (let i = 0; i < waterPolygons.length; i++) {
      const P = waterPolygons[i];
      if (lng < P.bb.minX || lng > P.bb.maxX || lat < P.bb.minY || lat > P.bb.maxY) continue;
      if (pointInPolygonWithHoles(lng, lat, P)) return true;
    }
    return false;
  }

  async function loadWaterMask() {
    waterPolygons = [];
    for (let i = 0; i < WATER_CANDIDATES.length; i++) {
      try {
        const r = await fetch(WATER_CANDIDATES[i], { cache: "no-store" });
        if (!r.ok) continue;
        const gj = await r.json();
        const n = (gj && gj.features) ? gj.features.length : 0;
        if (n > 0) {
          mergeWaterGeoJSON(gj);
          return;
        }
      } catch (_e) {}
    }
    appendBuiltInOceanMask();
  }

  function haversineKm(lat1, lon1, lat2, lon2) {
    const toRad = Math.PI / 180;
    const dLat = (lat2 - lat1) * toRad;
    const dLon = (lon2 - lon1) * toRad;
    const a =
      Math.sin(dLat / 2) ** 2 +
      Math.cos(lat1 * toRad) * Math.cos(lat2 * toRad) * Math.sin(dLon / 2) ** 2;
    return 6371 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }

  /** Gaussian kernel K(d/h); h in km, d in km. */
  function gaussianKernel(d, h) {
    if (h <= 0) return 0;
    return Math.exp(-(d * d) / (2 * h * h));
  }

  /**
   * Weighted KDE: f(x,y) = Σ w_i K(d_i) / Σ K(d_i) — continuous suitability surface.
   */
  function kdeWeightedValue(lat, lng, pts, hKm, cutoffKm) {
    let sumKw = 0;
    let sumK = 0;
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      const d = haversineKm(lat, lng, p.lat, p.lng);
      if (d > cutoffKm) continue;
      const k = gaussianKernel(d, hKm);
      if (k < 1e-12) continue;
      sumKw += k * p.score;
      sumK += k;
    }
    return sumK > 0 ? sumKw / sumK : 0;
  }

  function samplePointsForKde(visible, maxPts) {
    const out = [];
    const n = visible.length;
    if (n <= maxPts) {
      for (let i = 0; i < n; i++) {
        const f = visible[i];
        const lon = f.geometry.coordinates[0];
        const lat = f.geometry.coordinates[1];
        out.push({ lat: lat, lng: lon, score: scoreOf(f) });
      }
      return out;
    }
    const step = n / maxPts;
    for (let t = 0; t < maxPts; t++) {
      const f = visible[Math.min(n - 1, Math.floor(t * step))];
      const lon = f.geometry.coordinates[0];
      const lat = f.geometry.coordinates[1];
      out.push({ lat: lat, lng: lon, score: scoreOf(f) });
    }
    return out;
  }

  /** Bucket points into a coarse grid for fast neighborhood queries. */
  function bucketPoints(pts, south, west, north, east, bx, by) {
    const buckets = [];
    for (let i = 0; i < bx * by; i++) buckets[i] = [];
    const dx = (east - west) / bx;
    const dy = (north - south) / by;
    if (dx <= 0 || dy <= 0) return buckets;
    for (let p = 0; p < pts.length; p++) {
      const pt = pts[p];
      let ix = Math.floor((pt.lng - west) / dx);
      let iy = Math.floor((pt.lat - south) / dy);
      if (ix < 0) ix = 0;
      if (iy < 0) iy = 0;
      if (ix >= bx) ix = bx - 1;
      if (iy >= by) iy = by - 1;
      buckets[iy * bx + ix].push(pt);
    }
    return buckets;
  }

  function gatherPointsNear(buckets, bx, by, ix, iy, radius) {
    const acc = [];
    for (let di = -radius; di <= radius; di++) {
      for (let dj = -radius; dj <= radius; dj++) {
        const xi = ix + di;
        const yi = iy + dj;
        if (xi < 0 || yi < 0 || xi >= bx || yi >= by) continue;
        const b = buckets[yi * bx + xi];
        for (let k = 0; k < b.length; k++) acc.push(b[k]);
      }
    }
    return acc;
  }

  /**
   * Colour scale for stretched KDE values t ∈ [0,1]:
   * low → dark green, mid → yellow/orange, high → red (higher ML suitability = warmer colour).
   * Matches the idea of the sidebar legend (better sites use “hotter” colours).
   */
  function rgbaSuitability(t) {
    const v = Math.max(0, Math.min(1, t));
    let r;
    let g;
    let b;
    if (v < 0.5) {
      const u = v * 2;
      r = 20 + (234 - 20) * u;
      g = 83 + (179 - 83) * u;
      b = 45 + (8 - 45) * u;
    } else {
      const u = (v - 0.5) * 2;
      r = 234 + (239 - 234) * u;
      g = 179 + (68 - 179) * u;
      b = 8 + (68 - 8) * u;
    }
    return [Math.round(r), Math.round(g), Math.round(b), 210];
  }

  function buildHeatRasterCanvas(values, cols, rows, south, west, north, east) {
    const latStep = (north - south) / rows;
    const lonStep = (east - west) / cols;

    const rawLand = [];
    for (let row = 0; row < rows; row++) {
      const lat = south + (row + 0.5) * latStep;
      for (let col = 0; col < cols; col++) {
        const lng = west + (col + 0.5) * lonStep;
        if (waterPolygons.length && isOverWater(lng, lat)) continue;
        rawLand.push(values[row * cols + col]);
      }
    }
    const raw = rawLand.length ? rawLand : Array.prototype.slice.call(values);
    const pLo = percentile(raw, 5);
    const pHi = percentile(raw, 95);
    const span = Math.max(1e-9, pHi - pLo);

    const canvas = document.createElement("canvas");
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(cols, rows);
    const data = img.data;
    for (let row = 0; row < rows; row++) {
      const lat = south + (row + 0.5) * latStep;
      for (let col = 0; col < cols; col++) {
        const lng = west + (col + 0.5) * lonStep;
        const canvasRow = rows - 1 - row;
        const idx = (canvasRow * cols + col) * 4;
        if (isOverWater(lng, lat)) {
          data[idx] = 0;
          data[idx + 1] = 0;
          data[idx + 2] = 0;
          data[idx + 3] = 0;
          continue;
        }
        const v = values[row * cols + col];
        const t = (v - pLo) / span;
        const rgba = rgbaSuitability(t);
        data[idx] = rgba[0];
        data[idx + 1] = rgba[1];
        data[idx + 2] = rgba[2];
        data[idx + 3] = rgba[3];
      }
    }
    ctx.putImageData(img, 0, 0);
    const bounds = [
      [south, west],
      [north, east],
    ];
    return L.imageOverlay(canvas.toDataURL("image/png"), bounds, {
      opacity: 0.78,
      interactive: false,
      className: "bee-heat-layer",
    });
  }

  /**
   * Computes KDE on a grid over Kenya bounds; yields in chunks so the UI stays responsive.
   */
  function scheduleHeatRasterSurface(visible, token, onDone) {
    const south = DEFAULT_BOUNDS[0][0];
    const west = DEFAULT_BOUNDS[0][1];
    const north = DEFAULT_BOUNDS[1][0];
    const east = DEFAULT_BOUNDS[1][1];
    const cols = 84;
    const rows = 76;
    const hKm = 48;
    const cutoffKm = 3.5 * hKm;
    const bx = 28;
    const by = 26;
    const bucketRadius = 2;

    const pts = samplePointsForKde(visible, 4500);
    const buckets = bucketPoints(pts, south, west, north, east, bx, by);

    const latStep = (north - south) / rows;
    const lonStep = (east - west) / cols;
    const values = new Float32Array(cols * rows);

    let row = 0;
    const rowsPerFrame = 6;

    function frame() {
      if (token !== heatBuildToken) return;
      const end = Math.min(row + rowsPerFrame, rows);
      for (; row < end; row++) {
        const lat = south + (row + 0.5) * latStep;
        const iy = Math.min(by - 1, Math.max(0, Math.floor(((lat - south) / (north - south)) * by)));
        for (let c = 0; c < cols; c++) {
          const lng = west + (c + 0.5) * lonStep;
          if (isOverWater(lng, lat)) {
            values[row * cols + c] = 0;
            continue;
          }
          const ix = Math.min(bx - 1, Math.max(0, Math.floor(((lng - west) / (east - west)) * bx)));
          const near = gatherPointsNear(buckets, bx, by, ix, iy, bucketRadius);
          values[row * cols + c] = kdeWeightedValue(lat, lng, near, hKm, cutoffKm);
        }
      }
      if (row < rows) {
        requestAnimationFrame(frame);
      } else {
        if (token !== heatBuildToken) return;
        try {
          const layer = buildHeatRasterCanvas(values, cols, rows, south, west, north, east);
          onDone(layer);
        } catch (e) {
          console.error(e);
          onDone(null);
        }
      }
    }
    requestAnimationFrame(frame);
  }

  function routeLink(lat, lon) {
    if (!userLocation) return "";
    const u = "https://www.openstreetmap.org/directions?engine=fossgis_osrm_car" +
      `&route=${userLocation.lat}%2C${userLocation.lon}%3B${lat}%2C${lon}`;
    return `<div style="margin-top:8px"><a href="${u}" target="_blank" rel="noopener" style="color:#f2a900;text-decoration:none;font-weight:600">Route from my location ↗</a></div>`;
  }

  function popupHtml(f) {
    const p = props(f);
    const s = scoreOf(f);
    const lat = f.geometry.coordinates[1];
    const lon = f.geometry.coordinates[0];
    return (
      '<div style="min-width:220px;font-family:system-ui,sans-serif;color:#eef4fb">' +
      `<div style="font-size:22px;font-weight:700;color:${scoreColor(s)}">${s.toFixed(3)}</div>` +
      `<div style="font-size:11px;color:#99abc2;margin-bottom:8px">${tierLabel(s)}</div>` +
      '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:12px">' +
      `<div>Elevation<br><b>${fmt(p.elevation_m, 0)} m</b></div>` +
      `<div>Temp<br><b>${fmt(p.temp_c, 1)} C</b></div>` +
      `<div>Rain<br><b>${fmt(p.precip_mm, 0)} mm</b></div>` +
      `<div>Wind<br><b>${fmt(p.wind_ms, 1)} m/s</b></div>` +
      `<div>Water dist<br><b>${fmt(p.water_dist_m, 0)} m</b></div>` +
      `<div>Road dist<br><b>${fmt(p.road_dist_m, 0)} m</b></div>` +
      "</div>" +
      routeLink(lat, lon) +
      "</div>"
    );
  }

  function filterFeatures() {
    return allFeatures.filter(function (f) {
      const s = scoreOf(f);
      const el = elevOf(f);
      if (s < minScore) return false;
      if (el === null || el === undefined || isNaN(el) || el <= 0 || el > maxElev) return false;
      return true;
    });
  }

  function safeRemove(layer) {
    if (layer && map && map.hasLayer(layer)) map.removeLayer(layer);
  }

  function rebuildLayers() {
    const visible = filterFeatures();
    dotsLayer.clearLayers();
    clusterGroup.clearLayers();
    safeRemove(heatRasterLayer);
    heatRasterLayer = null;

    allFeatures.forEach(function (f) {
      const [lon, lat] = f.geometry.coordinates;
      const s = scoreOf(f);
      const col = scoreColor(s);
      const inArea = activeSearchBBox && featureInBBox(f, activeSearchBBox);
      const r = inArea ? (s >= 0.65 ? 9 : 7) : s >= 0.65 ? 6 : 4;
      const op = activeSearchBBox ? (inArea ? 0.92 : 0.22) : 0.8;

      const circle = L.circleMarker([lat, lon], {
        radius: r,
        color: col,
        fillColor: col,
        fillOpacity: op,
        weight: inArea ? 1 : 0,
      }).bindTooltip(`<b>${tierLabel(s)}</b> · ${s.toFixed(3)}`, {
        className: "bee-tip",
        sticky: true,
      }).bindPopup(popupHtml(f));
      dotsLayer.addLayer(circle);

      const icon = L.divIcon({
        html: `<div style="background:${col};width:11px;height:11px;border-radius:50%;border:2px solid rgba(0,0,0,.35)"></div>`,
        className: "",
        iconSize: [11, 11],
        iconAnchor: [5, 5],
      });
      clusterGroup.addLayer(L.marker([lat, lon], { icon: icon }).bindPopup(popupHtml(f)));
    });

    const token = ++heatBuildToken;
    if (visible.length) {
      scheduleHeatRasterSurface(visible, token, function (layer) {
        if (token !== heatBuildToken) return;
        heatRasterLayer = layer;
        if (mode === "heat") applyMode();
      });
    }

    updateStats(visible);
    applyMode();
    rebuildEvalObsLayer();
  }

  function applyMode() {
    safeRemove(dotsLayer);
    safeRemove(clusterGroup);
    safeRemove(heatRasterLayer);
    if (mode === "dots") map.addLayer(dotsLayer);
    else if (mode === "cluster") map.addLayer(clusterGroup);
    else if (mode === "heat") {
      if (heatRasterLayer) map.addLayer(heatRasterLayer);
    }
  }

  function updateStats(visible) {
    const primeT = (manifest && manifest.thresholds && manifest.thresholds.prime) || 0.8;
    const optT = (manifest && manifest.thresholds && manifest.thresholds.optimal) || 0.65;
    let prime = 0;
    let opt = 0;
    let sum = 0;
    visible.forEach(function (f) {
      const s = scoreOf(f);
      sum += s;
      if (s >= primeT) prime += 1;
      if (s >= optT) opt += 1;
    });
    document.getElementById("st-prime").textContent = prime.toLocaleString();
    document.getElementById("st-opt").textContent = opt.toLocaleString();
    document.getElementById("st-visible").textContent = visible.length.toLocaleString();
    document.getElementById("st-mean").textContent = visible.length ? (sum / visible.length).toFixed(3) : "-";
  }

  function updateValidationPanel() {
    if (!manifest || !manifest.ml) {
      document.getElementById("val-prime").textContent = "-";
      document.getElementById("val-opt").textContent = "-";
      document.getElementById("val-dist").textContent = "-";
      document.getElementById("val-n").textContent = "-";
      return;
    }
    document.getElementById("val-prime").textContent = `${manifest.ml.pct_eval_in_prime?.toFixed?.(1) ?? "-"}%`;
    document.getElementById("val-opt").textContent = `${manifest.ml.pct_eval_in_optimal_plus?.toFixed?.(1) ?? "-"}%`;
    document.getElementById("val-dist").textContent = `${manifest.ml.mean_dist_to_optimal_zone_km?.toFixed?.(1) ?? "-"} km`;
    document.getElementById("val-n").textContent = `${manifest.n_eval_observations ?? "-"}`;
  }

  function rebuildEvalObsLayer() {
    evalObsLayer.clearLayers();
    if (!showEvalObs) {
      if (map.hasLayer(evalObsLayer)) map.removeLayer(evalObsLayer);
      return;
    }

    evalObsFeatures.forEach(function (f) {
      if (!f.geometry || f.geometry.type !== "Point") return;
      const lon = f.geometry.coordinates[0];
      const lat = f.geometry.coordinates[1];
      const p = f.properties || {};
      const m = L.circleMarker([lat, lon], {
        radius: 5,
        color: "#f2a900",
        fillColor: "#facc15",
        fillOpacity: 0.95,
        weight: 1,
      });
      m.bindPopup(
        `<div style="font-family:system-ui,sans-serif;min-width:200px">` +
        `<b>Observed evaluation record</b><br/>` +
        `County: ${p.county || "Unknown"}<br/>` +
        `ML score near record: ${fmt(p.ml_suitability, 3)}<br/>` +
        `Physics score near record: ${fmt(p.suitability, 3)}<br/>` +
        `Nearest grid distance: ${fmt(p.nearest_grid_dist_m, 0)} m` +
        `</div>`
      );
      evalObsLayer.addLayer(m);
    });

    if (!map.hasLayer(evalObsLayer)) map.addLayer(evalObsLayer);
  }

  function buildTopList() {
    const ol = document.getElementById("top-list");
    ol.innerHTML = "";
    const sorted = filterFeatures()
      .map(function (f) { return { f: f, s: scoreOf(f) }; })
      .sort(function (a, b) { return b.s - a.s; })
      .slice(0, 25);
    sorted.forEach(function (item, idx) {
      const [lon, lat] = item.f.geometry.coordinates;
      const li = document.createElement("li");
      li.innerHTML = `<span class="rank" style="background:${scoreColor(item.s)}">${idx + 1}</span>` +
        `<div style="flex:1;min-width:0"><div style="font-size:11px;color:#99abc2">${lat.toFixed(3)}°, ${lon.toFixed(3)}°</div>` +
        `<div style="font-weight:700">${item.s.toFixed(3)}</div></div>`;
      li.addEventListener("click", function () { map.flyTo([lat, lon], 11, { duration: 1.0 }); });
      ol.appendChild(li);
    });
  }

  function updateNearestSites() {
    const list = document.getElementById("nearest-list");
    const empty = document.getElementById("nearest-empty");
    const summary = document.getElementById("nearby-summary");
    list.innerHTML = "";
    if (!userLocation) {
      empty.hidden = false;
      summary.hidden = true;
      return;
    }
    const nearestAll = filterFeatures()
      .map(function (f) {
        const [lon, lat] = f.geometry.coordinates;
        return { f: f, km: haversineKm(userLocation.lat, userLocation.lon, lat, lon) };
      })
      .sort(function (a, b) { return a.km - b.km; });

    const nearest = nearestAll.slice(0, 10);

    if (!nearest.length) {
      empty.hidden = false;
      empty.textContent = "No visible points for current filters.";
      summary.hidden = true;
      return;
    }
    empty.hidden = true;

    const c5 = nearestAll.filter(function (n) { return n.km <= 5; }).length;
    const c10 = nearestAll.filter(function (n) { return n.km <= 10; }).length;
    const c25 = nearestAll.filter(function (n) { return n.km <= 25; }).length;
    const nearestKm = nearestAll[0].km;
    summary.hidden = false;
    summary.innerHTML =
      `<b>Nearby now:</b> ${c5} within 5 km · ${c10} within 10 km · ${c25} within 25 km<br/>` +
      `<b>Closest site:</b> ${nearestKm.toFixed(2)} km away`;

    nearest.forEach(function (n, idx) {
      const [lon, lat] = n.f.geometry.coordinates;
      const s = scoreOf(n.f);
      const li = document.createElement("li");
      li.innerHTML = `<span class="rank" style="background:${scoreColor(s)}">${idx + 1}</span>` +
        `<div style="flex:1;min-width:0"><div style="font-size:11px;color:#99abc2">${n.km.toFixed(1)} km away</div>` +
        `<div style="font-weight:700">${tierLabel(s)} · ${s.toFixed(3)}</div></div>`;
      li.addEventListener("click", function () { map.flyTo([lat, lon], 12, { duration: 1.0 }); });
      list.appendChild(li);
    });
  }

  function populateCountyFilter() {
    // County filter removed - function kept for reference
  }

  function exportCsv() {
    const visible = filterFeatures();
    const header = "lat,lon,county,ml_suitability,elevation_m,temp_c,precip_mm,wind_ms,water_dist_m,road_dist_m";
    const rows = visible.map(function (f) {
      const [lon, lat] = f.geometry.coordinates;
      const p = props(f);
      return [lat, lon, countyOf(f), p.ml_suitability, p.elevation_m, p.temp_c, p.precip_mm, p.wind_ms, p.water_dist_m, p.road_dist_m]
        .map(function (v) { return v == null ? "" : v; })
        .join(",");
    });
    const blob = new Blob([header + "\n" + rows.join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "bee_visible_export.csv";
    a.click();
    URL.revokeObjectURL(a.href);
    toast("Exported " + visible.length + " points");
  }

  function initMap() {
    map = L.map("map", { zoomControl: false, preferCanvas: true }).setView([-0.5, 37.9], 6);
    L.control.zoom({ position: "bottomright" }).addTo(map);

    tiles.dark = L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", { attribution: "© OSM © CARTO", maxZoom: 19 });
    tiles.sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { attribution: "Esri", maxZoom: 19 });
    tiles.osm = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { attribution: "© OpenStreetMap", maxZoom: 19 });
    tiles.dark.addTo(map);
    searchAreaLayer.addTo(map);
    userLayer.addTo(map);
  }

  function setTile(name) {
    if (currentTile === name) return;
    if (tiles[currentTile]) tiles[currentTile].remove();
    tiles[name].addTo(map);
    currentTile = name;
    document.querySelectorAll("[data-tile]").forEach(function (b) {
      b.classList.toggle("active", b.getAttribute("data-tile") === name);
    });
  }

  function setupSearch() {
    const searchBox = document.getElementById("search-box");
    const resBox = document.getElementById("search-results");
    const searchClear = document.getElementById("search-clear");
    const searchLoading = document.getElementById("search-loading");
    let tmr;

    function clearSearchArea(silent) {
      activeSearchBBox = null;
      searchAreaLayer.clearLayers();
      document.getElementById("area-zones-panel").classList.add("hidden");
      rebuildLayers();
      if (!silent) toast("Area filter cleared");
    }

    document.getElementById("area-clear").addEventListener("click", function () {
      clearSearchArea(false);
    });

    function fmtNominatim(it) {
      const a = it.address || {};
      const primary = a.city || a.town || a.village || a.county || a.state || it.display_name.split(",")[0];
      const type = String(it.class || it.type || "place").replace(/_/g, " ");
      return { primary: primary, type: type, full: it.display_name };
    }

    function highlightSearch() {
      document.querySelectorAll(".search-result-btn").forEach(function (el, i) {
        el.setAttribute("aria-selected", i === searchPickIdx ? "true" : "false");
      });
    }

    function applySearchPick(item, title) {
      resBox.hidden = true;
      searchBox.setAttribute("aria-expanded", "false");
      searchPickIdx = -1;

      const lat = parseFloat(item.lat);
      const lon = parseFloat(item.lon);
      let south;
      let north;
      let west;
      let east;
      if (item.boundingbox && item.boundingbox.length >= 4) {
        south = parseFloat(item.boundingbox[0]);
        north = parseFloat(item.boundingbox[1]);
        west = parseFloat(item.boundingbox[2]);
        east = parseFloat(item.boundingbox[3]);
      } else {
        const d = 0.12;
        south = lat - d;
        north = lat + d;
        west = lon - d;
        east = lon + d;
      }
      activeSearchBBox = { south: south, north: north, west: west, east: east };
      searchAreaLayer.clearLayers();
      searchAreaLayer.addLayer(L.rectangle([[south, west], [north, east]], { className: "map-search-rect", interactive: false }));
      map.flyToBounds([[south, west], [north, east]], { padding: [28, 28], duration: 1.1, maxZoom: 11 });

      const inside = filterFeatures().filter(function (f) { return featureInBBox(f, activeSearchBBox); });
      const primeT = (manifest && manifest.thresholds && manifest.thresholds.prime) || 0.8;
      const optT = (manifest && manifest.thresholds && manifest.thresholds.optimal) || 0.65;
      let pr = 0;
      let op = 0;
      let sm = 0;
      inside.forEach(function (f) {
        const s = scoreOf(f);
        sm += s;
        if (s >= primeT) pr += 1;
        if (s >= optT) op += 1;
      });

      document.getElementById("area-title").textContent = title;
      document.getElementById("az-n").textContent = inside.length.toLocaleString();
      document.getElementById("az-prime").textContent = pr.toLocaleString();
      document.getElementById("az-opt").textContent = op.toLocaleString();
      document.getElementById("az-mean").textContent = inside.length ? (sm / inside.length).toFixed(3) : "-";

      const spots = document.getElementById("area-spots");
      spots.innerHTML = "";
      inside.slice().sort(function (a, b) { return scoreOf(b) - scoreOf(a); }).slice(0, 12).forEach(function (f) {
        const [lon1, lat1] = f.geometry.coordinates;
        const s = scoreOf(f);
        const row = document.createElement("button");
        row.type = "button";
        row.className = "area-spot-btn";
        row.innerHTML = `<span class="area-spot-sc" style="color:${scoreColor(s)}">${s.toFixed(2)}</span><span class="area-spot-xy">${lat1.toFixed(3)}°, ${lon1.toFixed(3)}°</span>`;
        row.addEventListener("click", function () { map.flyTo([lat1, lon1], 12, { duration: 1.0 }); });
        spots.appendChild(row);
      });

      document.getElementById("area-zones-panel").classList.remove("hidden");
      rebuildLayers();
      toast(inside.length + " bee grid points in this search area");
    }

    function renderSearchList(data) {
      resBox.innerHTML = "";
      searchList = data;
      data.forEach(function (item, idx) {
        const meta = fmtNominatim(item);
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "search-result-btn";
        btn.setAttribute("role", "option");
        btn.innerHTML = `<span class="sr-type">${meta.type}</span><span class="sr-title">${meta.primary}</span><span class="sr-sub">${meta.full}</span>`;
        btn.addEventListener("mouseenter", function () { searchPickIdx = idx; highlightSearch(); });
        btn.addEventListener("click", function () { applySearchPick(item, meta.primary); searchBox.value = meta.primary; });
        resBox.appendChild(btn);
      });
      searchPickIdx = data.length ? 0 : -1;
      highlightSearch();
    }

    searchBox.addEventListener("input", function () {
      searchClear.hidden = !searchBox.value.length;
      clearTimeout(tmr);
      const q = searchBox.value.trim();
      if (q.length < 2) {
        resBox.hidden = true;
        searchLoading.hidden = true;
        return;
      }
      tmr = setTimeout(function () {
        searchLoading.hidden = false;
        const url = "https://nominatim.openstreetmap.org/search?" + new URLSearchParams({ q: q + ", Kenya", format: "json", limit: "8", addressdetails: "1" });
        fetch(url, { headers: { "Accept-Language": "en" } })
          .then(function (r) { return r.json(); })
          .then(function (data) {
            searchLoading.hidden = true;
            if (!data || !data.length) {
              resBox.innerHTML = '<div class="search-empty">No matches. Try another name.</div>';
              resBox.hidden = false;
              searchList = [];
              return;
            }
            renderSearchList(data);
            resBox.hidden = false;
            searchBox.setAttribute("aria-expanded", "true");
          })
          .catch(function () {
            searchLoading.hidden = true;
            resBox.hidden = true;
          });
      }, 320);
    });

    searchClear.addEventListener("click", function () {
      searchBox.value = "";
      searchClear.hidden = true;
      resBox.hidden = true;
      searchBox.focus();
    });

    searchBox.addEventListener("keydown", function (e) {
      const open = !resBox.hidden && searchList.length;
      if (e.key === "Escape") {
        resBox.hidden = true;
        searchBox.setAttribute("aria-expanded", "false");
        return;
      }
      if (!open) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        searchPickIdx = Math.min(searchPickIdx + 1, searchList.length - 1);
        highlightSearch();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        searchPickIdx = Math.max(searchPickIdx - 1, 0);
        highlightSearch();
      } else if (e.key === "Enter" && searchPickIdx >= 0) {
        e.preventDefault();
        const item = searchList[searchPickIdx];
        const meta = fmtNominatim(item);
        applySearchPick(item, meta.primary);
        searchBox.value = meta.primary;
      }
    });

    document.addEventListener("click", function (e) {
      if (!e.target.closest(".search-wrap")) resBox.hidden = true;
    });

    return { clearSearchArea: clearSearchArea };
  }

  function setupUi(searchApi) {
    const minEl = document.getElementById("min-score");
    const minOut = document.getElementById("min-score-out");
    minEl.addEventListener("input", function () {
      minScore = minEl.valueAsNumber / 100;
      minOut.textContent = minScore.toFixed(2);
      rebuildLayers();
      buildTopList();
      updateNearestSites();
    });

    const elevEl = document.getElementById("max-elev");
    const elevOut = document.getElementById("max-elev-out");
    elevEl.addEventListener("input", function () {
      maxElev = elevEl.valueAsNumber;
      elevOut.textContent = String(maxElev);
      rebuildLayers();
      buildTopList();
      updateNearestSites();
    });

    document.querySelectorAll("[data-layer]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        mode = btn.getAttribute("data-layer");
        document.querySelectorAll("[data-layer]").forEach(function (b) {
          b.classList.toggle("active", b === btn);
        });
        applyMode();
      });
    });

    document.getElementById("toggle-eval-obs").addEventListener("click", function (e) {
      showEvalObs = !showEvalObs;
      e.target.classList.toggle("active", showEvalObs);
      e.target.textContent = showEvalObs ? "Hide eval observations" : "Show eval observations";
      rebuildEvalObsLayer();
    });

    document.querySelectorAll("[data-tile]").forEach(function (btn) {
      btn.addEventListener("click", function () { setTile(btn.getAttribute("data-tile")); });
    });

    document.getElementById("btn-export").addEventListener("click", exportCsv);

    document.getElementById("btn-fit").addEventListener("click", function () {
      searchApi.clearSearchArea(true);
      map.flyToBounds(DEFAULT_BOUNDS, { padding: [24, 24], duration: 1.2 });
    });

    document.getElementById("btn-locate").addEventListener("click", function () {
      if (!navigator.geolocation) {
        toast("Geolocation not supported");
        return;
      }
      toast("Locating...", 1400);
      navigator.geolocation.getCurrentPosition(
        function (pos) {
          const lat = pos.coords.latitude;
          const lon = pos.coords.longitude;
          userLocation = { lat: lat, lon: lon };
          userLayer.clearLayers();
          L.circle([lat, lon], {
            radius: Math.max(40, pos.coords.accuracy || 60),
            color: "#60a5fa",
            fillColor: "#60a5fa",
            fillOpacity: 0.18,
            weight: 1,
          }).addTo(userLayer);
          L.circleMarker([lat, lon], {
            radius: 8,
            color: "#60a5fa",
            fillColor: "#60a5fa",
            fillOpacity: 0.95,
            weight: 2,
          }).addTo(userLayer).bindPopup("You are here").openPopup();
          map.flyTo([lat, lon], 10, { duration: 1.1 });
          rebuildLayers();
          updateNearestSites();
          toast("Location shown + nearest sites updated");
        },
        function () { toast("Could not get location"); },
        { enableHighAccuracy: true, timeout: 12000 }
      );
    });

    let sbOpen = true;
    document.getElementById("btn-sidebar").addEventListener("click", function () {
      sbOpen = !sbOpen;
      document.getElementById("sidebar").classList.toggle("collapsed", !sbOpen);
      document.body.classList.toggle("panel-collapsed", !sbOpen);
      setTimeout(function () { map.invalidateSize(); }, 320);
    });

    document.getElementById("refresh-top").addEventListener("click", function () {
      buildTopList();
      updateNearestSites();
      toast("Lists refreshed");
    });
  }

  async function boot() {
    initMap();
    const searchApi = setupSearch();
    setupUi(searchApi);

    try {
      for (let i = 0; i < MAN_CANDIDATES.length; i++) {
        try {
          const r = await fetch(MAN_CANDIDATES[i], { cache: "no-store" });
          if (r.ok) {
            manifest = await r.json();
            break;
          }
        } catch (_ignore) {}
      }

      const geoResult = await fetchFirstOk(GEO_CANDIDATES);
      const geojson = await geoResult.response.json();
      allFeatures = (geojson.features || []).filter(function (f) {
        return f.geometry && f.geometry.type === "Point" && Array.isArray(f.geometry.coordinates);
      });

      document.getElementById("data-source").textContent = geoResult.url;

      try {
        const evalResult = await fetchFirstOk(EVAL_OBS_CANDIDATES);
        const evalGeo = await evalResult.response.json();
        evalObsFeatures = (evalGeo.features || []).filter(function (f) {
          return f.geometry && f.geometry.type === "Point" && Array.isArray(f.geometry.coordinates);
        });
      } catch (_e) {
        evalObsFeatures = [];
      }

      await loadWaterMask();

      updateValidationPanel();
      rebuildLayers();
      buildTopList();
      updateNearestSites();
      document.getElementById("loading").classList.add("hidden");
      toast("Loaded " + allFeatures.length.toLocaleString() + " points", 3200);
    } catch (e) {
      console.error(e);
      document.querySelector(".loading-text").textContent = "Could not load data files.";
      document.querySelector(".loading-hint").innerHTML =
        "From project folder run:<br><code style='color:#f2a900'>python -m http.server 8765</code><br>" +
        "Open <code>http://localhost:8765/web/index.html</code>";
      toast("Could not load data. Check server and file paths.", 5000);
    }
  }

  boot();
})();
