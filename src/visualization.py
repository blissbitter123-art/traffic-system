"""Matplotlib / Folium visualisation helpers shared by notebook and app.

Every helper returns a Figure or a Folium map; callers decide where to render
it (notebook inline, Streamlit container, saved PNG).
"""

from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless-safe default; Streamlit overrides as needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import SEVERITY_CLASSES

SEVERITY_COLORS = {"minor": "#2ecc71", "major": "#f39c12", "fatal": "#e74c3c"}


def _style_axes(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def severity_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of collision counts by severity class."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["accident_severity"].value_counts().reindex(SEVERITY_CLASSES)
    bars = ax.bar(counts.index, counts.values,
                  color=[SEVERITY_COLORS[c] for c in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 100, str(v), ha="center")
    _style_axes(ax, "Collision records by severity (full dataset)",
                "Severity class", "Number of collisions")
    fig.tight_layout()
    return fig


def time_distribution(df: pd.DataFrame) -> plt.Figure:
    """Collisions by hour of day, split by severity."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for cls in SEVERITY_CLASSES:
        sub = df[df["accident_severity"] == cls]
        ax.plot(sub.groupby("hour").size() / len(sub) * 100,
                label=cls, color=SEVERITY_COLORS[cls], marker="o", ms=3)
    _style_axes(ax, "Share of each severity class by hour of day",
                "Hour of day", "% of that class's records")
    ax.legend(title="Severity")
    fig.tight_layout()
    return fig


def monthly_trend(monthly_counts: pd.Series) -> plt.Figure:
    """City-level monthly collision totals over time."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(monthly_counts)), monthly_counts.values, marker="o", ms=3)
    step = max(1, len(monthly_counts) // 12)
    ax.set_xticks(range(0, len(monthly_counts), step))
    ax.set_xticklabels(monthly_counts.index[::step], rotation=45, ha="right")
    _style_axes(ax, "Total reported collisions per month",
                "Month", "Collisions")
    fig.tight_layout()
    return fig


def confusion_heatmap(cm: pd.DataFrame) -> plt.Figure:
    """Confusion-matrix heatmap for the severity model."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm.index)), cm.index)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm.values[i, j]), ha="center", va="center",
                    color="white" if cm.values[i, j] > cm.values.max() / 2 else "black")
    ax.set_title("Confusion matrix (held-out 2025 test set)", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def hotspot_scatter_map(hotspot_cells: pd.DataFrame) -> Any:
    """Folium map: circle size ~ total collisions, colour ~ trend label."""
    import folium
    from folium.plugins import HeatMap  # noqa: F401  (kept for notebook use)

    colour_map = {
        "Persistent hotspot": "red",
        "Emerging hotspot": "orange",
        "Stable": "blue",
        "Declining": "green",
        "Insufficient evidence": "gray",
    }
    center = [hotspot_cells["latitude"].mean(), hotspot_cells["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")
    max_col = hotspot_cells["total_collisions"].max() or 1
    for _, row in hotspot_cells.iterrows():
        radius = 4 + 18 * (row["total_collisions"] / max_col)
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=colour_map.get(row.get("trend_label", ""), "blue"),
            fill=True,
            fill_opacity=0.6,
            tooltip=(
                f"{row['cell_id']}<br>"
                f"total: {row['total_collisions']} | last 3m: {row.get('last_3m', 'n/a')}<br>"
                f"trend: {row.get('trend_label', 'n/a')}<br>"
                f"coverage: {row.get('coverage_band', 'n/a')}"
            ),
        ).add_to(m)
    return m


def proba_bar(probas: dict[str, float]) -> plt.Figure:
    """Horizontal bar chart of predicted severity probabilities."""
    fig, ax = plt.subplots(figsize=(6, 2.5))
    classes = list(probas.keys())
    vals = list(probas.values())
    ax.barh(classes[::-1], vals[::-1],
            color=[SEVERITY_COLORS[c] for c in classes[::-1]])
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center")
    ax.set_xlim(0, 1)
    _style_axes(ax, "Predicted severity probability", "Probability", "")
    fig.tight_layout()
    return fig


def google_maps_hotspot_html(hotspot_cells: pd.DataFrame, api_key: str, default_city: str = "All") -> str:
    """Generate clean, eye-pleasing Google Maps HTML with collision density overlays and responsive city filters."""
    import json

    cells_list = []
    for _, row in hotspot_cells.iterrows():
        cells_list.append({
            "cell_id": str(row.get("cell_id", "")),
            "lat": float(row.get("latitude", 0.0)),
            "lng": float(row.get("longitude", 0.0)),
            "total_collisions": int(row.get("total_collisions", 0)),
            "mean_monthly": float(row.get("mean_monthly", 0.0)),
            "last_3m": int(row.get("last_3m", 0)) if pd.notna(row.get("last_3m")) else 0,
            "trend_ratio": float(row.get("trend_ratio", 1.0)) if pd.notna(row.get("trend_ratio")) else 1.0,
            "trend_label": str(row.get("trend_label", "Stable")),
            "coverage_band": str(row.get("coverage_band", "Complete")),
        })

    cells_json = json.dumps(cells_list)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Traffic Hotspot Explorer</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{
            width: 100%;
            height: 100%;
            min-height: 600px;
            overflow: hidden;
            background: #f8fafc;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        #map-container {{
            width: 100%;
            height: 100%;
            min-height: 600px;
            position: relative;
        }}
        #map {{
            width: 100%;
            height: 100%;
            min-height: 600px;
            z-index: 1;
        }}

        /* Clean, Pleasant Glass Toolbar */
        .map-toolbar {{
            position: absolute;
            top: 14px;
            left: 14px;
            right: 14px;
            z-index: 1000;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 10px 16px;
            border-radius: 14px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);
        }}
        .toolbar-title {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13.5px;
            font-weight: 700;
            color: #0f172a;
        }}
        .live-chip {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
            animation: pulse-dot 2s infinite;
        }}
        @keyframes pulse-dot {{
            0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
            70% {{ transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }}
            100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
        }}

        .city-chips {{
            display: flex;
            gap: 6px;
            overflow-x: auto;
            max-width: 100%;
            padding: 2px 0;
        }}
        .btn-city {{
            background: #f1f5f9;
            color: #475569;
            border: 1px solid #e2e8f0;
            padding: 5px 12px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            white-space: nowrap;
        }}
        .btn-city:hover, .btn-city.active {{
            background: #2563eb;
            color: #ffffff;
            border-color: #2563eb;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
        }}

        .map-style-selector {{
            display: flex;
            gap: 6px;
            background: #f1f5f9;
            padding: 3px;
            border-radius: 9px;
            border: 1px solid #e2e8f0;
        }}
        .btn-style {{
            background: transparent;
            color: #475569;
            border: none;
            padding: 4px 10px;
            border-radius: 7px;
            font-size: 11.5px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .btn-style.active {{
            background: #ffffff;
            color: #0f172a;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.1);
        }}

        .btn-toggle {{
            background: #ffffff;
            color: #334155;
            border: 1px solid #cbd5e1;
            padding: 5px 11px;
            border-radius: 8px;
            font-size: 11.5px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .btn-toggle.active {{
            background: #0f172a;
            color: #ffffff;
            border-color: #0f172a;
        }}

        /* Clean Legend */
        .map-legend {{
            position: absolute;
            bottom: 20px;
            left: 14px;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 8px 14px;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 11.5px;
            color: #475569;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
        }}
        .legend-indicator {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }}

        /* Leaflet Popup Styling */
        .leaflet-popup-content-wrapper {{
            background: #ffffff !important;
            border-radius: 14px !important;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.15) !important;
            padding: 0 !important;
            border: 1px solid #e2e8f0 !important;
        }}
        .leaflet-popup-content {{
            margin: 0 !important;
            line-height: 1.4 !important;
        }}
        .leaflet-container a.leaflet-popup-close-button {{
            top: 8px;
            right: 8px;
            color: #94a3b8;
            font-size: 18px;
        }}
    </style>
</head>
<body>
    <div id="map-container">
        <div class="map-toolbar">
            <div class="toolbar-title">
                <span class="live-chip"></span>
                <span>Google Maps Hotspots</span>
                <span style="font-size:11px; color:#64748b; font-weight:500;">({len(hotspot_cells)} monitored sectors)</span>
            </div>

            <div class="city-chips">
                <button class="btn-city active" onclick="zoomTo('All')">🇮🇳 All India</button>
                <button class="btn-city" onclick="zoomTo('Mumbai')">Mumbai</button>
                <button class="btn-city" onclick="zoomTo('Delhi')">Delhi</button>
                <button class="btn-city" onclick="zoomTo('Bengaluru')">Bengaluru</button>
                <button class="btn-city" onclick="zoomTo('Hyderabad')">Hyderabad</button>
                <button class="btn-city" onclick="zoomTo('Chennai')">Chennai</button>
                <button class="btn-city" onclick="zoomTo('Kolkata')">Kolkata</button>
                <button class="btn-city" onclick="zoomTo('Pune')">Pune</button>
                <button class="btn-city" onclick="zoomTo('Chandigarh')">Chandigarh</button>
            </div>

            <div style="display:flex; align-items:center; gap:8px;">
                <div class="map-style-selector">
                    <button class="btn-style active" id="btn-road" onclick="setTileLayer('road')">🗺️ Google Roads</button>
                    <button class="btn-style" id="btn-satellite" onclick="setTileLayer('satellite')">🛰️ Satellite</button>
                    <button class="btn-style" id="btn-carto" onclick="setTileLayer('carto')">🏙️ Clean Light</button>
                </div>
                <div style="display:flex; gap:5px;">
                    <button class="btn-toggle active" id="btn-circles" onclick="toggleLayer('circles')">⭕ Radii</button>
                    <button class="btn-toggle active" id="btn-heatmap" onclick="toggleLayer('heatmap')">🔥 Heatmap</button>
                </div>
            </div>
        </div>

        <div id="map"></div>

        <div class="map-legend">
            <div><span class="legend-indicator" style="background:#ef4444;"></span> Persistent Hotspot</div>
            <div><span class="legend-indicator" style="background:#f59e0b;"></span> Emerging Risk (≥1.5x)</div>
            <div><span class="legend-indicator" style="background:#2563eb;"></span> Stable</div>
            <div><span class="legend-indicator" style="background:#10b981;"></span> Declining</div>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
    <script>
        const cells = {cells_json};
        const apiKey = "{api_key}";

        const cityCoords = {{
            'All': {{ lat: 21.5, lng: 79.5, zoom: 5 }},
            'Mumbai': {{ lat: 19.0760, lng: 72.8777, zoom: 11 }},
            'Delhi': {{ lat: 28.6139, lng: 77.2090, zoom: 11 }},
            'Bengaluru': {{ lat: 12.9716, lng: 77.5946, zoom: 11 }},
            'Hyderabad': {{ lat: 17.3850, lng: 78.4867, zoom: 11 }},
            'Chennai': {{ lat: 13.0827, lng: 80.2707, zoom: 11 }},
            'Kolkata': {{ lat: 22.5726, lng: 88.3639, zoom: 11 }},
            'Pune': {{ lat: 18.5204, lng: 73.8567, zoom: 11 }},
            'Chandigarh': {{ lat: 30.7333, lng: 76.7794, zoom: 11 }}
        }};

        // Initialize Map
        const map = L.map('map', {{
            center: [21.5, 79.5],
            zoom: 5,
            zoomControl: false
        }});

        L.control.zoom({{ position: 'bottomright' }}).addControl(map);

        // Tile Layers with Google Maps Key
        const tileLayers = {{
            road: L.tileLayer('https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}&key=' + apiKey, {{
                maxZoom: 20,
                attribution: '&copy; Google Maps'
            }}),
            satellite: L.tileLayer('https://mt1.google.com/vt/lyrs=y&x={{x}}&y={{y}}&z={{z}}&key=' + apiKey, {{
                maxZoom: 20,
                attribution: '&copy; Google Maps'
            }}),
            carto: L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                maxZoom: 19,
                subdomains: 'abcd',
                attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
            }})
        }};

        let currentTile = tileLayers.road;
        currentTile.addTo(map);

        function setTileLayer(name) {{
            if (currentTile) map.removeLayer(currentTile);
            currentTile = tileLayers[name] || tileLayers.road;
            currentTile.addTo(map);

            document.querySelectorAll('.btn-style').forEach(b => b.classList.remove('active'));
            const btn = document.getElementById('btn-' + name);
            if (btn) btn.classList.add('active');
        }}

        // Circles & Heatmap layers
        const circleGroup = L.layerGroup().addTo(map);
        const heatPoints = [];

        cells.forEach(c => {{
            let color = "#2563eb";
            let stroke = "#1d4ed8";
            let badgeBg = "#eff6ff";
            let badgeColor = "#1d4ed8";
            let badgeBorder = "#bfdbfe";

            if (c.trend_label === "Persistent hotspot") {{
                color = "#ef4444";
                stroke = "#dc2626";
                badgeBg = "#fef2f2";
                badgeColor = "#dc2626";
                badgeBorder = "#fecaca";
            }} else if (c.trend_label === "Emerging hotspot") {{
                color = "#f59e0b";
                stroke = "#d97706";
                badgeBg = "#fffbeb";
                badgeColor = "#d97706";
                badgeBorder = "#fde68a";
            }} else if (c.trend_label === "Declining") {{
                color = "#10b981";
                stroke = "#059669";
                badgeBg = "#f0fdf4";
                badgeColor = "#059669";
                badgeBorder = "#bbf7d0";
            }}

            const radius = Math.max(3000, Math.min(18000, c.total_collisions * 16));

            // Risk Radius Circle
            const circle = L.circle([c.lat, c.lng], {{
                color: stroke,
                weight: 1.5,
                fillColor: color,
                fillOpacity: 0.32,
                radius: radius
            }});

            // Center Pin Dot
            const marker = L.circleMarker([c.lat, c.lng], {{
                radius: 6,
                fillColor: color,
                color: '#ffffff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.95
            }});

            const popupContent = `
                <div style="font-family:'Plus Jakarta Sans',-apple-system,sans-serif; color:#0f172a; padding:14px 16px; min-width:230px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-weight:700; font-size:15px; color:#0f172a;">Sector ${{c.cell_id}}</span>
                        <span style="padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; background:${{badgeBg}}; color:${{badgeColor}}; border:1px solid ${{badgeBorder}};">${{c.trend_label}}</span>
                    </div>
                    <div style="font-size:11.5px; color:#64748b; margin-bottom:12px;">
                        Location: ${{c.lat.toFixed(3)}}°N, ${{c.lng.toFixed(3)}}°E
                    </div>
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-bottom:12px;">
                        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:6px 10px;">
                            <div style="font-size:9.5px; color:#64748b; font-weight:700; text-transform:uppercase;">TOTAL COLLISIONS</div>
                            <div style="font-size:17px; font-weight:800; color:#0f172a;">${{c.total_collisions}}</div>
                        </div>
                        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:6px 10px;">
                            <div style="font-size:9.5px; color:#64748b; font-weight:700; text-transform:uppercase;">MONTHLY MEAN</div>
                            <div style="font-size:17px; font-weight:800; color:#2563eb;">${{c.mean_monthly.toFixed(1)}}/mo</div>
                        </div>
                    </div>
                    <div style="font-size:11.5px; display:flex; justify-content:space-between; margin-bottom:4px; padding:2px 0;">
                        <span style="color:#64748b;">Trend Velocity vs Baseline:</span>
                        <strong style="color:${{c.trend_ratio >= 1.5 ? '#dc2626' : '#2563eb'}};">${{c.trend_ratio.toFixed(2)}}x</strong>
                    </div>
                    <div style="font-size:11.5px; display:flex; justify-content:space-between; padding:2px 0;">
                        <span style="color:#64748b;">Historical Coverage:</span>
                        <span style="color:#334155; font-weight:600;">${{c.coverage_band}}</span>
                    </div>
                </div>
            `;

            circle.bindPopup(popupContent);
            marker.bindPopup(popupContent);

            circleGroup.addLayer(circle);
            circleGroup.addLayer(marker);

            heatPoints.push([c.lat, c.lng, Math.min(1.0, c.total_collisions / 150)]);
        }});

        // Heatmap Layer
        const heatLayer = L.heatLayer(heatPoints, {{
            radius: 35,
            blur: 25,
            maxZoom: 10,
            gradient: {{ 0.2: '#2563eb', 0.5: '#f59e0b', 0.8: '#ef4444' }}
        }}).addTo(map);

        let showCircles = true;
        let showHeatmap = true;

        function toggleLayer(type) {{
            if (type === 'circles') {{
                showCircles = !showCircles;
                if (showCircles) map.addLayer(circleGroup);
                else map.removeLayer(circleGroup);
                document.getElementById('btn-circles').classList.toggle('active', showCircles);
            }} else if (type === 'heatmap') {{
                showHeatmap = !showHeatmap;
                if (showHeatmap) map.addLayer(heatLayer);
                else map.removeLayer(heatLayer);
                document.getElementById('btn-heatmap').classList.toggle('active', showHeatmap);
            }}
        }}

        function zoomTo(cityName) {{
            const target = cityCoords[cityName] || cityCoords['All'];
            map.flyTo([target.lat, target.lng], target.zoom, {{
                animate: true,
                duration: 1.2
            }});

            document.querySelectorAll('.btn-city').forEach(b => b.classList.remove('active'));
            const activeBtn = Array.from(document.querySelectorAll('.btn-city')).find(b => b.textContent.includes(cityName));
            if (activeBtn) activeBtn.classList.add('active');
        }}

        // Force resize recalculation in iframe
        setTimeout(() => {{
            map.invalidateSize();
        }}, 200);
        window.addEventListener('resize', () => {{
            map.invalidateSize();
        }});
    </script>
</body>
</html>"""



