# Traffic Safety Intelligence — Design System Specification

A unified, futuristic Cyberpunk / Mission-Control design system engineered for high-density spatial intelligence, telemetry dashboards, and emergency dispatch systems.

---

## 1. Core Visual Philosophy

- **Cosmic Canvas**: Deep layered midnight surfaces (`#070a12`, `#0a0f1d`) that eliminate eye strain and maximize the visual luminance of geospatial hotspot indicators.
- **Vibrant HUD Accents**: Cyberpunk Cyan (`#00f2fe`), Arterial Blue (`#38bdf8`), Hazard Amber (`#f59e0b`), Emergency Crimson (`#ef4444`), and Operational Emerald (`#10b981`).
- **Glassmorphic Depth**: Frosted glass containers with `backdrop-filter: blur(20px)`, subtle 1px border glows (`rgba(0, 242, 254, 0.25)`), and progressive drop shadows.
- **Live Telemetry Dynamics**: Micro-animations including pulsing beacons, expanding radar shockwaves, and rotating radar cones.

---

## 2. Color Tokens

### Background Palette
| Token | Hex | Usage |
|---|---|---|
| `--bg-base` | `#070a12` | Root dark canvas with subtle radial mesh |
| `--bg-surface-1` | `rgba(13, 20, 36, 0.85)` | Primary HUD cards, stat containers |
| `--bg-surface-2` | `rgba(20, 30, 52, 0.70)` | Secondary panels, modal drawers, table headers |
| `--border-subtle` | `rgba(255, 255, 255, 0.08)` | Structural dividers, card borders |
| `--border-glow` | `rgba(0, 242, 254, 0.28)` | Interactive focus states and high-priority tiles |

### Hazard & Status Accents
| Token | Hex | Semantics |
|---|---|---|
| `--status-critical` | `#ef4444` | Persistent Hotspots, High Severity, Priority 1 Dispatch |
| `--status-warning` | `#f59e0b` | Emerging Hotspots (>=1.5x surge), Major Injury likelihood |
| `--status-info` | `#00f2fe` | Radar Beacons, Connected APIs, Stable corridors |
| `--status-success` | `#10b981` | Declining Hotspots, System Online, Mitigation verified |
| `--status-purple` | `#a855f7` | Policy Weights, AI Explainability, Model comparisons |

---

## 3. Typography Scale

- **Primary Display Font**: `'Outfit', sans-serif` (300, 400, 600, 700, 800, 900)
- **Numerical & Metric Font**: `'Space Grotesk', sans-serif` (500, 700)
- **Code & Coordinate Telemetry**: `'JetBrains Mono', monospace` (400, 700)

```css
.hero-title    { font-size: 28px; font-weight: 900; letter-spacing: -0.5px; }
.panel-header  { font-size: 18px; font-weight: 800; letter-spacing: 0px; }
.metric-value  { font-size: 26px; font-weight: 800; font-family: 'Space Grotesk'; }
.telemetry-tag { font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono'; }
```

---

## 4. Components & Interactive Elements

### 1. Stat HUD Card
```html
<div class="stat-card">
  <div class="stat-label">
    <span>TOTAL COLLISIONS</span>
    <span class="stat-chip chip-cyan">INGESTED</span>
  </div>
  <div class="stat-value">20,000</div>
  <div class="stat-sub">Across 8 major Indian metro regions</div>
</div>
```

### 2. Animated Pulsing Radar Beacon
```css
@keyframes beacon-pulse {
  0%, 100% { transform: scale(0.85); opacity: 0.7; }
  50% { transform: scale(1.35); opacity: 1; }
}
```

### 3. Dual-Engine Map Layer
- Full-height responsive container with embedded Google Maps JS API and Leaflet CartoDB Dark Matter.
- Custom SVG circular markers with dynamic CSS shockwave rings (`@keyframes ring-shockwave`).

---

## 5. Screen Inventory for Stitch MCP

1. **Screen 01: Mission Control HUD** (`/`)
   - High-level situational awareness, DEFCON hazard meter, top 5 high-crash corridors leaderboard, validation benchmark matrix.
2. **Screen 02: Spatio-Temporal Geospatial Radar** (`/map`)
   - Full-bleed interactive dark map with city quick-pan pills, shockwave layers, heatmap toggle, and rapid response dispatch drawer.
3. **Screen 03: Neural Severity Diagnostic Console** (`/severity`)
   - 3-column input pod (Geometry, Atmosphere, Calendar) with real-time probability meters for Minor, Major, and Fatal outcomes.
4. **Screen 04: Time-Place Safety Twin Studio** (`/twin`)
   - Interactive policy weight sliders, before-vs-after intervention scenario simulator, safety budget allocator with societal loss savings projection.
5. **Screen 05: Early Warning Triage Board** (`/radar`)
   - 4-quadrant triage board (Persistent, Emerging, Stable, Declining) with velocity surge multipliers and automated patrol recommendations.
