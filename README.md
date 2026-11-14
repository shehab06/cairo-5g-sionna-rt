# Cairo 3D Building Map 5G Ray-Tracing Simulation with NVIDIA Sionna RT

A clean, modular Python pipeline to convert 2D building footprints with heights (QGIS GeoPackage) into a 3D Mitsuba 3 mesh scene, run an accelerated 5G ray-tracing coverage simulation using **NVIDIA Sionna RT 2.x**, and generate high-resolution 2D and 3D signal propagation heatmaps.

---

## 1. Project Structure

```
.
├── testarea1.gpkg             # Input QGIS GeoPackage (Heliopolis / Nasr City buildings)
├── build_scene.py             # Converts GeoPackage -> 3D Mitsuba 3 / Sionna RT scene
├── run_sionna_test.py         # Runs 5G ray tracing & produces 2D/3D visualizations
├── scene_utils.py             # Helper module for GPS (lat, lon) <-> Local (x, y) coordinates
├── cairo_scene/               # Generated Mitsuba 3 Scene
│   ├── scene.xml              # Mitsuba 3 scene definition with ITU radio materials
│   ├── scene_info.json        # Scene origin, UTM zone 36N metadata, and extent
│   └── meshes/
│       ├── buildings.ply      # Watertight 3D extruded building footprints (PLY)
│       └── ground.ply         # Flat ground plane mesh (top surface at Z = 0.0m)
├── outputs/                   # Generated High-Resolution Visualizations
│   ├── coverage_2d.png        # 2D RSRP coverage heatmap overlaid with building footprints
│   └── scene_3d.png           # 3D Mitsuba ray-traced snapshot with radio map overlay
└── README.md
```

---

## 2. Environment & Prerequisites

- **Python Version**: Python 3.11 (recommended for PyTorch + Dr.Jit + Mitsuba compatibility on Windows).
- **GPU**: NVIDIA GPU with CUDA support (e.g., GeForce GTX 1650 Ti or RTX series).
- **Core Dependencies**:
  - `sionna` & `sionna-rt` (v2.2.0, utilizing Dr.Jit CUDA backend)
  - `geopandas`, `shapely`, `pyproj`
  - `trimesh`, `mapbox-earcut`
  - `matplotlib`, `numpy`

### Installation

If setting up a new virtual environment:

```powershell
# Create Python 3.11 virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install required packages
pip install --upgrade pip
pip install geopandas trimesh mapbox-earcut shapely pyproj matplotlib numpy sionna
```

---

## 3. Workflow & Usage

### Step 1: Build the 3D Scene (`build_scene.py`)

This script processes the raw GeoPackage (`testarea1.gpkg`), performs CRS validation, filters invalid geometries, simplifies footprints, centers the scene around `(0.0, 0.0)`, extrudes buildings into 3D meshes, and generates the Mitsuba XML.

```powershell
python build_scene.py
```

**Key Features:**
- **Automatic CRS Detection**: Verifies if coordinates are in `EPSG:32636` (UTM zone 36N). Reprojects automatically if in `EPSG:4326`.
- **Polygon Simplification**: Simplifies footprints (`tolerance=0.5` m) to maintain ray-tracing performance while preserving building morphology.
- **Scene Centering**: Bounding-box center `(center_x, center_y)` is subtracted so coordinates are local meters around `(0.0, 0.0)`.
- **Watertight 3D Extrusion**: Extrudes footprints vertically by `Height` using vectorized earcut triangulation into `cairo_scene/meshes/buildings.ply`.
- **Ground Plane**: Creates a flat ground plane mesh with its top surface aligned at `Z = 0.0` m.
- **Scene Metadata**: Exports `cairo_scene/scene_info.json` containing the UTM center, GPS origin, and coordinate transformation utilities.

---

### Step 2: Run 5G Ray Tracing Simulation (`run_sionna_test.py`)

Loads the Mitsuba scene into Sionna RT, sets up 5G radio parameters, executes ray tracing, and generates visualizations.

```powershell
python run_sionna_test.py
```

**Simulation Parameters:**
- **Carrier Frequency**: `1.8 GHz` (LTE Band 3 / 5G NR Band n3, widely used in Cairo).
- **Radio Materials**: ITU Recommendation P.2040 evaluated at 1.8 GHz:
  - Buildings: ITU `concrete` ($\varepsilon_r = 5.24$, $\sigma = 0.073\,\text{S/m}$, `thickness=0.25` m).
  - Ground: ITU `very_dry_ground` ($\varepsilon_r = 3.0$, $\sigma = 0.001\,\text{S/m}$, `thickness=0.5` m).
- **Antenna Configuration**:
  - **3-Sector Macro Site**: 3 directional sectors at local `[0.0, 0.0, 28.0]` m spaced $120^\circ$ apart (yaw = $0^\circ, 120^\circ, 240^\circ$) covering all $360^\circ$ around the site.
  - **Array Geometry**: 4x4 `PlanarArray` with 3GPP `tr38901` directional pattern per sector at `44 dBm` ($25\text{ W}$) transmit power.
  - **Transmitter Markers**: Compact display radius (`display_radius=2.0` m) and `show_orientations=False` to keep surrounding building facades visible.
  - **Receiver**: 1x1 isotropic antenna array (`rx_array`) at pedestrian height $Z = 1.5$ m.
- **Ray-Tracing Solver**: `RadioMapSolver` with:
  - Simulation Grid: `2000m × 2000m` centered at `(0.0, 0.0, 1.5)`.
  - Cell Resolution: `5.0m × 5.0m` (160,000 measurement cells).
  - Ray Density: `10,000,000` rays per sector (`samples_per_tx=1e7`).
  - Propagation Mechanisms: Specular reflections, refraction, and edge **diffraction** (`diffraction=True`).

---

## 4. Output Visualizations

The simulation generates three complementary visualizations in `outputs/`:

1. **`outputs/coverage_2d.png`**:
   - High-resolution top-down 2D heatmap showing estimated 5G Best-Server composite RSRP (in dBm, from -125 dBm to -50 dBm) across the 2 km × 2 km coverage area.
   - 4,125 vector building footprints overlaid on top of the signal map to clearly visualize urban canyons, line-of-sight propagation, and building shadow zones.
   - Color-coded directional azimuth arrows for the 3 sectors ($0^\circ, 120^\circ, 240^\circ$).

2. **`outputs/scene_3d.png`**:
   - High-fidelity 3D perspective snapshot rendered with Mitsuba 3 and Sionna RT.
   - Dynamic camera angle $(X=220\text{m}, Y=-300\text{m}, Z=150\text{m})$ looking directly at the rooftop installation $(X=0\text{m}, Y=0\text{m}, Z=24\text{m})$.
   - Features prominent spherical transmitter markers (`display_radius=8.0` m) on the rooftop mast and a vibrant turbo colormap showing signal coverage down the avenues.

3. **`outputs/cairo_3d_viewer.html`**:
   - Standalone interactive 3D WebGL (Three.js) viewer that opens in any browser.
   - Interactive orbit, pan, and zoom controls.
   - True 3D extruded Cairo buildings, the central tower mast, 3-sector transmitter spheres, directional 3D radiation coverage cones, and the 5G RSRP heatmap texture draped across the terrain.

---

## 5. GPS Coordinate Conversion Helper

To position transmitters or receivers using real GPS coordinates (WGS84 latitude/longitude), use `scene_utils.py`:

```python
from scene_utils import latlon_to_scene_xy, scene_xy_to_latlon

# Convert GPS (latitude, longitude) to local scene (x, y) coordinates
local_x, local_y = latlon_to_scene_xy(lat=30.079753, lon=31.302006)
print(f"Local coordinates: x={local_x:.2f} m, y={local_y:.2f} m")

# Convert local scene coordinates back to GPS
gps_lat, gps_lon = scene_xy_to_latlon(x=150.0, y=-200.0)
print(f"GPS coordinates: Lat={gps_lat:.6f}, Lon={gps_lon:.6f}")
```
