"""
run_sionna_test.py
Loads Cairo 3D building scene into Sionna RT, runs a 5G ray-tracing coverage simulation
with a 3-sector macro site, high ray density, and diffraction, producing high-resolution
2D and 3D visualizations.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import geopandas as gpd
from shapely.geometry import box

# Detect Sionna and Sionna RT version and API variant
try:
    import sionna
    import sionna.rt as rt
    sionna_version = getattr(sionna, "__version__", "unknown")
    is_v2 = hasattr(rt, "RadioMapSolver") or hasattr(rt, "ITURadioMaterial")
    print(f"Detected Sionna version: {sionna_version} (API: {'Sionna RT 1.x/2.x' if is_v2 else 'Sionna legacy 0.19'})")
except ImportError as e:
    raise ImportError(f"Sionna RT is not installed. Please install sionna: {e}")


def load_cairo_scene(scene_xml_path="cairo_scene/scene.xml", frequency_hz=1.8e9):
    """Load the Mitsuba 3 XML scene into Sionna RT and configure frequency & materials."""
    if not os.path.isfile(scene_xml_path):
        raise FileNotFoundError(f"Scene XML file '{scene_xml_path}' not found. Run build_scene.py first.")

    print(f"Loading Sionna scene from: {scene_xml_path}...")
    t0 = time.time()
    
    try:
        scene = rt.load_scene(scene_xml_path, merge_shapes=False)
    except TypeError:
        scene = rt.load_scene(scene_xml_path)

    scene.frequency = frequency_hz
    if hasattr(scene, "synthetic_array"):
        scene.synthetic_array = True

    # Assign ITU Radio Materials: concrete (thickness=0.25) and very_dry_ground (thickness=0.5)
    print("Configuring ITU radio materials: concrete (thickness=0.25m), very_dry_ground (thickness=0.5m)...")
    if hasattr(rt, "ITURadioMaterial"):
        mat_concrete = rt.ITURadioMaterial(name="concrete", itu_type="concrete", thickness=0.25)
        mat_ground = rt.ITURadioMaterial(name="very_dry_ground", itu_type="very_dry_ground", thickness=0.5)
        
        if "concrete" not in scene.radio_materials:
            scene.add(mat_concrete)
        if "very_dry_ground" not in scene.radio_materials:
            scene.add(mat_ground)

        if "buildings" in scene.objects:
            scene.get("buildings").radio_material = scene.radio_materials["concrete"]
        if "ground" in scene.objects:
            scene.get("ground").radio_material = scene.radio_materials["very_dry_ground"]
    elif hasattr(rt, "RadioMaterial"):
        mat_concrete = rt.RadioMaterial(name="concrete", relative_permittivity=5.24, conductivity=0.123)
        mat_ground = rt.RadioMaterial(name="very_dry_ground", relative_permittivity=3.0, conductivity=0.001)
        scene.radio_materials["concrete"] = mat_concrete
        scene.radio_materials["very_dry_ground"] = mat_ground

    print(f"Scene loaded successfully in {time.time()-t0:.2f}s. Objects: {list(scene.objects.keys())}")
    return scene


def setup_3sector_site(scene, tower_pos=(0.0, 0.0, 28.0), tx_power_dbm=44.0,
                       sector_yaws=(0.0, 120.0, 240.0), display_radius=8.0):
    """
    Set up a 3-sector 5G macro site:
    - 3 directional sectors spaced 120 deg apart (yaw = 0 deg, 120 deg, 240 deg)
    - 4x4 PlanarArray with 3GPP TR 38.901 pattern per sector
    - Clearly visible transmitter spheres (display_radius=8.0m) positioned around the rooftop mast
    """
    print(f"\nConfiguring 3-Sector 5G Macro Site at tower position [x={tower_pos[0]:.1f}, y={tower_pos[1]:.1f}, z={tower_pos[2]:.1f}]m...")
    print(f" - Antenna Array: 4x4 PlanarArray (3GPP TR 38.901 directional pattern)")
    print(f" - Transmit Power: {tx_power_dbm} dBm per sector")
    print(f" - Marker Display Radius: {display_radius}m (clearly visible sphere shape in 3D)")

    scene.tx_array = rt.PlanarArray(
        num_rows=4,
        num_cols=4,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="V"
    )

    scene.rx_array = rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V"
    )

    sector_colors = [(1.0, 0.15, 0.15), (0.15, 0.85, 0.2), (0.2, 0.45, 1.0)]
    transmitters = []

    # Small 2m offset in sector azimuth direction so sectors form a realistic mast cluster
    offset_dist = 2.5

    for i, yaw_deg in enumerate(sector_yaws):
        yaw_rad = float(np.radians(yaw_deg))
        dx = float(offset_dist * np.cos(yaw_rad))
        dy = float(offset_dist * np.sin(yaw_rad))
        pos = [tower_pos[0] + dx, tower_pos[1] + dy, tower_pos[2]]

        tx_name = f"gNodeB_Cairo_Sec{i+1}_{int(yaw_deg)}deg"
        tx = rt.Transmitter(
            name=tx_name,
            position=pos,
            orientation=[yaw_rad, 0.0, 0.0],
            power_dbm=float(tx_power_dbm),
            color=sector_colors[i % len(sector_colors)],
            display_radius=float(display_radius)
        )
        scene.add(tx)
        transmitters.append(tx)
        print(f"   * Sector {i+1}: '{tx_name}' at [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] | Azimuth = {yaw_deg:.0f}°")

    return transmitters


def compute_coverage(scene, center=(0.0, 0.0, 1.5), size=(2000.0, 2000.0),
                     cell_size=(5.0, 5.0), max_depth=3, samples_per_tx=10_000_000,
                     diffraction=True):
    """
    Compute 2D radio coverage / path-gain map using Sionna RT's RadioMapSolver
    with high ray density (samples_per_tx=1e7) and diffraction enabled.
    """
    print(f"\n--- Running 5G Ray Tracing Simulation ---")
    print(f"Coverage Area: {size[0]}m x {size[1]}m, Cell size: {cell_size[0]}m x {cell_size[1]}m ({(size[0]/cell_size[0])*(size[1]/cell_size[1]):,.0f} cells)")
    print(f"Measurement height: {center[2]}m, Max depth: {max_depth}")
    print(f"Ray Density: {samples_per_tx:,.0f} rays per sector | Specular reflection: True | Diffraction: {diffraction}")
    t0 = time.time()

    if hasattr(rt, "RadioMapSolver"):
        solver = rt.RadioMapSolver()
        rm = solver(
            scene=scene,
            center=list(center),
            orientation=[0.0, 0.0, 0.0],
            size=list(size),
            cell_size=list(cell_size),
            samples_per_tx=int(samples_per_tx),
            max_depth=max_depth,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            diffraction=diffraction
        )
    else:
        rm = scene.coverage_map(
            center=list(center),
            size=list(size),
            cell_size=list(cell_size),
            max_depth=max_depth,
            reflection=True,
            diffraction=diffraction
        )

    print(f"Coverage map simulation completed in {time.time()-t0:.2f}s!")
    return rm


def plot_2d_coverage_with_buildings(rm, scene_info, output_path, tower_pos=(0.0, 0.0, 28.0),
                                    sector_yaws=(0.0, 120.0, 240.0), size=(2000.0, 2000.0),
                                    center=(0.0, 0.0, 1.5), tx_power_dbm=44.0):
    """
    Save high-resolution 2D top-down path-gain / RSRP heatmap overlaid with building footprints.
    Computes best-server composite RSRP across all sectors and visualizes sector boresight arrows.
    """
    print(f"\nGenerating 2D coverage visualization...")
    t0 = time.time()
    
    if hasattr(rm, "path_gain"):
        try:
            pg = rm.path_gain.numpy()
        except AttributeError:
            pg = np.array(rm.path_gain)
        
        if pg.ndim == 3 and pg.shape[0] > 1:
            print(f"Computing 3-sector Best Server composite coverage across {pg.shape[0]} sectors...")
            pg_best = np.max(pg, axis=0)
        elif pg.ndim == 3:
            pg_best = pg[0]
        else:
            pg_best = pg
    elif hasattr(rm, "as_tensor"):
        pg_best = np.squeeze(rm.as_tensor().numpy())
    else:
        pg_best = np.array(rm)

    eps = 1e-20
    pg_db = 10.0 * np.log10(np.maximum(pg_best, eps))
    rsrp_dbm = tx_power_dbm + pg_db

    half_w = size[0] / 2.0
    half_h = size[1] / 2.0
    extent = [
        center[0] - half_w, center[0] + half_w,
        center[1] - half_h, center[1] + half_h
    ]

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

    vmin, vmax = -125.0, -50.0  # cellular RSRP range in dBm
    im = ax.imshow(
        rsrp_dbm,
        origin="lower",
        extent=extent,
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear"
    )

    gpkg_path = scene_info.get("source_gpkg", "")
    if gpkg_path and os.path.isfile(gpkg_path):
        try:
            cx_utm = scene_info["center_utm_x"]
            cy_utm = scene_info["center_utm_y"]
            bbox_utm = box(
                cx_utm - half_w, cy_utm - half_h,
                cx_utm + half_w, cy_utm + half_h
            )
            gdf_local = gpd.read_file(gpkg_path, bbox=bbox_utm)
            if not gdf_local.empty:
                gdf_local["geometry"] = gdf_local["geometry"].translate(xoff=-cx_utm, yoff=-cy_utm)
                gdf_local.boundary.plot(ax=ax, color="#111111", linewidth=0.35, alpha=0.7)
                gdf_local.plot(ax=ax, facecolor="#222222", alpha=0.35)
                print(f"Overlaid {len(gdf_local):,} building footprints onto 2D map.")
        except Exception as e:
            print(f"Note: building footprint overlay skipped: {e}")

    ax.scatter(
        [tower_pos[0]], [tower_pos[1]],
        color="crimson", edgecolor="white", s=140, marker="o",
        linewidth=2.0, label=f"3-Sector gNodeB Tower (Z={tower_pos[2]:.0f}m)", zorder=12
    )

    arrow_len = 130.0  # meters
    sector_colors = ["#ff3333", "#33cc33", "#3388ff"]
    for i, yaw_deg in enumerate(sector_yaws):
        yaw_rad = np.radians(yaw_deg)
        dx = arrow_len * np.cos(yaw_rad)
        dy = arrow_len * np.sin(yaw_rad)
        col = sector_colors[i % len(sector_colors)]
        ax.annotate(
            "",
            xy=(tower_pos[0] + dx, tower_pos[1] + dy),
            xytext=(tower_pos[0], tower_pos[1]),
            arrowprops=dict(
                arrowstyle="->",
                color=col,
                lw=2.5,
                mutation_scale=15
            ),
            zorder=13
        )
        ax.plot([], [], color=col, lw=2.5, label=f"Sec {i+1} Azimuth: {int(yaw_deg)}°")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Best-Server 5G/4G RSRP (dBm)  |  Carrier: 1.8 GHz (Band 3 / n3)", fontsize=11, fontweight="bold")
    
    ax.set_title(
        f"Cairo (Heliopolis / Nasr City) - 3-Sector Macro Ray-Tracing Coverage\n"
        f"Carrier: 1.8 GHz (Band 3 / n3) | Sectors: 0°, 120°, 240° | 10M Rays/Sector (160k cells @ 5m)",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Local X Coordinate (meters from scene origin)", fontsize=10)
    ax.set_ylabel("Local Y Coordinate (meters from scene origin)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35, color="white")
    ax.legend(loc="upper right", framealpha=0.92, facecolor="white", edgecolor="#cccccc")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved 2D coverage visualization: {output_path} ({time.time()-t0:.2f}s)")


def render_3d_snapshot(scene, rm, output_path="outputs/scene_3d.png",
                       cam_pos=(220.0, -300.0, 150.0), look_at=(0.0, 0.0, 24.0),
                       resolution=(1280, 720)):
    """
    Render high-quality 3D snapshot showing the rooftop transmitter spheres
    and the surrounding street coverage clearly in 3D.
    """
    print(f"\nRendering 3D Perspective Snapshot: camera={cam_pos} -> look_at={look_at}...")
    t0 = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cam = rt.Camera(position=list(cam_pos), look_at=list(look_at))

    try:
        scene.render_to_file(
            camera=cam,
            filename=output_path,
            radio_map=rm,
            num_samples=64,
            resolution=resolution,
            rm_metric="path_gain",
            rm_db_scale=True,
            rm_cmap="turbo",
            rm_vmin=-130.0,
            rm_vmax=-60.0,
            show_devices=True,
            show_orientations=False
        )
        print(f"Saved 3D snapshot: {output_path} ({time.time()-t0:.2f}s)")
    except Exception as e:
        print(f"Warning: Primary render with radio_map encountered error: {e}")
        scene.render_to_file(camera=cam, filename=output_path, num_samples=64, resolution=resolution, show_devices=True)
        print(f"Saved fallback 3D snapshot: {output_path}")


def generate_interactive_3d_html(rm, scene_info, output_path="outputs/cairo_3d_viewer.html",
                                 tower_pos=(0.0, 0.0, 28.0), sector_yaws=(0.0, 120.0, 240.0),
                                 tx_power_dbm=44.0, size=(2000.0, 2000.0)):
    """
    Generate a standalone interactive 3D WebGL (Three.js) HTML visualizer:
    - Interactive 3D orbit, pan, and zoom controls.
    - True 3D extruded Cairo buildings around the tower.
    - Spherical transmitter markers with directional radiation cones for all 3 sectors (0°, 120°, 240°).
    - 2D RSRP coverage heatmap draped onto the ground surface with an interactive colormap legend.
    """
    print(f"\nGenerating Standalone Interactive 3D WebGL Viewer ({output_path})...")
    
    # Extract path gain array and save as a low-res image for the WebGL terrain texture
    if hasattr(rm, "path_gain"):
        try:
            pg = rm.path_gain.numpy()
        except AttributeError:
            pg = np.array(rm.path_gain)
        if pg.ndim == 3 and pg.shape[0] > 1:
            pg_best = np.max(pg, axis=0)
        elif pg.ndim == 3:
            pg_best = pg[0]
        else:
            pg_best = pg
    elif hasattr(rm, "as_tensor"):
        pg_best = np.squeeze(rm.as_tensor().numpy())
    else:
        pg_best = np.array(rm)

    eps = 1e-20
    pg_db = 10.0 * np.log10(np.maximum(pg_best, eps))
    rsrp = tx_power_dbm + pg_db
    vmin, vmax = -125.0, -50.0
    norm_rsrp = np.clip((rsrp - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = plt.get_cmap("turbo")
    rgba = cmap(norm_rsrp)
    
    # Save texture as base64 data URI to eliminate CORS issues on file://
    import base64
    import io
    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png", origin="lower")
    b64_texture = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    # Load nearby buildings (within 450m) to embed as 3D geometry in Three.js
    gpkg_path = scene_info.get("source_gpkg", "testarea1.gpkg")
    building_data = []
    if gpkg_path and os.path.isfile(gpkg_path):
        try:
            cx_utm = scene_info["center_utm_x"] + tower_pos[0]
            cy_utm = scene_info["center_utm_y"] + tower_pos[1]
            r = 450.0
            bbox_utm = box(cx_utm - r, cy_utm - r, cx_utm + r, cy_utm + r)
            gdf_local = gpd.read_file(gpkg_path, bbox=bbox_utm)
            if not gdf_local.empty:
                gdf_local = gdf_local.explode(index_parts=False)
                gdf_local["geometry"] = gdf_local["geometry"].translate(xoff=-scene_info["center_utm_x"], yoff=-scene_info["center_utm_y"])
                gdf_local["geometry"] = gdf_local["geometry"].simplify(0.5)
                height_col = scene_info.get("height_column", "Height")
                for row in gdf_local.itertuples():
                    geom = row.geometry
                    if geom.is_empty or not geom.is_valid:
                        continue
                    h = float(getattr(row, height_col, 15.0))
                    if geom.geom_type == "Polygon":
                        coords = [[round(p[0], 2), round(p[1], 2)] for p in geom.exterior.coords[:-1]]
                        if len(coords) >= 3:
                            building_data.append({"c": coords, "h": round(h, 1)})
        except Exception as e:
            print(f"Note: HTML building embedding skipped: {e}")
    print(f"Embedded {len(building_data):,} 3D buildings into interactive 3D WebGL viewer.")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Cairo 5G 3D Coverage Viewer</title>
    <style>
        body {{ margin: 0; overflow: hidden; background: #0a0e17; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #fff; }}
        #info {{ position: absolute; top: 15px; left: 15px; background: rgba(13, 20, 36, 0.85); backdrop-filter: blur(10px); padding: 16px 20px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); max-width: 320px; z-index: 100; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
        #info h2 {{ margin: 0 0 8px 0; font-size: 17px; color: #38bdf8; font-weight: 600; }}
        #info p {{ margin: 4px 0; font-size: 12px; color: #94a3b8; line-height: 1.4; }}
        .badge {{ display: inline-block; padding: 2px 7px; border-radius: 4px; font-weight: bold; font-size: 11px; margin-right: 4px; }}
        .sec1 {{ background: rgba(239, 68, 68, 0.25); color: #f87171; border: 1px solid #ef4444; }}
        .sec2 {{ background: rgba(34, 197, 94, 0.25); color: #4ade80; border: 1px solid #22c55e; }}
        .sec3 {{ background: rgba(59, 130, 246, 0.25); color: #60a5fa; border: 1px solid #3b82f6; }}
        #legend {{ position: absolute; bottom: 20px; right: 20px; background: rgba(13, 20, 36, 0.85); backdrop-filter: blur(8px); padding: 12px 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); z-index: 100; }}
        .cbar {{ width: 180px; height: 12px; border-radius: 3px; background: linear-gradient(to right, #30123b, #4662d8, #1ae4b6, #a2fc3c, #faba39, #e23412, #7a0403); }}
        .cbar-labels {{ display: flex; justify-content: space-between; font-size: 10px; color: #94a3b8; margin-top: 4px; }}
        #instructions {{ position: absolute; bottom: 20px; left: 15px; font-size: 11px; color: #64748b; background: rgba(0,0,0,0.4); padding: 6px 10px; border-radius: 6px; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="info">
        <h2>Cairo 1.8 GHz 3D Coverage Viewer</h2>
        <p><strong>Site:</strong> 3-Sector Macro Tower (X = {tower_pos[0]:.0f}m, Y = {tower_pos[1]:.0f}m, Z = {tower_pos[2]:.0f}m)</p>
        <p><strong>Carrier:</strong> 1.8 GHz (Band 3 / n3) | Power: 44 dBm/sec</p>
        <div style="margin-top: 8px;">
            <span class="badge sec1">Sec 1: 0°</span>
            <span class="badge sec2">Sec 2: 120°</span>
            <span class="badge sec3">Sec 3: 240°</span>
        </div>
        <p style="margin-top: 10px; font-size: 11px; color: #cbd5e1;">Real-time 3D simulation with 30M rays, reflections & diffraction.</p>
    </div>

    <div id="legend">
        <div style="font-size: 11px; font-weight: bold; margin-bottom: 4px;">5G RSRP Coverage (dBm)</div>
        <div class="cbar"></div>
        <div class="cbar-labels"><span>-125 dBm</span><span>-90 dBm</span><span>-50 dBm</span></div>
    </div>

    <div id="instructions">Left Drag: Rotate | Right Drag: Pan | Scroll: Zoom</div>

    <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0e17);
        scene.fog = new THREE.FogExp2(0x0a0e17, 0.001);

        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 4000);
        camera.position.set({tower_pos[0]} + 220, {tower_pos[2]} + 160, {tower_pos[1]} + 280);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set({tower_pos[0]}, {tower_pos[2]}, {tower_pos[1]});
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        // Lighting
        const ambLight = new THREE.AmbientLight(0xffffff, 0.65);
        scene.add(ambLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
        dirLight.position.set({tower_pos[0]} + 200, 400, {tower_pos[1]} + 300);
        scene.add(dirLight);

        // Ground with Heatmap Texture (Base64 data URI eliminates CORS)
        const texLoader = new THREE.TextureLoader();
        texLoader.load('{b64_texture}', function(texture) {{
            texture.magFilter = THREE.LinearFilter;
            texture.minFilter = THREE.LinearFilter;
            const groundGeo = new THREE.PlaneGeometry({size[0]}, {size[1]});
            const groundMat = new THREE.MeshLambertMaterial({{
                map: texture,
                side: THREE.DoubleSide
            }});
            const ground = new THREE.Mesh(groundGeo, groundMat);
            ground.rotation.x = -Math.PI / 2;
            ground.position.set({tower_pos[0]}, 0, {tower_pos[1]});
            scene.add(ground);
        }});
        }});

        // 3D Buildings
        const buildingData = {json.dumps(building_data)};
        const bldgMat = new THREE.MeshStandardMaterial({{
            color: 0x475569,
            roughness: 0.5,
            metalness: 0.2,
            flatShading: true
        }});
        const edgeMat = new THREE.LineBasicMaterial({{ color: 0x94a3b8, transparent: true, opacity: 0.7 }});

        buildingData.forEach(b => {{
            const shape = new THREE.Shape();
            const pts = b.c || b.coords;
            pts.forEach((p, idx) => {{
                if (idx === 0) shape.moveTo(p[0], -p[1]);
                else shape.lineTo(p[0], -p[1]);
            }});
            const extrudeSettings = {{
                depth: b.h,
                bevelEnabled: false
            }};
            const geom = new THREE.ExtrudeGeometry(shape, extrudeSettings);
            const mesh = new THREE.Mesh(geom, bldgMat);
            mesh.rotation.x = -Math.PI / 2;
            mesh.position.y = 0;
            scene.add(mesh);

            const edgesGeom = new THREE.EdgesGeometry(geom, 25);
            const edges = new THREE.LineSegments(edgesGeom, edgeMat);
            edges.rotation.x = -Math.PI / 2;
            scene.add(edges);
        }});

        // Tower Mast
        const mastGeo = new THREE.CylinderGeometry(0.8, 1.2, {tower_pos[2]}, 16);
        const mastMat = new THREE.MeshStandardMaterial({{ color: 0x64748b, metalness: 0.8, roughness: 0.2 }});
        const mast = new THREE.Mesh(mastGeo, mastMat);
        mast.position.set({tower_pos[0]}, {tower_pos[2]} / 2, {tower_pos[1]});
        scene.add(mast);

        // 3-Sector Transmitter Spheres & Radiation Cones
        const sectors = [
            {{ name: "Sec 1", yaw: 0, color: 0xef4444 }},
            {{ name: "Sec 2", yaw: 120, color: 0x22c55e }},
            {{ name: "Sec 3", yaw: 240, color: 0x3b82f6 }}
        ];

        sectors.forEach(s => {{
            const rad = s.yaw * Math.PI / 180;
            const offDist = 3.5;
            const txX = {tower_pos[0]} + offDist * Math.cos(rad);
            const txZ = {tower_pos[1]} + offDist * Math.sin(rad);
            const txY = {tower_pos[2]};

            // Glowing Sphere Marker
            const sphereGeo = new THREE.SphereGeometry(3.0, 32, 32);
            const sphereMat = new THREE.MeshStandardMaterial({{
                color: s.color,
                emissive: s.color,
                emissiveIntensity: 0.7,
                roughness: 0.2
            }});
            const sphere = new THREE.Mesh(sphereGeo, sphereMat);
            sphere.position.set(txX, txY, txZ);
            scene.add(sphere);

            // Directional Radiation Coverage Cone
            const coneGeo = new THREE.ConeGeometry(18, 55, 32, 1, true);
            const coneMat = new THREE.MeshBasicMaterial({{
                color: s.color,
                transparent: true,
                opacity: 0.22,
                wireframe: false,
                side: THREE.DoubleSide
            }});
            const cone = new THREE.Mesh(coneGeo, coneMat);
            cone.position.set(txX + 25 * Math.cos(rad), txY - 8, txZ + 25 * Math.sin(rad));
            cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(Math.cos(rad), -0.2, Math.sin(rad)).normalize());
            scene.add(cone);
        }});

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved Standalone Interactive 3D WebGL Viewer: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run 5G Sionna RT ray-tracing simulation on Cairo scene.")
    parser.add_argument("--scene", default="cairo_scene/scene.xml", help="Path to scene.xml")
    parser.add_argument("--scene_info", default="cairo_scene/scene_info.json", help="Path to scene_info.json")
    parser.add_argument("--frequency", type=float, default=1.8e9, help="Carrier frequency in Hz (default: 1.8e9)")
    parser.add_argument("--lat", type=float, default=None, help="Tower GPS latitude WGS84")
    parser.add_argument("--lon", type=float, default=None, help="Tower GPS longitude WGS84")
    parser.add_argument("--tower_x", type=float, default=0.0, help="Tower local X coordinate (m)")
    parser.add_argument("--tower_y", type=float, default=0.0, help="Tower local Y coordinate (m)")
    parser.add_argument("--tower_z", type=float, default=28.0, help="Tower rooftop height Z (m)")
    parser.add_argument("--tx_power", type=float, default=44.0, help="Transmit power in dBm per sector")
    parser.add_argument("--tx_display_radius", type=float, default=8.0, help="Radius of transmitter sphere marker (m)")
    parser.add_argument("--sim_size", type=float, default=2000.0, help="Coverage map width & height (m)")
    parser.add_argument("--cell_size", type=float, default=5.0, help="Grid cell resolution (m)")
    parser.add_argument("--max_depth", type=int, default=3, help="Max ray tracing depth")
    parser.add_argument("--samples_per_tx", type=int, default=10_000_000, help="Rays launched per sector (default: 1e7)")
    parser.add_argument("--diffraction", action="store_true", default=True, help="Enable edge/wedge diffraction")
    parser.add_argument("--output_dir", default="outputs", help="Output directory for visualizations")
    args = parser.parse_args()

    # Automatically map GPS latitude/longitude to local metric coordinates if provided
    if args.lat is not None and args.lon is not None:
        from scene_utils import latlon_to_scene_xy
        args.tower_x, args.tower_y = latlon_to_scene_xy(args.lat, args.lon)
        print(f"Mapped GPS ({args.lat:.6f}, {args.lon:.6f}) -> Tower Local Position: X={args.tower_x:.2f}m, Y={args.tower_y:.2f}m")

    scene_info = {}
    if os.path.isfile(args.scene_info):
        with open(args.scene_info, "r", encoding="utf-8") as f:
            scene_info = json.load(f)

    # 1. Load scene and configure materials
    scene = load_cairo_scene(args.scene, frequency_hz=args.frequency)

    # 2. Setup 3-Sector Macro Site
    tower_pos = (args.tower_x, args.tower_y, args.tower_z)
    sector_yaws = (0.0, 120.0, 240.0)
    setup_3sector_site(
        scene,
        tower_pos=tower_pos,
        tx_power_dbm=args.tx_power,
        sector_yaws=sector_yaws,
        display_radius=args.tx_display_radius
    )

    # 3. Compute 2D radio coverage map centered around tower
    rm = compute_coverage(
        scene=scene,
        center=(args.tower_x, args.tower_y, 1.5),
        size=(args.sim_size, args.sim_size),
        cell_size=(args.cell_size, args.cell_size),
        max_depth=args.max_depth,
        samples_per_tx=args.samples_per_tx,
        diffraction=args.diffraction
    )

    # 4. Generate 2D coverage heatmap
    coverage_2d_path = os.path.join(args.output_dir, "coverage_2d.png")
    plot_2d_coverage_with_buildings(
        rm=rm,
        scene_info=scene_info,
        output_path=coverage_2d_path,
        tower_pos=tower_pos,
        sector_yaws=sector_yaws,
        size=(args.sim_size, args.sim_size),
        center=(args.tower_x, args.tower_y, 1.5),
        tx_power_dbm=args.tx_power
    )

    # 5. Generate Standalone Interactive 3D HTML Viewer FIRST
    viewer_path = os.path.join(args.output_dir, "cairo_3d_viewer.html")
    generate_interactive_3d_html(
        rm=rm,
        scene_info=scene_info,
        output_path=viewer_path,
        tower_pos=tower_pos,
        sector_yaws=sector_yaws,
        tx_power_dbm=args.tx_power,
        size=(args.sim_size, args.sim_size)
    )

    # 6. Render High-Quality 3D Snapshot
    scene_3d_path = os.path.join(args.output_dir, "scene_3d.png")
    try:
        render_3d_snapshot(
            scene=scene,
            rm=rm,
            output_path=scene_3d_path,
            cam_pos=(args.tower_x + 220.0, args.tower_y - 300.0, args.tower_z + 130.0),
            look_at=(args.tower_x, args.tower_y, args.tower_z - 4.0),
            resolution=(1280, 720)
        )
    except Exception as e:
        print(f"Note: 3D snapshot render skipped: {e}")

    print("\n=======================================================")
    print("Enhanced 3D Visualization Pipeline Completed!")
    print(f"2D Coverage Heatmap:      {os.path.abspath(coverage_2d_path)}")
    print(f"3D Scene Snapshot:        {os.path.abspath(scene_3d_path)}")
    print(f"Interactive 3D WebGL:     {os.path.abspath(viewer_path)}")
    print("=======================================================")


if __name__ == "__main__":
    main()
