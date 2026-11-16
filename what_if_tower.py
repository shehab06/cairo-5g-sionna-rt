"""
what_if_tower.py
Digital Network Twin "What-If" Scenario Engine for Tower Configuration Optimization.
Simulates and compares baseline vs. proposed macro tower parameters:
- Tower mast height (e.g. 28m vs 38m)
- Mechanical / electrical downtilt (e.g. 0 deg vs 6 deg)
- Transmit power (e.g. 44 dBm vs 46 dBm)
- Sector azimuths and antenna geometry
Generates side-by-side RSRP heatmaps, Delta Gain/Loss map, and interactive 3D comparison viewer.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm
import geopandas as gpd
from shapely.geometry import box

try:
    import sionna
    import sionna.rt as rt
except ImportError as e:
    raise ImportError(f"Sionna RT is not installed: {e}")


def load_cairo_scene(scene_xml_path="cairo_scene/scene.xml", frequency_hz=1.8e9):
    """Load the Mitsuba 3 XML scene into Sionna RT."""
    if not os.path.isfile(scene_xml_path):
        raise FileNotFoundError(f"Scene XML '{scene_xml_path}' not found. Run build_scene.py first.")

    try:
        scene = rt.load_scene(scene_xml_path, merge_shapes=False)
    except TypeError:
        scene = rt.load_scene(scene_xml_path)

    scene.frequency = frequency_hz
    if hasattr(scene, "synthetic_array"):
        scene.synthetic_array = True

    # Configure materials
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

    return scene


def configure_and_simulate(scene, tower_pos, sector_yaws, downtilt_deg, tx_power_dbm,
                           sim_size, cell_size, samples_per_tx, max_depth, tag="Scenario"):
    """
    Configure tower sectors with given height, azimuth, tilt, and power, then execute ray tracing.
    """
    print(f"\n--- Running Ray-Tracing for {tag} ---")
    print(f"Mast Height: {tower_pos[2]:.1f}m | Downtilt: {downtilt_deg:.1f}° | TX Power: {tx_power_dbm:.1f} dBm/sec")
    print(f"Sectors: {sector_yaws}° | Ray Density: {samples_per_tx:,.0f} rays/sec")
    t0 = time.time()

    # Clear existing transmitters
    for name in list(scene.transmitters.keys()):
        scene.remove(name)

    # Antenna arrays
    scene.tx_array = rt.PlanarArray(
        num_rows=4, num_cols=4, vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="tr38901", polarization="V"
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V"
    )

    tilt_rad = float(np.radians(downtilt_deg))
    offset_dist = 2.5
    sector_colors = [(1.0, 0.2, 0.2), (0.2, 0.85, 0.2), (0.2, 0.45, 1.0)]

    for i, yaw_deg in enumerate(sector_yaws):
        yaw_rad = float(np.radians(yaw_deg))
        dx = float(offset_dist * np.cos(yaw_rad))
        dy = float(offset_dist * np.sin(yaw_rad))
        pos = [tower_pos[0] + dx, tower_pos[1] + dy, tower_pos[2]]

        tx = rt.Transmitter(
            name=f"{tag}_Sec{i+1}",
            position=pos,
            orientation=[yaw_rad, tilt_rad, 0.0],
            power_dbm=float(tx_power_dbm),
            color=sector_colors[i % len(sector_colors)],
            display_radius=8.0
        )
        scene.add(tx)

    solver = rt.RadioMapSolver()
    rm = solver(
        scene=scene,
        center=[tower_pos[0], tower_pos[1], 1.5],
        orientation=[0.0, 0.0, 0.0],
        size=[sim_size, sim_size],
        cell_size=[cell_size, cell_size],
        samples_per_tx=int(samples_per_tx),
        max_depth=max_depth,
        specular_reflection=True,
        diffraction=True
    )
    print(f"{tag} ray tracing completed in {time.time()-t0:.2f}s!")

    # Extract best-server path gain
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

    eps = 1e-20
    pg_db = 10.0 * np.log10(np.maximum(pg_best, eps))
    rsrp_dbm = tx_power_dbm + pg_db
    return rsrp_dbm, rm


def calculate_kpis(rsrp, threshold_good=-95.0, threshold_usable=-105.0, threshold_dead=-115.0):
    """Compute standard telecom coverage KPIs."""
    total_cells = rsrp.size
    pct_good = (rsrp >= threshold_good).sum() / total_cells * 100.0
    pct_usable = (rsrp >= threshold_usable).sum() / total_cells * 100.0
    pct_dead = (rsrp < threshold_dead).sum() / total_cells * 100.0
    mean_rsrp = float(np.mean(rsrp))
    return {
        "pct_good": pct_good,
        "pct_usable": pct_usable,
        "pct_dead": pct_dead,
        "mean_rsrp": mean_rsrp
    }


def plot_what_if_comparison(rsrp_a, rsrp_b, delta, kpis_a, kpis_b, scene_info,
                            params_a, params_b, output_path, sim_size=2000.0,
                            tower_pos=(0.0, 0.0)):
    """
    Plot 3-panel figure: Baseline RSRP, What-If RSRP, and Delta Gain/Loss Map.
    """
    print(f"\nGenerating 3-Panel What-If Comparison Visualizations...")
    t0 = time.time()
    half = sim_size / 2.0
    extent = [tower_pos[0] - half, tower_pos[0] + half, tower_pos[1] - half, tower_pos[1] + half]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5), dpi=300)

    # 1. Overlay building footprints if available
    gpkg_path = scene_info.get("source_gpkg", "")
    gdf_local = None
    if gpkg_path and os.path.isfile(gpkg_path):
        try:
            cx_utm = scene_info["center_utm_x"]
            cy_utm = scene_info["center_utm_y"]
            bbox_utm = box(cx_utm + extent[0], cy_utm + extent[2], cx_utm + extent[1], cy_utm + extent[3])
            gdf_local = gpd.read_file(gpkg_path, bbox=bbox_utm)
            if not gdf_local.empty:
                gdf_local["geometry"] = gdf_local["geometry"].translate(xoff=-cx_utm, yoff=-cy_utm)
        except Exception:
            pass

    # --- Panel 1: Baseline Config A ---
    ax1 = axes[0]
    im1 = ax1.imshow(rsrp_a, origin="lower", extent=extent, cmap="turbo", vmin=-125.0, vmax=-50.0, interpolation="bilinear")
    if gdf_local is not None:
        gdf_local.boundary.plot(ax=ax1, color="#111111", linewidth=0.25, alpha=0.6)
        gdf_local.plot(ax=ax1, facecolor="#222222", alpha=0.3)
    ax1.scatter([tower_pos[0]], [tower_pos[1]], color="red", edgecolor="white", s=90, marker="^", zorder=10)
    ax1.set_title(
        f"Baseline (Config A)\nHeight: {params_a['height']:.0f}m | Tilt: {params_a['tilt']:.0f}° | Power: {params_a['power']:.0f} dBm\n"
        f"Good (>-95 dBm): {kpis_a['pct_good']:.1f}% | Dead (<-115 dBm): {kpis_a['pct_dead']:.1f}%",
        fontsize=11, fontweight="bold", pad=8
    )
    ax1.set_xlabel("Local X (m)")
    ax1.set_ylabel("Local Y (m)")
    ax1.grid(True, linestyle="--", alpha=0.3, color="white")
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label("RSRP (dBm)", fontsize=10)

    # --- Panel 2: What-If Config B ---
    ax2 = axes[1]
    im2 = ax2.imshow(rsrp_b, origin="lower", extent=extent, cmap="turbo", vmin=-125.0, vmax=-50.0, interpolation="bilinear")
    if gdf_local is not None:
        gdf_local.boundary.plot(ax=ax2, color="#111111", linewidth=0.25, alpha=0.6)
        gdf_local.plot(ax=ax2, facecolor="#222222", alpha=0.3)
    ax2.scatter([tower_pos[0]], [tower_pos[1]], color="lime", edgecolor="black", s=90, marker="^", zorder=10)
    ax2.set_title(
        f"What-If (Config B - Optimized)\nHeight: {params_b['height']:.0f}m | Tilt: {params_b['tilt']:.0f}° | Power: {params_b['power']:.0f} dBm\n"
        f"Good (>-95 dBm): {kpis_b['pct_good']:.1f}% | Dead (<-115 dBm): {kpis_b['pct_dead']:.1f}%",
        fontsize=11, fontweight="bold", pad=8
    )
    ax2.set_xlabel("Local X (m)")
    ax2.grid(True, linestyle="--", alpha=0.3, color="white")
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("RSRP (dBm)", fontsize=10)

    # --- Panel 3: Delta Gain/Loss Map (Config B - Config A) ---
    ax3 = axes[2]
    # Diverging colormap centered at 0 dB
    # Clip delta to [-20, +20] dB for maximum visual clarity
    v_abs = 15.0
    norm = TwoSlopeNorm(vmin=-v_abs, vcenter=0.0, vmax=v_abs)
    im3 = ax3.imshow(delta, origin="lower", extent=extent, cmap="RdYlGn", norm=norm, interpolation="bilinear")
    if gdf_local is not None:
        gdf_local.boundary.plot(ax=ax3, color="#111111", linewidth=0.25, alpha=0.6)
        gdf_local.plot(ax=ax3, facecolor="#222222", alpha=0.3)
    ax3.scatter([tower_pos[0]], [tower_pos[1]], color="blue", edgecolor="white", s=90, marker="o", zorder=10)
    
    mean_gain = float(np.mean(delta))
    pct_improved = (delta > 1.0).sum() / delta.size * 100.0
    ax3.set_title(
        f"Delta Coverage Gain / Loss (Δ RSRP = B - A)\n"
        f"Avg Delta: {mean_gain:+.2f} dB | Streets Improved: {pct_improved:.1f}%\n"
        f"Green = Signal Boost | Red = Overshoot Reduction",
        fontsize=11, fontweight="bold", pad=8
    )
    ax3.set_xlabel("Local X (m)")
    ax3.grid(True, linestyle="--", alpha=0.3, color="white")
    cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.set_label("Δ RSRP Gain / Loss (dB)", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"Cairo Digital Network Twin — Tower Parameter 'What-If' Optimization (1.8 GHz)\n"
        f"Comparing Baseline (Z={params_a['height']:.0f}m, 0° tilt, {params_a['power']:.0f}dBm) vs. "
        f"Optimized (Z={params_b['height']:.0f}m, {params_b['tilt']:.0f}° tilt, {params_b['power']:.0f}dBm)",
        fontsize=13, fontweight="bold", y=0.98
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved 3-Panel What-If Comparison: {output_path} ({time.time()-t0:.2f}s)")


def generate_interactive_3d_comparison_html(rsrp_a, rsrp_b, delta, scene_info,
                                            params_a, params_b, kpis_a, kpis_b,
                                            output_path="outputs/what_if_3d_viewer.html",
                                            sim_size=2000.0, tower_pos=(0.0, 0.0)):
    """
    Generate an interactive 3D WebGL viewer allowing instant toggling between:
    - [Baseline Config A]
    - [What-If Config B]
    - [Delta Gain/Loss Map]
    """
    print(f"\nGenerating Interactive 3D What-If Viewer ({output_path})...")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save normalized textures and convert to Base64 to bypass CORS when opened via file://
    import base64
    import io

    vmin, vmax = -125.0, -50.0
    cmap_turbo = plt.get_cmap("turbo")
    tex_a = cmap_turbo(np.clip((rsrp_a - vmin) / (vmax - vmin), 0.0, 1.0))
    tex_b = cmap_turbo(np.clip((rsrp_b - vmin) / (vmax - vmin), 0.0, 1.0))
    cmap_delta = plt.get_cmap("RdYlGn")
    norm_delta = np.clip((delta - (-15.0)) / (30.0), 0.0, 1.0)
    tex_delta = cmap_delta(norm_delta)

    def to_b64(img_array):
        buf = io.BytesIO()
        plt.imsave(buf, img_array, format="png", origin="lower")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    b64_a = to_b64(tex_a)
    b64_b = to_b64(tex_b)
    b64_d = to_b64(tex_delta)

    # Load buildings (crucial: explode MultiPolygons so footprints are parsed)
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
            print(f"Note: building loading error: {e}")
    print(f"Embedded {len(building_data):,} 3D buildings into interactive What-If viewer.")

    mean_gain = float(np.mean(delta))
    pct_improved = (delta > 1.0).sum() / delta.size * 100.0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Cairo Digital Twin — Tower What-If Comparison</title>
    <style>
        body {{ margin: 0; overflow: hidden; background: #0a0e17; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #fff; }}
        #control-panel {{ position: absolute; top: 15px; left: 15px; background: rgba(13, 20, 36, 0.9); backdrop-filter: blur(10px); padding: 18px 22px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.15); max-width: 350px; z-index: 100; box-shadow: 0 10px 30px rgba(0,0,0,0.6); }}
        #control-panel h2 {{ margin: 0 0 6px 0; font-size: 17px; color: #38bdf8; }}
        #control-panel p {{ margin: 3px 0; font-size: 12px; color: #94a3b8; }}
        .btn-group {{ display: flex; gap: 8px; margin: 12px 0 10px 0; }}
        .btn {{ flex: 1; padding: 8px 10px; border-radius: 6px; font-weight: 600; font-size: 11px; cursor: pointer; border: 1px solid rgba(255,255,255,0.2); background: rgba(255,255,255,0.08); color: #fff; transition: all 0.2s ease; text-align: center; }}
        .btn:hover {{ background: rgba(255,255,255,0.18); }}
        .btn.active {{ background: #0284c7; border-color: #38bdf8; color: #fff; box-shadow: 0 0 12px rgba(2,132,199,0.5); }}
        .kpi-card {{ background: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px; margin-top: 10px; font-size: 12px; }}
        .kpi-val {{ font-weight: bold; color: #4ade80; }}
        #legend {{ position: absolute; bottom: 20px; right: 20px; background: rgba(13, 20, 36, 0.9); backdrop-filter: blur(8px); padding: 12px 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.15); z-index: 100; }}
        .cbar {{ width: 190px; height: 12px; border-radius: 3px; }}
        .cbar-turbo {{ background: linear-gradient(to right, #30123b, #4662d8, #1ae4b6, #a2fc3c, #faba39, #e23412, #7a0403); }}
        .cbar-delta {{ background: linear-gradient(to right, #d73027, #fc8d59, #fee08b, #ffffbf, #d9ef8b, #91cf60, #1a9850); }}
        .cbar-labels {{ display: flex; justify-content: space-between; font-size: 10px; color: #94a3b8; margin-top: 4px; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="control-panel">
        <h2>What-If Tower Optimizer</h2>
        <p><strong>Carrier:</strong> 1.8 GHz | Simulation Grid: 2000m × 2000m</p>
        
        <div class="btn-group">
            <div id="btn-a" class="btn" onclick="setMode('a')">Baseline (28m)</div>
            <div id="btn-b" class="btn active" onclick="setMode('b')">What-If (38m)</div>
            <div id="btn-delta" class="btn" onclick="setMode('delta')">Δ Gain Map</div>
        </div>

        <div class="kpi-card">
            <div id="info-text">
                <div><strong>Mast Height:</strong> <span id="val-h">{params_b['height']:.0f}m</span></div>
                <div><strong>Downtilt:</strong> <span id="val-t">{params_b['tilt']:.0f}°</span> | <strong>Power:</strong> <span id="val-p">{params_b['power']:.0f} dBm</span></div>
                <div style="margin-top: 4px;"><strong>Good Coverage (&gt;-95 dBm):</strong> <span class="kpi-val" id="val-kpi">{kpis_b['pct_good']:.1f}%</span> (Baseline: {kpis_a['pct_good']:.1f}%)</div>
                <div><strong>Dead Zone (&lt;-115 dBm):</strong> <span style="color:#f87171;" id="val-dead">{kpis_b['pct_dead']:.1f}%</span> (Baseline: {kpis_a['pct_dead']:.1f}%)</div>
            </div>
        </div>
    </div>

    <div id="legend">
        <div id="legend-title" style="font-size: 11px; font-weight: bold; margin-bottom: 4px;">RSRP Coverage (dBm)</div>
        <div id="cbar-el" class="cbar cbar-turbo"></div>
        <div id="cbar-labels" class="cbar-labels"><span>-125 dBm</span><span>-90 dBm</span><span>-50 dBm</span></div>
    </div>

    <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0e17);
        scene.fog = new THREE.FogExp2(0x0a0e17, 0.001);

        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 4000);
        camera.position.set({tower_pos[0]} + 240, 180, {tower_pos[1]} + 260);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set({tower_pos[0]}, 30, {tower_pos[1]});
        controls.enableDamping = true;

        scene.add(new THREE.AmbientLight(0xffffff, 0.7));
        const dir = new THREE.DirectionalLight(0xffffff, 0.8);
        dir.position.set({tower_pos[0]} + 200, 400, {tower_pos[1]} + 300);
        scene.add(dir);

        // Ground Plane & Embedded Textures (Base64 eliminates CORS)
        const texLoader = new THREE.TextureLoader();
        const texA = texLoader.load('{b64_a}');
        const texB = texLoader.load('{b64_b}');
        const texDelta = texLoader.load('{b64_d}');

        const groundGeo = new THREE.PlaneGeometry({sim_size}, {sim_size});
        const groundMat = new THREE.MeshLambertMaterial({{ map: texB, side: THREE.DoubleSide }});
        const ground = new THREE.Mesh(groundGeo, groundMat);
        ground.rotation.x = -Math.PI / 2;
        ground.position.set({tower_pos[0]}, 0, {tower_pos[1]});
        scene.add(ground);

        // 3D Buildings with Architectural Shading and Sharp Edge Highlights
        const buildingData = {json.dumps(building_data)};
        const bldgMat = new THREE.MeshStandardMaterial({{ color: 0x475569, roughness: 0.5, metalness: 0.2, flatShading: true }});
        const edgeMat = new THREE.LineBasicMaterial({{ color: 0x94a3b8, transparent: true, opacity: 0.7 }});

        buildingData.forEach(b => {{
            const shape = new THREE.Shape();
            const pts = b.c || b.coords;
            pts.forEach((p, idx) => {{ if (idx === 0) shape.moveTo(p[0], -p[1]); else shape.lineTo(p[0], -p[1]); }});
            const geom = new THREE.ExtrudeGeometry(shape, {{ depth: b.h, bevelEnabled: false }});
            const mesh = new THREE.Mesh(geom, bldgMat);
            mesh.rotation.x = -Math.PI / 2;
            scene.add(mesh);

            const edgesGeom = new THREE.EdgesGeometry(geom, 25);
            const edges = new THREE.LineSegments(edgesGeom, edgeMat);
            edges.rotation.x = -Math.PI / 2;
            scene.add(edges);
        }});

        // Dynamic Mast
        let mastHeight = {params_b['height']};
        const mastGeo = new THREE.CylinderGeometry(0.8, 1.2, mastHeight, 16);
        const mastMat = new THREE.MeshStandardMaterial({{ color: 0x64748b, metalness: 0.8 }});
        const mast = new THREE.Mesh(mastGeo, mastMat);
        mast.position.set({tower_pos[0]}, mastHeight / 2, {tower_pos[1]});
        scene.add(mast);

        // Tower Sphere Beacon
        const sphereGeo = new THREE.SphereGeometry(3.5, 32, 32);
        const sphereMat = new THREE.MeshStandardMaterial({{ color: 0x38bdf8, emissive: 0x0284c7, emissiveIntensity: 0.8 }});
        const sphere = new THREE.Mesh(sphereGeo, sphereMat);
        sphere.position.set({tower_pos[0]}, mastHeight, {tower_pos[1]});
        scene.add(sphere);

        function updateMast(h) {{
            mast.scale.y = h / {params_b['height']};
            mast.position.set({tower_pos[0]}, h / 2, {tower_pos[1]});
            sphere.position.set({tower_pos[0]}, h, {tower_pos[1]});
        }}

        function setMode(mode) {{
            document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
            const cbarEl = document.getElementById('cbar-el');
            const cbarLabels = document.getElementById('cbar-labels');
            const legendTitle = document.getElementById('legend-title');

            if (mode === 'a') {{
                document.getElementById('btn-a').classList.add('active');
                groundMat.map = texA;
                groundMat.needsUpdate = true;
                updateMast({params_a['height']});
                document.getElementById('val-h').innerText = "{params_a['height']:.0f}m";
                document.getElementById('val-t').innerText = "{params_a['tilt']:.0f}°";
                document.getElementById('val-p').innerText = "{params_a['power']:.0f} dBm";
                document.getElementById('val-kpi').innerText = "{kpis_a['pct_good']:.1f}%";
                document.getElementById('val-dead').innerText = "{kpis_a['pct_dead']:.1f}%";
                cbarEl.className = "cbar cbar-turbo";
                legendTitle.innerText = "RSRP Coverage (dBm)";
                cbarLabels.innerHTML = "<span>-125 dBm</span><span>-90 dBm</span><span>-50 dBm</span>";
            }} else if (mode === 'b') {{
                document.getElementById('btn-b').classList.add('active');
                groundMat.map = texB;
                groundMat.needsUpdate = true;
                updateMast({params_b['height']});
                document.getElementById('val-h').innerText = "{params_b['height']:.0f}m";
                document.getElementById('val-t').innerText = "{params_b['tilt']:.0f}°";
                document.getElementById('val-p').innerText = "{params_b['power']:.0f} dBm";
                document.getElementById('val-kpi').innerText = "{kpis_b['pct_good']:.1f}%";
                document.getElementById('val-dead').innerText = "{kpis_b['pct_dead']:.1f}%";
                cbarEl.className = "cbar cbar-turbo";
                legendTitle.innerText = "RSRP Coverage (dBm)";
                cbarLabels.innerHTML = "<span>-125 dBm</span><span>-90 dBm</span><span>-50 dBm</span>";
            }} else if (mode === 'delta') {{
                document.getElementById('btn-delta').classList.add('active');
                groundMat.map = texDelta;
                groundMat.needsUpdate = true;
                updateMast({params_b['height']});
                document.getElementById('val-h').innerText = "Δ Gain View";
                document.getElementById('val-t').innerText = "Avg: {mean_gain:+.2f} dB";
                document.getElementById('val-p').innerText = "{pct_improved:.1f}% Improved";
                document.getElementById('val-kpi').innerText = "+{kpis_b['pct_good'] - kpis_a['pct_good']:.1f}% Gain";
                document.getElementById('val-dead').innerText = "{kpis_b['pct_dead'] - kpis_a['pct_dead']:.1f}% Dead";
                cbarEl.className = "cbar cbar-delta";
                legendTitle.innerText = "Δ Signal Gain / Loss (dB)";
                cbarLabels.innerHTML = "<span>-15 dB (Loss)</span><span>0 dB</span><span>+15 dB (Gain)</span>";
            }}
        }}

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
        f.write(html)
    print(f"Saved Interactive 3D What-If Comparison Viewer: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run What-If Tower Configuration Optimization Scenario.")
    parser.add_argument("--scene", default="cairo_scene/scene.xml")
    parser.add_argument("--scene_info", default="cairo_scene/scene_info.json")
    parser.add_argument("--frequency", type=float, default=1.8e9)
    parser.add_argument("--lat", type=float, default=None, help="Tower GPS latitude WGS84")
    parser.add_argument("--lon", type=float, default=None, help="Tower GPS longitude WGS84")
    parser.add_argument("--tower_x", type=float, default=0.0, help="Tower local X coordinate (m)")
    parser.add_argument("--tower_y", type=float, default=0.0, help="Tower local Y coordinate (m)")
    # Config A (Baseline)
    parser.add_argument("--height_a", type=float, default=28.0, help="Baseline tower mast height Z (m)")
    parser.add_argument("--tilt_a", type=float, default=0.0, help="Baseline mechanical downtilt (deg)")
    parser.add_argument("--power_a", type=float, default=44.0, help="Baseline transmit power (dBm)")
    # Config B (What-If Optimized)
    parser.add_argument("--height_b", type=float, default=38.0, help="What-If tower mast height Z (m)")
    parser.add_argument("--tilt_b", type=float, default=6.0, help="What-If mechanical downtilt (deg)")
    parser.add_argument("--power_b", type=float, default=46.0, help="What-If transmit power (dBm)")
    # Shared simulation settings
    parser.add_argument("--sim_size", type=float, default=2000.0)
    parser.add_argument("--cell_size", type=float, default=5.0)
    parser.add_argument("--samples_per_tx", type=int, default=10_000_000)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--output_dir", default="outputs")
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

    sector_yaws = (0.0, 120.0, 240.0)

    # 1. Run Baseline (Config A)
    scene_a = load_cairo_scene(args.scene, frequency_hz=args.frequency)
    params_a = {"height": args.height_a, "tilt": args.tilt_a, "power": args.power_a}
    rsrp_a, _ = configure_and_simulate(
        scene=scene_a,
        tower_pos=(args.tower_x, args.tower_y, args.height_a),
        sector_yaws=sector_yaws,
        downtilt_deg=args.tilt_a,
        tx_power_dbm=args.power_a,
        sim_size=args.sim_size,
        cell_size=args.cell_size,
        samples_per_tx=args.samples_per_tx,
        max_depth=args.max_depth,
        tag="ConfigA_Baseline"
    )

    # Cleanly synchronize CUDA stream and release memory to prevent Dr.Jit async task conflicts
    import gc
    try:
        import drjit as dr
        dr.sync_thread()
    except Exception:
        pass
    del scene_a
    gc.collect()
    try:
        import drjit as dr
        dr.sync_thread()
    except Exception:
        pass

    # 2. Run What-If (Config B) - Use a fresh scene instance to prevent Dr.Jit pointer reuse errors
    scene_b = load_cairo_scene(args.scene, frequency_hz=args.frequency)
    params_b = {"height": args.height_b, "tilt": args.tilt_b, "power": args.power_b}
    rsrp_b, _ = configure_and_simulate(
        scene=scene_b,
        tower_pos=(args.tower_x, args.tower_y, args.height_b),
        sector_yaws=sector_yaws,
        downtilt_deg=args.tilt_b,
        tx_power_dbm=args.power_b,
        sim_size=args.sim_size,
        cell_size=args.cell_size,
        samples_per_tx=args.samples_per_tx,
        max_depth=args.max_depth,
        tag="ConfigB_Optimized"
    )

    try:
        import drjit as dr
        dr.sync_thread()
    except Exception:
        pass
    del scene_b
    gc.collect()

    # 3. Compute Delta Map: B - A
    delta_rsrp = rsrp_b - rsrp_a

    # 4. Calculate Telecom KPIs
    kpis_a = calculate_kpis(rsrp_a)
    kpis_b = calculate_kpis(rsrp_b)

    print("\n" + "="*60)
    print("TELECOM WHAT-IF KPI SCORECARD")
    print("="*60)
    print(f"Parameter              | Baseline (Config A) | What-If (Config B)  | Delta Gain")
    print(f"-----------------------+---------------------+---------------------+-----------")
    print(f"Mast Height            | {args.height_a:6.1f} m            | {args.height_b:6.1f} m            | {args.height_b - args.height_a:+5.1f} m")
    print(f"Mechanical Downtilt    | {args.tilt_a:6.1f}°            | {args.tilt_b:6.1f}°            | {args.tilt_b - args.tilt_a:+5.1f}°")
    print(f"Transmit Power         | {args.power_a:6.1f} dBm          | {args.power_b:6.1f} dBm          | {args.power_b - args.power_a:+5.1f} dB")
    print(f"Good Coverage (>-95)   | {kpis_a['pct_good']:6.1f} %          | {kpis_b['pct_good']:6.1f} %          | {kpis_b['pct_good'] - kpis_a['pct_good']:+5.1f} %")
    print(f"Usable Coverage (>-105)| {kpis_a['pct_usable']:6.1f} %          | {kpis_b['pct_usable']:6.1f} %          | {kpis_b['pct_usable'] - kpis_a['pct_usable']:+5.1f} %")
    print(f"Dead Zones (<-115 dBm) | {kpis_a['pct_dead']:6.1f} %          | {kpis_b['pct_dead']:6.1f} %          | {kpis_b['pct_dead'] - kpis_a['pct_dead']:+5.1f} %")
    print(f"Average Signal Level   | {kpis_a['mean_rsrp']:6.1f} dBm        | {kpis_b['mean_rsrp']:6.1f} dBm        | {kpis_b['mean_rsrp'] - kpis_a['mean_rsrp']:+5.1f} dB")
    print("="*60)

    # 5. Generate 3-Panel Visualizations
    comp_png_path = os.path.join(args.output_dir, "what_if_tower_comparison.png")
    plot_what_if_comparison(
        rsrp_a=rsrp_a,
        rsrp_b=rsrp_b,
        delta=delta_rsrp,
        kpis_a=kpis_a,
        kpis_b=kpis_b,
        scene_info=scene_info,
        params_a=params_a,
        params_b=params_b,
        output_path=comp_png_path,
        sim_size=args.sim_size,
        tower_pos=(args.tower_x, args.tower_y)
    )

    # 6. Generate Interactive 3D WebGL Comparison Viewer
    viewer_path = os.path.join(args.output_dir, "what_if_3d_viewer.html")
    generate_interactive_3d_comparison_html(
        rsrp_a=rsrp_a,
        rsrp_b=rsrp_b,
        delta=delta_rsrp,
        scene_info=scene_info,
        params_a=params_a,
        params_b=params_b,
        kpis_a=kpis_a,
        kpis_b=kpis_b,
        output_path=viewer_path,
        sim_size=args.sim_size,
        tower_pos=(args.tower_x, args.tower_y)
    )

    print("\n=======================================================")
    print("What-If Tower Optimization Completed Successfully!")
    print(f"3-Panel Comparison Map: {os.path.abspath(comp_png_path)}")
    print(f"Interactive 3D Viewer:  {os.path.abspath(viewer_path)}")
    print("=======================================================")


if __name__ == "__main__":
    main()
