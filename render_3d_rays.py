"""
render_3d_rays.py
Simulates and visualizes 3D physical RF ray beams bouncing off Cairo buildings:
- Traces Line-of-Sight (LoS), single/double reflections, and edge diffraction.
- Renders high-resolution 3D ray-traced snapshot (outputs/scene_3d_rays.png).
- Generates standalone interactive 3D WebGL viewer (outputs/cairo_3d_rays_viewer.html)
  with glowing laser ray paths, bounce points, and interactive filters.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
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


def setup_transmitters_and_receivers(scene, tower_pos=(0.0, 0.0, 28.0), tx_power_dbm=44.0):
    """
    Set up the 3-sector tower and place mobile user receivers at diverse street locations:
    - Open boulevard (LoS)
    - Side streets behind apartment blocks (1st-order reflections)
    - Urban canyons between opposing buildings (2nd-order reflections)
    - Shadowed alleys (diffraction around building corners)
    """
    print("Setting up 3-Sector Tower and Mobile User Receivers...")
    
    scene.tx_array = rt.PlanarArray(
        num_rows=4, num_cols=4, vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="tr38901", polarization="V"
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V"
    )

    # 3 Sectors
    sector_yaws = [0.0, 120.0, 240.0]
    sector_colors = [(1.0, 0.2, 0.2), (0.2, 0.85, 0.2), (0.2, 0.45, 1.0)]
    offset_dist = 2.5

    for i, yaw_deg in enumerate(sector_yaws):
        yaw_rad = float(np.radians(yaw_deg))
        dx = float(offset_dist * np.cos(yaw_rad))
        dy = float(offset_dist * np.sin(yaw_rad))
        pos = [tower_pos[0] + dx, tower_pos[1] + dy, tower_pos[2]]

        tx = rt.Transmitter(
            name=f"Tower_Sec{i+1}_{int(yaw_deg)}deg",
            position=pos,
            orientation=[yaw_rad, float(np.radians(6.0)), 0.0],  # 6 deg downtilt
            power_dbm=float(tx_power_dbm),
            color=sector_colors[i],
            display_radius=6.0
        )
        scene.add(tx)

    # Strategic receiver locations across Cairo streets relative to tower position
    rx_offsets = [
        {"name": "UE1 (Open Boulevard - LoS)", "offset": [80.0, 50.0, 1.5], "color": (0.2, 0.9, 0.3)},
        {"name": "UE2 (Side Alley - 1st Bounce)", "offset": [140.0, -90.0, 1.5], "color": (1.0, 0.8, 0.2)},
        {"name": "UE3 (Behind Tower - Reflected)", "offset": [-90.0, 120.0, 1.5], "color": (0.3, 0.8, 1.0)},
        {"name": "UE4 (Deep Shadow - Diffracted)", "offset": [-150.0, -110.0, 1.5], "color": (0.9, 0.3, 0.9)},
        {"name": "UE5 (Street Corridor - 2nd Bounce)", "offset": [190.0, 70.0, 1.5], "color": (1.0, 0.5, 0.1)},
        {"name": "UE6 (Intersection Corner)", "offset": [50.0, -170.0, 1.5], "color": (0.2, 0.9, 0.8)},
        {"name": "UE7 (Courtyard Receiver)", "offset": [-60.0, -150.0, 1.5], "color": (0.4, 0.6, 1.0)},
        {"name": "UE8 (Avenue North)", "offset": [110.0, 150.0, 1.5], "color": (0.5, 1.0, 0.4)}
    ]

    rx_scenarios = []
    for j, rx_info in enumerate(rx_offsets):
        rx_pos = [
            tower_pos[0] + rx_info["offset"][0],
            tower_pos[1] + rx_info["offset"][1],
            rx_info["offset"][2]
        ]
        rx_scenarios.append({
            "name": rx_info["name"],
            "pos": rx_pos,
            "color": rx_info["color"]
        })
        rx = rt.Receiver(
            name=f"rx_{j+1}",
            position=rx_pos,
            color=rx_info["color"],
            display_radius=3.0
        )
        scene.add(rx)

    print(f"Placed 3 Transmitters and {len(rx_scenarios)} Mobile Receivers across street locations.")
    return rx_scenarios


def compute_physical_paths(scene, max_depth=3):
    """Run Sionna RT's PathSolver to calculate physical multi-bounce ray trajectories."""
    print("\nTracing 3D physical ray trajectories with Sionna RT PathSolver...")
    print(f" - Max Bounces: {max_depth} | Specular Reflections: True | Diffraction: True")
    t0 = time.time()

    solver = rt.PathSolver()
    paths = solver(
        scene=scene,
        max_depth=max_depth,
        samples_per_src=100_000,
        los=True,
        specular_reflection=True,
        diffraction=True
    )
    print(f"Physical 3D ray paths computed in {time.time()-t0:.2f}s!")
    return paths


def extract_ray_segments(paths, max_paths_to_export=300):
    """
    Extract 3D polyline coordinates and interaction types from Sionna Paths:
    - Type 0: Line-of-Sight (LoS)
    - Type 1: Specular Reflection (bouncing off building facade)
    - Type 4: Edge Diffraction (bending around rooftop/corner)
    """
    verts = paths.vertices.numpy()  # [num_tx, num_rx, max_depth, num_paths, 3]
    types = paths.interactions.numpy()  # [num_tx, num_rx, max_depth, num_paths]
    sources = paths.sources.numpy().T  # [num_tx, 3]
    targets = paths.targets.numpy().T  # [num_rx, 3]

    num_tx, num_rx, max_depth, num_paths, _ = verts.shape
    ray_data = []

    for tx_i in range(num_tx):
        tx_pos = [float(c) for c in sources[tx_i]]
        for rx_j in range(num_rx):
            rx_pos = [float(c) for c in targets[rx_j]]

            for p in range(num_paths):
                seg = verts[tx_i, rx_j, :, p, :]
                interaction_seq = types[tx_i, rx_j, :, p]

                # Extract valid non-zero vertices along path
                path_pts = [tx_pos]
                valid_bounces = []
                for d in range(max_depth):
                    pt = seg[d]
                    if not np.allclose(pt, 0.0) and not np.isnan(pt[0]):
                        path_pts.append(pt.tolist())
                        valid_bounces.append(int(interaction_seq[d]))
                path_pts.append(rx_pos)

                if len(path_pts) >= 2:
                    # Classify primary path type
                    if len(valid_bounces) == 0 or 0 in valid_bounces:
                        path_type = "LoS"
                        color = "#00ff66"  # Neon Green
                    elif 4 in valid_bounces:
                        path_type = "Diffraction"
                        color = "#ff00ff"  # Magenta
                    elif len(valid_bounces) == 1:
                        path_type = "Single-Reflection"
                        color = "#ffaa00"  # Gold / Orange
                    else:
                        path_type = "Multi-Reflection"
                        color = "#00ddff"  # Cyan / Blue

                    ray_data.append({
                        "tx": int(tx_i),
                        "rx": int(rx_j),
                        "type": path_type,
                        "color": color,
                        "bounces": len(path_pts) - 2,
                        "points": path_pts
                    })

    # Sort rays to keep a balanced, visually stunning mix of LoS, reflections, and diffractions
    los_rays = [r for r in ray_data if r["type"] == "LoS"]
    refl_rays = [r for r in ray_data if "Reflection" in r["type"]]
    diff_rays = [r for r in ray_data if r["type"] == "Diffraction"]

    selected_rays = (
        los_rays[:30] +
        refl_rays[:max_paths_to_export - 60] +
        diff_rays[:30]
    )
    print(f"Extracted {len(ray_data):,} total paths. Selected top {len(selected_rays)} visual beam paths:")
    print(f" - Line-of-Sight rays: {len([r for r in selected_rays if r['type'] == 'LoS'])}")
    print(f" - Facade reflection rays: {len([r for r in selected_rays if 'Reflection' in r['type']])}")
    print(f" - Corner diffraction rays: {len([r for r in selected_rays if r['type'] == 'Diffraction'])}")
    return selected_rays


def render_mitsuba_3d_rays(scene, paths, output_path, cam_pos=(180.0, -240.0, 130.0), look_at=(0.0, 0.0, 20.0)):
    """Render physical ray trajectories using Sionna RT's native renderer."""
    print(f"\nRendering Mitsuba 3D Ray-Beam Snapshot ({output_path})...")
    t0 = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cam = rt.Camera(position=list(cam_pos), look_at=list(look_at))

    try:
        scene.render_to_file(
            camera=cam,
            filename=output_path,
            paths=paths,
            num_samples=64,
            resolution=(1280, 720),
            show_devices=True,
            show_orientations=False
        )
        print(f"Saved Mitsuba 3D Ray-Beam Snapshot: {output_path} ({time.time()-t0:.2f}s)")
    except Exception as e:
        print(f"Note: Mitsuba snapshot render skipped or unsupported on this GPU: {e}")


def generate_interactive_ray_viewer_html(scene_info, rx_scenarios, selected_rays,
                                        output_path="outputs/cairo_3d_rays_viewer.html",
                                        tower_pos=(0.0, 0.0, 28.0)):
    """
    Generate an interactive 3D WebGL viewer showing:
    - 3D Cairo buildings with clean architectural semi-transparency and crisp outlines.
    - The central 28m mast and 3-sector transmitter.
    - Glowing 3D laser ray beams with animated pulse and bounce markers on building walls.
    - Interactive controls to filter LoS, 1st-reflection, 2nd-reflection, and diffraction.
    """
    print(f"\nGenerating Interactive 3D WebGL Ray-Beam Viewer ({output_path})...")
    
    # Load nearby buildings (within 500m) to embed
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
                # Crucial: explode MultiPolygons into Polygons
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
            print(f"Note: building embedding error: {e}")
    print(f"Embedded {len(building_data):,} 3D buildings into interactive WebGL viewer.")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Cairo 3D RF Ray-Beam Tracing Visualizer</title>
    <style>
        body {{ margin: 0; overflow: hidden; background: #080c14; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #fff; }}
        #control-panel {{ position: absolute; top: 15px; left: 15px; background: rgba(11, 18, 32, 0.9); backdrop-filter: blur(12px); padding: 18px 22px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.15); max-width: 340px; z-index: 100; box-shadow: 0 10px 30px rgba(0,0,0,0.6); }}
        #control-panel h2 {{ margin: 0 0 6px 0; font-size: 17px; color: #38bdf8; }}
        #control-panel p {{ margin: 3px 0; font-size: 12px; color: #94a3b8; }}
        .filter-row {{ display: flex; align-items: center; justify-content: space-between; margin: 8px 0; font-size: 12px; }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 6px; vertical-align: middle; }}
        .toggle-btn {{ cursor: pointer; padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: bold; border: 1px solid rgba(255,255,255,0.2); background: rgba(255,255,255,0.08); color: #fff; }}
        .toggle-btn.active {{ background: #0284c7; border-color: #38bdf8; }}
        #instructions {{ position: absolute; bottom: 20px; left: 15px; font-size: 11px; color: #64748b; background: rgba(0,0,0,0.4); padding: 6px 12px; border-radius: 6px; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="control-panel">
        <h2>3D Physical Ray-Beam Tracing</h2>
        <p><strong>Frequency:</strong> 1.8 GHz | 3-Sector Macro Site (Z = {tower_pos[2]:.0f}m)</p>
        <p style="margin-top: 8px; font-size: 11px; color: #cbd5e1;">Visualizing real multipath reflection and diffraction rays bouncing into street canyons:</p>

        <div style="margin-top: 12px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px;">
            <div class="filter-row">
                <span><span class="legend-dot" style="background: #00ff66;"></span>Line-of-Sight (LoS)</span>
                <span id="btn-los" class="toggle-btn active" onclick="toggleRayType('LoS')">Visible</span>
            </div>
            <div class="filter-row">
                <span><span class="legend-dot" style="background: #ffaa00;"></span>1st-Bounce Reflection</span>
                <span id="btn-refl1" class="toggle-btn active" onclick="toggleRayType('Single-Reflection')">Visible</span>
            </div>
            <div class="filter-row">
                <span><span class="legend-dot" style="background: #00ddff;"></span>2nd-Bounce Reflection</span>
                <span id="btn-refl2" class="toggle-btn active" onclick="toggleRayType('Multi-Reflection')">Visible</span>
            </div>
            <div class="filter-row">
                <span><span class="legend-dot" style="background: #ff00ff;"></span>Corner Diffraction</span>
                <span id="btn-diff" class="toggle-btn active" onclick="toggleRayType('Diffraction')">Visible</span>
            </div>
        </div>
    </div>

    <div id="instructions">Left Drag: Orbit 3D | Right Drag: Pan | Scroll: Zoom into Street</div>

    <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x080c14);
        scene.fog = new THREE.FogExp2(0x080c14, 0.0012);

        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 3000);
        camera.position.set({tower_pos[0]} + 200, 160, {tower_pos[1]} + 220);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set({tower_pos[0]}, 20, {tower_pos[1]});
        controls.enableDamping = true;

        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
        dirLight.position.set({tower_pos[0]} + 150, 300, {tower_pos[1]} + 200);
        scene.add(dirLight);

        // Ground Plane
        const groundGeo = new THREE.PlaneGeometry(1600, 1600);
        const groundMat = new THREE.MeshStandardMaterial({{ color: 0x0f172a, roughness: 0.9 }});
        const ground = new THREE.Mesh(groundGeo, groundMat);
        ground.rotation.x = -Math.PI / 2;
        ground.position.set({tower_pos[0]}, 0, {tower_pos[1]});
        scene.add(ground);

        // Street Grid
        const grid = new THREE.GridHelper(1600, 80, 0x1e293b, 0x1e293b);
        grid.position.set({tower_pos[0]}, 0.2, {tower_pos[1]});
        scene.add(grid);

        // Stylized Semi-Transparent 3D Buildings with Sharp Edges
        const buildingData = {json.dumps(building_data)};
        const bldgMat = new THREE.MeshStandardMaterial({{
            color: 0x334155,
            roughness: 0.4,
            metalness: 0.3,
            transparent: true,
            opacity: 0.65
        }});
        const edgeMat = new THREE.LineBasicMaterial({{ color: 0x64748b, transparent: true, opacity: 0.8 }});

        buildingData.forEach(b => {{
            const shape = new THREE.Shape();
            const pts = b.c || b.coords;
            pts.forEach((p, idx) => {{ if (idx === 0) shape.moveTo(p[0], -p[1]); else shape.lineTo(p[0], -p[1]); }});
            const geom = new THREE.ExtrudeGeometry(shape, {{ depth: b.h, bevelEnabled: false }});
            const mesh = new THREE.Mesh(geom, bldgMat);
            mesh.rotation.x = -Math.PI / 2;
            scene.add(mesh);

            // Add sharp building edges
            const edgesGeom = new THREE.EdgesGeometry(geom, 25);
            const edges = new THREE.LineSegments(edgesGeom, edgeMat);
            edges.rotation.x = -Math.PI / 2;
            scene.add(edges);
        }});

        // Tower Mast & Transmitter Beacons
        const mastGeo = new THREE.CylinderGeometry(0.8, 1.2, {tower_pos[2]}, 16);
        const mastMat = new THREE.MeshStandardMaterial({{ color: 0x94a3b8, metalness: 0.8 }});
        const mast = new THREE.Mesh(mastGeo, mastMat);
        mast.position.set({tower_pos[0]}, {tower_pos[2]} / 2, {tower_pos[1]});
        scene.add(mast);

        const txColors = [0xef4444, 0x22c55e, 0x3b82f6];
        [0, 120, 240].forEach((yaw, i) => {{
            const rad = yaw * Math.PI / 180;
            const txSphere = new THREE.Mesh(
                new THREE.SphereGeometry(3.0, 32, 32),
                new THREE.MeshStandardMaterial({{ color: txColors[i], emissive: txColors[i], emissiveIntensity: 0.9 }})
            );
            txSphere.position.set({tower_pos[0]} + 3.0 * Math.cos(rad), {tower_pos[2]}, {tower_pos[1]} + 3.0 * Math.sin(rad));
            scene.add(txSphere);
        }});

        // Receivers (Mobile Users)
        const rxScenarios = {json.dumps(rx_scenarios)};
        rxScenarios.forEach(rx => {{
            const rxSphere = new THREE.Mesh(
                new THREE.SphereGeometry(2.5, 32, 32),
                new THREE.MeshStandardMaterial({{ color: 0x00ffaa, emissive: 0x00ffaa, emissiveIntensity: 0.8 }})
            );
            rxSphere.position.set(rx.pos[0], rx.pos[2], rx.pos[1]);
            scene.add(rxSphere);
        }});

        // Physical 3D Ray Beams
        const rayData = {json.dumps(selected_rays)};
        const rayObjects = [];

        rayData.forEach(r => {{
            const pts = r.points.map(p => new THREE.Vector3(p[0], p[2], p[1]));
            const curve = new THREE.CatmullRomCurve3(pts, false, 'catmullrom', 0);
            
            // Ray Line Tube
            const tubeGeo = new THREE.TubeGeometry(curve, pts.length * 6, 0.45, 8, false);
            const tubeMat = new THREE.MeshBasicMaterial({{
                color: new THREE.Color(r.color),
                transparent: true,
                opacity: 0.85
            }});
            const rayMesh = new THREE.Mesh(tubeGeo, tubeMat);
            rayMesh.userData = {{ type: r.type }};
            scene.add(rayMesh);
            rayObjects.push(rayMesh);

            // Add Bounce Spheres on Building Walls
            for (let k = 1; k < pts.length - 1; k++) {{
                const bounceSphere = new THREE.Mesh(
                    new THREE.SphereGeometry(1.2, 16, 16),
                    new THREE.MeshBasicMaterial({{ color: new THREE.Color(r.color) }})
                );
                bounceSphere.position.copy(pts[k]);
                bounceSphere.userData = {{ type: r.type }};
                scene.add(bounceSphere);
                rayObjects.push(bounceSphere);
            }}
        }});

        // Filter Toggle Handler
        const visibilityState = {{
            'LoS': true,
            'Single-Reflection': true,
            'Multi-Reflection': true,
            'Diffraction': true
        }};

        function toggleRayType(type) {{
            visibilityState[type] = !visibilityState[type];
            
            // Update button UI
            const btnMap = {{
                'LoS': 'btn-los',
                'Single-Reflection': 'btn-refl1',
                'Multi-Reflection': 'btn-refl2',
                'Diffraction': 'btn-diff'
            }};
            const btn = document.getElementById(btnMap[type]);
            if (visibilityState[type]) {{
                btn.className = 'toggle-btn active';
                btn.innerText = 'Visible';
            }} else {{
                btn.className = 'toggle-btn';
                btn.innerText = 'Hidden';
            }}

            // Update 3D ray visibility
            rayObjects.forEach(obj => {{
                if (obj.userData && obj.userData.type === type) {{
                    obj.visible = visibilityState[type];
                }}
            }});
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
    print(f"Saved Interactive 3D WebGL Ray-Beam Viewer: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Simulate and visualize 3D physical RF ray beams bouncing off Cairo buildings.")
    parser.add_argument("--scene", default="cairo_scene/scene.xml")
    parser.add_argument("--scene_info", default="cairo_scene/scene_info.json")
    parser.add_argument("--frequency", type=float, default=1.8e9)
    parser.add_argument("--lat", type=float, default=None, help="Tower GPS latitude WGS84")
    parser.add_argument("--lon", type=float, default=None, help="Tower GPS longitude WGS84")
    parser.add_argument("--tower_x", type=float, default=0.0, help="Tower local X coordinate (m)")
    parser.add_argument("--tower_y", type=float, default=0.0, help="Tower local Y coordinate (m)")
    parser.add_argument("--tower_z", type=float, default=28.0)
    parser.add_argument("--tx_power", type=float, default=44.0)
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

    # 1. Load scene
    scene = load_cairo_scene(args.scene, frequency_hz=args.frequency)

    # 2. Setup transmitters & receivers
    tower_pos = (args.tower_x, args.tower_y, args.tower_z)
    rx_scenarios = setup_transmitters_and_receivers(
        scene=scene,
        tower_pos=tower_pos,
        tx_power_dbm=args.tx_power
    )

    # 3. Compute physical 3D ray paths
    paths = compute_physical_paths(scene, max_depth=args.max_depth)

    # 4. Extract 3D polyline segments
    selected_rays = extract_ray_segments(paths, max_paths_to_export=250)

    # 5. Generate Standalone Interactive 3D WebGL Ray Viewer FIRST
    html_path = os.path.join(args.output_dir, "cairo_3d_rays_viewer.html")
    generate_interactive_ray_viewer_html(
        scene_info=scene_info,
        rx_scenarios=rx_scenarios,
        selected_rays=selected_rays,
        output_path=html_path,
        tower_pos=tower_pos
    )

    # 6. Render Mitsuba 3D snapshot with physical rays (optional / fallback safe)
    png_path = os.path.join(args.output_dir, "scene_3d_rays.png")
    render_mitsuba_3d_rays(
        scene=scene,
        paths=paths,
        output_path=png_path,
        cam_pos=(180.0, -230.0, 120.0),
        look_at=(0.0, 0.0, 20.0)
    )

    print("\n=======================================================")
    print("3D Ray-Beam Simulation & Visualization Completed!")
    print(f"Mitsuba 3D Ray Snapshot: {os.path.abspath(png_path)}")
    print(f"Interactive 3D WebGL:    {os.path.abspath(html_path)}")
    print("=======================================================")


if __name__ == "__main__":
    main()
