"""
build_scene.py
Converts QGIS GeoPackage building footprints into a 3D Mitsuba 3 / Sionna RT scene.
"""

import os
import sys
import glob
import json
import time
import argparse
import numpy as np
import geopandas as gpd
import shapely.geometry as sg
import trimesh
import pyproj
import mapbox_earcut as earcut


def find_gpkg_file(explicit_path=None):
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path
    gpkg_files = glob.glob("*.gpkg")
    if gpkg_files:
        return gpkg_files[0]
    raise FileNotFoundError("No .gpkg file found in workspace directory.")


def latlon_to_local(lat: float, lon: float, center_x: float, center_y: float, epsg: int = 32636):
    """Convert GPS WGS84 (lat, lon) to local scene coordinates (x, y) in meters."""
    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return float(utm_x - center_x), float(utm_y - center_y)


def local_to_latlon(x: float, y: float, center_x: float, center_y: float, epsg: int = 32636):
    """Convert local scene coordinates (x, y) in meters to GPS WGS84 (lat, lon)."""
    transformer = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    utm_x = x + center_x
    utm_y = y + center_y
    lon, lat = transformer.transform(utm_x, utm_y)
    return float(lat), float(lon)


def extrude_polygons(gdf, height_col="Height"):
    """
    Extrude 2D building footprints into 3D watertight meshes using mapbox-earcut
    and trimesh. Returns a single concatenated trimesh.Trimesh.
    """
    all_vertices = []
    all_faces = []
    vert_count = 0
    total_buildings = len(gdf)
    
    print(f"Extruding {total_buildings} building footprints...")
    t0 = time.time()
    
    for row in gdf.itertuples():
        geom = row.geometry
        if geom.is_empty or not geom.is_valid or geom.geom_type != "Polygon":
            continue
        
        h = float(getattr(row, height_col))
        if h <= 0.0:
            continue
            
        ext_coords = np.array(geom.exterior.coords)
        if len(ext_coords) < 4:
            continue
        ext = ext_coords[:-1]  # remove duplicate closing vertex
        n = len(ext)
        if n < 3:
            continue
            
        rings = np.array([n], dtype=np.uint32)
        try:
            tri_top = earcut.triangulate_float64(ext, rings)
        except Exception:
            continue
            
        if len(tri_top) == 0:
            continue
            
        # Vertices: bottom (z=0) and top (z=h)
        v_bottom = np.column_stack([ext, np.zeros(n, dtype=np.float32)])
        v_top = np.column_stack([ext, np.full(n, h, dtype=np.float32)])
        verts = np.vstack([v_bottom, v_top])
        
        # Faces
        top_faces = tri_top.reshape(-1, 3) + n
        bot_faces = tri_top.reshape(-1, 3)[:, ::-1]  # outward normal pointing down
        
        # Wall quad faces
        idx = np.arange(n)
        idx_next = (idx + 1) % n
        wall1 = np.column_stack([idx, idx_next, n + idx_next])
        wall2 = np.column_stack([idx, n + idx_next, n + idx])
        walls = np.vstack([wall1, wall2])
        
        faces = np.vstack([top_faces, bot_faces, walls]) + vert_count
        
        all_vertices.append(verts)
        all_faces.append(faces)
        vert_count += 2 * n

    if not all_vertices:
        raise ValueError("No valid building meshes were generated.")
        
    combined_vertices = np.vstack(all_vertices)
    combined_faces = np.vstack(all_faces)
    
    mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces, process=False)
    print(f"Extrusion complete in {time.time()-t0:.2f}s: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces.")
    return mesh


def create_ground_mesh(min_x, max_x, min_y, max_y, margin=200.0, depth=1.0):
    """
    Create a flat ground box mesh whose top surface sits exactly at Z = 0.0 m.
    """
    gx_min = min_x - margin
    gx_max = max_x + margin
    gy_min = min_y - margin
    gy_max = max_y + margin
    
    extents = [gx_max - gx_min, gy_max - gy_min, depth]
    ground = trimesh.creation.box(extents=extents)
    # Box center is at (0, 0, 0), so translate top surface to Z = 0.0
    center_ground_x = (gx_min + gx_max) / 2.0
    center_ground_y = (gy_min + gy_max) / 2.0
    ground.apply_translation([center_ground_x, center_ground_y, -depth / 2.0])
    return ground


def generate_mitsuba_xml(output_xml_path):
    """Generate Mitsuba 3 / Sionna RT scene XML with ITU radio materials."""
    xml_content = """<scene version="2.1.0">
    <!-- ITU Radio Materials -->
    <bsdf type="itu-radio-material" id="concrete">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.25"/>
    </bsdf>

    <bsdf type="itu-radio-material" id="very_dry_ground">
        <string name="type" value="very_dry_ground"/>
        <float name="thickness" value="0.5"/>
    </bsdf>

    <!-- 3D Cairo Buildings Mesh -->
    <shape type="ply" id="buildings">
        <string name="filename" value="meshes/buildings.ply"/>
        <boolean name="face_normals" value="true"/>
        <ref id="concrete" name="bsdf"/>
    </shape>

    <!-- Flat Ground Plane Mesh (Top Surface at Z = 0.0m) -->
    <shape type="ply" id="ground">
        <string name="filename" value="meshes/ground.ply"/>
        <boolean name="face_normals" value="true"/>
        <ref id="very_dry_ground" name="bsdf"/>
    </shape>
</scene>
"""
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(xml_content)
    print(f"Generated Mitsuba 3 scene XML: {output_xml_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Cairo building footprints GeoPackage to 3D Sionna scene.")
    parser.add_argument("--input", default=None, help="Path to input GeoPackage (.gpkg)")
    parser.add_argument("--output_dir", default="cairo_scene", help="Output directory for scene")
    parser.add_argument("--max_radius", type=float, default=None,
                        help="Optional radius in meters from center to crop buildings (e.g. 2000.0)")
    parser.add_argument("--simplify_tolerance", type=float, default=0.5,
                        help="Simplification tolerance in meters (default: 0.5)")
    args = parser.parse_args()

    gpkg_path = find_gpkg_file(args.input)
    print(f"Loading GeoPackage from: {gpkg_path}")
    gdf = gpd.read_file(gpkg_path)
    print(f"Loaded {len(gdf):,} raw features.")

    # 1. Check and reproject CRS if necessary
    print(f"Input CRS: {gdf.crs}")
    if gdf.crs is None or gdf.crs.to_epsg() == 4326:
        print("Reprojecting coordinates from EPSG:4326 to EPSG:32636 (UTM zone 36N)...")
        gdf = gdf.to_crs(epsg=32636)
        print(f"New CRS: {gdf.crs}")
    elif gdf.crs.to_epsg() != 32636:
        print(f"Re-projecting from {gdf.crs} to EPSG:32636...")
        gdf = gdf.to_crs(epsg=32636)

    # 2. Check height column
    height_col = None
    for c in gdf.columns:
        if c.lower() == "height":
            height_col = c
            break
    if height_col is None:
        raise ValueError(f"No height column found in dataset! Available columns: {gdf.columns.tolist()}")
    print(f"Using height column: '{height_col}'")

    # 3. Filter invalid geometries and Height <= 1.0
    initial_count = len(gdf)
    gdf = gdf[gdf[height_col] > 1.0]
    gdf = gdf[gdf.geometry.notnull() & gdf.geometry.is_valid]
    print(f"Filtered {initial_count - len(gdf)} invalid or <= 1.0m height features. Remaining: {len(gdf):,}")

    # 4. Simplify footprints to keep triangle count fast for GPU ray tracing
    print(f"Simplifying polygon footprints (tolerance={args.simplify_tolerance} m)...")
    t0 = time.time()
    gdf["geometry"] = gdf["geometry"].simplify(args.simplify_tolerance, preserve_topology=True)
    print(f"Simplification done in {time.time()-t0:.2f}s.")

    # 5. Explode multi-polygons
    gdf = gdf.explode(index_parts=False)

    # 6. Center coordinates around (0, 0)
    total_bounds = gdf.total_bounds
    min_x, min_y, max_x, max_y = total_bounds
    center_x = float((min_x + max_x) / 2.0)
    center_y = float((min_y + max_y) / 2.0)
    print(f"Original Bounding Box (UTM 36N): X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}]")
    print(f"Scene Center (UTM 36N): ({center_x:.2f}, {center_y:.2f})")

    # Translate geometry to local origin
    gdf["geometry"] = gdf["geometry"].translate(xoff=-center_x, yoff=-center_y)

    # Optional spatial filtering if requested
    if args.max_radius is not None:
        r = float(args.max_radius)
        print(f"Filtering buildings within radius of {r}m from center...")
        gdf = gdf[gdf.geometry.intersects(sg.box(-r, -r, r, r))]
        print(f"Remaining buildings within simulation area: {len(gdf):,}")

    local_bounds = gdf.total_bounds
    l_min_x, l_min_y, l_max_x, l_max_y = local_bounds
    print(f"Centered Local Bounding Box: X=[{l_min_x:.2f}, {l_max_x:.2f}], Y=[{l_min_y:.2f}, {l_max_y:.2f}]")

    # Compute GPS coordinates of scene center
    center_lat, center_lon = local_to_latlon(0.0, 0.0, center_x, center_y, epsg=32636)
    print(f"Scene Center GPS (WGS84): Lat={center_lat:.6f}, Lon={center_lon:.6f}")

    # Create directories
    mesh_dir = os.path.join(args.output_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    # 7. Save scene_info.json
    scene_info = {
        "crs": "EPSG:32636",
        "source_gpkg": os.path.basename(gpkg_path),
        "center_utm_x": center_x,
        "center_utm_y": center_y,
        "center_gps_lat": center_lat,
        "center_gps_lon": center_lon,
        "utm_zone": 36,
        "original_bounds_utm": [float(min_x), float(min_y), float(max_x), float(max_y)],
        "local_bounds_m": [float(l_min_x), float(l_min_y), float(l_max_x), float(l_max_y)],
        "num_buildings": len(gdf),
        "height_column": height_col,
        "height_stats": {
            "min": float(gdf[height_col].min()),
            "max": float(gdf[height_col].max()),
            "mean": float(gdf[height_col].mean())
        },
        "python_gps_transform": {
            "latlon_to_local": "latlon_to_local(lat, lon, center_x, center_y, epsg=32636)",
            "local_to_latlon": "local_to_latlon(x, y, center_x, center_y, epsg=32636)"
        }
    }
    scene_info_path = os.path.join(args.output_dir, "scene_info.json")
    with open(scene_info_path, "w", encoding="utf-8") as f:
        json.dump(scene_info, f, indent=2)
    print(f"Saved scene info to: {scene_info_path}")

    # 8. Extrude building footprints & export buildings.ply
    buildings_mesh = extrude_polygons(gdf, height_col=height_col)
    buildings_ply_path = os.path.join(mesh_dir, "buildings.ply")
    print(f"Exporting buildings mesh to: {buildings_ply_path}...")
    buildings_mesh.export(buildings_ply_path)
    print(f"Exported buildings.ply ({os.path.getsize(buildings_ply_path) / 1024 / 1024:.2f} MB)")

    # 9. Create flat ground plane mesh with top surface at Z = 0.0 m
    ground_mesh = create_ground_mesh(l_min_x, l_max_x, l_min_y, l_max_y, margin=300.0, depth=1.0)
    ground_ply_path = os.path.join(mesh_dir, "ground.ply")
    ground_mesh.export(ground_ply_path)
    print(f"Exported ground mesh to: {ground_ply_path} (Top surface at Z = 0.0 m)")

    # 10. Generate Mitsuba XML
    scene_xml_path = os.path.join(args.output_dir, "scene.xml")
    generate_mitsuba_xml(scene_xml_path)

    print("\nScene conversion completed successfully!")
    print(f"Output files in '{args.output_dir}':")
    print(f" - {scene_info_path}")
    print(f" - {scene_xml_path}")
    print(f" - {buildings_ply_path}")
    print(f" - {ground_ply_path}")


if __name__ == "__main__":
    main()
