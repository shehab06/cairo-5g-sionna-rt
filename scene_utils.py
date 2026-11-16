"""
scene_utils.py
Helper functions to convert GPS (latitude, longitude) coordinates to local scene (x, y) coordinates.
"""

import json
import os
import pyproj


def load_scene_origin(scene_info_path="cairo_scene/scene_info.json"):
    """Load origin and projection parameters from scene_info.json."""
    if not os.path.isfile(scene_info_path):
        raise FileNotFoundError(f"Scene info file '{scene_info_path}' not found. Run build_scene.py first.")
    with open(scene_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def latlon_to_scene_xy(lat: float, lon: float, scene_info_path="cairo_scene/scene_info.json"):
    """
    Convert GPS WGS84 (latitude, longitude) to local scene coordinates (x, y) in meters.
    """
    info = load_scene_origin(scene_info_path)
    center_x = info["center_utm_x"]
    center_y = info["center_utm_y"]
    crs = info.get("crs", "EPSG:32636")
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return float(utm_x - center_x), float(utm_y - center_y)


def scene_xy_to_latlon(x: float, y: float, scene_info_path="cairo_scene/scene_info.json"):
    """
    Convert local scene coordinates (x, y) in meters back to GPS WGS84 (latitude, longitude).
    """
    info = load_scene_origin(scene_info_path)
    center_x = info["center_utm_x"]
    center_y = info["center_utm_y"]
    crs = info.get("crs", "EPSG:32636")
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    utm_x = x + center_x
    utm_y = y + center_y
    lon, lat = transformer.transform(utm_x, utm_y)
    return float(lat), float(lon)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert between GPS coordinates and local scene metric coordinates.")
    parser.add_argument("--lat", type=float, help="GPS WGS84 Latitude")
    parser.add_argument("--lon", type=float, help="GPS WGS84 Longitude")
    parser.add_argument("--x", type=float, help="Local Scene X (meters)")
    parser.add_argument("--y", type=float, help="Local Scene Y (meters)")
    args = parser.parse_args()

    if args.lat is not None and args.lon is not None:
        x, y = latlon_to_scene_xy(args.lat, args.lon)
        print(f"GPS: ({args.lat:.6f}, {args.lon:.6f}) -> Local Scene: X = {x:.2f} m, Y = {y:.2f} m")
    elif args.x is not None and args.y is not None:
        lat, lon = scene_xy_to_latlon(args.x, args.y)
        print(f"Local Scene: (X = {args.x:.2f} m, Y = {args.y:.2f} m) -> GPS: ({lat:.6f}, {lon:.6f})")
    else:
        parser.print_help()
