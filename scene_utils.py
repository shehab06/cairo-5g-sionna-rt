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
