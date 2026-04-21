import io
from dataclasses import dataclass
from math import cos, radians
from typing import Optional

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium import plugins
from shapely.geometry import LineString, MultiPoint, Polygon, MultiPolygon
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from streamlit_folium import st_folium


MAPBOX_TOKEN = "pk.eyJ1IjoiZmxhc2hvcDAwNyIsImEiOiJjbW44a2s5MzcwYm5vMnFzZGloMGpodDI2In0.HO3qwCL8N4YSH3PmwVc3mw"
M2_PER_GUNTHA = 101.17141056
FT_TO_M = 0.3048


st.set_page_config(page_title="Farm Area Calculator", layout="wide")


@dataclass
class LocalProjection:
    lat0: float
    lon0: float
    meters_per_deg_lat: float
    meters_per_deg_lon: float


# ---------------------------
# Helpers
# ---------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


def find_first_matching_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def make_projection(lat0: float, lon0: float) -> LocalProjection:
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * cos(radians(lat0))
    return LocalProjection(
        lat0=float(lat0),
        lon0=float(lon0),
        meters_per_deg_lat=float(meters_per_deg_lat),
        meters_per_deg_lon=float(meters_per_deg_lon),
    )


def latlon_to_xy(lat: np.ndarray, lng: np.ndarray, proj: LocalProjection) -> tuple[np.ndarray, np.ndarray]:
    x = (np.asarray(lng, dtype=float) - proj.lon0) * proj.meters_per_deg_lon
    y = (np.asarray(lat, dtype=float) - proj.lat0) * proj.meters_per_deg_lat
    return x, y


def xy_to_latlon(x: np.ndarray, y: np.ndarray, proj: LocalProjection) -> tuple[np.ndarray, np.ndarray]:
    lat = proj.lat0 + (np.asarray(y, dtype=float) / proj.meters_per_deg_lat)
    lng = proj.lon0 + (np.asarray(x, dtype=float) / proj.meters_per_deg_lon)
    return lat, lng


def rotate_xy(x: np.ndarray, y: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    xr = x * c + y * s
    yr = -x * s + y * c
    return xr, yr


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    if name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(content))

    raise ValueError("Unsupported file format. Upload .xlsx or .csv")


def normalize_input_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(raw_df)

    lat_col = find_first_matching_column(df, ["lat", "latitude", "gps_lat", "gps latitude", "latitute"])
    lon_col = find_first_matching_column(df, ["lng", "lon", "long", "longitude", "gps_lng", "gps_lon", "gps longitude"])
    time_col = find_first_matching_column(df, ["timestamp", "time", "date", "datetime", "created_at", "recorded_at"])
    speed_col = find_first_matching_column(df, ["speed", "speed_km_h", "speed km/h", "speed_kmph", "speed (km/h)", "speed_kmh"])
    sat_col = find_first_matching_column(df, ["satellites", "sats", "gps_sats", "no_of_satellites"])

    if lat_col is None or lon_col is None:
        raise ValueError(
            "Could not detect latitude/longitude columns. Use one of these names: lat, latitude, lng, lon, long, longitude."
        )

    out = pd.DataFrame()
    out["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    out["lng"] = pd.to_numeric(df[lon_col], errors="coerce")

    if time_col is not None:
        out["Timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        out["Timestamp"] = pd.date_range(start=pd.Timestamp.now().floor("s"), periods=len(df), freq="s")

    if speed_col is not None:
        out["speed_kmh_input"] = pd.to_numeric(df[speed_col], errors="coerce")
    else:
        out["speed_kmh_input"] = np.nan

    if sat_col is not None:
        out["satellites"] = pd.to_numeric(df[sat_col], errors="coerce")
    else:
        out["satellites"] = np.nan

    out = out.dropna(subset=["lat", "lng", "Timestamp"]).copy()
    out = out[(out["lat"].between(-90, 90)) & (out["lng"].between(-180, 180))].copy()

    if out.empty:
        raise ValueError("No valid rows found after cleaning. Check lat/lng/timestamp values.")

    out = out.sort_values("Timestamp").drop_duplicates(subset=["Timestamp", "lat", "lng"]).reset_index(drop=True)
    return out


def rolling_median(values: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    return values.rolling(window=window, center=True, min_periods=1).median()


def prepare_gps_data(
    gps_df: pd.DataFrame,
    smoothing_window: int,
    jump_threshold_m: float,
    min_satellites: int,
) -> tuple[pd.DataFrame, pd.DataFrame, LocalProjection]:
    df = gps_df.copy().reset_index(drop=True)
    proj = make_projection(df["lat"].mean(), df["lng"].mean())

    x_raw, y_raw = latlon_to_xy(df["lat"].to_numpy(), df["lng"].to_numpy(), proj)
    df["x_raw"] = x_raw
    df["y_raw"] = y_raw

    df["dt_s"] = df["Timestamp"].diff().dt.total_seconds().fillna(1.0)
    df.loc[df["dt_s"] <= 0, "dt_s"] = 1.0

    df["step_m_raw"] = np.sqrt(df["x_raw"].diff().fillna(0.0) ** 2 + df["y_raw"].diff().fillna(0.0) ** 2)
    df["speed_kmh_calc_raw"] = (df["step_m_raw"] / df["dt_s"]) * 3.6
    df["speed_kmh"] = df["speed_kmh_input"].where(df["speed_kmh_input"].notna(), df["speed_kmh_calc_raw"])

    df["jump_rejected"] = df["step_m_raw"] > float(jump_threshold_m)
    df["sat_rejected"] = False
    if "satellites" in df.columns and df["satellites"].notna().any() and min_satellites > 0:
        df["sat_rejected"] = df["satellites"].fillna(0) < int(min_satellites)

    keep_mask = ~(df["jump_rejected"] | df["sat_rejected"])
    clean = df.loc[keep_mask].copy().reset_index(drop=True)

    if len(clean) < 5:
        raise ValueError("Too many points were removed during cleaning. Reduce jump threshold or satellite filter.")

    clean["x"] = rolling_median(clean["x_raw"], smoothing_window)
    clean["y"] = rolling_median(clean["y_raw"], smoothing_window)
    clean["lat_smooth"], clean["lng_smooth"] = xy_to_latlon(clean["x"].to_numpy(), clean["y"].to_numpy(), proj)

    clean["dt_s"] = clean["Timestamp"].diff().dt.total_seconds().fillna(1.0)
    clean.loc[clean["dt_s"] <= 0, "dt_s"] = 1.0
    clean["step_m"] = np.sqrt(clean["x"].diff().fillna(0.0) ** 2 + clean["y"].diff().fillna(0.0) ** 2)
    clean["speed_kmh_calc"] = (clean["step_m"] / clean["dt_s"]) * 3.6
    clean["speed_kmh"] = clean["speed_kmh_input"].where(clean["speed_kmh_input"].notna(), clean["speed_kmh_calc"])

    move_mask = (clean["step_m"] > 0.05) | clean.index.isin([0])
    clean = clean.loc[move_mask].copy().reset_index(drop=True)

    clean["point_index"] = np.arange(len(clean))
    return clean, df, proj


def add_field_labels(clean_df: pd.DataFrame, eps_meters: float, min_samples: int) -> pd.DataFrame:
    df = clean_df.copy()
    coords = df[["x", "y"]].to_numpy()
    if len(coords) < max(3, min_samples):
        raise ValueError("Not enough cleaned points for clustering.")

    model = DBSCAN(eps=float(eps_meters), min_samples=int(min_samples))
    df["field_id"] = model.fit_predict(coords)
    return df


def angle_diff_deg(a: float, b: float) -> float:
    diff = abs(float(a) - float(b)) % 180.0
    return min(diff, 180.0 - diff)


def dominant_axial_angle_deg(angles_deg: np.ndarray, weights: np.ndarray) -> float:
    if len(angles_deg) == 0:
        return 0.0
    doubled = np.deg2rad(np.asarray(angles_deg, dtype=float) * 2.0)
    weights = np.asarray(weights, dtype=float)
    c = np.sum(weights * np.cos(doubled))
    s = np.sum(weights * np.sin(doubled))
    angle = (0.5 * np.rad2deg(np.arctan2(s, c))) % 180.0
    return float(angle)


def polygon_to_latlng_rings(geom, proj: LocalProjection) -> list[list[list[float]]]:
    if geom is None or geom.is_empty:
        return []

    polys: list[Polygon]
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    else:
        return []

    rings = []
    for poly in polys:
        coords = np.asarray(poly.exterior.coords)
        lat, lng = xy_to_latlon(coords[:, 0], coords[:, 1], proj)
        rings.append([[float(a), float(b)] for a, b in zip(lat, lng)])
    return rings


def build_segments(field_df: pd.DataFrame, max_segment_m: float, max_time_gap_s: float) -> list[dict]:
    segs: list[dict] = []
    if len(field_df) < 2:
        return segs

    f = field_df.sort_values("Timestamp").reset_index(drop=True)
    for i in range(1, len(f)):
        p0 = f.iloc[i - 1]
        p1 = f.iloc[i]
        dx = float(p1["x"] - p0["x"])
        dy = float(p1["y"] - p0["y"])
        step_m = float(np.hypot(dx, dy))
        if step_m < 0.25 or step_m > float(max_segment_m):
            continue
        gap_s = float(max(0.0, (p1["Timestamp"] - p0["Timestamp"]).total_seconds()))
        if gap_s > float(max_time_gap_s):
            continue

        speed_kmh = float(max(p0.get("speed_kmh", 0.0), p1.get("speed_kmh", 0.0)))
        angle_deg = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
        mid_x = float((p0["x"] + p1["x"]) / 2.0)
        mid_y = float((p0["y"] + p1["y"]) / 2.0)

        segs.append(
            {
                "geom": LineString([(float(p0["x"]), float(p0["y"])), (float(p1["x"]), float(p1["y"]))]),
                "length_m": step_m,
                "angle_deg": angle_deg,
                "gap_s": gap_s,
                "speed_kmh": speed_kmh,
                "mid_x": mid_x,
                "mid_y": mid_y,
                "start_idx": int(p0["point_index"]),
                "end_idx": int(p1["point_index"]),
            }
        )
    return segs


def classify_operation_segments(
    segments: list[dict],
    working_width_m: float,
    operation_max_speed_kmh: float,
    angle_tolerance_deg: float,
    min_row_segment_length_m: float,
    min_row_total_length_m: float,
    min_row_segment_count: int,
    band_merge_bins: int,
) -> tuple[list[dict], list[dict], float]:
    if not segments:
        return [], [], 0.0

    usable = [s for s in segments if s["speed_kmh"] <= float(operation_max_speed_kmh) and s["length_m"] >= float(min_row_segment_length_m)]
    if not usable:
        return [], segments, 0.0

    dominant_angle = dominant_axial_angle_deg(
        np.array([s["angle_deg"] for s in usable], dtype=float),
        np.array([s["length_m"] for s in usable], dtype=float),
    )

    aligned = [s for s in usable if angle_diff_deg(s["angle_deg"], dominant_angle) <= float(angle_tolerance_deg)]
    if not aligned:
        return [], segments, dominant_angle

    band_size = max(0.75, float(working_width_m) * 0.8)
    mx = np.array([s["mid_x"] for s in aligned], dtype=float)
    my = np.array([s["mid_y"] for s in aligned], dtype=float)
    _, v = rotate_xy(mx, my, dominant_angle)
    band_ids = np.floor(v / band_size).astype(int)

    band_stats: dict[int, dict[str, float]] = {}
    for seg, bid in zip(aligned, band_ids):
        stat = band_stats.setdefault(int(bid), {"length": 0.0, "count": 0.0})
        stat["length"] += float(seg["length_m"])
        stat["count"] += 1.0

    good_bands = {
        bid
        for bid, stat in band_stats.items()
        if stat["length"] >= float(min_row_total_length_m) and stat["count"] >= float(min_row_segment_count)
    }

    expanded_good_bands = set(good_bands)
    for bid in list(good_bands):
        for shift in range(1, int(band_merge_bins) + 1):
            if (bid - shift) in band_stats:
                expanded_good_bands.add(bid - shift)
            if (bid + shift) in band_stats:
                expanded_good_bands.add(bid + shift)

    operation_segments: list[dict] = []
    travel_segments: list[dict] = []
    for seg, bid in zip(aligned, band_ids):
        if int(bid) in expanded_good_bands:
            operation_segments.append(seg)
        else:
            travel_segments.append(seg)

    # anything usable but not aligned is travel
    aligned_ids = {id(s) for s in aligned}
    for seg in usable:
        if id(seg) not in aligned_ids:
            travel_segments.append(seg)

    # fast / short / bad-gap segments count as travel too
    usable_ids = {id(s) for s in usable}
    for seg in segments:
        if id(seg) not in usable_ids:
            travel_segments.append(seg)

    return operation_segments, travel_segments, dominant_angle


def build_field_shape_geometry(
    operation_segments: list[dict],
    working_width_m: float,
    fill_gap_m: float,
    smooth_m: float,
):
    if not operation_segments:
        return None, None

    strip_geom = unary_union([
        seg["geom"].buffer(float(working_width_m) / 2.0, cap_style=2, join_style=2)
        for seg in operation_segments
    ])
    if strip_geom.is_empty:
        return None, None

    shape_geom = strip_geom
    if float(fill_gap_m) > 0:
        shape_geom = shape_geom.buffer(float(fill_gap_m) / 2.0, join_style=2).buffer(-float(fill_gap_m) / 2.0, join_style=2)
    if float(smooth_m) > 0:
        shape_geom = shape_geom.buffer(float(smooth_m), join_style=1).buffer(-float(smooth_m), join_style=1)

    if isinstance(shape_geom, MultiPolygon):
        shape_geom = max(shape_geom.geoms, key=lambda g: g.area)
    return strip_geom, shape_geom


def summarize_fields(
    df: pd.DataFrame,
    proj: LocalProjection,
    working_width_ft: float,
    operation_max_speed_kmh: float,
    max_segment_m: float,
    max_time_gap_s: float,
    min_field_gunthas: float,
    angle_tolerance_deg: float,
    min_row_segment_length_m: float,
    min_row_total_length_m: float,
    min_row_segment_count: int,
    band_merge_bins: int,
    fill_gap_m: float,
    smooth_m: float,
) -> tuple[pd.DataFrame, dict[int, object], dict[int, object], dict[int, float], dict[int, pd.DataFrame], dict[int, pd.DataFrame], list[int]]:
    width_m = float(working_width_ft) * FT_TO_M
    results = []
    strip_polygons: dict[int, object] = {}
    shape_polygons: dict[int, object] = {}
    field_angles: dict[int, float] = {}
    operation_points_map: dict[int, pd.DataFrame] = {}
    travel_points_map: dict[int, pd.DataFrame] = {}

    valid_cluster_ids = [int(fid) for fid in sorted(df["field_id"].unique()) if int(fid) != -1]
    for fid in valid_cluster_ids:
        field_df = df[df["field_id"] == fid].copy().sort_values("Timestamp")
        if len(field_df) < 3:
            continue

        segments = build_segments(field_df=field_df, max_segment_m=max_segment_m, max_time_gap_s=max_time_gap_s)
        if not segments:
            continue

        operation_segments, travel_segments, dominant_angle = classify_operation_segments(
            segments=segments,
            working_width_m=width_m,
            operation_max_speed_kmh=operation_max_speed_kmh,
            angle_tolerance_deg=angle_tolerance_deg,
            min_row_segment_length_m=min_row_segment_length_m,
            min_row_total_length_m=min_row_total_length_m,
            min_row_segment_count=min_row_segment_count,
            band_merge_bins=band_merge_bins,
        )
        if not operation_segments:
            continue

        strip_geom, shape_geom = build_field_shape_geometry(
            operation_segments=operation_segments,
            working_width_m=width_m,
            fill_gap_m=fill_gap_m,
            smooth_m=smooth_m,
        )
        if strip_geom is None or strip_geom.is_empty:
            continue

        strip_area_m2 = float(strip_geom.area)
        shape_area_m2 = float(shape_geom.area) if shape_geom is not None and not shape_geom.is_empty else 0.0

        op_idx = sorted({s["start_idx"] for s in operation_segments}.union({s["end_idx"] for s in operation_segments}))
        tr_idx = sorted({s["start_idx"] for s in travel_segments}.union({s["end_idx"] for s in travel_segments}))
        op_points = field_df[field_df["point_index"].isin(op_idx)].copy().sort_values("Timestamp")
        travel_points = field_df[field_df["point_index"].isin(tr_idx)].copy().sort_values("Timestamp")

        operation_time_min = float(sum(s["gap_s"] for s in operation_segments) / 60.0)
        travel_time_min = float(sum(s["gap_s"] for s in travel_segments) / 60.0)
        operation_length_m = float(sum(s["length_m"] for s in operation_segments))
        travel_length_m = float(sum(s["length_m"] for s in travel_segments))

        if (strip_area_m2 / M2_PER_GUNTHA) < float(min_field_gunthas):
            continue

        strip_polygons[fid] = strip_geom
        shape_polygons[fid] = shape_geom
        field_angles[fid] = dominant_angle
        operation_points_map[fid] = op_points
        travel_points_map[fid] = travel_points

        centroid_source = op_points if not op_points.empty else field_df
        results.append(
            {
                "Field ID": fid,
                "4 ft Strip Area (Gunthas)": strip_area_m2 / M2_PER_GUNTHA,
                "4 ft Strip Area (m²)": strip_area_m2,
                "Field Shape Area (Gunthas)": shape_area_m2 / M2_PER_GUNTHA,
                "Field Shape Area (m²)": shape_area_m2,
                "Operation Length (m)": operation_length_m,
                "Travel Length (m)": travel_length_m,
                "Operation Time (Minutes)": operation_time_min,
                "Travel Time (Minutes)": travel_time_min,
                "Point Count": int(len(field_df)),
                "Operation Point Count": int(len(op_points)),
                "Travel Point Count": int(len(travel_points)),
                "Dominant Row Angle (deg)": dominant_angle,
                "Start Date": field_df["Timestamp"].min(),
                "End Date": field_df["Timestamp"].max(),
                "Centroid Lat": float(centroid_source["lat_smooth"].mean()),
                "Centroid Lng": float(centroid_source["lng_smooth"].mean()),
            }
        )

    if not results:
        raise ValueError(
            "No valid field areas found. Reduce minimum points, reduce angle filter, increase max speed slightly, or reduce minimum field area."
        )

    summary = pd.DataFrame(results).sort_values("Start Date").reset_index(drop=True)
    ordered_ids = summary["Field ID"].tolist()
    return summary, strip_polygons, shape_polygons, field_angles, operation_points_map, travel_points_map, ordered_ids


def create_map(
    clean_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    strip_polygons: dict[int, object],
    shape_polygons: dict[int, object],
    operation_points_map: dict[int, pd.DataFrame],
    travel_points_map: dict[int, pd.DataFrame],
    proj: LocalProjection,
    use_satellite: bool,
    show_raw_points: bool,
    show_rejected_points: bool,
    focus_field: str,
) -> folium.Map:
    center = [float(clean_df["lat_smooth"].mean()), float(clean_df["lng_smooth"].mean())]

    if use_satellite:
        tiles = (
            f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/256/{{z}}/{{x}}/{{y}}"
            f"?access_token={MAPBOX_TOKEN}"
        )
        m = folium.Map(location=center, zoom_start=18, tiles=tiles, attr="Mapbox")
    else:
        m = folium.Map(location=center, zoom_start=18, tiles="OpenStreetMap")

    plugins.Fullscreen().add_to(m)
    plugins.MeasureControl(position="topright", primary_length_unit="meters").add_to(m)

    bounds_pts: list[list[float]] = []

    if show_raw_points:
        raw_points = raw_df[["lat", "lng"]].dropna().values.tolist()
        for lat, lng in raw_points:
            folium.CircleMarker(
                location=(float(lat), float(lng)),
                radius=1,
                color="#757575",
                fill=True,
                fill_color="#757575",
                fill_opacity=0.3,
                opacity=0.3,
            ).add_to(m)

    if show_rejected_points:
        rejected = raw_df[raw_df["jump_rejected"] | raw_df["sat_rejected"]]
        for _, row in rejected.iterrows():
            folium.CircleMarker(
                location=(float(row["lat"]), float(row["lng"])),
                radius=3,
                color="#ff1744",
                fill=True,
                fill_color="#ff1744",
                fill_opacity=0.9,
                tooltip="Rejected point",
            ).add_to(m)

    for fid in summary_df["Field ID"].tolist():
        if focus_field != "All Fields" and focus_field != f"Field {fid}":
            continue

        op_points = operation_points_map.get(fid, pd.DataFrame())
        travel_points = travel_points_map.get(fid, pd.DataFrame())

        if not travel_points.empty:
            travel_path = travel_points[["lat_smooth", "lng_smooth"]].values.tolist()
            if len(travel_path) >= 2:
                folium.PolyLine(
                    travel_path,
                    weight=3,
                    color="#ff6d00",
                    opacity=0.9,
                    tooltip=f"Field {fid} travel / on-road path",
                ).add_to(m)
                bounds_pts.extend([[float(a), float(b)] for a, b in travel_path])

        if not op_points.empty:
            op_path = op_points[["lat_smooth", "lng_smooth"]].values.tolist()
            if len(op_path) >= 2:
                folium.PolyLine(
                    op_path,
                    weight=4,
                    color="#00e676",
                    opacity=0.95,
                    tooltip=f"Field {fid} operation path",
                ).add_to(m)
                bounds_pts.extend([[float(a), float(b)] for a, b in op_path])

        strip_geom = strip_polygons.get(fid)
        for ring in polygon_to_latlng_rings(strip_geom, proj):
            folium.Polygon(
                locations=ring,
                color="#42a5f5",
                weight=2,
                fill=True,
                fill_color="#42a5f5",
                fill_opacity=0.16,
                tooltip=f"Field {fid} 4 ft strip area",
            ).add_to(m)
            bounds_pts.extend(ring)

        shape_geom = shape_polygons.get(fid)
        for ring in polygon_to_latlng_rings(shape_geom, proj):
            folium.Polygon(
                locations=ring,
                color="#ffd600",
                weight=3,
                fill=True,
                fill_color="#ffd600",
                fill_opacity=0.12,
                tooltip=f"Field {fid} field shape area",
            ).add_to(m)
            bounds_pts.extend(ring)

        field_row = summary_df[summary_df["Field ID"] == fid].iloc[0]
        folium.Marker(
            location=(float(field_row["Centroid Lat"]), float(field_row["Centroid Lng"])),
            tooltip=f"Field {fid}",
            popup=(
                f"Field {fid}<br>"
                f"4 ft strip area: {field_row['4 ft Strip Area (Gunthas)']:.2f} guntha<br>"
                f"Field shape area: {field_row['Field Shape Area (Gunthas)']:.2f} guntha"
            ),
        ).add_to(m)
        bounds_pts.append([float(field_row["Centroid Lat"]), float(field_row["Centroid Lng"])])

    if bounds_pts:
        lats = [p[0] for p in bounds_pts]
        lngs = [p[1] for p in bounds_pts]
        sw = [min(lats), min(lngs)]
        ne = [max(lats), max(lngs)]
        if sw != ne:
            m.fit_bounds([sw, ne], padding=(25, 25))

    return m


def process_data(
    gps_df: pd.DataFrame,
    eps_meters: float,
    min_samples: int,
    min_field_gunthas: float,
    use_satellite: bool,
    smoothing_window: int,
    jump_threshold_m: float,
    min_satellites: int,
    working_width_ft: float,
    operation_max_speed_kmh: float,
    max_segment_m: float,
    max_time_gap_s: float,
    angle_tolerance_deg: float,
    min_row_segment_length_m: float,
    min_row_total_length_m: float,
    min_row_segment_count: int,
    band_merge_bins: int,
    fill_gap_m: float,
    smooth_m: float,
    show_raw_points: bool,
    show_rejected_points: bool,
    focus_field: str,
):
    clean_df, raw_df, proj = prepare_gps_data(
        gps_df=gps_df,
        smoothing_window=smoothing_window,
        jump_threshold_m=jump_threshold_m,
        min_satellites=min_satellites,
    )
    clustered_df = add_field_labels(clean_df, eps_meters=eps_meters, min_samples=min_samples)

    (
        summary_df,
        strip_polygons,
        shape_polygons,
        field_angles,
        operation_points_map,
        travel_points_map,
        ordered_ids,
    ) = summarize_fields(
        df=clustered_df,
        proj=proj,
        working_width_ft=working_width_ft,
        operation_max_speed_kmh=operation_max_speed_kmh,
        max_segment_m=max_segment_m,
        max_time_gap_s=max_time_gap_s,
        min_field_gunthas=min_field_gunthas,
        angle_tolerance_deg=angle_tolerance_deg,
        min_row_segment_length_m=min_row_segment_length_m,
        min_row_total_length_m=min_row_total_length_m,
        min_row_segment_count=min_row_segment_count,
        band_merge_bins=band_merge_bins,
        fill_gap_m=fill_gap_m,
        smooth_m=smooth_m,
    )

    total_strip_area = float(summary_df["4 ft Strip Area (Gunthas)"].sum())
    total_shape_area = float(summary_df["Field Shape Area (Gunthas)"].sum())
    total_operation_time = float(summary_df["Operation Time (Minutes)"].sum())
    total_travel_dist = float(summary_df["Travel Length (m)"].sum() / 1000.0)
    total_travel_time = float(summary_df["Travel Time (Minutes)"].sum())

    map_obj = create_map(
        clean_df=clustered_df,
        raw_df=raw_df,
        summary_df=summary_df,
        strip_polygons=strip_polygons,
        shape_polygons=shape_polygons,
        operation_points_map=operation_points_map,
        travel_points_map=travel_points_map,
        proj=proj,
        use_satellite=use_satellite,
        show_raw_points=show_raw_points,
        show_rejected_points=show_rejected_points,
        focus_field=focus_field,
    )

    rejected_points = int((raw_df["jump_rejected"] | raw_df["sat_rejected"]).sum())
    return (
        map_obj,
        summary_df,
        clustered_df,
        total_strip_area,
        total_shape_area,
        total_operation_time,
        total_travel_dist,
        total_travel_time,
        rejected_points,
        field_angles,
    )


# ---------------------------
# UI
# ---------------------------
st.title("Farm Area Calculator from GPS File")
st.caption("Green = operation path, orange = travel/on-road path, yellow = field shape, blue = 4 ft strip.")

with st.sidebar:
    st.header("Settings")
    primary_area_method = st.selectbox("Primary area method", ["Field Shape", "4 ft Strip"], index=0)
    working_width_ft = st.number_input("Working width (ft)", min_value=1.0, max_value=20.0, value=4.0, step=0.5)
    eps_meters = st.number_input("Cluster radius (meters)", min_value=1.0, max_value=100.0, value=8.0, step=1.0)
    min_samples = st.number_input("Minimum points in cluster", min_value=3, max_value=500, value=12, step=1)
    min_field_gunthas = st.number_input("Minimum field area (gunthas)", min_value=0.0, max_value=10000.0, value=0.5, step=0.1)
    smoothing_window = st.number_input("GPS smoothing window", min_value=1, max_value=21, value=5, step=2)
    jump_threshold_m = st.number_input("Reject GPS jumps above (m)", min_value=5.0, max_value=100.0, value=20.0, step=1.0)
    min_satellites = st.number_input("Minimum satellites (0 = ignore)", min_value=0, max_value=20, value=0, step=1)
    operation_max_speed_kmh = st.number_input("Operation max speed (km/h)", min_value=1.0, max_value=40.0, value=8.0, step=0.5)
    max_segment_m = st.number_input("Max segment length to count (m)", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
    max_time_gap_s = st.number_input("Max time gap to connect points (s)", min_value=1.0, max_value=120.0, value=20.0, step=1.0)

    st.subheader("Operation / row filter")
    angle_tolerance_deg = st.number_input("Main row angle tolerance (deg)", min_value=5.0, max_value=60.0, value=18.0, step=1.0)
    min_row_segment_length_m = st.number_input("Minimum row segment length (m)", min_value=0.5, max_value=20.0, value=1.5, step=0.5)
    min_row_total_length_m = st.number_input("Minimum row total length per band (m)", min_value=1.0, max_value=200.0, value=8.0, step=1.0)
    min_row_segment_count = st.number_input("Minimum row segments per band", min_value=1, max_value=50, value=3, step=1)
    band_merge_bins = st.number_input("Band merge bins", min_value=0, max_value=5, value=1, step=1)

    st.subheader("Field shape")
    fill_gap_m = st.number_input("Fill gap between rows (m)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
    smooth_m = st.number_input("Field-shape smooth (m)", min_value=0.0, max_value=10.0, value=0.4, step=0.1)

    actual_area_guntha = st.number_input("Actual area for comparison (guntha, optional)", min_value=0.0, max_value=100000.0, value=11.30, step=0.1)
    use_satellite = st.toggle("Use satellite map", value=True)
    show_raw_points = st.toggle("Show raw points", value=False)
    show_rejected_points = st.toggle("Show rejected points", value=True)

uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

with st.expander("Required file format", expanded=False):
    st.markdown(
        """
        **Minimum required columns**
        - `lat`
        - `lng`

        **Optional columns**
        - `timestamp`
        - `speed_km_h`
        - `satellites`

        **Also accepted**
        - `latitude` instead of `lat`
        - `lon`, `long`, or `longitude` instead of `lng`
        - `time`, `date`, `datetime`, or `created_at` instead of `timestamp`
        """
    )

if uploaded_file is not None:
    try:
        raw_df = load_uploaded_data(uploaded_file)
        gps_df = normalize_input_dataframe(raw_df)

        preview_clean_df, _, _ = prepare_gps_data(
            gps_df=gps_df,
            smoothing_window=int(smoothing_window),
            jump_threshold_m=float(jump_threshold_m),
            min_satellites=int(min_satellites),
        )
        preview_clustered_df = add_field_labels(preview_clean_df, eps_meters=float(eps_meters), min_samples=int(min_samples))
        preview_fields = [int(fid) for fid in sorted(preview_clustered_df["field_id"].unique()) if int(fid) != -1]
        focus_options = ["All Fields"] + [f"Field {fid}" for fid in preview_fields]
        focus_field = st.selectbox("Map focus", options=focus_options, index=0)

        (
            map_obj,
            result_df,
            clustered_df,
            total_strip_area,
            total_shape_area,
            total_operation_time,
            total_travel_dist,
            total_travel_time,
            rejected_points,
            field_angles,
        ) = process_data(
            gps_df=gps_df,
            eps_meters=float(eps_meters),
            min_samples=int(min_samples),
            min_field_gunthas=float(min_field_gunthas),
            use_satellite=use_satellite,
            smoothing_window=int(smoothing_window),
            jump_threshold_m=float(jump_threshold_m),
            min_satellites=int(min_satellites),
            working_width_ft=float(working_width_ft),
            operation_max_speed_kmh=float(operation_max_speed_kmh),
            max_segment_m=float(max_segment_m),
            max_time_gap_s=float(max_time_gap_s),
            angle_tolerance_deg=float(angle_tolerance_deg),
            min_row_segment_length_m=float(min_row_segment_length_m),
            min_row_total_length_m=float(min_row_total_length_m),
            min_row_segment_count=int(min_row_segment_count),
            band_merge_bins=int(band_merge_bins),
            fill_gap_m=float(fill_gap_m),
            smooth_m=float(smooth_m),
            show_raw_points=show_raw_points,
            show_rejected_points=show_rejected_points,
            focus_field=focus_field,
        )

        primary_area = total_shape_area if primary_area_method == "Field Shape" else total_strip_area
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Field Shape Area", f"{total_shape_area:.2f} Gunthas")
        c2.metric("4 ft Strip Area", f"{total_strip_area:.2f} Gunthas")
        c3.metric("Primary Area", f"{primary_area:.2f} Gunthas")
        c4.metric("Operation Time", f"{total_operation_time:.2f} min")
        c5.metric("Rejected Points", f"{rejected_points}")

        t1, t2 = st.columns(2)
        t1.metric("Travel Distance", f"{total_travel_dist:.3f} km")
        t2.metric("Travel Time", f"{total_travel_time:.2f} min")

        if actual_area_guntha > 0:
            primary_diff = primary_area - float(actual_area_guntha)
            primary_pct = (primary_diff / float(actual_area_guntha)) * 100.0
            strip_diff = total_strip_area - float(actual_area_guntha)
            strip_pct = (strip_diff / float(actual_area_guntha)) * 100.0
            shape_diff = total_shape_area - float(actual_area_guntha)
            shape_pct = (shape_diff / float(actual_area_guntha)) * 100.0

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Actual Area", f"{actual_area_guntha:.2f} Gunthas")
            e2.metric("Primary Error", f"{primary_diff:+.2f} Gunthas", delta=f"{primary_pct:+.1f}%")
            e3.metric("Strip Error", f"{strip_diff:+.2f} Gunthas", delta=f"{strip_pct:+.1f}%")
            e4.metric("Field Shape Error", f"{shape_diff:+.2f} Gunthas", delta=f"{shape_pct:+.1f}%")

        st.subheader("Map")
        st_folium(map_obj, width=None, height=760, returned_objects=[])

        st.subheader("Field Summary")
        display_df = result_df.copy()
        display_df["Start Date"] = pd.to_datetime(display_df["Start Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        display_df["End Date"] = pd.to_datetime(display_df["End Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        numeric_cols = [
            "4 ft Strip Area (Gunthas)",
            "4 ft Strip Area (m²)",
            "Field Shape Area (Gunthas)",
            "Field Shape Area (m²)",
            "Operation Length (m)",
            "Travel Length (m)",
            "Operation Time (Minutes)",
            "Travel Time (Minutes)",
            "Dominant Row Angle (deg)",
            "Centroid Lat",
            "Centroid Lng",
        ]
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(3)

        st.dataframe(display_df, use_container_width=True)

        with st.expander("Cleaned point sample"):
            sample_cols = ["Timestamp", "lat", "lng", "lat_smooth", "lng_smooth", "speed_kmh", "step_m", "field_id"]
            st.dataframe(clustered_df[sample_cols].head(300), use_container_width=True)

        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Result CSV",
            data=csv_bytes,
            file_name="farm_area_summary.csv",
            mime="text/csv",
        )

        st.info(
            "If the on-road line is still entering the field area, first reduce 'Operation max speed', then reduce 'Main row angle tolerance', then reduce 'Fill gap between rows'."
        )

    except Exception as e:
        st.error(str(e))
else:
    st.info("Upload your file to start.")
