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

    lat_col = find_first_matching_column(
        df,
        ["lat", "latitude", "gps_lat", "gps latitude", "latitute"],
    )
    lon_col = find_first_matching_column(
        df,
        ["lng", "lon", "long", "longitude", "gps_lng", "gps_lon", "gps longitude"],
    )
    time_col = find_first_matching_column(
        df,
        ["timestamp", "time", "date", "datetime", "created_at", "recorded_at"],
    )
    speed_col = find_first_matching_column(
        df,
        ["speed", "speed_km_h", "speed km/h", "speed_kmph", "speed (km/h)", "speed_kmh"],
    )
    sat_col = find_first_matching_column(
        df,
        ["satellites", "sats", "gps_sats", "no_of_satellites"],
    )

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
        out["Timestamp"] = pd.date_range(
            start=pd.Timestamp.now().floor("s"),
            periods=len(df),
            freq="s",
        )

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



def build_operation_segments(
    field_df: pd.DataFrame,
    operation_max_speed_kmh: float,
    max_segment_m: float,
    max_time_gap_s: float,
) -> list[dict]:
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
        gap_s = max(0.0, (p1["Timestamp"] - p0["Timestamp"]).total_seconds())
        speed_kmh = float(max(p0.get("speed_kmh", 0.0), p1.get("speed_kmh", 0.0)))

        if step_m < 0.25:
            continue
        if step_m > max_segment_m:
            continue
        if gap_s > max_time_gap_s:
            continue
        if speed_kmh > operation_max_speed_kmh:
            continue

        angle_deg = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
        segs.append(
            {
                "geom": LineString([(float(p0["x"]), float(p0["y"])), (float(p1["x"]), float(p1["y"]))]),
                "length_m": step_m,
                "angle_deg": angle_deg,
                "gap_s": gap_s,
                "speed_kmh": speed_kmh,
            }
        )

    return segs


def estimate_dominant_angle_deg(segments: list[dict], bin_size_deg: float = 5.0) -> float:
    if not segments:
        return 0.0

    bin_count = int(180.0 / bin_size_deg)
    scores = np.zeros(bin_count, dtype=float)
    for seg in segments:
        idx = int((seg["angle_deg"] % 180.0) // bin_size_deg) % bin_count
        scores[idx] += float(seg["length_m"])

    best_idx = int(np.argmax(scores))
    candidate_angles = []
    candidate_weights = []
    best_center = (best_idx + 0.5) * bin_size_deg
    for seg in segments:
        if angle_diff_deg(seg["angle_deg"], best_center) <= bin_size_deg:
            candidate_angles.append(seg["angle_deg"])
            candidate_weights.append(seg["length_m"])

    if not candidate_angles:
        return best_center

    doubled = np.deg2rad(np.asarray(candidate_angles) * 2.0)
    weights = np.asarray(candidate_weights)
    mean_x = np.sum(np.cos(doubled) * weights)
    mean_y = np.sum(np.sin(doubled) * weights)
    angle = (np.degrees(np.arctan2(mean_y, mean_x)) / 2.0) % 180.0
    return float(angle)


def get_main_direction_segments(
    segments: list[dict],
    row_angle_tolerance_deg: float,
    min_row_segment_m: float,
) -> tuple[list[dict], float]:
    if not segments:
        return [], 0.0

    dominant_angle_deg = estimate_dominant_angle_deg(segments)
    main_segments = [
        seg
        for seg in segments
        if seg["length_m"] >= float(min_row_segment_m)
        and angle_diff_deg(seg["angle_deg"], dominant_angle_deg) <= float(row_angle_tolerance_deg)
    ]

    if not main_segments:
        main_segments = segments.copy()

    return main_segments, dominant_angle_deg


def remove_polygon_holes(geom):
    if geom.is_empty:
        return geom

    if isinstance(geom, Polygon):
        return Polygon(geom.exterior)
    if isinstance(geom, MultiPolygon):
        cleaned = [Polygon(poly.exterior) for poly in geom.geoms if not poly.is_empty]
        return unary_union(cleaned) if cleaned else geom
    return geom


def build_field_shape_geometry(
    main_segments: list[dict],
    working_width_m: float,
    fill_gap_m: float,
    smooth_m: float,
):
    if not main_segments:
        return None

    core_geom = unary_union(
        [
            seg["geom"].buffer(working_width_m / 2.0, cap_style=2, join_style=2)
            for seg in main_segments
        ]
    )

    field_shape = core_geom
    if fill_gap_m > 0:
        field_shape = field_shape.buffer(fill_gap_m, cap_style=2, join_style=2).buffer(
            -fill_gap_m, cap_style=2, join_style=2
        )

    if smooth_m > 0:
        field_shape = field_shape.buffer(smooth_m, join_style=2).buffer(-smooth_m, join_style=2)

    field_shape = remove_polygon_holes(field_shape)
    field_shape = field_shape.buffer(0)
    if field_shape.is_empty:
        return None
    return field_shape


def polygon_to_latlng_rings(geom, proj: LocalProjection) -> list[list[list[float]]]:
    if geom is None or geom.is_empty:
        return []

    polys = []
    if geom.geom_type == "Polygon":
        polys = [geom]
    elif geom.geom_type == "MultiPolygon":
        polys = list(geom.geoms)
    else:
        return []

    rings = []
    for poly in polys:
        coords = np.asarray(poly.exterior.coords)
        lat, lng = xy_to_latlon(coords[:, 0], coords[:, 1], proj)
        rings.append([[float(a), float(b)] for a, b in zip(lat, lng)])
    return rings


def summarize_fields(
    df: pd.DataFrame,
    proj: LocalProjection,
    working_width_ft: float,
    operation_max_speed_kmh: float,
    max_segment_m: float,
    max_time_gap_s: float,
    min_field_gunthas: float,
    field_fill_gap_m: float,
    field_shape_smooth_m: float,
    row_angle_tolerance_deg: float,
    min_row_segment_m: float,
) -> tuple[pd.DataFrame, dict[int, dict], list[int]]:
    width_m = float(working_width_ft) * FT_TO_M
    results = []
    geometries: dict[int, dict] = {}

    valid_cluster_ids = [int(fid) for fid in sorted(df["field_id"].unique()) if int(fid) != -1]
    for fid in valid_cluster_ids:
        field_df = df[df["field_id"] == fid].copy().sort_values("Timestamp")
        if len(field_df) < 3:
            continue

        segs = build_operation_segments(
            field_df=field_df,
            operation_max_speed_kmh=operation_max_speed_kmh,
            max_segment_m=max_segment_m,
            max_time_gap_s=max_time_gap_s,
        )
        if not segs:
            continue

        main_segments, dominant_angle_deg = get_main_direction_segments(
            segs,
            row_angle_tolerance_deg=row_angle_tolerance_deg,
            min_row_segment_m=min_row_segment_m,
        )

        swept_geom = unary_union([seg["geom"].buffer(width_m / 2.0, cap_style=2, join_style=2) for seg in segs])
        swept_area_m2 = float(swept_geom.area)

        field_shape_geom = build_field_shape_geometry(
            main_segments=main_segments,
            working_width_m=width_m,
            fill_gap_m=field_fill_gap_m,
            smooth_m=field_shape_smooth_m,
        )
        field_shape_area_m2 = float(field_shape_geom.area) if field_shape_geom is not None else 0.0

        pts = MultiPoint([(float(x), float(y)) for x, y in field_df[["x", "y"]].to_numpy()])
        footprint_geom = pts.convex_hull if not pts.is_empty else None
        footprint_area_m2 = float(footprint_geom.area) if footprint_geom is not None else 0.0

        op_time_min = float(field_df["dt_s"].clip(lower=0).sum() / 60.0)
        strip_length_m = float(sum(seg["length_m"] for seg in segs))
        main_strip_length_m = float(sum(seg["length_m"] for seg in main_segments))

        keep_area_guntha = max(swept_area_m2, field_shape_area_m2) / M2_PER_GUNTHA
        if keep_area_guntha < float(min_field_gunthas):
            continue

        geometries[fid] = {
            "swept": swept_geom,
            "field_shape": field_shape_geom,
            "footprint": footprint_geom,
        }
        results.append(
            {
                "Field ID": fid,
                "4 ft Strip Area (Gunthas)": swept_area_m2 / M2_PER_GUNTHA,
                "4 ft Strip Area (m²)": swept_area_m2,
                "Field Shape Area (Gunthas)": field_shape_area_m2 / M2_PER_GUNTHA,
                "Field Shape Area (m²)": field_shape_area_m2,
                "Footprint Area (Gunthas)": footprint_area_m2 / M2_PER_GUNTHA,
                "Footprint Area (m²)": footprint_area_m2,
                "Operation Length (m)": strip_length_m,
                "Main Row Length (m)": main_strip_length_m,
                "Time (Minutes)": op_time_min,
                "Point Count": int(len(field_df)),
                "Main Row Segment Count": int(len(main_segments)),
                "Dominant Angle (deg)": dominant_angle_deg,
                "Start Date": field_df["Timestamp"].min(),
                "End Date": field_df["Timestamp"].max(),
                "Centroid Lat": float(field_df["lat_smooth"].mean()),
                "Centroid Lng": float(field_df["lng_smooth"].mean()),
            }
        )

    if not results:
        raise ValueError(
            "No valid field areas found. Increase cluster radius, reduce minimum samples, or reduce minimum field area."
        )

    summary = pd.DataFrame(results).sort_values("Start Date").reset_index(drop=True)
    ordered_ids = summary["Field ID"].tolist()
    return summary, geometries, ordered_ids


def compute_travel_metrics(df: pd.DataFrame, ordered_ids: list[int]) -> pd.DataFrame:
    out = []
    for i, fid in enumerate(ordered_ids):
        row = {"Field ID": fid, "Travel Distance (km)": np.nan, "Travel Time (min)": np.nan}
        if i < len(ordered_ids) - 1:
            cur = df[df["field_id"] == fid].sort_values("Timestamp")
            nxt = df[df["field_id"] == ordered_ids[i + 1]].sort_values("Timestamp")
            if not cur.empty and not nxt.empty:
                x1, y1 = float(cur.iloc[-1]["x"]), float(cur.iloc[-1]["y"])
                x2, y2 = float(nxt.iloc[0]["x"]), float(nxt.iloc[0]["y"])
                row["Travel Distance (km)"] = float(np.hypot(x2 - x1, y2 - y1) / 1000.0)
                row["Travel Time (min)"] = float(max(0.0, (nxt.iloc[0]["Timestamp"] - cur.iloc[-1]["Timestamp"]).total_seconds() / 60.0))
        out.append(row)
    return pd.DataFrame(out)


def create_map(
    clean_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    geometries: dict[int, dict],
    proj: LocalProjection,
    use_satellite: bool,
    show_raw_points: bool,
    show_rejected_points: bool,
    show_strip_overlay: bool,
    show_field_shape_overlay: bool,
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

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Google Satellite",
        overlay=False,
        control=True,
    ).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    plugins.Fullscreen().add_to(m)
    plugins.MeasureControl(position="topright", primary_length_unit="meters").add_to(m)

    plot_df = clean_df.copy()
    if focus_field != "All Fields":
        selected_id = int(focus_field.split()[-1])
        plot_df = plot_df[plot_df["field_id"] == selected_id].copy()

    field_colors = [
        "#00c853", "#2962ff", "#ff6d00", "#aa00ff", "#00b8d4", "#d50000", "#64dd17", "#ffd600"
    ]

    bounds_pts: list[list[float]] = []

    if show_raw_points:
        raw_points = raw_df[["lat", "lng"]].dropna().values.tolist()
        for lat, lng in raw_points:
            folium.CircleMarker(
                location=(float(lat), float(lng)),
                radius=1,
                color="#666666",
                fill=True,
                fill_color="#666666",
                fill_opacity=0.35,
                opacity=0.35,
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

    for idx, fid in enumerate(summary_df["Field ID"].tolist()):
        if focus_field != "All Fields" and focus_field != f"Field {fid}":
            continue

        color = field_colors[idx % len(field_colors)]
        field_df = plot_df[plot_df["field_id"] == fid].sort_values("Timestamp")
        path = field_df[["lat_smooth", "lng_smooth"]].values.tolist()

        if len(path) >= 2:
            folium.PolyLine(path, weight=3, color=color, opacity=0.95, tooltip=f"Field {fid} path").add_to(m)
            bounds_pts.extend([[float(a), float(b)] for a, b in path])

        geom_pack = geometries.get(fid, {})

        if show_strip_overlay:
            for ring in polygon_to_latlng_rings(geom_pack.get("swept"), proj):
                folium.Polygon(
                    locations=ring,
                    color=color,
                    weight=2,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.18,
                    tooltip=f"Field {fid} strip area",
                ).add_to(m)
                bounds_pts.extend(ring)

        if show_field_shape_overlay:
            for ring in polygon_to_latlng_rings(geom_pack.get("field_shape"), proj):
                folium.Polygon(
                    locations=ring,
                    color="#ffd600",
                    weight=3,
                    fill=False,
                    tooltip=f"Field {fid} field-shape area",
                ).add_to(m)
                bounds_pts.extend(ring)

        field_row = summary_df[summary_df["Field ID"] == fid].iloc[0]
        folium.Marker(
            location=(float(field_row["Centroid Lat"]), float(field_row["Centroid Lng"])),
            tooltip=f"Field {fid}",
            popup=(
                f"Field {fid}<br>"
                f"4 ft strip area: {field_row['4 ft Strip Area (Gunthas)']:.2f} guntha<br>"
                f"Field shape area: {field_row['Field Shape Area (Gunthas)']:.2f} guntha<br>"
                f"Footprint area: {field_row['Footprint Area (Gunthas)']:.2f} guntha"
            ),
        ).add_to(m)
        bounds_pts.append([float(field_row["Centroid Lat"]), float(field_row["Centroid Lng"])] )

    if bounds_pts:
        lats = [p[0] for p in bounds_pts]
        lngs = [p[1] for p in bounds_pts]
        sw = [min(lats), min(lngs)]
        ne = [max(lats), max(lngs)]
        if sw != ne:
            m.fit_bounds([sw, ne], padding=(20, 20))

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
    field_fill_gap_m: float,
    field_shape_smooth_m: float,
    row_angle_tolerance_deg: float,
    min_row_segment_m: float,
    show_raw_points: bool,
    show_rejected_points: bool,
    show_strip_overlay: bool,
    show_field_shape_overlay: bool,
    focus_field: str,
) -> tuple[folium.Map, pd.DataFrame, pd.DataFrame, float, float, float, float, float, int]:
    clean_df, raw_df, proj = prepare_gps_data(
        gps_df=gps_df,
        smoothing_window=smoothing_window,
        jump_threshold_m=jump_threshold_m,
        min_satellites=min_satellites,
    )
    clustered_df = add_field_labels(clean_df, eps_meters=eps_meters, min_samples=min_samples)

    summary_df, geometries, ordered_ids = summarize_fields(
        df=clustered_df,
        proj=proj,
        working_width_ft=working_width_ft,
        operation_max_speed_kmh=operation_max_speed_kmh,
        max_segment_m=max_segment_m,
        max_time_gap_s=max_time_gap_s,
        min_field_gunthas=min_field_gunthas,
        field_fill_gap_m=field_fill_gap_m,
        field_shape_smooth_m=field_shape_smooth_m,
        row_angle_tolerance_deg=row_angle_tolerance_deg,
        min_row_segment_m=min_row_segment_m,
    )

    travel_df = compute_travel_metrics(clustered_df, ordered_ids)
    summary_df = summary_df.merge(travel_df, on="Field ID", how="left")

    total_strip_area = float(summary_df["4 ft Strip Area (Gunthas)"].sum())
    total_field_shape_area = float(summary_df["Field Shape Area (Gunthas)"].sum())
    total_time = float(summary_df["Time (Minutes)"].sum())
    total_travel_dist = float(np.nansum(summary_df["Travel Distance (km)"]))
    total_travel_time = float(np.nansum(summary_df["Travel Time (min)"]))

    map_obj = create_map(
        clean_df=clustered_df,
        raw_df=raw_df,
        summary_df=summary_df,
        geometries=geometries,
        proj=proj,
        use_satellite=use_satellite,
        show_raw_points=show_raw_points,
        show_rejected_points=show_rejected_points,
        show_strip_overlay=show_strip_overlay,
        show_field_shape_overlay=show_field_shape_overlay,
        focus_field=focus_field,
    )

    rejected_points = int((raw_df["jump_rejected"] | raw_df["sat_rejected"]).sum())
    return (
        map_obj,
        summary_df,
        clustered_df,
        total_strip_area,
        total_field_shape_area,
        total_time,
        total_travel_dist,
        total_travel_time,
        rejected_points,
    )


# ---------------------------
# UI
# ---------------------------
st.title("Farm Area Calculator from GPS File")
st.caption("Now includes a field-shape area that converts repeated parallel operation rows into a farm-like polygon.")

with st.sidebar:
    st.header("Settings")
    working_width_ft = st.number_input("Working width (ft)", min_value=1.0, max_value=20.0, value=4.0, step=0.5)
    eps_meters = st.number_input("Cluster radius (meters)", min_value=1.0, max_value=100.0, value=8.0, step=1.0)
    min_samples = st.number_input("Minimum points in cluster", min_value=3, max_value=500, value=12, step=1)
    min_field_gunthas = st.number_input("Minimum field area (gunthas)", min_value=0.0, max_value=10000.0, value=0.5, step=0.1)
    smoothing_window = st.number_input("GPS smoothing window", min_value=1, max_value=21, value=5, step=2)
    jump_threshold_m = st.number_input("Reject GPS jumps above (m)", min_value=5.0, max_value=100.0, value=20.0, step=1.0)
    min_satellites = st.number_input("Minimum satellites (0 = ignore)", min_value=0, max_value=20, value=0, step=1)
    operation_max_speed_kmh = st.number_input("Operation max speed (km/h)", min_value=1.0, max_value=40.0, value=10.0, step=0.5)
    max_segment_m = st.number_input("Max segment length to count (m)", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
    max_time_gap_s = st.number_input("Max time gap to connect points (s)", min_value=1.0, max_value=120.0, value=20.0, step=1.0)

    st.subheader("Field-shape tuning")
    field_fill_gap_m = st.number_input("Fill gap between rows (m)", min_value=0.0, max_value=30.0, value=2.0, step=0.25)
    field_shape_smooth_m = st.number_input("Field-shape smooth (m)", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
    row_angle_tolerance_deg = st.number_input("Main row angle tolerance (deg)", min_value=1.0, max_value=60.0, value=18.0, step=1.0)
    min_row_segment_m = st.number_input("Minimum row segment length (m)", min_value=0.0, max_value=20.0, value=1.5, step=0.25)

    actual_area_guntha = st.number_input("Actual area for comparison (guntha, optional)", min_value=0.0, max_value=100000.0, value=11.30, step=0.1)
    primary_area_method = st.selectbox("Primary area method", ["Field Shape", "4 ft Strip"], index=0)

    use_satellite = st.toggle("Use satellite map", value=True)
    show_raw_points = st.toggle("Show raw points", value=False)
    show_rejected_points = st.toggle("Show rejected points", value=True)
    show_strip_overlay = st.toggle("Show strip overlay", value=False)
    show_field_shape_overlay = st.toggle("Show field-shape overlay", value=True)

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

        **Example**

        | lat | lng | timestamp | speed_km_h | satellites |
        |---|---:|---|---:|---:|
        | 18.520430 | 73.856744 | 2026-04-18 10:00:01 | 3.4 | 6 |
        | 18.520431 | 73.856760 | 2026-04-18 10:00:02 | 3.2 | 6 |
        | 18.520433 | 73.856781 | 2026-04-18 10:00:03 | 3.1 | 7 |
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
            total_field_shape_area,
            total_time,
            total_travel_dist,
            total_travel_time,
            rejected_points,
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
            field_fill_gap_m=float(field_fill_gap_m),
            field_shape_smooth_m=float(field_shape_smooth_m),
            row_angle_tolerance_deg=float(row_angle_tolerance_deg),
            min_row_segment_m=float(min_row_segment_m),
            show_raw_points=show_raw_points,
            show_rejected_points=show_rejected_points,
            show_strip_overlay=show_strip_overlay,
            show_field_shape_overlay=show_field_shape_overlay,
            focus_field=focus_field,
        )

        primary_total_area = total_field_shape_area if primary_area_method == "Field Shape" else total_strip_area

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(f"{primary_area_method} Area", f"{primary_total_area:.2f} Gunthas")
        c2.metric("4 ft Strip Area", f"{total_strip_area:.2f} Gunthas")
        c3.metric("Field Shape Area", f"{total_field_shape_area:.2f} Gunthas")
        c4.metric("Operation Time", f"{total_time:.2f} min")
        c5.metric("Rejected Points", f"{rejected_points}")

        c6, c7 = st.columns(2)
        c6.metric("Travel Distance", f"{total_travel_dist:.3f} km")
        c7.metric("Travel Time", f"{total_travel_time:.2f} min")

        if actual_area_guntha > 0:
            strip_diff = total_strip_area - float(actual_area_guntha)
            strip_err_pct = (strip_diff / float(actual_area_guntha)) * 100.0
            field_diff = total_field_shape_area - float(actual_area_guntha)
            field_err_pct = (field_diff / float(actual_area_guntha)) * 100.0
            p_diff = primary_total_area - float(actual_area_guntha)
            p_err_pct = (p_diff / float(actual_area_guntha)) * 100.0

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Actual Area", f"{actual_area_guntha:.2f} Gunthas")
            e2.metric("Primary Error", f"{p_diff:+.2f} Gunthas", delta=f"{p_err_pct:+.1f}%")
            e3.metric("Strip Error", f"{strip_diff:+.2f} Gunthas", delta=f"{strip_err_pct:+.1f}%")
            e4.metric("Field Shape Error", f"{field_diff:+.2f} Gunthas", delta=f"{field_err_pct:+.1f}%")

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
            "Footprint Area (Gunthas)",
            "Footprint Area (m²)",
            "Operation Length (m)",
            "Main Row Length (m)",
            "Time (Minutes)",
            "Travel Distance (km)",
            "Travel Time (min)",
            "Centroid Lat",
            "Centroid Lng",
            "Dominant Angle (deg)",
        ]
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(3)

        st.dataframe(display_df, use_container_width=True)

        with st.expander("How field-shape area is built"):
            st.markdown(
                """
                - keeps only operation clusters
                - finds the dominant row direction inside each cluster
                - ignores short turning segments that do not match that direction
                - buffers the row segments by working width
                - fills gaps between nearby rows to make one farm-like shape
                - measures that resulting polygon as **Field Shape Area**
                """
            )

        with st.expander("Cleaned point sample"):
            sample_cols = [
                "Timestamp", "lat", "lng", "lat_smooth", "lng_smooth", "speed_kmh", "step_m", "field_id"
            ]
            st.dataframe(clustered_df[sample_cols].head(300), use_container_width=True)

        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Result CSV",
            data=csv_bytes,
            file_name="farm_area_summary.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(str(e))
else:
    st.info("Upload your file to start.")
