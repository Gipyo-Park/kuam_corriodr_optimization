"""API adapter for the authoritative UAM corridor optimizer.

The optimization implementation lives in ``MAIN_uam_corridor_optimizer.py``.
This module only validates API-owned inputs, invokes that implementation with
those overrides, and converts its authoritative Excel output to the API shape.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


API_SERVICE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_SERVICE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# API runs must never open the standalone click-waypoint UI.
os.environ.setdefault("WP_CLICK_MODE", "0")

import MAIN_uam_corridor_optimizer as _core  # noqa: E402


DEFAULT_START_VERTIPORT = (35.603386, 129.078025, 150.0)
DEFAULT_END_VERTIPORT = (35.603386, 129.078025, 150.0)
DEFAULT_TRANSITION_ANGLE_DEG = 6.0

PATH_SCOPE_FIXED_ONLY = "takeoff_transition_end_to_landing_transition_start"
PATH_SCOPE_FULL = "start_vertiport_to_end_vertiport"

ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


def project_path(*parts: str) -> Path:
    """Return a project-root-relative path independent of process CWD."""
    return PROJECT_ROOT.joinpath(*parts)


def _emit_progress(
    callback: ProgressCallback,
    percent: float,
    stage: str,
    message: str,
    **details: Any,
) -> None:
    if callback is None:
        return
    event: Dict[str, Any] = {
        "event": "progress",
        "percent": int(np.clip(round(float(percent)), 0, 100)),
        "stage": str(stage),
        "message": str(message),
    }
    if details:
        event["details"] = details
    callback(event)


def _forward_core_progress(callback: ProgressCallback, event: Dict[str, Any]) -> None:
    """Translate the core's progress event to the existing SSE event contract."""
    if callback is None:
        return
    payload = dict(event)
    payload["event"] = str(payload.get("event", "progress"))
    payload["percent"] = int(
        np.clip(payload.pop("progress", payload.get("percent", 0)), 0, 100)
    )
    callback(payload)


def _finite_number(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite.")
    return result


def _parse_angle(value: Any, field_name: str) -> float:
    angle = _finite_number(value, field_name)
    if not 0.0 < angle < 90.0:
        raise ValueError(f"{field_name} must be greater than 0 and less than 90 degrees.")
    return angle


def _parse_lat_lon(lat: Any, lon: Any, field_name: str) -> tuple[float, float]:
    lat_value = _finite_number(lat, f"{field_name}.lat")
    lon_value = _finite_number(lon, f"{field_name}.lon")
    if not -90.0 <= lat_value <= 90.0:
        raise ValueError(f"{field_name}.lat must be between -90 and 90.")
    if not -180.0 <= lon_value <= 180.0:
        raise ValueError(f"{field_name}.lon must be between -180 and 180.")
    return lat_value, lon_value


def _parse_vertiport(value: Any, field_name: str, default: Sequence[float]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=float).copy()
    if not isinstance(value, dict) or not isinstance(value.get("lla"), dict):
        raise ValueError(f"{field_name} must be {{lla: {{lat, lon, alt_m}}}}.")
    lla = value["lla"]
    try:
        lat, lon = _parse_lat_lon(lla["lat"], lla["lon"], f"{field_name}.lla")
        alt_m = _finite_number(lla["alt_m"], f"{field_name}.lla.alt_m")
    except KeyError as exc:
        raise ValueError(f"{field_name}.lla requires lat, lon, and alt_m.") from exc
    return np.array([lat, lon, alt_m], dtype=float)


def _parse_corridor_points(raw_points: Any, cruise_altitude_m: float) -> np.ndarray:
    if raw_points is None:
        return np.empty((0, 3), dtype=float)
    if not isinstance(raw_points, (list, tuple, np.ndarray)):
        raise ValueError("corridor_points must be a list.")

    points: List[List[float]] = []
    for index, point in enumerate(raw_points):
        field_name = f"corridor_points[{index}]"
        if isinstance(point, dict):
            try:
                lat, lon = _parse_lat_lon(point["lat"], point["lon"], field_name)
            except KeyError as exc:
                raise ValueError(f"{field_name} requires lat and lon.") from exc
        elif isinstance(point, (list, tuple, np.ndarray)):
            values = np.asarray(point, dtype=object).reshape(-1)
            if values.size not in (2, 3):
                raise ValueError(f"{field_name} must be [lat, lon] or [lat, lon, alt].")
            lat, lon = _parse_lat_lon(values[0], values[1], field_name)
        else:
            raise ValueError(f"{field_name} must be an object or coordinate list.")
        # API corridor points always lie on the requested cruise MSL plane.
        points.append([lat, lon, float(cruise_altitude_m)])

    if not points:
        return np.empty((0, 3), dtype=float)
    return np.asarray(points, dtype=float).reshape(-1, 3)


def _parse_no_fly_zones_to_bbox(no_fly_zones: Any) -> np.ndarray:
    """Convert supported NFZ forms to [lon_min, lon_max, lat_min, lat_max]."""
    if no_fly_zones is None:
        return np.empty((0, 4), dtype=float)
    if not isinstance(no_fly_zones, (list, tuple, np.ndarray)):
        raise ValueError("no_fly_zones must be a list.")

    boxes: List[List[float]] = []
    for index, zone in enumerate(no_fly_zones):
        field_name = f"no_fly_zones[{index}]"
        if isinstance(zone, (list, tuple, np.ndarray)):
            values = np.asarray(zone, dtype=object).reshape(-1)
            if values.size != 4:
                raise ValueError(f"{field_name} must contain four bbox values.")
            box = [_finite_number(v, field_name) for v in values]
        elif isinstance(zone, dict) and isinstance(zone.get("bbox"), dict):
            bbox = zone["bbox"]
            try:
                box = [
                    _finite_number(bbox["lon_min"], f"{field_name}.bbox.lon_min"),
                    _finite_number(bbox["lon_max"], f"{field_name}.bbox.lon_max"),
                    _finite_number(bbox["lat_min"], f"{field_name}.bbox.lat_min"),
                    _finite_number(bbox["lat_max"], f"{field_name}.bbox.lat_max"),
                ]
            except KeyError as exc:
                raise ValueError(f"{field_name}.bbox is incomplete.") from exc
        elif isinstance(zone, dict) and isinstance(zone.get("bbox"), (list, tuple)):
            values = list(zone["bbox"])
            if len(values) != 4:
                raise ValueError(f"{field_name}.bbox must contain four values.")
            box = [_finite_number(v, f"{field_name}.bbox") for v in values]
        elif isinstance(zone, dict) and isinstance(zone.get("center"), dict):
            center = zone["center"]
            try:
                center_lat, center_lon = _parse_lat_lon(
                    center["lat"], center["lon"], f"{field_name}.center"
                )
            except KeyError as exc:
                raise ValueError(f"{field_name}.center requires lat and lon.") from exc
            radius_km = _finite_number(zone.get("radius_km", 1.0), f"{field_name}.radius_km")
            if radius_km < 0.0:
                raise ValueError(f"{field_name}.radius_km must be non-negative.")
            radius_m = radius_km * 1000.0
            d_lat = radius_m / 111000.0
            meters_per_lon = 111000.0 * math.cos(math.radians(center_lat))
            d_lon = radius_m / meters_per_lon if abs(meters_per_lon) > 1e-9 else d_lat
            box = [center_lon - d_lon, center_lon + d_lon, center_lat - d_lat, center_lat + d_lat]
        else:
            raise ValueError(f"{field_name} requires bbox or center/radius_km.")

        lon_min, lon_max, lat_min, lat_max = box
        if lon_min > lon_max or lat_min > lat_max:
            raise ValueError(f"{field_name} bbox minimum must not exceed maximum.")
        boxes.append([lon_min, lon_max, lat_min, lat_max])

    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def _parse_airspace(
    raw_airspace: Any,
    start_vertiport: np.ndarray,
    end_vertiport: np.ndarray,
) -> Dict[str, Any]:
    if raw_airspace is None:
        return {}
    if not isinstance(raw_airspace, dict):
        raise ValueError("airspace_info must be an object.")
    if not raw_airspace:
        return {}
    center = raw_airspace.get("center")
    if not isinstance(center, dict):
        raise ValueError("airspace_info.center must contain lat and lon.")
    try:
        center_lat, center_lon = _parse_lat_lon(
            center["lat"], center["lon"], "airspace_info.center"
        )
    except KeyError as exc:
        raise ValueError("airspace_info.center requires lat and lon.") from exc
    center_alt = _finite_number(
        center.get("alt_m", 0.5 * (start_vertiport[2] + end_vertiport[2])),
        "airspace_info.center.alt_m",
    )
    radius_km = _finite_number(raw_airspace.get("radius_km"), "airspace_info.radius_km")
    if radius_km <= 0.0:
        raise ValueError("airspace_info.radius_km must be greater than 0.")
    result: Dict[str, Any] = {
        "center": [center_lat, center_lon, center_alt],
        "radius_km": radius_km,
    }
    if raw_airspace.get("alt_min_m") is not None:
        result["alt_min_m"] = _finite_number(raw_airspace["alt_min_m"], "airspace_info.alt_min_m")
    if raw_airspace.get("alt_max_m") is not None:
        result["alt_max_m"] = _finite_number(raw_airspace["alt_max_m"], "airspace_info.alt_max_m")
    if (
        "alt_min_m" in result
        and "alt_max_m" in result
        and result["alt_max_m"] <= result["alt_min_m"]
    ):
        raise ValueError("airspace_info.alt_max_m must be greater than alt_min_m.")
    return result


def _unpack_request_payload(path_request: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the public request and return only current engine inputs."""
    if not isinstance(path_request, dict):
        raise ValueError("Request payload must be a JSON object.")

    start_point = _parse_vertiport(
        path_request.get("start_vertiport"),
        "start_vertiport",
        DEFAULT_START_VERTIPORT,
    )
    end_point = _parse_vertiport(
        path_request.get("end_vertiport"),
        "end_vertiport",
        DEFAULT_END_VERTIPORT,
    )
    cruise_altitude_m = _finite_number(
        path_request.get("cruise_altitude_m"), "cruise_altitude_m"
    )
    takeoff_angle = _parse_angle(
        path_request.get("takeoff_climb_angle_deg", DEFAULT_TRANSITION_ANGLE_DEG),
        "takeoff_climb_angle_deg",
    )
    landing_angle = _parse_angle(
        path_request.get("landing_descent_angle_deg", DEFAULT_TRANSITION_ANGLE_DEG),
        "landing_descent_angle_deg",
    )

    raw_corridor = path_request.get("corridor_points")
    if not raw_corridor:
        # Preserve the existing legacy alias without adding default waypoints.
        raw_corridor = path_request.get("waypoints") or []
    corridor_points = _parse_corridor_points(raw_corridor, cruise_altitude_m)
    airspace = _parse_airspace(path_request.get("airspace_info"), start_point, end_point)
    no_fly_zones = _parse_no_fly_zones_to_bbox(path_request.get("no_fly_zones", []))

    # takeoff_end, landing_end, and min_corridor_distance_km are deliberately
    # never read.  Legacy clients may send them, but they cannot affect a run.
    return {
        "start_point": start_point.tolist(),
        "end_point": end_point.tolist(),
        "airspace_info": airspace,
        "no_fly_zones": no_fly_zones.tolist(),
        "corridor_points": corridor_points.tolist(),
        "cruise_altitude_m": cruise_altitude_m,
        "takeoff_climb_angle_deg": takeoff_angle,
        "landing_descent_angle_deg": landing_angle,
    }


def _load_risk_maps() -> None:
    """Warm API startup caches and fail early when required project data is absent."""
    ground_path = project_path("ground_risk_data", "Modified_high_res_affected_population_GRC.npy")
    bird_path = project_path("air_risk_data", "bird_riskmap_springfall_3d.npy")
    moc_dir = project_path("260608_MOC")
    noise_path = project_path("noise_data", "noise_lden_grid.npy")
    for label, path in (
        ("Ground risk map", ground_path),
        ("Bird risk map", bird_path),
        ("MOC directory", moc_dir),
        ("Noise grid", noise_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    np.load(str(ground_path), allow_pickle=True)
    np.load(str(bird_path), allow_pickle=True)
    moc_files = sorted(moc_dir.glob("UAM_MOC_XYZ_risk_fixedAGL*.npy"))
    if not moc_files:
        raise FileNotFoundError(f"No fixed-AGL MOC files found in: {moc_dir}")
    np.load(str(moc_files[0]), allow_pickle=True)
    np.load(str(noise_path), allow_pickle=True)


def _path_scope_for_mode(transition_structure_mode: str) -> str:
    if transition_structure_mode == _core.TRANSITION_STRUCTURE_FIXED_ONLY:
        return PATH_SCOPE_FIXED_ONLY
    return PATH_SCOPE_FULL


def _load_route_artifact(run_dir: Path, transition_structure_mode: str) -> Dict[str, Any]:
    """Read and validate the authoritative Route_Data sheet."""
    xlsx_path = Path(run_dir) / "route_data.xlsx"
    if not xlsx_path.exists():
        raise RuntimeError(f"Optimization route artifact is missing: {xlsx_path}")
    try:
        route_df = pd.read_excel(xlsx_path, sheet_name="Route_Data")
    except Exception as exc:
        raise RuntimeError(f"Failed to read Route_Data from {xlsx_path}") from exc

    required = ["Flight_Phase", "Lat", "Lon", "Altitude_MSL_m"]
    missing = [column for column in required if column not in route_df.columns]
    if missing:
        raise RuntimeError("Route_Data is missing required column(s): " + ", ".join(missing))
    route = route_df[["Lat", "Lon", "Altitude_MSL_m"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    phases = route_df["Flight_Phase"].astype(str).to_numpy(dtype=object)
    if route.shape[0] == 0 or not np.all(np.isfinite(route)):
        raise RuntimeError("Route_Data contains no valid optimized path rows.")

    if transition_structure_mode == _core.TRANSITION_STRUCTURE_FIXED_ONLY:
        hidden_phases = {
            _core.FLIGHT_PHASE_VERTIPORT,
            _core.FLIGHT_PHASE_TAKEOFF_STAGE1,
            _core.FLIGHT_PHASE_LANDING_STAGE1,
        }
        exposed_hidden = sorted(set(str(v) for v in phases) & hidden_phases)
        if exposed_hidden:
            raise RuntimeError(
                "Fixed-straight-only Route_Data exposed hidden transition rows: "
                + ", ".join(exposed_hidden)
            )
    else:
        if phases[0] != _core.FLIGHT_PHASE_VERTIPORT or phases[-1] != _core.FLIGHT_PHASE_VERTIPORT:
            raise RuntimeError(
                "Full-scope Route_Data must begin and end with a vertiport row."
            )
    return {
        "path": route.tolist(),
        "phases": [str(value) for value in phases.tolist()],
        "row_count": int(route.shape[0]),
        "excel_path": str(xlsx_path),
    }


def _load_optimal_path_from_excel(run_dir: Path) -> List[List[float]]:
    """Backward-compatible route loader using params.json to determine scope."""
    params_path = Path(run_dir) / "params.json"
    if not params_path.exists():
        raise RuntimeError(f"Optimization parameters are missing: {params_path}")
    with open(params_path, "r", encoding="utf-8") as stream:
        params = json.load(stream)
    mode = str(params.get("transition_structure_mode_effective") or params.get("transition_structure_mode"))
    return _load_route_artifact(Path(run_dir), mode)["path"]


def run_path_engine(
    start_point: Optional[Sequence[float]] = None,
    end_point: Optional[Sequence[float]] = None,
    airspace_info: Optional[Dict[str, Any]] = None,
    no_fly_zones: Optional[Sequence[Sequence[float]]] = None,
    corridor_points: Optional[Sequence[Sequence[float]]] = None,
    cruise_altitude_m: Optional[float] = None,
    takeoff_climb_angle_deg: float = DEFAULT_TRANSITION_ANGLE_DEG,
    landing_descent_angle_deg: float = DEFAULT_TRANSITION_ANGLE_DEG,
    max_attempts: int = 1,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    """Run the latest main optimizer with API-owned inputs only."""
    if cruise_altitude_m is None:
        raise ValueError("cruise_altitude_m is required.")
    cruise_altitude_m = _finite_number(cruise_altitude_m, "cruise_altitude_m")
    takeoff_climb_angle_deg = _parse_angle(
        takeoff_climb_angle_deg, "takeoff_climb_angle_deg"
    )
    landing_descent_angle_deg = _parse_angle(
        landing_descent_angle_deg, "landing_descent_angle_deg"
    )
    start = np.asarray(
        DEFAULT_START_VERTIPORT if start_point is None else start_point, dtype=float
    ).reshape(-1)
    end = np.asarray(
        DEFAULT_END_VERTIPORT if end_point is None else end_point, dtype=float
    ).reshape(-1)
    if start.size != 3 or end.size != 3 or not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise ValueError("start_point and end_point must be finite [lat, lon, alt] values.")
    corridor = _parse_corridor_points(corridor_points, cruise_altitude_m)
    nfz = _parse_no_fly_zones_to_bbox(no_fly_zones)

    attempts = int(max(1, max_attempts))
    last_run_dir: Optional[Path] = None
    _emit_progress(progress_callback, 2, "request_validation", "Path request validated.")
    for attempt in range(1, attempts + 1):
        ok, feasible_count, run_dir = _core.attempt_run_once(
            start_vertiport_override=start.tolist(),
            end_vertiport_override=end.tolist(),
            airspace_info_override=dict(airspace_info or {}),
            forbidden_zones_override=nfz.tolist(),
            corridor_points_override=corridor.tolist(),
            cruise_altitude_m_override=cruise_altitude_m,
            takeoff_climb_angle_deg_override=takeoff_climb_angle_deg,
            landing_descent_angle_deg_override=landing_descent_angle_deg,
            project_root=PROJECT_ROOT,
            use_clicked_waypoints_override=False,
            progress_callback=lambda event: _forward_core_progress(progress_callback, event),
            return_run_dir=True,
        )
        last_run_dir = Path(run_dir).resolve()
        if not ok:
            continue

        results_path = last_run_dir / "results.pkl"
        if not results_path.exists():
            raise RuntimeError(f"Optimization results are missing: {results_path}")
        with open(results_path, "rb") as stream:
            result_obj = pickle.load(stream)
        transition_structure_mode = str(
            result_obj.get("transition_structure_mode_effective")
            or result_obj.get("transition_structure_mode")
        )
        if transition_structure_mode not in _core.TRANSITION_STRUCTURE_MODES:
            raise RuntimeError(
                f"Unknown transition structure mode in results: {transition_structure_mode!r}"
            )
        route_artifact = _load_route_artifact(last_run_dir, transition_structure_mode)
        result_path = np.asarray(result_obj.get("route_data_path", []), dtype=float).reshape(-1, 3)
        excel_path = np.asarray(route_artifact["path"], dtype=float).reshape(-1, 3)
        if result_path.shape != excel_path.shape or not np.allclose(
            result_path, excel_path, rtol=0.0, atol=0.05
        ):
            raise RuntimeError("results.pkl route_data_path differs from Excel Route_Data.")

        path_scope = _path_scope_for_mode(transition_structure_mode)
        _emit_progress(progress_callback, 100, "complete", "Optimization result is ready.")
        return {
            "success": True,
            "attempt": attempt,
            "feasible_count": int(feasible_count),
            "run_dir": str(last_run_dir),
            "optimal_path": route_artifact["path"],
            "route_flight_phases": route_artifact["phases"],
            "route_data_row_count": route_artifact["row_count"],
            "transition_structure_mode": transition_structure_mode,
            "path_scope": path_scope,
            "result": result_obj,
            "artifact_files": sorted(
                str(path) for path in last_run_dir.glob("**/*") if path.is_file()
            ),
        }

    return {
        "success": False,
        "attempt": attempts,
        "feasible_count": 0,
        "run_dir": None if last_run_dir is None else str(last_run_dir),
        "optimal_path": [],
        "route_flight_phases": [],
        "route_data_row_count": 0,
        "transition_structure_mode": None,
        "path_scope": None,
        "result": None,
        "artifact_files": [],
    }


def find_optimal_path(
    path_request: Dict[str, Any],
    progress_callback: ProgressCallback = None,
) -> List[List[float]]:
    """Compatibility helper returning only route waypoints."""
    return find_optimal_path_with_artifacts(path_request, progress_callback)["optimal_path"]


def find_optimal_path_with_artifacts(
    path_request: Dict[str, Any],
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    """Validate a public request and return the route plus run artifacts."""
    _emit_progress(progress_callback, 1, "request_validation", "Validating path request.")
    request = _unpack_request_payload(path_request)
    return run_path_engine(
        start_point=request["start_point"],
        end_point=request["end_point"],
        airspace_info=request["airspace_info"],
        no_fly_zones=request["no_fly_zones"],
        corridor_points=request["corridor_points"],
        cruise_altitude_m=request["cruise_altitude_m"],
        takeoff_climb_angle_deg=request["takeoff_climb_angle_deg"],
        landing_descent_angle_deg=request["landing_descent_angle_deg"],
        max_attempts=1,
        progress_callback=progress_callback,
    )


# Compatibility alias for local integrations that imported the old duplicate.
attempt_run_once = _core.attempt_run_once
cleanup_matplotlib_tk = _core.cleanup_matplotlib_tk


if __name__ == "__main__":
    raise SystemExit(
        "path_engine.py is an API adapter. Run MAIN_uam_corridor_optimizer.py "
        "for standalone optimization or start api_service.api_server."
    )
