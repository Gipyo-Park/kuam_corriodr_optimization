import requests
import json
import os
import webbrowser
from pathlib import Path

API_SERVICE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_SERVICE_DIR.parent

SERVER = "http://127.0.0.1:8000"  # 로컬 테스트용 
# SERVER = "https://03d0-114-70-4-161.ngrok-free.app"  # ngrok 쓸 경우 URL 교체

LOG_DIR = str(API_SERVICE_DIR / "logs")

payload_optimal_corridor = {
    "start_vertiport": {
        "lla": {"lat": 35.6033361, "lon": 129.0776917, "alt_m": 150.0},
    },
    "end_vertiport": {
        "lla": {"lat": 35.6033361, "lon": 129.0776917, "alt_m": 150.0},
    },
    # Optional; None keeps the automatic transition endpoint.
    # "takeoff_end": {"lla": {"lat": 35.59468397, "lon": 129.07515721}},
    # "landing_end": {"lla": {"lat": 35.59701567, "lon": 129.08585995}},
    "takeoff_end": None,
    "landing_end": None,
    "airspace_info": {
        "center": {"lat": 35.6033361, "lon": 129.0776917},
        "radius_km": 5.0,
    },
    # [lon_min, lon_max, lat_min, lat_max] # optional (default: [])
    "no_fly_zones": [                  
        # {"bbox": [129.0700, 129.0820, 35.5980, 35.6100]},  
    ],
    # optional middle waypoints (missing/null/empty uses no middle waypoints)
    "corridor_points": [
        {"lat": 35.6165628, "lon": 129.1174075, "alt_m": 600.0},
        {"lat": 35.6125953, "lon": 129.1271681, "alt_m": 600.0},
        {"lat": 35.5709934, "lon": 129.1042811, "alt_m": 600.0},
        {"lat": 35.5693508, "lon": 129.0849280, "alt_m": 600.0},
        {"lat": 35.5980918, "lon": 129.1098345, "alt_m": 600.0},
        {"lat": 35.6009654, "lon": 129.0968764, "alt_m": 600.0},
        {"lat": 35.5777004, "lon": 129.0787014, "alt_m": 600.0},
        {"lat": 35.5901549, "lon": 129.0696139, "alt_m": 600.0},
        {"lat": 35.6156051, "lon": 129.0442026, "alt_m": 600.0},
        {"lat": 35.6329778, "lon": 129.0531218, "alt_m": 600.0},
        {"lat": 35.6213509, "lon": 129.0687725, "alt_m": 600.0},
    ],
    "cruise_altitude_m": 600.0,
    "min_corridor_distance_km": 30.0,
}

# tmp_trajectory.txt 전체 waypoint를 예시로 사용
DATA_DIR = PROJECT_ROOT / "data"

def _load_trajectory():
    pts = []
    trajectory_path = DATA_DIR / "tmp_trajectory.txt"
    if not trajectory_path.exists():
        return pts
    with open(trajectory_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return pts

payload_ground_risk = {
    "waypoints": _load_trajectory()
}

def post_and_log(endpoint, payload):
    url = f"{SERVER}{endpoint}"
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        body = resp.json()
        wps = body.get("waypoints", [])
        if wps:
            print(f"waypoints: {len(wps)}개, start={wps[0]}, end={wps[-1]}")
        else:
            print("waypoints: 0개")
        return body
    except Exception as e:
        print(f"Error: {e}")
        return {}


def _print_diagnostic_event(event):
    checks = event.get("checks", {}) or {}
    overall = checks.get("overall_feasible", {}) or {}
    current = event.get("current", "?")
    total = event.get("total", "?")
    feasible = overall.get("passed", 0)
    target = overall.get("target", 0)
    state = str(event.get("state", ""))
    if state == "completed":
        print(
            f"[server/diagnostic] initial population completed on retry "
            f"{current}/{total}: feasible_candidates={feasible} (required={target})"
        )
        return
    if state == "failed":
        print(
            f"[server/diagnostic] initial population failed after "
            f"{current}/{total} retries: feasible_candidates={feasible} (required={target})"
        )
    else:
        print(
            f"[server/diagnostic] initial population retry {current}/{total}: "
            f"feasible_candidates={feasible} (required={target})"
        )

    blockers = list(event.get("blockers", []) or [])[:3]
    if blockers:
        blocker_text = ", ".join(
            f"{item.get('code')} {item.get('failed', 0)}/{item.get('evaluated', 0)}"
            for item in blockers
        )
        print(f"  blockers: {blocker_text}")
        actions = []
        for item in blockers:
            action = str(item.get("suggested_action", "")).strip()
            if action and action not in actions:
                actions.append(action)
        if actions:
            print(f"  suggested: {' | '.join(actions)}")


def post_optimized_path_stream(payload):
    url = f"{SERVER}/optimized-path/stream"
    final_event = None
    try:
        with requests.post(url, json=payload, stream=True, timeout=(10, 1800)) as resp:
            resp.raise_for_status()
            print("[request] streaming connected")

            for raw_line in resp.iter_lines(decode_unicode=True, chunk_size=1):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue

                data_str = line[len("data:"):].strip()
                if not data_str:
                    continue
                try:
                    ev = json.loads(data_str)
                except Exception:
                    print(f"[stream/raw] {data_str}")
                    continue

                event_type = str(ev.get("event", ""))
                if event_type == "progress":
                    percent = ev.get("percent", 0)
                    stage = ev.get("stage", "")
                    current = ev.get("current")
                    total = ev.get("total")
                    count_text = f" ({current}/{total})" if current is not None and total is not None else ""
                    print(f"[server/progress] {percent}% {stage}{count_text}: {ev.get('message', '')}")
                elif event_type == "diagnostic":
                    _print_diagnostic_event(ev)
                elif event_type == "error":
                    print(
                        f"[server/error] status={ev.get('status_code')} "
                        f"type={ev.get('error_type')} stage={ev.get('stage')} "
                        f"progress={ev.get('percent')}% id={ev.get('error_id')}: "
                        f"{ev.get('message', '')}"
                    )
                elif event_type == "status":
                    stage = ev.get("stage")
                    percent = ev.get("percent")
                    state_text = f" {percent}% {stage}" if stage is not None and percent is not None else ""
                    print(f"[server/status]{state_text}: {ev.get('message', '')}")
                elif event_type == "accepted":
                    print(f"[server] {ev.get('message', '')}")
                elif event_type == "result":
                    final_event = ev
                    ok = bool(ev.get("ok", False))
                    if ok:
                        response = ev.get("response", {})
                        wps = response.get("waypoints", [])
                        if wps:
                            print(f"[server/result] success, waypoints={len(wps)}, start={wps[0]}, end={wps[-1]}")
                        else:
                            print("[server/result] success, waypoints=0")
                    else:
                        print(
                            f"[server/result] failed (status={ev.get('status_code')}, "
                            f"id={ev.get('error_id')}): {ev.get('detail')}"
                        )
    except Exception as e:
        print(f"Error(stream): {e}")
        return {}

    if final_event and bool(final_event.get("ok", False)):
        return final_event.get("response", {})
    return final_event or {}


def save_log(filename, data):
    path = os.path.join(LOG_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")

if __name__ == "__main__":
    print("=== POST /optimized-path/stream ===")
    save_log("request_optimal_corridor.json", payload_optimal_corridor)
    result = post_optimized_path_stream(payload_optimal_corridor)
    save_log("response_optimized_path.json", result)

    print("\n=== POST /2D-ground-risk-map ===")
    save_log("request_2D_ground_risk_map.json", payload_ground_risk)
    resp = requests.post(f"{SERVER}/2D-ground-risk-map", json=payload_ground_risk, timeout=30)
    risk_data = resp.json()
    save_log("response_2D_ground_risk_map.json", risk_data)
    print(f"features: {len(risk_data.get('features', []))}")

    print("\n=== 2D Ground Risk Map 시각화 ===")
    webbrowser.open(f"{SERVER}/2D-ground-risk-map-view")
