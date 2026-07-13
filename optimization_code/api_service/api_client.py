import requests
import json
import os
from pathlib import Path

# --- 서버 정보 ---
# api_server.py가 실행 중인 주소
SERVER_URL = "http://127.0.0.1:8000" # 로컬에서 테스트 시
# SERVER_URL = "https://bff54cdf0390.ngrok-free.app" # 손교수님 서버 주소

API_SERVICE_DIR = Path(__file__).resolve().parent
LOG_DIR = str(API_SERVICE_DIR / "logs")

# /optimized-path is a long-running optimization endpoint.
REQUEST_TIMEOUTS = {
    "/optimized-path": 600,
}

USE_STREAMING = True

# 시작점 고도 기반.
payload_optimal_corridor = {
    # Required fields
    "start_vertiport": {
        "lla": {"lat": 35.6033361, "lon": 129.0776917, "alt_m": 150.0},
    },
    # Required fields
    "end_vertiport": {
        "lla": {"lat": 35.6033361, "lon": 129.0776917, "alt_m": 150.0},
    },
    # Required fields
    "airspace_info": {
        "center": {"lat": 35.6033361, "lon": 129.0776917, "alt_m": 150.0},
        "radius_km": 5.0,
        "altitude_min_m": 100.0,
        "altitude_max_m": 1000.0,
    },
    # Required fields
    "cruise_altitude_m": 600.0,
    # optional (default: []) [lon_min, lon_max, lat_min, lat_max]
    "no_fly_zones": [
        # {"bbox": [129.0700, 129.0820, 35.5980, 35.6100]},
    ],
    # optional middle waypoints (if empty, start/end만으로 계산)
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
    # optional minimum corridor distance constraint (default: 0, meaning no constraint)
    "min_corridor_distance_km": 0.0,
}


def post_and_log(endpoint, payload):
    url = f"{SERVER_URL}{endpoint}"
    timeout = int(REQUEST_TIMEOUTS.get(endpoint, 30))
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
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
        return {
            "error": str(e),
            "endpoint": endpoint,
            "timeout_sec": timeout,
        }


def check_server_health():
    url = f"{SERVER_URL}/"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        body = resp.json()
        print(f"[health] connected: {body}")
        return True
    except Exception as e:
        print(f"[health] failed: {e}")
        return False


def post_and_log_stream(payload):
    url = f"{SERVER_URL}/optimized-path/stream"
    final_event = None
    try:
        with requests.post(url, json=payload, stream=True, timeout=(10, 1800)) as resp:
            resp.raise_for_status()
            print("[client] streaming connected")

            for raw_line in resp.iter_lines(decode_unicode=True):
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
                if event_type == "log":
                    print(f"[server/log] {ev.get('message', '')}")
                elif event_type == "status":
                    print(f"[server/status] {ev.get('message', '')}")
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
                        print(f"[server/result] failed (status={ev.get('status_code')}): {ev.get('detail')}")
                else:
                    print(f"[server/{event_type}] {ev}")

    except Exception as e:
        print(f"Error(stream): {e}")
        return {
            "error": str(e),
            "endpoint": "/optimized-path/stream",
        }

    if final_event and bool(final_event.get("ok", False)):
        return final_event.get("response", {})
    return final_event or {}


def save_log(filename, data):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


def download_excel_if_available(response_body):
    if not isinstance(response_body, dict):
        return None

    download_path = response_body.get("excel_download_path")
    file_name = response_body.get("excel_file_name") or "route_data.xlsx"
    if not download_path:
        return None

    url = f"{SERVER_URL}{download_path}"
    save_path = os.path.join(LOG_DIR, file_name)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with requests.get(url, timeout=120, stream=True) as resp:
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
        print(f"Saved Excel: {save_path}")
        return save_path
    except Exception as e:
        print(f"Excel download failed: {e}")
        return None


def test_find_path():
    mode = "/optimized-path/stream" if USE_STREAMING else "/optimized-path"
    print(f"=== POST {mode} ===")
    if not check_server_health():
        print("Server health check failed. Start api_server.py first.")
        return

    save_log("request_optimal_corridor.json", payload_optimal_corridor)
    if USE_STREAMING:
        result = post_and_log_stream(payload_optimal_corridor)
    else:
        result = post_and_log("/optimized-path", payload_optimal_corridor)

    download_excel_if_available(result)
    save_log("response_optimized_path.json", result)


if __name__ == "__main__":
    test_find_path()
