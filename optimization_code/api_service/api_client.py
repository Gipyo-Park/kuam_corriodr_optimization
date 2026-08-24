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
    "/optimized-path": 1000,
}

USE_STREAMING = True

# 시작점 고도 기반.
payload_optimal_corridor = {
    # 필수: 시작 버티포트 위도·경도·MSL 고도(m)
    "start_vertiport": {
        "lla": {"lat": 35.603386, "lon": 129.078025, "alt_m": 150.0},
    },
    # 필수: 도착 버티포트 위도·경도·MSL 고도(m)
    "end_vertiport": {
        "lla": {"lat": 35.603386, "lon": 129.078025, "alt_m": 150.0},
    },
    # 필수: 순항 MSL 고도(m)
    "cruise_altitude_m": 600.0,
    # 선택: 이륙 상승각·착륙 하강각(deg), 생략 시 각각 6.0
    "takeoff_climb_angle_deg": 6.0,
    "landing_descent_angle_deg": 6.0,
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
            print(
                f"transition_structure_mode={body.get('transition_structure_mode')} | "
                f"path_scope={body.get('path_scope')}"
            )
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
        engine_ready = bool(body.get("engine_ready", False))
        if engine_ready:
            print(f"[health] ready: {body}")
        else:
            print(f"[health] degraded: {body}")
        return engine_ready
    except Exception as e:
        print(f"[health] failed: {e}")
        return False


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


def post_and_log_stream(payload):
    url = f"{SERVER_URL}/optimized-path/stream"
    final_event = None
    try:
        with requests.post(url, json=payload, stream=True, timeout=(10, 1800)) as resp:
            resp.raise_for_status()
            print("[client] streaming connected")

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
                elif event_type == "warning":
                    print(
                        f"[server/warning] code={ev.get('code')} "
                        f"stage={ev.get('stage')} progress={ev.get('percent')}%: "
                        f"{ev.get('message', '')}"
                    )
                elif event_type == "error":
                    print(
                        f"[server/error] status={ev.get('status_code')} "
                        f"type={ev.get('error_type')} stage={ev.get('stage')} "
                        f"progress={ev.get('percent')}% id={ev.get('error_id')}: "
                        f"{ev.get('message', '')}"
                    )
                    for detail in ev.get("details", []) or []:
                        print(
                            f"  - {detail.get('field')}: {detail.get('message')} "
                            f"[{detail.get('type')}]"
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
                            print(
                                "[server/result] "
                                f"transition_structure_mode={response.get('transition_structure_mode')} | "
                                f"path_scope={response.get('path_scope')}"
                            )
                        else:
                            print("[server/result] success, waypoints=0")
                    else:
                        print(
                            f"[server/result] failed (status={ev.get('status_code')}, "
                            f"id={ev.get('error_id')}): {ev.get('detail')}"
                        )
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
