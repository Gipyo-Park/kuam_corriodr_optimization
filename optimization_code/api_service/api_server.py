import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import queue
import sys
import threading
import time
import traceback
import uuid
import json
from pathlib import Path

# path_engine.py에서 핵심 함수를 가져옵니다.
try:
    from .path_engine import find_optimal_path_with_artifacts, _load_risk_maps
except ImportError:
    from path_engine import find_optimal_path_with_artifacts, _load_risk_maps

# --- FastAPI 앱 초기화 ---
app = FastAPI(
    title="K-UAM Optimal Pathfinding API",
    description="API to find the optimal UAM path between two points.",
    version="1.0.0",
)

# --- 데이터 모델 정의 (Pydantic) ---
class LLA(BaseModel):
    lat: float
    lon: float
    alt_m: float


class LatLon(BaseModel):
    lat: float
    lon: float


class EndpointLatLon(BaseModel):
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)

    model_config = ConfigDict(extra="forbid")


class TransitionEndpoint(BaseModel):
    lla: EndpointLatLon

    model_config = ConfigDict(extra="forbid")


class WaypointLLA(BaseModel):
    lat: float
    lon: float
    alt_m: Optional[float] = None


class Vertiport(BaseModel):
    lla: LLA


class AirspaceInfo(BaseModel):
    center: LatLon
    radius_km: float


class PathRequest(BaseModel):
    """API 요청 본문 모델 (api_request.py 스키마)"""
    start_vertiport: Optional[Vertiport] = None
    end_vertiport: Optional[Vertiport] = None
    takeoff_end: Optional[TransitionEndpoint] = None
    landing_end: Optional[TransitionEndpoint] = None
    airspace_info: Optional[AirspaceInfo] = None
    # Preferred NFZ format: {"bbox": [lon_min, lon_max, lat_min, lat_max]}
    # Backward compatible: {"center": {lat, lon}, "radius_km": 1.0}
    no_fly_zones: List[Any] = Field(default_factory=list)
    # Optional middle waypoints. Missing/null/empty runs without middle waypoints.
    corridor_points: Optional[List[WaypointLLA]] = None
    # Backward-compatible alias for corridor_points.
    waypoints: Optional[List[WaypointLLA]] = None
    # Required explicit cruise altitude used by planner altitude_levels.
    cruise_altitude_m: float
    min_corridor_distance_km: Optional[float] = 0.0

    # OpenAPI /docs display example only. Runtime defaults live in path_engine.py.
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start_vertiport": {
                    "lla": {"lat": 35.6033361, "lon": 129.0776917, "alt_m": 150.0}
                },
                "end_vertiport": {
                    "lla": {"lat": 35.6033361, "lon": 129.0776917, "alt_m": 150.0}
                },
                "takeoff_end": {
                    "lla": {
                        "lat": 35.59468397,
                        "lon": 129.07515721
                    }
                },
                "landing_end": {
                    "lla": {
                        "lat": 35.59701567,
                        "lon": 129.08585995
                    }
                },
                "airspace_info": {
                    "center": {"lat": 35.6033361, "lon": 129.0776917},
                    "radius_km": 5.0
                },
                "cruise_altitude_m": 600.0,
                "no_fly_zones": [
                    {"bbox": [129.0600, 129.0700, 35.5950, 35.6050]}
                ],
                "corridor_points": [],
                "min_corridor_distance_km": 0.0
            }
        }
    )

class PathResponse(BaseModel):
    """API 응답 본문 모델"""
    message: str
    waypoint_count: int
    waypoints: List[Tuple[float, float, float]]
    run_dir: Optional[str] = None
    excel_file_name: Optional[str] = None
    excel_download_path: Optional[str] = None


_EXCEL_ARTIFACTS: Dict[str, str] = {}
SSE_HEARTBEAT_SECONDS = 10.0


def _normalize_no_fly_zones(no_fly_zones: List[Any]) -> List[Any]:
    normalized: List[Any] = []
    for i, zone in enumerate(no_fly_zones, start=1):
        if isinstance(zone, dict):
            bbox = zone.get("bbox")
            if isinstance(bbox, list) and len(bbox) != 4:
                raise ValueError(
                    f"no_fly_zones[{i}].bbox must be [lon_min, lon_max, lat_min, lat_max]."
                )
            normalized.append(zone)
            continue

        if isinstance(zone, (list, tuple)):
            if len(zone) != 4:
                raise ValueError(
                    f"no_fly_zones[{i}] list must be [lon_min, lon_max, lat_min, lat_max]."
                )
            normalized.append([float(v) for v in zone])
            continue

        raise ValueError(
            f"Invalid no_fly_zones[{i}] format. Use dict with bbox or 4-length list."
        )
    return normalized


def _build_engine_request(request: PathRequest) -> Dict[str, Any]:
    corridor_points = request.corridor_points or []
    legacy_waypoints = request.waypoints or []
    points = corridor_points if corridor_points else legacy_waypoints
    engine_request: Dict[str, Any] = {
        "no_fly_zones": _normalize_no_fly_zones(request.no_fly_zones),
        "min_corridor_distance_km": float(request.min_corridor_distance_km or 0.0),
        "corridor_points": [p.model_dump(exclude_none=True) for p in points],
        "cruise_altitude_m": float(request.cruise_altitude_m),
    }
    if request.start_vertiport is not None:
        engine_request["start_vertiport"] = {"lla": request.start_vertiport.lla.model_dump()}
    if request.end_vertiport is not None:
        engine_request["end_vertiport"] = {"lla": request.end_vertiport.lla.model_dump()}
    if request.takeoff_end is not None:
        engine_request["takeoff_end"] = request.takeoff_end.model_dump()
    if request.landing_end is not None:
        engine_request["landing_end"] = request.landing_end.model_dump()
    if request.airspace_info is not None:
        engine_request["airspace_info"] = request.airspace_info.model_dump()
    return engine_request


def _to_path_response_payload(engine_output: Dict[str, Any]) -> Dict[str, Any]:
    # ==================== 최적화된 회랑 데이터 추출 ====================
    # 중요: 다음 데이터는 모두 최적화된 회랑을 나타내며, 일관성 있게 최적화되어 있습니다:
    # 
    # 1. Excel 데이터 (route_data.xlsx의 Route_Data 시트):
    #    - 완전한 정보 포함: Type, Segment, Risk, Distance, TF/RF markers, Arc center
    #    - Transition 포인트 제외 (Takeoff_Path_End, Landing_Path_Start, Landing_Path_End)
    #    - TF/RF 경계 중복 제거 (연속된 TF→RF 전환 시 중복 마커 제거)
    #
    # 2. JSON Response (현재 응답, client가 수신):
    #    - 소스: engine_output["optimal_path"] = 엑셀의 핵심 데이터 (Lat, Lon, Alt_m)
    #    - 형식: [[lat, lon, alt_m], [lat, lon, alt_m], ...]
    #    - 포인트 개수: len(waypoints) = excel의 행 수
    #
    # 3. waypoint_count 정확성:
    #    - waypoint_count = len(optimal_path) = 엑셀 포인트 개수
    #    - 3가지 데이터의 포인트 개수가 항상 동일함을 보장
    #
    # 결론: Excel, JSON Response, Engine Output의 3가지가 동일한 최적화된 회랑을 나타냅니다.
    
    path = engine_output.get("optimal_path", [])
    if not path:
        raise HTTPException(
            status_code=404,
            detail="Could not find a feasible optimal path with the given parameters.",
        )
    run_dir = engine_output.get("run_dir")
    excel_file_name = None
    excel_download_path = None

    artifact_files = engine_output.get("artifact_files", []) or []
    excel_candidates = [
        str(p) for p in artifact_files
        if str(p).lower().endswith(".xlsx")
    ]
    if excel_candidates:
        preferred = next(
            (p for p in excel_candidates if Path(p).name.lower() == "route_data.xlsx"),
            excel_candidates[0],
        )
        excel_file_name = Path(preferred).name
        run_id = Path(str(run_dir)).name if run_dir else None
        if run_id:
            _EXCEL_ARTIFACTS[run_id] = str(preferred)
            excel_download_path = f"/artifacts/excel/{run_id}"

    return {
        "message": "Optimal path found successfully.",
        # ==================== waypoint_count: 엑셀 포인트 개수 ====================
        # 이 개수는 다음을 보장합니다:
        # - 엑셀의 Route_Data 행 수와 동일
        # - Transition 포인트 제외 후의 최적화된 회랑
        # - TF/RF 경계 중복 제거 후의 최종 포인트 수
        "waypoint_count": len(path),
        # ==================== waypoints: 엑셀 회랑 데이터 ====================
        # 각 waypoint는 [lat, lon, alt_m] 형식
        # 소스: engine_output["optimal_path"] (엑셀 Route_Data 시트의 Lat/Lon/Alt_m)
        # 순서: 엑셀에 저장된 순서 유지
        "waypoints": [tuple(waypoint) for waypoint in path],
        "run_dir": run_dir,
        "excel_file_name": excel_file_name,
        "excel_download_path": excel_download_path,
    }


# --- 서버 시작 시 이벤트 핸들러 ---
@app.on_event("startup")
async def startup_event():
    """
    서버가 시작될 때 무거운 Risk Map 데이터들을 미리 로드합니다.
    이렇게 하면 첫 요청 시 지연이 발생하는 것을 방지할 수 있습니다.
    """
    print("Server is starting up...")
    _load_risk_maps()
    print("Risk maps loaded successfully. Server is ready.")


# --- API 엔드포인트 정의 ---
@app.get("/", summary="Health Check")
async def read_root():
    """서버가 정상적으로 실행 중인지 확인하는 기본 엔드포인트입니다."""
    return {"status": "K-UAM Pathfinding API is running."}


@app.post("/optimized-path", response_model=PathResponse, summary="Find Optimal Path")
async def get_optimal_path(request: PathRequest):
    """
    시작점과 끝점을 받아 최적의 비행 경로(Waypoint 리스트)를 계산하여 반환합니다.
    """
    print(
        "Received optimized-path request: "
        f"start={(request.start_vertiport.lla if request.start_vertiport else None)} "
        f"end={(request.end_vertiport.lla if request.end_vertiport else None)}"
    )
    
    try:
        # path_engine의 메인 함수 호출
        engine_request = _build_engine_request(request)
        engine_output = find_optimal_path_with_artifacts(engine_request)
        return _to_path_response_payload(engine_output)

    except HTTPException:
        raise

    except ValueError as e:
        # 입력 검증 실패는 422(Unprocessable Entity)로 반환
        print(f"Invalid pathfinding request: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        # 경로 탐색 중 예상치 못한 에러 발생 시
        print(f"An error occurred during pathfinding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimized-path/stream", summary="Find Optimal Path With Streaming Progress and Diagnostics")
async def get_optimal_path_stream(request: PathRequest):
    """
    Stream status, progress, initial-population diagnostics, errors, and the final
    result through SSE. Detailed engine logs and tracebacks stay on the server.
    """
    event_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    done = threading.Event()
    state: Dict[str, Any] = {
        "engine_output": None,
        "error": None,
        "error_id": None,
        "error_message": None,
        "percent": 0,
        "stage": "accepted",
        "latest_diagnostic": None,
    }

    def _engine_event_callback(event: Dict[str, Any]) -> None:
        percent = int(max(int(state["percent"]), min(100, max(0, int(event.get("percent", 0))))))
        engine_event = dict(event)
        event_type = str(engine_event.get("event", "progress"))
        if event_type not in {"progress", "diagnostic"}:
            event_type = "progress"
        engine_event["event"] = event_type
        engine_event["percent"] = percent
        state["percent"] = percent
        state["stage"] = str(engine_event.get("stage", state["stage"]))
        if event_type == "diagnostic":
            state["latest_diagnostic"] = engine_event
        event_queue.put(engine_event)

    def _record_exception(error: Exception) -> None:
        error_id = uuid.uuid4().hex
        error_message = str(error).strip() or (
            f"{type(error).__name__} occurred without an error message."
        )
        state["error"] = error
        state["error_id"] = error_id
        state["error_message"] = error_message
        print(
            f"[path-engine/error] error_id={error_id} "
            f"type={type(error).__name__} stage={state['stage']} "
            f"percent={state['percent']} message={error_message}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)

    def _worker() -> None:
        try:
            event_queue.put({"event": "status", "message": "Request accepted. Building engine request..."})
            state["stage"] = "input_validation"
            engine_request = _build_engine_request(request)
            event_queue.put({"event": "status", "message": "Path optimization started."})
            state["stage"] = "optimization"
            state["engine_output"] = find_optimal_path_with_artifacts(
                engine_request,
                progress_callback=_engine_event_callback,
            )
        except Exception as e:
            _record_exception(e)
        finally:
            done.set()

    threading.Thread(target=_worker, daemon=True).start()

    async def _event_generator():
        yield f"data: {json.dumps({'event': 'accepted', 'message': 'Streaming started'}, ensure_ascii=False)}\n\n"
        last_event_at = time.monotonic()
        stream_started_at = last_event_at

        while not done.is_set() or not event_queue.empty():
            flushed = False
            while True:
                try:
                    ev = event_queue.get_nowait()
                except queue.Empty:
                    break
                flushed = True
                last_event_at = time.monotonic()
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

            if not flushed:
                now = time.monotonic()
                if not done.is_set() and now - last_event_at >= SSE_HEARTBEAT_SECONDS:
                    heartbeat = {
                        "event": "status",
                        "stage": state["stage"],
                        "percent": int(state["percent"]),
                        "elapsed_seconds": int(max(0.0, now - stream_started_at)),
                        "message": "Optimization is still running.",
                    }
                    last_event_at = now
                    yield f"data: {json.dumps(heartbeat, ensure_ascii=False)}\n\n"
                    continue
                await asyncio.sleep(0.15)

        if state.get("error") is not None:
            error = state["error"]
            status_code = 422 if isinstance(error, ValueError) else 500
            error_id = str(state["error_id"])
            error_message = str(state["error_message"])
            error_payload = {
                "event": "error",
                "status_code": status_code,
                "error_type": type(error).__name__,
                "stage": state["stage"],
                "percent": int(state["percent"]),
                "message": error_message,
                "error_id": error_id,
            }
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
            result_payload = {
                "event": "result",
                "ok": False,
                "status_code": status_code,
                "detail": error_message,
                "error_id": error_id,
            }
            yield f"data: {json.dumps(result_payload, ensure_ascii=False)}\n\n"
            return

        engine_output = state.get("engine_output") or {}
        path = engine_output.get("optimal_path", [])
        if not path:
            latest_diagnostic = state.get("latest_diagnostic")
            initial_population_failed = bool(
                isinstance(latest_diagnostic, dict)
                and latest_diagnostic.get("stage") == "initial_population"
                and latest_diagnostic.get("state") == "failed"
            )
            if initial_population_failed:
                error_type = "NoFeasibleInitialPopulationError"
                message = "No feasible initial population was found after all retries."
            else:
                error_type = "NoFeasiblePathError"
                message = "Could not find a feasible optimal path with the given parameters."
            error_id = uuid.uuid4().hex
            print(
                f"[path-engine/error] error_id={error_id} "
                f"type={error_type} stage={state['stage']} "
                f"percent={state['percent']} message={message}",
                file=sys.stderr,
                flush=True,
            )
            error_payload = {
                "event": "error",
                "status_code": 404,
                "error_type": error_type,
                "stage": state["stage"],
                "percent": int(state["percent"]),
                "message": message,
                "error_id": error_id,
                "run_dir": engine_output.get("run_dir"),
            }
            if initial_population_failed:
                error_payload["diagnostic"] = latest_diagnostic
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
            result_payload = {
                "event": "result",
                "ok": False,
                "status_code": 404,
                "detail": message,
                "error_id": error_id,
                "run_dir": engine_output.get("run_dir"),
            }
            if initial_population_failed:
                result_payload["diagnostic"] = latest_diagnostic
            yield f"data: {json.dumps(result_payload, ensure_ascii=False)}\n\n"
            return

        try:
            state["stage"] = "result_processing"
            response_payload = _to_path_response_payload(engine_output)
            payload = {
                "event": "result",
                "ok": True,
                "response": response_payload,
                "run_dir": engine_output.get("run_dir"),
                "attempt": engine_output.get("attempt"),
                "feasible_count": engine_output.get("feasible_count"),
            }
            result_event = json.dumps(payload, ensure_ascii=False)
        except Exception as e:
            _record_exception(e)
            status_code = 422 if isinstance(e, ValueError) else 500
            error_payload = {
                "event": "error",
                "status_code": status_code,
                "error_type": type(e).__name__,
                "stage": state["stage"],
                "percent": int(state["percent"]),
                "message": state["error_message"],
                "error_id": state["error_id"],
            }
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
            result_payload = {
                "event": "result",
                "ok": False,
                "status_code": status_code,
                "detail": state["error_message"],
                "error_id": state["error_id"],
            }
            yield f"data: {json.dumps(result_payload, ensure_ascii=False)}\n\n"
            return

        yield f"data: {result_event}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/artifacts/excel/{run_id}", summary="Download Excel Artifact")
async def download_excel_artifact(run_id: str):
    excel_path = _EXCEL_ARTIFACTS.get(str(run_id))
    if excel_path is None:
        raise HTTPException(status_code=404, detail="Excel artifact not found for given run_id.")

    p = Path(excel_path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Excel artifact file is missing.")

    return FileResponse(
        path=str(p),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=p.name,
    )


@app.post("/find-path", response_model=PathResponse, summary="Find Optimal Path (Legacy Alias)")
async def get_optimal_path_legacy_alias(request: PathRequest):
    """Compatibility alias for older clients."""
    return await get_optimal_path(request)


# --- 서버 실행 ---
if __name__ == "__main__":
    # 터미널에서 `python api_server.py` 명령으로 서버를 실행할 수 있습니다.
    uvicorn.run(app, host="127.0.0.1", port=8000)
