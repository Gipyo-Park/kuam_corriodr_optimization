import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import contextlib
import queue
import threading
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


class WaypointLLA(BaseModel):
    lat: float
    lon: float
    alt_m: Optional[float] = None


class Vertiport(BaseModel):
    lla: LLA


class AirspaceInfo(BaseModel):
    center: LLA
    radius_km: float
    altitude_min_m: float
    altitude_max_m: float


class PathRequest(BaseModel):
    """API 요청 본문 모델 (api_request.py 스키마)"""
    start_vertiport: Optional[Vertiport] = None
    end_vertiport: Optional[Vertiport] = None
    airspace_info: Optional[AirspaceInfo] = None
    # Preferred NFZ format: {"bbox": [lon_min, lon_max, lat_min, lat_max]}
    # Backward compatible: {"center": {lat, lon, alt_m}, "radius_km": 1.0}
    no_fly_zones: List[Any] = Field(default_factory=list)
    # Optional middle waypoints. If empty, optimization runs with start/end only.
    corridor_points: List[WaypointLLA] = Field(default_factory=list)
    # Backward-compatible alias for corridor_points.
    waypoints: List[WaypointLLA] = Field(default_factory=list)
    # Required explicit cruise altitude used by planner altitude_levels.
    cruise_altitude_m: float
    min_corridor_distance_km: Optional[float] = 0.0

    class Config:
        schema_extra = {
            "example": {
                "start_vertiport": {
                    "lla": {"lat": 35.6033361, "lon": 129.0776917, "alt_m": 150.0}
                },
                "end_vertiport": {
                    "lla": {"lat": 35.6033361, "lon": 129.0776917, "alt_m": 150.0}
                },
                "airspace_info": {
                    "center": {"lat": 35.6033361, "lon": 129.0776917, "alt_m": 150.0},
                    "radius_km": 5.0,
                    "altitude_min_m": 100.0,
                    "altitude_max_m": 1000.0
                },
                "cruise_altitude_m": 600.0,
                "no_fly_zones": [
                    {"bbox": [129.0600, 129.0700, 35.5950, 35.6050]}
                ],
                "corridor_points": [
                    {"lat": 35.6201083, "lon": 129.1191806, "alt_m": 600.0}
                ],
                "min_corridor_distance_km": 0.0
            }
        }

class PathResponse(BaseModel):
    """API 응답 본문 모델"""
    message: str
    waypoint_count: int
    waypoints: List[Tuple[float, float, float]]
    run_dir: Optional[str] = None
    excel_file_name: Optional[str] = None
    excel_download_path: Optional[str] = None


_EXCEL_ARTIFACTS: Dict[str, str] = {}


class _QueueStreamWriter:
    def __init__(self, event_queue: "queue.Queue[Dict[str, Any]]"):
        self._q = event_queue
        self._buf = ""

    def write(self, s: str) -> int:
        text = str(s)
        if not text:
            return 0
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                self._q.put({"event": "log", "message": line})
        return len(text)

    def flush(self) -> None:
        if self._buf.strip():
            self._q.put({"event": "log", "message": self._buf.strip()})
        self._buf = ""


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
    points = request.corridor_points if len(request.corridor_points) > 0 else request.waypoints
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


@app.post("/optimized-path/stream", summary="Find Optimal Path With Streaming Logs")
async def get_optimal_path_stream(request: PathRequest):
    """
    SSE 스트림으로 서버 진행 로그(초기화/Gen/재시도)를 전달하고,
    마지막에 result 이벤트로 최종 결과를 보냅니다.
    """
    event_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    done = threading.Event()
    state: Dict[str, Any] = {"engine_output": None, "error": None}

    def _worker() -> None:
        try:
            event_queue.put({"event": "status", "message": "Request accepted. Building engine request..."})
            engine_request = _build_engine_request(request)
            event_queue.put({"event": "status", "message": "Path optimization started."})
            writer = _QueueStreamWriter(event_queue)
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                state["engine_output"] = find_optimal_path_with_artifacts(engine_request)
            writer.flush()
        except Exception as e:
            state["error"] = str(e)
        finally:
            done.set()

    threading.Thread(target=_worker, daemon=True).start()

    async def _event_generator():
        yield f"data: {json.dumps({'event': 'accepted', 'message': 'Streaming started'}, ensure_ascii=False)}\n\n"

        while not done.is_set() or not event_queue.empty():
            flushed = False
            while True:
                try:
                    ev = event_queue.get_nowait()
                except queue.Empty:
                    break
                flushed = True
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

            if not flushed:
                await asyncio.sleep(0.15)

        if state.get("error") is not None:
            payload = {
                "event": "result",
                "ok": False,
                "status_code": 500,
                "detail": state["error"],
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            return

        engine_output = state.get("engine_output") or {}
        path = engine_output.get("optimal_path", [])
        if not path:
            payload = {
                "event": "result",
                "ok": False,
                "status_code": 404,
                "detail": "Could not find a feasible optimal path with the given parameters.",
                "run_dir": engine_output.get("run_dir"),
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            return

        response_payload = _to_path_response_payload(engine_output)
        payload = {
            "event": "result",
            "ok": True,
            "response": response_payload,
            "run_dir": engine_output.get("run_dir"),
            "attempt": engine_output.get("attempt"),
            "feasible_count": engine_output.get("feasible_count"),
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


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
