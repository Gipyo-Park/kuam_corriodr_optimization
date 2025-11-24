import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple

# path_engine.py에서 핵심 함수를 가져옵니다.
from path_engine import find_optimal_path, _load_risk_maps

# --- FastAPI 앱 초기화 ---
app = FastAPI(
    title="K-UAM Optimal Pathfinding API",
    description="API to find the optimal UAM path between two points.",
    version="1.0.0",
)

# --- 데이터 모델 정의 (Pydantic) ---
class PathRequest(BaseModel):
    """API 요청 본문 모델"""
    start_point: Tuple[float, float, float]
    end_point: Tuple[float, float, float]
    corridor_points: List[Tuple[float, float, float]] = None

    class Config:
        schema_extra = {
            "example": {
                "start_point": [35.584536, 129.107647, 500],
                "end_point": [35.624958, 129.133552, 500],
                "corridor_points": []
            }
        }

class PathResponse(BaseModel):
    """API 응답 본문 모델"""
    message: str
    waypoint_count: int
    optimal_path: List[Tuple[float, float, float]]


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


@app.post("/find-path", response_model=PathResponse, summary="Find Optimal Path")
async def get_optimal_path(request: PathRequest):
    """
    시작점과 끝점을 받아 최적의 비행 경로(Waypoint 리스트)를 계산하여 반환합니다.
    """
    print(f"Received pathfinding request from {request.start_point} to {request.end_point}")
    
    try:
        # path_engine의 메인 함수 호출
        path = find_optimal_path(
            start_point=list(request.start_point),
            end_point=list(request.end_point),
            corridor_points=[list(p) for p in request.corridor_points] if request.corridor_points else None
        )

        if not path:
            # 경로를 찾지 못한 경우
            raise HTTPException(
                status_code=404, 
                detail="Could not find a feasible optimal path with the given parameters."
            )
        
        # 성공적으로 경로를 찾은 경우
        response = {
            "message": "Optimal path found successfully.",
            "waypoint_count": len(path),
            "optimal_path": [tuple(waypoint) for waypoint in path]
        }
        return response

    except Exception as e:
        # 경로 탐색 중 예상치 못한 에러 발생 시
        print(f"An error occurred during pathfinding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- 서버 실행 ---
if __name__ == "__main__":
    # 터미널에서 `python api_server.py` 명령으로 서버를 실행할 수 있습니다.
    uvicorn.run(app, host="0.0.0.0", port=8000)