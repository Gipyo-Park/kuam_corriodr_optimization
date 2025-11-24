import requests
import json

# --- 서버 정보 ---
# api_server.py가 실행 중인 주소
SERVER_URL = "http://127.0.0.1:8000" # 로컬에서 테스트 시
# SERVER_URL = "https://bff54cdf0390.ngrok-free.app" # 손교수님 서버 주소

def test_find_path():
    """
    /find-path 엔드포인트에 테스트 요청을 보내고 결과를 확인합니다.
    """
    endpoint = "/find-path"
    url = f"{SERVER_URL}{endpoint}"

    # --- 테스트할 요청 데이터 ---
    # path_engine.py의 테스트 케이스와 동일한 데이터 사용
    payload = {
        "start_point": [35.6033361, 129.0776917, 500], # vertiport
        "end_point": [35.6249583, 129.1335528, 500],
        # "corridor_points": []  # 경유지가 없는 경우 빈 리스트 또는 null

        "corridor_points": [
            [35.5845917, 129.0936472, 500.0],
            [35.6026528, 129.1130667, 500.0],
            [35.6326806, 129.1238583, 500.0],
            [35.6249583, 129.1335528, 500.0],
            [35.6034750, 129.1268194, 500.0],
            [35.5845361, 129.1076472, 500.0],
            [35.5692361, 129.1085306, 500.0],
            [35.5546444, 129.0936972, 500.0],
            [35.5586722, 129.0816611, 500.0],
            [35.5784750, 129.0916889, 500.0],
            [35.5843722, 129.0770000, 500.0],
            [35.6163861, 129.0613944, 500.0],
            [35.6212528, 129.0725444, 500.0],
            [35.6109972, 129.0711889, 500.0]
        ]
    }

    print(f"--- Sending request to {url} ---")
    print("Request Payload:")
    print(json.dumps(payload, indent=2))

    try:
        # POST 요청 보내기 (타임아웃을 길게 설정, 경로 탐색은 오래 걸릴 수 있음)
        response = requests.post(url, json=payload, timeout=300)

        # HTTP 상태 코드가 200 (OK)이 아닌 경우 에러 발생
        response.raise_for_status()

        # 성공적인 응답 처리
        response_data = response.json()
        print("\n--- Received successful response ---")
        print(f"Message: {response_data.get('message')}")
        print(f"Waypoint Count: {response_data.get('waypoint_count')}")
        
        print("\nOptimal Path Waypoints:")
        # 보기 좋게 일부만 출력
        path = response_data.get('optimal_path', [])
        for i, waypoint in enumerate(path):
            if i < 5 or i > len(path) - 6: # 처음 5개와 마지막 5개만 출력
                print(f"  {i+1}: {waypoint}")
            elif i == 5:
                print("  ...")

    except requests.exceptions.HTTPError as e:
        # 서버가 4xx 또는 5xx 에러를 반환한 경우
        print(f"\n--- Error: Received status code {e.response.status_code} ---")
        try:
            # 에러 응답에 포함된 상세 메시지 출력
            error_details = e.response.json()
            print(f"Detail: {error_details.get('detail')}")
        except json.JSONDecodeError:
            print(f"Response Body: {e.response.text}")

    except requests.exceptions.RequestException as e:
        # 타임아웃, 연결 실패 등 네트워크 관련 에러
        print(f"\n--- Request Failed: {e} ---")
        print("Please ensure the API server is running and accessible.")


if __name__ == "__main__":
    test_find_path()