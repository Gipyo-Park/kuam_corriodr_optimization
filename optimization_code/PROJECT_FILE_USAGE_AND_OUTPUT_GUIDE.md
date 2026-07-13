# Project File Usage And Output Guide

작성 기준: `MAIN_uam_corridor_optimizer.py`를 현재 메인 실행 파일로 보고, 루트 폴더와 주요 하위 폴더를 확인해서 정리했다. 이 문서는 파일 삭제용 명령서가 아니라, 어떤 파일이 어떤 역할을 하는지 이해하기 위한 설명서다.

## 핵심 실행 흐름

가장 중요한 실행 파일은 `MAIN_uam_corridor_optimizer.py`다.

이 파일을 실행하면 다음 흐름으로 진행된다.

1. 지상 위험도, 조류 공중 위험도, MOC 장애물 위험도, 소음 위험도 데이터를 읽는다.
2. 출발/도착 버티포트, 이착륙 전이 경로, 중간 waypoint, 비상착륙지, 금지구역, 공역 조건을 구성한다.
3. backbone 주변에 후보 안전 노드를 만든다.
4. 초기 경로 후보를 만들고 RF turn 적용 가능성과 제약조건을 검사한다.
5. NSGA-III 방식으로 거리, 지상 위험, 공중 위험, 소음 위험을 동시에 줄이는 경로를 탐색한다.
6. 대표 경로와 균형 경로를 뽑고, 그림/엑셀/pickle/json 결과를 `runs/<실행시각>` 폴더에 저장한다.

실행 명령 예:

```powershell
python MAIN_uam_corridor_optimizer.py
```

## 루트 파일

| 파일 | 역할 | 실행/사용 시 생기는 것 | 정리 판단 |
|---|---|---|---|
| `MAIN_uam_corridor_optimizer.py` | 현재 메인 UAM corridor optimizer. 데이터 로딩, waypoint 구성, 안전 노드 생성, RF turn, NSGA-III 최적화, 결과 저장까지 담당 | `runs/<timestamp>/params.json`, `results.pkl`, `route_data.xlsx`, 여러 `fig*.png`, `gen_snapshots/` 생성 | 최우선 보존 |
| `crossover_GP.py` | 유전 알고리즘 crossover 연산. 부모 경로 두 개의 중간 노드를 조합해 자식 경로 생성 | 단독 실행 산출물 없음. 메인/엔진에서 import | 보존 |
| `mutation_GP.py` | 경로의 중간 노드를 변이시키는 연산. 주변 safe node를 우선 사용하고, 없으면 작은 위치 perturbation 적용 | 단독 실행 산출물 없음. 메인/엔진에서 import | 보존 |
| `fast_non_dominated_sort.py` | NSGA 계열 Pareto front 정렬 함수 | 단독 실행 산출물 없음. 메인/엔진에서 import | 보존 |
| `generate_initial_population_GP.py` | 초기 경로 후보 population 생성 보조 함수 | 단독 실행 산출물 없음. 메인/엔진에서 import | 보존 |
| `generate_reference_points.py` | NSGA-III reference point 생성 | 단독 실행 산출물 없음. 메인/엔진에서 import | 보존 |
| `normalize_objectives.py` | 목적함수 값 정규화. balanced solution 선택 등에 사용 | 단독 실행 산출물 없음. 메인/엔진에서 import | 보존 |
| `niching_selection.py` | NSGA-III niching selection. front에서 다음 세대 개체를 고르는 데 사용 | 단독 실행 산출물 없음. 메인/엔진에서 import | 보존 |
| `evaluate_objectives_with_constraints_GP.py` | 거리, 지상 위험, 공중 위험, 소음 위험 목적함수와 NFZ/MOC/거리/고도/자기교차 등 제약 검사 | 단독 실행 산출물 없음. 메인/엔진에서 import | 보존 |
| `evaluate_objectives_GP.py` | 예전/기본 목적함수 평가 코드로 보임. 현재 메인은 `evaluate_objectives_with_constraints_GP.py`를 직접 사용 | 단독 실행 산출물 없음 | 보류 또는 archive 후보 |
| `evaluate_objectives_uniform.py` | uniform 평가 방식 실험 코드로 보임. 현재 메인 직접 import 없음 | 단독 실행 산출물 없음 | archive 후보 |
| `rf_turn.py` | waypoint 경로에 RF turn을 붙여 TF/RF segment로 변환. 회전반경, tangent clamp, look-ahead scaling 계산 | 단독 실행 산출물 없음. 메인/엔진에서 import | 보존 |
| `takeoff_landing_sector.py` | 계절별 이륙/착륙 허용 섹터 mask와 섹터 검증 함수 | 단독 실행 산출물 없음. 메인/엔진에서 import | 보존 |
| `debug_rf.py` | RF turn 디버깅용 스크립트로 보임 | 실행 시 디버그 출력/그림이 생길 수 있음. 메인 직접 의존 없음 | 보류 또는 archive 후보 |

## 데이터 폴더

| 폴더/파일 | 역할 | 누가 쓰는가 | 실행/사용 시 생기는 것 | 정리 판단 |
|---|---|---|---|---|
| `ground_risk_data/Modified_high_res_affected_population_GRC.npy` | 지상 위험도 원본. 메인에서는 `[:, :, 0, 3:]` 구간을 꺼내 정규화해 사용 | `MAIN_uam_corridor_optimizer.py`, `api_service/path_engine.py`, visualization 도구 | 읽기 전용 입력 데이터 | 보존 |
| `air_risk_data/bird_riskmap_springfall_3d.npy` | v20 기준 조류 공중 위험도 3D 지도. 현재 메인 기본값 | 메인, API 엔진, 조류 시각화 | 읽기 전용 입력 데이터 | 보존 |
| `air_risk_data/bird_riskmap_summer_3d.npy` | 여름 조류 위험도 지도. 현재 메인 기본값은 아님 | 계절 분석/향후 변경 시 사용 가능 | 읽기 전용 입력 데이터 | 보류 |
| `air_risk_data/bird_riskmap_winter_3d.npy` | 겨울 조류 위험도 지도. 현재 메인 기본값은 아님 | 계절 분석/향후 변경 시 사용 가능 | 읽기 전용 입력 데이터 | 보류 |
| `air_risk_data/UAM_MOC_3D_Risk_Map.npy` | 예전 MOC 3D 위험도 데이터로 보임. 현재 메인은 `260608_MOC` 폴더를 사용 | 현재 메인 직접 사용 없음 | 읽기 전용 입력 데이터 | archive 후보 |
| `air_risk_data/참고.txt` | air risk 데이터 관련 메모 | 사람이 참고 | 없음 | 보류 |
| `260608_MOC/UAM_MOC_XYZ_risk_fixedAGL100.npy` | AGL 100m MOC binary obstacle map | 메인/엔진이 고도에 맞춰 선택 가능 | 읽기 전용 입력 데이터 | 보존 |
| `260608_MOC/UAM_MOC_XYZ_risk_fixedAGL200.npy` | AGL 200m MOC binary obstacle map | 동일 | 읽기 전용 입력 데이터 | 보존 |
| `260608_MOC/UAM_MOC_XYZ_risk_fixedAGL300.npy` | AGL 300m MOC binary obstacle map | 동일 | 읽기 전용 입력 데이터 | 보존 |
| `260608_MOC/UAM_MOC_XYZ_risk_fixedAGL400.npy` | AGL 400m MOC binary obstacle map | 동일 | 읽기 전용 입력 데이터 | 보존 |
| `260608_MOC/UAM_MOC_XYZ_risk_fixedAGL500.npy` | AGL 500m MOC binary obstacle map | 동일 | 읽기 전용 입력 데이터 | 보존 |
| `260608_MOC/UAM_MOC_XYZ_risk_fixedAGL600.npy` | AGL 600m MOC binary obstacle map. 현재 cruise 750 MSL, vertiport 150 MSL이면 이 파일이 선택됨 | 메인/엔진/시각화 | 읽기 전용 입력 데이터 | 보존 |
| `260608_MOC/UAM_MOC_XYZ_risk_fixedAGL700.npy` | AGL 700m MOC binary obstacle map | 고도 변경 시 사용 가능 | 읽기 전용 입력 데이터 | 보존 |
| `260608_MOC/UAM_MOC_XYZ_risk_fixedAGL800.npy` | AGL 800m MOC binary obstacle map | 고도 변경 시 사용 가능 | 읽기 전용 입력 데이터 | 보존 |
| `260608_MOC/UAM_MOC_XYZ_risk_fixedAGL900.npy` | AGL 900m MOC binary obstacle map | 고도 변경 시 사용 가능 | 읽기 전용 입력 데이터 | 보존 |
| `noise_data/noise_lden_grid.npy` | 메인에서 직접 사용하는 소음 위험도 격자. dB 값을 정규화해서 목적함수에 반영 | 메인, API 엔진, noise 시각화 | 읽기 전용 입력 데이터 | 보존 |
| `noise_data/noise_output_lden.csv` | 소음 원본/중간 CSV로 보임 | `visualization_tools/visualize_noise_map.py`의 변환 입력 가능 | 변환 시 `noise_lden_grid.npy` 생성 가능 | 보존 권장 |
| `noise_data/noise_output_lden_with_coords.csv` | 좌표 포함 소음 CSV | noise 시각화/검증 | 그림/NPY 재생성 가능 | 보존 권장 |
| `noise_data/noise_output_lden_with_coords.xlsx` | 좌표 포함 소음 엑셀 | 사람이 확인하기 좋은 원본/중간 자료 | 없음 | 보류 |
| `noise_data/grid_waypoints_snake.csv` | 소음 격자/waypoint 관련 CSV로 보임 | 현재 메인 직접 사용 없음 | 없음 | 보류 |

## API 폴더

| 파일 | 역할 | 실행/사용 시 생기는 것 | 정리 판단 |
|---|---|---|---|
| `api_service/__init__.py` | `api_service`를 Python package로 인식시키는 파일. 상대 import에 필요 | 직접 산출물 없음 | 보존 |
| `api_service/path_engine.py` | `MAIN_uam_corridor_optimizer.py`와 같은 최적화 흐름을 API에서 호출 가능하게 만든 엔진 | API 요청 시 `runs/<timestamp>` 결과 생성 | 보존 |
| `api_service/api_server.py` | FastAPI 서버. `/optimized-path`, `/optimized-path/stream`, Excel 다운로드 endpoint 제공 | 서버 실행. 요청이 들어오면 `runs/<timestamp>` 생성 | 보존 |
| `api_service/api_client.py` | API 호출 예제/클라이언트. 서버에 요청 보내고 응답/Excel 저장 | `logs/` 등에 request/response 로그 또는 다운로드 파일 생성 가능 | 보류 |
| `api_service/api_request.py` | API 요청 및 브라우저 테스트용 스크립트 | `logs/`에 request/response JSON 저장 가능, 브라우저 열 수 있음 | 보류 |

API 서버 실행 예:

```powershell
python -m api_service.api_server
```

또는 파일 직접 실행 방식이 코드에서 지원될 수 있다.

## 시각화 도구 폴더

| 파일 | 역할 | 실행/사용 시 생기는 것 | 정리 판단 |
|---|---|---|---|
| `visualization_tools/visualize_ground_risk_map.py` | `ground_risk_data/Modified_high_res_affected_population_GRC.npy`를 v20 지도 범위 기준으로 시각화 | `figure/Modified_ground_risk_heatmaps.png`, `figure/Modified_ground_risk_alignment_compare.png` 생성 | 보존 또는 도구 폴더 유지 |
| `visualization_tools/visualize_bird_riskmap_springfall_3d.py` | `bird_riskmap_springfall_3d.npy`를 v20 격자/범위에 맞춰 시각화 | `figure/bird_riskmap_springfall_3d.png`, alignment 비교 그림 생성 | 보존 또는 도구 폴더 유지 |
| `visualization_tools/visualize_UAM_MOC_3D_Risk_Map.py` | `260608_MOC` fixed AGL MOC 파일을 v20 범위에 맞춰 시각화 | `figure/UAM_MOC_fixedAGL*_map.png` 류 그림 생성 | 보존 또는 도구 폴더 유지 |
| `visualization_tools/visualize_noise_map.py` | 소음 CSV를 지도/격자로 시각화하고 `noise_lden_grid.npy`를 만들 수 있는 도구 | `figure/noise_analysis/*.png`, `noise_data/noise_lden_grid.npy` 생성/갱신 가능 | 보존 |

주의: `visualize_noise_map.py`는 단순 그림 생성뿐 아니라 `noise_lden_grid.npy`를 다시 저장할 수 있다. 소음 입력을 바꾼 뒤 재생성할 때만 실행하는 편이 좋다.

## Wind Data 폴더

| 파일/폴더 | 역할 | 실행/사용 시 생기는 것 | 정리 판단 |
|---|---|---|---|
| `wind_data/AirRisk_Data_1.mat` ... `AirRisk_Data_12.mat` | 월별 바람/공중 자료. wind 분석 스크립트가 직접 읽는다 | 읽기 전용 입력 데이터 | wind 분석을 유지하면 보존 |
| `wind_data/vertiport_wind_plot2.py` | 월별 `.mat` 바람 자료를 읽어 고도/월/계절별 wind plot과 sector recommendation 생성 | `wind_data/python_outputs/*.png`, `wind_sector_recommendations.json` 생성 | 보존 |
| `wind_data/vertiport_wind_plot2.m` | MATLAB 원본/참고 코드 | MATLAB에서 실행 가능 | archive 후보 또는 참고용 보존 |
| `wind_data/plot_new_moc_top6.py` | `260608_MOC`, wind `.mat`, 조류/지상 위험도 자료를 함께 보고 섹터 조합을 평가/시각화 | `wind_data/python_outputs/*.png`, `*.csv`, `*.txt`, `*.json` 생성 | 보존 |
| `wind_data/python_outputs/` | wind 분석 결과 폴더 | 위 두 스크립트 실행 결과 누적 | 출력물. 필요한 결과만 보존 |
| `wind_data/__pycache__/` | Python bytecode cache | Python 실행 시 자동 생성 | 삭제해도 자동 재생성 |

## 결과/작업 폴더

| 폴더 | 역할 | 실행/사용 시 생기는 것 | 정리 판단 |
|---|---|---|---|
| `runs/` | 메인 최적화/API 엔진 실행 결과가 시간별 폴더로 저장되는 곳 | `params.json`, `results.pkl`, `route_data.xlsx`, `fig*.png`, `gen_snapshots/` | 출력물. 성공 케이스만 남기고 정리 가능 |
| `figure/` | 시각화 도구가 저장하는 공용 그림 폴더 | risk map, noise map 등 PNG | 출력물. 보고서용만 보존 가능 |
| `logs/` | API client/request 실행 로그 폴더 | request/response JSON, 다운로드 파일 가능 | 출력물. 필요 시 정리 가능 |
| `archive/` | 예전 `main_JS_1218_v*.py`와 emergency simulation 백업 | 직접 실행하지 않는 과거 버전 보관 | 보존 또는 별도 백업 후 정리 |
| `.venv/` | Python 가상환경 | package 설치 파일 | 프로젝트 실행 환경. 다른 환경이 확실하면 제외/재생성 가능 |
| `.vscode/` | VS Code 설정 | IDE 설정 | 보류 |
| `__pycache__/` | 루트 Python bytecode cache | Python 실행 시 자동 생성 | 삭제해도 자동 재생성 |

## Archive 폴더

| 파일 패턴 | 역할 | 정리 판단 |
|---|---|---|
| `archive/main_JS_1218_v2.py` ... `archive/main_JS_1218_v19.py` | `MAIN_uam_corridor_optimizer.py` 이전 버전 기록 | 현재 실행에는 불필요. 히스토리 보관용 |
| `archive/main_JS_1218_v15_1.py`, `archive/main_JS_1218_v15_2.py`, `archive/main_JS_1218_v15_JS.py` | v15 파생 버전 | 현재 실행에는 불필요. 히스토리 보관용 |
| `archive/MAIN_for_EMERGENCY_SIM.py` | emergency simulation용 과거/별도 스크립트 | 현재 메인 최적화와 별도 목적. 필요성 확인 후 보관/정리 |

## MAIN 실행 결과물 상세

`MAIN_uam_corridor_optimizer.py`를 실행하면 `runs/<YYYYMMDD_HHMMSS>/`가 만들어진다.

주요 결과물:

| 결과물 | 의미 |
|---|---|
| `params.json` | 실행에 사용된 파라미터, 데이터 경로, waypoint, transition, risk map metadata, airspace/NFZ 정보 |
| `results.pkl` | Python에서 다시 불러올 수 있는 전체 결과 객체. population, objective values, representative paths, 위험도 지도 일부 포함 |
| `route_data.xlsx` | 최종 balanced optimal corridor 경로 표. `Route_Data`, `CR_Points`, `RF_Centers`, `Input_Points`, `Summary`, `Airspace_Info`, `NFZ_Info` sheet 포함 |
| `fig_route_from_excel_map.png` | `route_data.xlsx` 기반 최종 경로 지도 |
| `fig1_safe_nodes.png` | backbone 주변 후보 safe node |
| `fig1p_safe_nodes_compare_diag.png` | safe node 필터 방식 비교 진단 |
| `fig1b_moc_binary.png` | MOC 금지 영역 지도 |
| `fig1c_waypoint_moc_safety.png` | waypoint와 MOC 위치 비교 |
| `fig2_sample_init.png` | 초기 후보 경로 샘플 |
| `fig2b_init_before_after_rf.png` | RF 적용 전/후 초기 경로 비교 |
| `fig3_pareto.png` | Pareto objective scatter |
| `fig4_optimal_corridor.png` | 대표 최적 corridor 지도 |
| `fig4_*_corridor.png` | 거리/지상위험/공중위험/소음위험 기준 대표 경로 그림 |
| `fig5_balanced_only.png` | balanced corridor 단독 그림 |
| `fig5b_balanced_fig4_style.png` | Fig4 스타일 balanced corridor |
| `fig5c_balanced_cr_rfc_labels.png` | CR/RFC label 포함 balanced corridor |
| `gen_snapshots/` | 세대별 진화 경로 snapshot |

## 정리 우선순위

삭제를 바로 권하는 것이 아니라, 정리 판단 기준만 적는다.

| 우선순위 | 대상 | 이유 |
|---|---|---|
| 먼저 정리 가능 | `__pycache__/`, `visualization_tools/__pycache__/`, `wind_data/__pycache__/` | 자동 생성 cache |
| 출력물 정리 가능 | 오래된 `runs/`, 오래된 `figure/`, 오래된 `logs/`, `wind_data/python_outputs/` | 재실행하면 다시 생기는 결과물 |
| archive 유지 권장 | `archive/main_JS_1218_v*.py` | 현재 실행에는 필요 없지만 과거 비교/복구용 |
| 신중히 보존 | `260608_MOC`, `ground_risk_data`, `noise_data/noise_lden_grid.npy`, `air_risk_data/bird_riskmap_springfall_3d.npy` | 현재 메인/API 핵심 입력 데이터 |
| 코드 핵심 보존 | `MAIN_uam_corridor_optimizer.py`, NSGA/RF/helper 모듈, `api_service/path_engine.py` | 현재 실행 경로에 직접 필요 |

