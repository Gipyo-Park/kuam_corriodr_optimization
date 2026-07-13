# MAIN UAM Corridor Optimizer Detailed Guide

대상 파일: `MAIN_uam_corridor_optimizer.py`

이 문서는 이 파일을 처음 보는 사람이 “어디를 바꾸면 무엇이 바뀌는지” 이해할 수 있도록 정리한 설명서다. 코드 자체는 수정하지 않았다.

## 한 줄 요약

`MAIN_uam_corridor_optimizer.py`는 울산 지역 UAM 회랑을 만들기 위해 출발/도착 버티포트와 waypoint를 기준으로 후보 경로를 만들고, 거리/지상 위험/공중 위험/소음 위험을 동시에 고려해서 최적 경로를 찾는 메인 실행 파일이다.

## 전체 실행 순서

1. 라이브러리와 최적화 helper 모듈을 import한다.
2. 이착륙 전이 경로, RF turn, 지도/위험도 처리 함수들을 정의한다.
3. `attempt_run_once()` 안에서 실제 실행 파라미터를 설정한다.
4. 지상 위험도 `ground_risk_data/Modified_high_res_affected_population_GRC.npy`를 읽는다.
5. 조류 위험도 `air_risk_data/bird_riskmap_springfall_3d.npy`를 읽는다.
6. MOC 장애물 지도 `260608_MOC/UAM_MOC_XYZ_risk_fixedAGL*.npy` 중 현재 AGL 고도에 맞는 파일을 읽는다.
7. 소음 지도 `noise_data/noise_lden_grid.npy`를 읽는다.
8. 출발/도착 버티포트, 이착륙 섹터, 전이 경로, 중간 waypoint를 구성한다.
9. backbone 주변에 safe node 후보를 만든다.
10. 초기 population을 만들고 RF turn과 제약조건을 검사한다.
11. NSGA-III를 실행해 Pareto 경로 후보를 진화시킨다.
12. 대표 경로와 balanced 경로를 뽑는다.
13. 그림, 엑셀, json, pickle 결과를 `runs/<timestamp>`에 저장한다.
14. feasible 경로가 하나도 없으면 새 run folder를 만들고 다시 시도한다.

## 핵심 개념

| 용어 | 의미 |
|---|---|
| `vertiport` | 출발/도착 버티포트. `[lat, lon, alt_m]` 형식이며 고도는 MSL 기준 |
| `altitude_levels` | 순항 고도 목록. 현재는 보통 하나의 고도만 사용 |
| `AGL` | 지표/버티포트 기준 고도. 코드에서는 `MSL - start_vertiport_alt`로 계산 |
| `takeoff_complete` | 이륙 전이 구간이 끝나는 지점. 순항 경로의 시작점처럼 사용 |
| `landing_entry` | 착륙 전이 구간이 시작되는 지점. 순항 경로의 끝점처럼 사용 |
| `waypoints` | 사용자가 지정한 중간 경유점 |
| `backbone` | 최적화가 반드시 지나가야 하는 큰 기준점 배열 |
| `safe nodes` | backbone 주변에서 후보로 쓸 수 있는 안전 노드 |
| `TF` | Track-to-Fix 직선 구간 |
| `RF` | Radius-to-Fix 회전 구간 |
| `MOC` | 장애물/최소 장애물 회피 기준 기반 금지 cell. 코드에서는 `1`이면 corridor 금지로 처리 |
| `NFZ` | No-Fly Zone. 사용자가 직접 넣는 금지 사각형 영역 |

## Import되는 내부 모듈

| 모듈 | 메인에서 하는 일 |
|---|---|
| `crossover_GP.crossover_gp` | 부모 경로 두 개를 섞어 자식 경로 생성 |
| `mutation_GP.mutation_gp` | 경로 중간 노드를 변이 |
| `fast_non_dominated_sort.fast_non_dominated_sort` | Pareto front 정렬 |
| `generate_initial_population_GP.generate_initial_population_gp` | 초기 경로 후보 생성 |
| `generate_reference_points.generate_reference_points` | NSGA-III reference points 생성 |
| `normalize_objectives.normalize_objectives` | 목적함수 정규화 |
| `niching_selection.niching_selection` | NSGA-III niching selection |
| `evaluate_objectives_with_constraints_GP.evaluate_objectives_with_constraints_gp` | 목적함수 계산과 제약조건 검사 |
| `rf_turn.apply_rf_turns` | waypoint 경로에 RF turn 적용 |
| `takeoff_landing_sector.*` | 계절별 이착륙 섹터 허용 여부 검증 |

## 지도와 데이터 범위

현재 메인 기본 지도 범위:

| 파라미터 | 현재값 | 의미 |
|---|---:|---|
| `lat_lim` | `[35.535, 35.652]` | 평가/시각화에 쓰는 위도 범위 |
| `lon_lim` | `[129.020, 129.150]` | 평가/시각화에 쓰는 경도 범위 |

이 값을 바꾸면 지상 위험도, 조류 위험도, MOC, 소음 지도와 좌표가 서로 맞아야 한다. 범위만 임의로 바꾸면 위험도 map과 실제 위치가 어긋날 수 있다.

## 주요 입력 데이터

| 파라미터/경로 | 의미 | 바꾸면 생기는 변화 |
|---|---|---|
| `ground_risk_path = Path("ground_risk_data") / "Modified_high_res_affected_population_GRC.npy"` | 지상 위험도 입력 | 다른 지상 위험도 파일을 쓰게 된다. shape/방향이 맞지 않으면 에러 또는 잘못된 위험도 평가가 나온다 |
| `bird_airrisk_path = Path("air_risk_data") / "bird_riskmap_springfall_3d.npy"` | 조류 공중 위험도 입력 | 계절별 조류 위험도를 바꾸려면 summer/winter 파일로 변경 가능하다. altitude vector가 맞아야 한다 |
| `moc_airrisk_dir = Path("260608_MOC")` | fixed AGL MOC 파일들이 있는 폴더 | 이 폴더 안의 `UAM_MOC_XYZ_risk_fixedAGL*.npy` 중 현재 AGL에 맞는 파일을 고른다 |
| `noise_npy_path = Path("noise_data") / "noise_lden_grid.npy"` | 소음 위험도 입력 | 다른 소음 지도를 쓰게 된다. lat/lon extent 차이는 경고로 출력된다 |
| `noise_floor_db = 0.0` | 소음 dB에서 floor처럼 빼는 값 | 값을 올리면 낮은 소음 값이 0에 가까워져 소음 목적함수 영향이 달라진다 |

## 버티포트와 waypoint 파라미터

| 파라미터 | 현재값 | 의미 | 바꾸면 생기는 변화 |
|---|---:|---|---|
| `start_vertiport_default` | `[35.603386, 129.078025, 150.0]` | 출발 버티포트 `[lat, lon, MSL고도]` | 출발 위치와 AGL 기준이 바뀐다. MOC 선택 고도도 `altitude_levels - start_alt` 기준으로 달라진다 |
| `end_vertiport_default` | `[35.6316511, 129.0535480, 150.0]` | 도착 버티포트 `[lat, lon, MSL고도]` | 도착 위치가 바뀌고 전체 경로 방향/거리/위험도가 바뀐다 |
| `takeoff_end_lla` | `[35.59468397, 129.07515721, altitude_levels[0]]` | `use_takeoff_landing_transition=False`이고 waypoint가 있을 때 쓰는 수동 이륙 완료점 | 수동 모드에서 순항 시작점이 바뀐다 |
| `landing_end_lla` | `[35.59701567, 129.08585995, altitude_levels[0]]` | `use_takeoff_landing_transition=False`이고 waypoint가 있을 때 쓰는 수동 착륙 진입점 | 수동 모드에서 순항 끝점이 바뀐다 |
| `corridor_lat_default` / `corridor_lon_default` | 현재 빈 배열 | 중간 waypoint 위도/경도 목록 | 비어 있으면 중간 waypoint 없이 출발/도착 기준으로 최적화. 값이 있으면 그 waypoint 순서를 지나가는 backbone 생성 |
| `waypoint_alt_fixed_m` | `altitude_levels[0]` | 중간 waypoint 고도 | waypoint의 순항 고도가 바뀐다 |
| `use_clicked_waypoints` | `False` | 지도 클릭으로 waypoint를 입력할지 여부 | `True`면 실행 중 클릭 창에서 waypoint를 찍고, 결과를 `clicked_waypoints.json/csv`로 저장 |
| `min_clicked_waypoints` | `0` | 클릭 waypoint 최소 개수 | 클릭 수가 이보다 적으면 fallback waypoint를 사용 |
| `clicked_wp_map_zoom` | `13` | 클릭 지도 zoom | 클릭 지도 배율만 바뀐다. 최적화 수학에는 직접 영향 없음 |

주의할 점:

- `corridor_lat_default`와 `corridor_lon_default`는 반드시 같은 개수여야 한다.
- waypoint가 없고 `use_takeoff_landing_transition=True`이면 start vertiport에서 takeoff transition을 타고 순항 구간으로 들어간 뒤 end vertiport로 landing transition을 붙인다.
- waypoint가 없고 `use_takeoff_landing_transition=False`이면 start/end vertiport 자체를 backbone으로 사용한다.

## 고도 파라미터

| 파라미터 | 현재값 | 의미 | 바꾸면 생기는 변화 |
|---|---:|---|---|
| `altitude_levels` | `[750.0]` | 순항 고도 MSL | 전체 경로 고도, 조류 위험도 altitude slice, 소음 altitude slice, MOC fixed AGL 선택이 바뀐다 |
| `airspace_alt_min_m` | `100.0` | 허용 공역 최저 고도 MSL | 순항 고도가 이보다 낮으면 실행 전 에러 |
| `airspace_alt_max_m` | `1000.0` | 허용 공역 최고 고도 MSL | 순항 고도가 이보다 높으면 실행 전 에러 |
| `delta_z_max` | `max(100.0, abs(altitude_levels - 150.0)+5.0)` | 인접 노드 간 허용 고도 변화량 | 고도 변화가 큰 경로를 허용/제한하는 제약. 현재 단일 순항고도라 큰 영향은 제한적 |

AGL 계산 예:

- 출발 버티포트 고도: 150m MSL
- 순항 고도: 750m MSL
- AGL: 750 - 150 = 600m
- 따라서 현재 기본 설정에서는 `UAM_MOC_XYZ_risk_fixedAGL600.npy`가 선택된다.

## 이착륙 전이 파라미터

| 파라미터 | 현재값 | 의미 | 바꾸면 생기는 변화 |
|---|---:|---|---|
| `use_takeoff_landing_transition` | `True` | 자동 이륙/착륙 전이 경로를 붙일지 여부 | `True`면 버티포트에서 순항고도까지 전이 profile 생성. `False`면 수동 endpoint 또는 start/end 직접 사용 |
| `transition_mode` | `"distance"` | 전이 형상 계산 방식 | `"distance"`는 밑변 거리 고정, `"angle"`은 경사각 고정 |
| `takeoff_distance_m` | `1000.0` | 이륙 전이 수평거리 | 커질수록 이륙 완료점이 버티포트에서 멀어진다. 경사각은 완만해진다 |
| `landing_distance_m` | `1000.0` | 착륙 전이 수평거리 | 커질수록 착륙 진입점이 도착 버티포트에서 멀어진다 |
| `takeoff_angle_deg` | `8.0` | angle 모드에서 이륙 경사각 | angle 모드일 때만 직접 쓰인다. 작을수록 필요한 수평거리가 길어진다 |
| `landing_angle_deg` | `8.0` | angle 모드에서 착륙 경사각 | angle 모드일 때만 직접 쓰인다 |
| `transition_sample_spacing_m` | `100.0` | 전이 profile 점 간격 | 작을수록 전이 경로 점이 더 촘촘해지고 Excel row가 많아진다 |
| `transition_speed_takeoff_mps` | `50.0` | Excel 기록용 이륙 전이 속도 | 경로 형상 계산에는 직접 영향 없음 |
| `transition_speed_landing_mps` | `50.0` | Excel 기록용 착륙 전이 속도 | 경로 형상 계산에는 직접 영향 없음 |

초보자용 이해:

- 전이 경로는 “버티포트에서 곧바로 순항 경로를 시작하지 않고, 지정 섹터 방향으로 일정 거리 이동하면서 순항 고도까지 올라가는/내려가는 구간”이다.
- `transition_mode="distance"`에서는 사용자가 수평거리 1000m를 정하고, 코드는 고도 차이를 보고 실제 각도를 계산한다.
- `transition_mode="angle"`에서는 사용자가 각도 8도를 정하고, 코드는 그 각도를 만족하는 수평거리를 계산한다.

## 이착륙 섹터 파라미터

| 파라미터 | 현재값 | 의미 | 바꾸면 생기는 변화 |
|---|---:|---|---|
| `sector_mode_enabled` | `False` | 계절별 허용 섹터 mask를 강제할지 여부 | `True`면 `sector_season`에 따라 허용되지 않는 섹터 입력 시 에러 |
| `sector_season` | `"annual"` | 섹터 허용 mask 기준 계절 | `sector_mode_enabled=True`일 때만 검증에 사용 |
| `takeoff_sector_user` | `7` | 이륙 섹터 번호 | 이륙 전이 방향이 바뀐다 |
| `landing_sector_user` | `5` | 착륙 섹터 번호 | 착륙 전이 방향이 바뀐다 |
| `sector_half_width_deg` | `15.0` | 지도에 그리는 섹터 wedge 반폭 | 시각화 wedge 폭이 바뀐다. 경로 계산 방향은 섹터 중심선 기준 |

섹터 번호는 1~12이며, 12개 방향을 시계방향으로 나눈다. 섹터를 바꾸면 `takeoff_heading_deg`, `landing_heading_deg`가 자동으로 바뀐다.

## 공역/금지구역/비상착륙 파라미터

| 파라미터 | 현재값 | 의미 | 바꾸면 생기는 변화 |
|---|---:|---|---|
| `airspace_center_lla` | `[35.6033361, 129.0776917, 150.0]` | 원형 공역 중심 | 후보 노드와 경로가 이 중심 반경 안에 있어야 한다 |
| `airspace_radius_km` | `5.0` | 공역 반경 | 작게 하면 feasible 경로가 줄거나 없어질 수 있다. 크게 하면 탐색 범위가 넓어진다 |
| `use_forbidden_zones` | `True` | NFZ 사용 여부 | `True`면 `forbidden_zones_input` 영역을 피한다 |
| `forbidden_zones_input` | 현재 빈 배열 | 금지 사각형 `[lon_min, lon_max, lat_min, lat_max]` | 값을 넣으면 해당 영역을 통과하는 경로가 제약 위반 |
| `check_corridor_nfz` | `True` | 경로 선뿐 아니라 corridor 폭까지 고려해 NFZ 검사 | `True`면 더 엄격하다 |
| `use_emergency_points` | `True` | 비상착륙지 사용 여부 | fallback 후보나 표시용으로 사용된다 |
| `emergency_points_input` | 3개 지점 | 비상착륙 후보 지점 | 지도에 표시되고 safe node 부족 시 후보 pool fallback에 영향 가능 |
| `emergency_strip_m` | `500.0` | 비상착륙 관련 완화 strip 폭 | 관련 후보 처리의 공간 폭에 영향 |

NFZ 예시:

```python
forbidden_zones_input = np.array([
    [129.08, 129.10, 35.59, 35.61],
], dtype=float)
```

이렇게 넣으면 경도 129.08~129.10, 위도 35.59~35.61 사각형을 금지구역으로 본다.

## Corridor 폭과 safe node 파라미터

| 파라미터 | 현재값 | 의미 | 바꾸면 생기는 변화 |
|---|---:|---|---|
| `W_half` | `296.0` | corridor 반폭. 주석상 TSE=148m의 2배 | 커지면 corridor가 넓어지고 NFZ/MOC/self-overlap 검사에서 더 엄격해질 수 있다 |
| `W_buf` | `1250.0` | backbone 주변 safe node 생성 buffer 폭 | 커지면 더 넓은 영역에서 후보 노드를 만든다. 계산량 증가 가능 |
| `node_grid_resolution_m` | `100.0` | safe node 격자 간격 | 작게 하면 후보가 촘촘해지고 계산량 증가. 크게 하면 후보가 성기고 feasible 경로가 줄 수 있다 |
| `MIN_SAFE_NODES_TARGET` | `200` | segment별 최소 safe node 목표 수 | 크게 하면 더 많은 후보를 확보하려고 한다 |
| `SAFE_NODE_AIRRISK_MAX_LIST` | `[0.1,0.2,0.3,0.4,0.5]` | safe node 필터 기준 | percentile 모드에서는 10%,20%... 백분위 기준. absolute 모드에서는 절대 위험도 threshold |
| `USE_PERCENTILE_SAFE_NODE_FILTER` | `True` | safe node 필터 방식을 percentile로 쓸지 여부 | `True`면 각 segment에서 상대적으로 낮은 위험 노드를 확보하기 쉬움 |
| `min_seg_for_extra_nodes_m` | `1500.0` | 짧은 segment에서 extra node 생성을 생략하는 기준 | 크게 하면 짧은 구간에는 후보 노드를 덜 넣어 단순해진다 |

초보자용 이해:

- `W_buf`는 “어디까지 후보 점을 뿌릴 것인가”에 가깝다.
- `W_half`는 “최종 corridor 폭을 검사할 때 반폭을 얼마로 볼 것인가”에 가깝다.
- `node_grid_resolution_m`을 줄이면 경로가 더 정교해질 수 있지만 오래 걸린다.

## 초기 population 파라미터

| 파라미터 | 현재값 | 의미 | 바꾸면 생기는 변화 |
|---|---:|---|---|
| `N_init` | `1000` | 초기 후보를 만들 때 목표 후보 수 | 크게 하면 초기 feasible을 찾을 확률은 늘지만 느려진다 |
| `min_feasible_init_solutions` | `1` | 최적화 시작에 필요한 최소 feasible 초기 후보 수 | 크게 하면 시작 전 검사가 엄격해지고 retry가 늘 수 있다 |
| `max_init_retries` | `300` | 초기 feasible population 찾기 재시도 횟수 | 크게 하면 오래 기다리지만 성공 가능성 증가 |
| `wp_perturb_radius_m` | `100.0` | waypoint perturbation 반경 | 크게 하면 backbone 주변에서 더 멀리 흔들린 초기 경로 생성 |
| `wp_perturb_steps` | `10` | perturbation 반복 단계 | 커질수록 waypoint 이동이 단계적으로 적용된다 |
| `min_extra_nodes_per_seg` | `0` | segment별 추가 노드 최소 개수 | 늘리면 경로가 더 많은 중간점으로 구성된다 |
| `max_extra_nodes_per_seg` | `1` | segment별 추가 노드 최대 개수 | 늘리면 더 복잡한 경로가 가능하지만 RF/제약 실패 가능성도 늘 수 있다 |
| `use_wp_skip_generator` | `False` | 중간 waypoint 일부 skip 초기해 생성기를 섞을지 여부 | `True`면 waypoint를 일부 생략한 초기 후보도 만들 수 있다 |
| `init_pop_skip_mix_ratio` | `0.5` | skip generator 혼합 비율 | `use_wp_skip_generator=True`일 때만 의미 있음 |
| `wp_skip_prob` | `0.00` | waypoint skip 확률 | 커질수록 중간 waypoint를 덜 지나는 초기 후보가 늘어난다 |

주의:

- 현재 코드에서는 mandatory waypoint order를 유지하는 흐름이 있으므로, waypoint를 반드시 통과시키려면 `enforce_mandatory_wp_order=True` 유지가 안전하다.
- waypoint가 비어 있으면 skip generator는 자동으로 사실상 비활성화된다.

## NSGA-III 최적화 파라미터

| 파라미터 | 현재값 | 의미 | 바꾸면 생기는 변화 |
|---|---:|---|---|
| `N_pop` | `50` | 세대별 population 크기 | 크게 하면 다양성이 늘지만 계산 시간 증가 |
| `Nmax` | `10` | 세대 수 | 크게 하면 더 오래 최적화하고 개선 가능성이 있지만 시간이 늘어난다 |
| `offspring_ratio` | `0.6` | 한 세대에서 만드는 offspring 비율 | 높이면 새 후보가 많아지고 탐색성이 커진다 |
| `require_rf_for_parent_selection` | `True` | parent selection에 RF feasible 조건을 요구 | `True`면 더 현실적인 후보 위주로 진화하지만 후보 부족 가능 |
| `mutation_rate` | `0.20` | mutation 발생 확률 | 높이면 경로 변화가 많아지고 탐색성 증가, 너무 높으면 안정성 저하 |

## Mutation 관련 파라미터

| 파라미터 | 현재값 | 의미 | 바꾸면 생기는 변화 |
|---|---:|---|---|
| `use_local_safe_resample` | `True` | 변이 시 주변 safe node를 우선 사용할지 여부 | `True`면 위험도가 낮은 후보로 변이하기 쉬움 |
| `local_resample_prob` | `0.70` | local safe node 변이 사용 확률 | 높이면 safe node 기반 변이가 많아진다 |
| `local_strip_width_m` | `500.0` | 변이 후보를 찾는 segment 주변 strip 폭 | 커지면 더 넓은 후보에서 선택 |
| `local_radius_m` | `500.0` | 현재 노드 주변 후보 반경 | 커지면 한 번에 더 멀리 변이 가능 |
| `local_max_tries` | `5` | local 후보 탐색 시도 횟수 | 커지면 후보를 찾을 가능성 증가, 시간 약간 증가 |
| `risk_weight_boost` | `True` | 낮은 위험 후보를 더 자주 고르게 할지 | `True`면 risk가 낮은 safe node가 선택될 확률 증가 |
| `risk_weight_strength` | `2.0` | 위험도 가중 선택 강도 | 커질수록 낮은 위험 후보 선호가 강해진다 |

## RF turn 파라미터

| 파라미터 | 현재값 | 의미 | 바꾸면 생기는 변화 |
|---|---:|---|---|
| `speed_max_kmh` | `300.0` | RF turn radius 계산에 쓰는 기준 속도 | 빠를수록 회전반경이 커져 짧은 segment에서 RF 실패 가능성 증가 |
| `ground_speed_mps` | `speed_max_kmh / 3.6` | m/s 단위 속도 | `speed_max_kmh`에서 자동 계산 |
| `bank_angle_deg` | `25.0` | 선회 bank angle | 커질수록 같은 속도에서 회전반경이 작아짐. 너무 크면 비현실적일 수 있음 |
| `rf_base_turn_radius_m` | 속도/뱅크각에서 계산 | 기본 RF 회전반경 | 직접 수정하지 않고 속도/뱅크각으로 조절하는 것이 안전 |
| `num_arc_points` | `30` | RF arc를 몇 점으로 그릴지 | 크면 arc가 부드럽고 Excel row 증가 |
| `look_ahead` | `True` | 짧은 segment 주변에서 회전반경 scaling 적용 | `True`면 짧은 구간에서 RF가 들어맞기 쉬움 |
| `look_ahead_threshold_m` | `2000.0` | 이 길이보다 짧으면 radius scaling 시작 | 크게 하면 더 많은 segment에서 radius 축소 |
| `look_ahead_min_scale` | `0.11` | RF radius 최소 축소 비율 | 작게 하면 더 작은 회전반경 허용. 너무 작으면 비현실적일 수 있음 |
| `look_ahead_window` | `3` | 주변 몇 segment까지 보고 scale할지 | 크게 하면 주변 구간까지 고려 |
| `rf_use_boundary_heading` | `False` | 이착륙 전이 heading을 RF 첫/마지막 corner에 강제할지 | `True`면 전이 방향과 순항 RF 연결이 더 엄격해짐 |
| `rf_debug_level` | `"off"` | RF 디버그 출력 | `"summary"`나 `"detail"`로 바꾸면 RF 실패 원인 출력 증가 |
| `rf_allow_tangent_clamp` | `True` | segment가 짧을 때 tangent 길이를 줄여 맞출지 | `True`면 feasible이 늘 수 있지만 clamp가 생긴다 |
| `rf_corner_fit_margin` | `0.95` | 인접 segment 중 tangent가 쓸 수 있는 최대 비율 | 낮추면 더 보수적, 높이면 segment 끝까지 더 많이 사용 |
| `rf_corner_min_tangent_m` | `1.0` | 최소 tangent 길이 | 너무 크면 짧은 segment에서 실패 가능 |
| `rf_min_turn_angle_deg` | `0.5` | 이보다 작은 각도는 직선으로 처리 | 크게 하면 작은 회전은 RF를 생략 |

RF turn radius 기본식:

```text
R = v^2 / (g * tan(bank_angle))
```

따라서 속도가 올라가면 반경은 제곱으로 커진다. 경로 segment가 짧은데 속도를 크게 하면 `min_dist_ok` 또는 RF feasibility가 나빠질 수 있다.

## 목적함수와 위험도 가중치

코드에서 `objective_names`는 다음 네 가지다.

```python
objective_names = ["Distance", "Ground Risk", "Air Risk", "Noise Risk"]
```

가중치 변수는 `params_dict`에 `w_dist`, `w_ground`, `w_air`, `w_noise`로 저장된다. 현재 코드값은 `w_dist=0.1`, `w_ground=1.0`, `w_air=2.0`, `w_noise=0.1`이다. 이 값들은 목적함수 계산에 들어간다.

| 파라미터 | 현재값 | 의미 | 크게 하면 |
|---|---:|---|---|
| `w_dist` | `0.1` | 거리 목적함수 가중치 | 짧은 경로를 더 선호 |
| `w_ground` | `1.0` | 지상 위험도 가중치 | 인구/지상 위험이 낮은 곳을 더 선호 |
| `w_air` | `2.0` | 공중 조류 위험도 가중치 | 조류 위험이 낮은 곳을 더 선호 |
| `w_noise` | `0.1` | 소음 위험도 가중치 | 소음 위험이 낮은 곳을 더 선호 |

주의: NSGA-III는 단일 가중합만 최적화하는 방식이 아니라 여러 목적함수를 동시에 다룬다. 그래도 각 목적함수 내부 scaling/가중치가 결과 대표점 선택과 값에 영향을 준다.

## 제약조건 파라미터

| 파라미터 | 현재값 | 의미 | 바꾸면 생기는 변화 |
|---|---:|---|---|
| `flight_dist_limit` | `100000.0` | segment 비행거리 상한 | 낮추면 긴 segment가 제약 위반 |
| `check_corridor_moc` | `True` | MOC 금지 영역 검사 | `True`면 MOC=1 cell을 corridor가 지나가면 제약 위반 |
| `check_corridor_self_overlap` | `False` | corridor 자기교차/겹침 검사 | `True`면 더 엄격해질 수 있음 |
| `check_corridor_nfz` | `True` | NFZ corridor 폭 검사 | `True`면 금지구역 주변까지 더 보수적으로 검사 |
| `min_corridor_distance_km` | `0.0` | 전체 회랑 최소 거리 | 0이면 비활성. 값을 올리면 너무 짧은 경로 제외 |
| `airspace_radius_km` | `5.0` | 원형 공역 반경 | 작게 하면 공역 밖 경로/노드가 제거되어 실패 가능성 증가 |

## Waypoint가 없을 때 동작

현재 의도된 동작:

| 상황 | backbone 구성 |
|---|---|
| waypoint 있음 + transition ON | `[takeoff_complete, waypoint들, landing_entry]` |
| waypoint 없음 + transition ON | `[takeoff_complete, landing_entry]`, 결과에는 start/end vertiport와 transition profile이 붙음 |
| waypoint 있음 + transition OFF | `[takeoff_end_lla, waypoint들, landing_end_lla]` |
| waypoint 없음 + transition OFF | `[start_vertiport, end_vertiport]` |

즉 “버티포트 두 개만 찍는 경우”는 중간 waypoint 배열이 비어 있는 상황으로 처리된다.

## 결과 저장 구조

매 실행마다 다음 폴더가 만들어진다.

```text
runs/YYYYMMDD_HHMMSS/
```

주요 파일:

| 파일 | 내용 |
|---|---|
| `params.json` | 실행 설정값과 데이터 metadata |
| `results.pkl` | Python 객체로 저장한 전체 결과 |
| `route_data.xlsx` | 최종 balanced 경로 표와 요약 |
| `fig_route_from_excel_map.png` | Excel 경로 기반 지도 |
| `fig1_safe_nodes.png` | safe node 지도 |
| `fig1p_safe_nodes_compare_diag.png` | safe node 필터 비교 |
| `fig1b_moc_binary.png` | MOC binary map |
| `fig1c_waypoint_moc_safety.png` | waypoint와 MOC 비교 |
| `fig2_sample_init.png` | 초기 후보 샘플 |
| `fig2b_init_before_after_rf.png` | RF 적용 전/후 비교 |
| `fig3_pareto.png` | Pareto front |
| `fig4_optimal_corridor.png` | 대표 최적 경로 |
| `fig5_balanced_only.png` | balanced 경로 단독 |
| `fig5b_balanced_fig4_style.png` | Fig4 스타일 balanced 경로 |
| `fig5c_balanced_cr_rfc_labels.png` | CR/RFC label 포함 |
| `gen_snapshots/` | 세대별 진화 과정 그림 |

## 자주 바꾸는 설정 추천

처음 실험할 때는 아래만 주로 바꾸는 것이 안전하다.

| 하고 싶은 일 | 바꿀 파라미터 |
|---|---|
| 출발/도착 위치 변경 | `start_vertiport_default`, `end_vertiport_default` |
| 순항 고도 변경 | `altitude_levels` |
| 중간 waypoint 추가 | `corridor_lat_default`, `corridor_lon_default` |
| 지도 클릭으로 waypoint 지정 | `use_clicked_waypoints=True` |
| 이륙/착륙 방향 변경 | `takeoff_sector_user`, `landing_sector_user` |
| 이착륙 전이 거리 변경 | `takeoff_distance_m`, `landing_distance_m` |
| 계산을 빠르게 테스트 | `N_init`, `N_pop`, `Nmax`를 줄임 |
| 더 정교하게 탐색 | `N_init`, `N_pop`, `Nmax`를 늘림 |
| 후보 노드를 촘촘하게 | `node_grid_resolution_m`을 줄임 |
| RF 실패가 많을 때 | `speed_max_kmh` 낮추기, `look_ahead_min_scale` 낮추기, `rf_debug_level="summary"` |
| 공역 때문에 실패할 때 | `airspace_center_lla`, `airspace_radius_km`, waypoint 위치 확인 |

## 실패 메시지 해석

| 메시지/상황 | 가능한 원인 | 확인할 것 |
|---|---|---|
| `Backbone waypoints are outside airspace constraints` | 버티포트/waypoint/전이 endpoint가 공역 반경 또는 고도 범위를 벗어남 | `airspace_center_lla`, `airspace_radius_km`, `airspace_alt_min_m/max_m`, waypoint 좌표 |
| `Cruise altitude is outside configured airspace altitude range` | `altitude_levels`가 공역 고도 제한 밖 | `altitude_levels`, `airspace_alt_min_m`, `airspace_alt_max_m` |
| 초기 retry에서 `both_feasible: 0` 반복 | RF turn, 최소거리, MOC, NFZ, 공역 중 하나가 너무 빡빡함 | 로그의 `constraint_fail_breakdown`, `min_dist_ok`, `rf_feasible`, `airspace_ok` |
| `MOC` 관련 실패 | 경로 또는 corridor 폭이 MOC=1 금지 cell을 통과 | `fig1b_moc_binary.png`, `fig1c_waypoint_moc_safety.png` 확인 |
| 실행은 됐지만 그림이 너무 복잡함 | waypoint/extra node/RF arc point가 많음 | `max_extra_nodes_per_seg`, `num_arc_points`, `N_pop`, `Nmax` |

## 초보자용 수정 순서

새 실험을 할 때는 이 순서로 바꾸는 것이 덜 헷갈린다.

1. `start_vertiport_default`, `end_vertiport_default`를 원하는 위치로 바꾼다.
2. `altitude_levels`를 원하는 순항 고도로 바꾼다.
3. `use_takeoff_landing_transition=True`를 유지하고 `takeoff_sector_user`, `landing_sector_user`를 고른다.
4. 중간 경유점이 필요하면 `corridor_lat_default`, `corridor_lon_default`를 채운다.
5. 빠른 테스트는 `N_init=200`, `N_pop=20`, `Nmax=3`처럼 줄여서 먼저 실행한다.
6. feasible이 확인되면 `N_init`, `N_pop`, `Nmax`를 다시 늘려 정식 실행한다.
7. 결과는 새로 생긴 `runs/<timestamp>`에서 `params.json`, `route_data.xlsx`, `fig5c_balanced_cr_rfc_labels.png`를 먼저 확인한다.

## 건드릴 때 조심할 부분

| 부분 | 이유 |
|---|---|
| `lat_lim`, `lon_lim` | 모든 risk map 정렬과 직접 관련된다 |
| `load_fixed_agl_moc_maps()` | EPSG:5179 좌표의 MOC 데이터를 v20 격자에 맞추는 핵심 로직 |
| `_aggregate_path_risks()`, `_aggregate_path_noise()` | 목적함수 값이 달라지는 부분 |
| `_apply_rf_corridor_path()` | 실제 최종 경로 형상과 RF segment가 만들어지는 부분 |
| `_export_route_outputs()` | Excel/그림/summary 산출물 형식이 결정되는 부분 |
| `attempt_run_once()` 아래쪽 retry loop | feasible 실패 시 재시도하는 구조 |

## 최소 확인 명령

문법만 확인:

```powershell
python -m py_compile MAIN_uam_corridor_optimizer.py
```

실행:

```powershell
python MAIN_uam_corridor_optimizer.py
```

API 엔진까지 같이 확인:

```powershell
python -m py_compile api_service/path_engine.py api_service/api_server.py
```
