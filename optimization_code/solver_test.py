import numpy as np
import heapq
import matplotlib.pyplot as plt

# =========================
# 1) 입력
# =========================
vertiport = np.array([35.6033361, 129.0776917, 150.0])

corridor_lat = np.array([
    35.5845917, 35.6026528, 35.6326806, 35.6249583, 35.6034750,
    35.5845361, 35.5692361, 35.5546444, 35.5586722, 35.5784750,
    35.5843722, 35.6163861, 35.6212528, 35.6109972
])

corridor_lon = np.array([
    129.0936472, 129.1130667, 129.1238583, 129.1335528, 129.1268194,
    129.1076472, 129.1085306, 129.0936972, 129.0816611, 129.0916889,
    129.0770000, 129.0613944, 129.0725444, 129.0711889
])

# 경유지 순서대로 강제 통과: vertiport -> corridor -> vertiport
waypoints_ll = [(vertiport[0], vertiport[1])]
waypoints_ll += list(zip(corridor_lat, corridor_lon))
waypoints_ll += [(vertiport[0], vertiport[1])]

# =========================
# 2) 좌표 변환 (latlon -> local xy meter)
# =========================
def ll_to_xy(lat, lon, lat0, lon0):
    R = 6371000.0
    lat = np.deg2rad(lat); lon = np.deg2rad(lon)
    lat0 = np.deg2rad(lat0); lon0 = np.deg2rad(lon0)
    x = (lon - lon0) * np.cos(lat0) * R
    y = (lat - lat0) * R
    return x, y

def xy_to_ll(x, y, lat0, lon0):
    R = 6371000.0
    lat0r = np.deg2rad(lat0); lon0r = np.deg2rad(lon0)
    lat = y / R + lat0r
    lon = x / (R * np.cos(lat0r)) + lon0r
    return np.rad2deg(lat), np.rad2deg(lon)

lat0 = waypoints_ll[0][0]
lon0 = waypoints_ll[0][1]

wps_xy = [ll_to_xy(lat, lon, lat0, lon0) for (lat, lon) in waypoints_ll]
wps_xy = np.array(wps_xy)  # shape (K,2)

# =========================
# 3) 격자 생성
# =========================
grid_res_m = 80.0       # 격자 해상도 (작게 하면 더 촘촘, 느려짐)
margin_m = 600.0        # 전체 경로 주변 여유
diag_ok = True          # 8방향 이동 True

xmin = np.min(wps_xy[:,0]) - margin_m
xmax = np.max(wps_xy[:,0]) + margin_m
ymin = np.min(wps_xy[:,1]) - margin_m
ymax = np.max(wps_xy[:,1]) + margin_m

nx = int(np.ceil((xmax - xmin) / grid_res_m)) + 1
ny = int(np.ceil((ymax - ymin) / grid_res_m)) + 1

def xy_to_ij(x, y):
    i = int(np.round((x - xmin) / grid_res_m))
    j = int(np.round((y - ymin) / grid_res_m))
    i = max(0, min(nx - 1, i))
    j = max(0, min(ny - 1, j))
    return i, j

def ij_to_xy(i, j):
    x = xmin + i * grid_res_m
    y = ymin + j * grid_res_m
    return x, y

# =========================
# 4) 장애물 (원형 금지구역) 옵션
#    금지구역 없으면 빈 리스트로 두면 됨
# =========================
# 예시:
# forbidden_circles = [
#     (1000.0, -500.0, 400.0),  # (cx, cy, r) in meters, local xy 기준
# ]
forbidden_circles = []

def is_blocked_xy(x, y):
    for cx, cy, r in forbidden_circles:
        if (x - cx)**2 + (y - cy)**2 <= r**2:
            return True
    return False

blocked = np.zeros((nx, ny), dtype=bool)
if forbidden_circles:
    for i in range(nx):
        for j in range(ny):
            x, y = ij_to_xy(i, j)
            blocked[i, j] = is_blocked_xy(x, y)

# =========================
# 5) A* (격자 기반)
# =========================
def astar(start, goal):
    si, sj = start
    gi, gj = goal
    if blocked[si, sj] or blocked[gi, gj]:
        return None

    if diag_ok:
        moves = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    else:
        moves = [(-1,0),(1,0),(0,-1),(0,1)]

    def h(i, j):
        dx = (i - gi) * grid_res_m
        dy = (j - gj) * grid_res_m
        return np.hypot(dx, dy)

    INF = 1e18
    g = np.full((nx, ny), INF, dtype=float)
    parent = np.full((nx, ny, 2), -1, dtype=int)
    closed = np.zeros((nx, ny), dtype=bool)

    pq = []
    g[si, sj] = 0.0
    heapq.heappush(pq, (h(si, sj), 0.0, si, sj))

    while pq:
        fcur, gcur, i, j = heapq.heappop(pq)
        if closed[i, j]:
            continue
        closed[i, j] = True

        if (i, j) == (gi, gj):
            path = []
            ci, cj = gi, gj
            while not (ci == si and cj == sj):
                path.append((ci, cj))
                pi, pj = parent[ci, cj]
                if pi < 0:
                    return None
                ci, cj = pi, pj
            path.append((si, sj))
            path.reverse()
            return path

        for di, dj in moves:
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                continue
            if blocked[ni, nj] or closed[ni, nj]:
                continue

            step = grid_res_m * (np.sqrt(2.0) if (di != 0 and dj != 0) else 1.0)
            ng = gcur + step
            if ng < g[ni, nj]:
                g[ni, nj] = ng
                parent[ni, nj] = (i, j)
                heapq.heappush(pq, (ng + h(ni, nj), ng, ni, nj))

    return None

# =========================
# 6) 경유지 강제 통과 경로 생성
# =========================
wps_ij = [xy_to_ij(x, y) for (x, y) in wps_xy]

full_path_ij = []
for k in range(len(wps_ij) - 1):
    seg = astar(wps_ij[k], wps_ij[k + 1])
    if seg is None:
        raise RuntimeError(f"Segment {k} -> {k+1} infeasible. grid_res_m 줄이거나 margin/금지구역 확인.")
    if k > 0:
        seg = seg[1:]  # 중복점 제거
    full_path_ij.extend(seg)

full_path_xy = np.array([ij_to_xy(i, j) for (i, j) in full_path_ij])
full_path_ll = np.array([xy_to_ll(x, y, lat0, lon0) for (x, y) in full_path_xy])  # (lat, lon)

print("Total nodes:", len(full_path_ll))

# =========================
# 7) 결과 확인 플롯
# =========================
plt.figure(figsize=(7, 7))
plt.plot(full_path_ll[:,1], full_path_ll[:,0], linewidth=1.5, label="Grid path")
plt.scatter([p[1] for p in waypoints_ll], [p[0] for p in waypoints_ll], s=25, label="Waypoints")
plt.scatter([vertiport[1]], [vertiport[0]], s=60, label="Vertiport")
plt.xlabel("lon"); plt.ylabel("lat")
plt.legend()
plt.axis("equal")
plt.show()

# full_path_ll 가 최종 (lat, lon) 경로
