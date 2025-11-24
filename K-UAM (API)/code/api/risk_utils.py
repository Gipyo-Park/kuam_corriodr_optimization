import numpy as np
from math import floor, atan2, degrees
from itertools import product
from pyproj import Transformer

transformer_lla_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
transformer_utm_to_lla = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)


def lla_to_utm(lat, lon, alt):
    x, y = transformer_lla_to_utm.transform(lon, lat)
    return (x, y, alt)

def utm_to_lla(x, y, alt):
    """Single point conversion UTM -> LLA"""
    lon, lat = transformer_utm_to_lla.transform(x, y)
    return (lat, lon, alt)

def batch_lla_to_utm(points, inverse=False):
    """
    Convert a batch of points.
    - If inverse=False: LLA -> UTM
    - If inverse=True : UTM -> LLA
    """
    if not inverse:
        return [lla_to_utm(lat, lon, alt) for (lat, lon, alt) in points]
    else:
        return [utm_to_lla(x, y, alt) for (x, y, alt) in points]

def split_path_by_interval(points, interval=500):
    """points: [(x, y, z)], interval(m)"""
    segments = []
    for i in range(len(points)-1):
        p1 = np.array(points[i])
        p2 = np.array(points[i+1])
        length = np.linalg.norm(p2-p1)
        nseg = max(1, int(length // interval))
        for j in range(nseg):
            a0 = j / nseg
            a1 = (j+1) / nseg
            pt0 = (1-a0)*p1 + a0*p2
            pt1 = (1-a1)*p1 + a1*p2
            segments.append((tuple(pt0), tuple(pt1)))
    return segments

def find_bounds_and_weights(value, grid):
    if value <= grid[0]:
        return 0, 0, 1.0
    if value >= grid[-1]:
        return len(grid)-2, len(grid)-1, 0.0
    for i in range(len(grid)-1):
        if grid[i] <= value < grid[i+1]:
            low, high = i, i+1
            w = (value - grid[low]) / (grid[high] - grid[low])
            return low, high, w

def interpolate_risk_path(data, x1, y1, z1, x2, y2, z2):
    heading = (degrees(atan2(y2 - y1, x2 - x1)) + 360) % 360
    return interpolate_risk_with_heading(data, x1, y1, z1, heading)

def interpolate_risk_with_heading(data, x, y, z, heading):
    lat_grid = data[:, 0, 0, 0]
    lon_grid = data[0, :, 0, 1]
    alt_grid = data[0, 0, :, 2]

    angle_idx_low = int(floor(heading / 45)) % 8
    angle_idx_high = (angle_idx_low + 1) % 8
    angle_weight = (heading % 45) / 45

    i_lat0, i_lat1, w_lat = find_bounds_and_weights(x, lat_grid)
    i_lon0, i_lon1, w_lon = find_bounds_and_weights(y, lon_grid)
    i_alt0, i_alt1, w_alt = find_bounds_and_weights(z, alt_grid)

    risk_values = []
    weights = []

    for di, dj, dk in product([0, 1], repeat=3):
        i = i_lat0 + di
        j = i_lon0 + dj
        k = i_alt0 + dk
        w = ((1 - w_lat) if di == 0 else w_lat) * \
            ((1 - w_lon) if dj == 0 else w_lon) * \
            ((1 - w_alt) if dk == 0 else w_alt)

        r1 = data[i, j, k, 3 + angle_idx_low]
        r2 = data[i, j, k, 3 + angle_idx_high]
        r = (1 - angle_weight) * r1 + angle_weight * r2

        risk_values.append(r)
        weights.append(w)

    final_risk = sum(r * w for r, w in zip(risk_values, weights))
    return final_risk

def emergency_risk_along_path(data, plane_utm, emergency_sites_utm, interval=100):
    all_risks = []
    for site in emergency_sites_utm:
        segs = split_path_by_interval([plane_utm, site], interval)
        risks = [
            interpolate_risk_with_heading(
                data,
                *seg[0],
                (degrees(atan2(seg[1][1] - seg[0][1], seg[1][0] - seg[0][0])) + 360) % 360
            )
            for seg in segs
        ]
        all_risks.append({
            "to_site": site,
            "risks": risks,
            "mean_risk": float(np.mean(risks)) if risks else None
        })
    return all_risks
