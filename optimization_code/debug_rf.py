import numpy as np
from rf_turn import apply_rf_turns, _latlon_to_local_m

corridor_lat = np.array([35.5944587, 35.6195580, 35.6218142, 35.5595903, 35.5671301, 35.5887324, 35.5931814, 35.6185184, 35.6249109])
corridor_lon = np.array([129.0977958, 129.1153758, 129.1266116, 129.0849662, 129.0776521, 129.0691071, 129.0665565, 129.0512209, 129.0536710])
vertiport = np.array([35.6033361, 129.0776917, 150.0])
waypoints = np.column_stack([corridor_lat, corridor_lon, np.full_like(corridor_lat, vertiport[2])])

def _sector_angle(sector_1, n=12):
    i = int(sector_1) - 1
    w = 2.0 * np.pi / n
    return np.deg2rad(90.0) - (i + 0.5) * w

def _move_latlon(lat0, lon0, heading_rad, dist_m):
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(lat0))
    return float(lat0 + dist_m * np.sin(heading_rad) / m_lat), float(lon0 + dist_m * np.cos(heading_rad) / m_lon)

d = 350.0 / np.tan(np.deg2rad(25.0))
to_lat, to_lon = _move_latlon(vertiport[0], vertiport[1], _sector_angle(11), d)
ld_lat, ld_lon = _move_latlon(vertiport[0], vertiport[1], _sector_angle(6), d)
takeoff_complete = np.array([to_lat, to_lon, vertiport[2]])
landing_entry = np.array([ld_lat, ld_lon, vertiport[2]])

def test_backbone(name, bb):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    N = bb.shape[0]
    for i, p in enumerate(bb):
        print(f"  [{i}] lat={p[0]:.6f} lon={p[1]:.6f}")
    
    rf = apply_rf_turns(bb, 70.65, 25.0, 30)
    print(f"  RF feasible: {rf['feasible']}, segments: {len(rf['segments'])}")
    
    ref_lat = float(np.mean(bb[:, 0]))
    ref_lon = float(np.mean(bb[:, 1]))
    local = _latlon_to_local_m(bb, ref_lat, ref_lon)
    R = (70.65**2) / (9.80665 * np.tan(np.deg2rad(25.0)))
    
    seg_lengths = [np.linalg.norm(local[i+1,:2] - local[i,:2]) for i in range(N-1)]
    
    tds = [0.0]
    for i in range(1, N-1):
        v_in = local[i,:2] - local[i-1,:2]
        v_out = local[i+1,:2] - local[i,:2]
        n_in = np.linalg.norm(v_in)
        n_out = np.linalg.norm(v_out)
        if n_in < 1e-6 or n_out < 1e-6:
            tds.append(0.0)
            continue
        cos_inner = np.clip(np.dot(v_in, v_out) / (n_in * n_out), -1.0, 1.0)
        delta = np.arccos(cos_inner)  # FIXED formula
        if abs(delta) < np.deg2rad(0.5):
            tds.append(0.0)
            continue
        td = R * np.tan(abs(delta) / 2.0)
        tds.append(td)
        print(f"  WP[{i}]: delta={np.rad2deg(delta):.1f}deg, td={td:.1f}m, seg_prev={seg_lengths[i-1]:.0f}m, seg_next={seg_lengths[i]:.0f}m")
    tds.append(0.0)
    
    all_ok = True
    for j in range(N-1):
        needed = tds[j] + tds[j+1]
        avail = seg_lengths[j]
        ok = needed <= avail + 0.01
        if not ok:
            all_ok = False
        tag = "OK" if ok else "FAIL"
        if not ok:
            print(f"  Seg[{j}->{j+1}]: needed={needed:.0f}m > avail={avail:.0f}m  {tag}")
    if all_ok:
        print("  All segments OK!")
    print(f"  R = {R:.1f}m")

# Test 1: Original WP order, backbone = [takeoff, WP1..WP9, landing]
print("\n=== Test 1: Original WP order + takeoff first ===")
bb1 = np.vstack([takeoff_complete, waypoints, landing_entry])
test_backbone("takeoff -> WP1..WP9 -> landing", bb1)

# Test 2: Reversed WP order  
print("\n=== Test 2: Reversed WP order + takeoff first ===")
bb2 = np.vstack([takeoff_complete, waypoints[::-1], landing_entry])
test_backbone("takeoff -> WP9..WP1 -> landing", bb2)

# Test 3: CW sorted  
centroid_lat = float(np.mean(waypoints[:, 0]))
centroid_lon = float(np.mean(waypoints[:, 1]))
_cos_lat = np.cos(np.deg2rad(centroid_lat))
_dlat = waypoints[:, 0] - centroid_lat
_dlon = waypoints[:, 1] - centroid_lon
_bearings = np.rad2deg(np.arctan2(_dlon * _cos_lat, _dlat)) % 360.0
_to_dlat = takeoff_complete[0] - centroid_lat
_to_dlon = takeoff_complete[1] - centroid_lon
_to_bearing = np.rad2deg(np.arctan2(_to_dlon * _cos_lat, _to_dlat)) % 360.0
_rel_bearings = (_bearings - _to_bearing) % 360.0
_cw_order = np.argsort(_rel_bearings)
bb3 = np.vstack([takeoff_complete, waypoints[_cw_order], landing_entry])
test_backbone(f"takeoff -> CW sorted WPs -> landing (order: {_cw_order.tolist()})", bb3)

