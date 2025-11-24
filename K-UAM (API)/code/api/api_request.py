import requests
import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.animation import FuncAnimation
from math import floor, atan2, degrees
import json, os, datetime

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

SERVER = "http://10.74.11.159:8000"

route_lla = [
    [35.603336, 129.077692, 0.0],
    [35.584592, 129.093647, 300.0],
    [35.602653, 129.113067, 450.0],
    [35.632681, 129.123858, 450.0],
    [35.624958, 129.133553, 450.0],
    [35.603475, 129.126819, 450.0],
    [35.584536, 129.107647, 450.0],
    [35.569236, 129.108531, 450.0],
    [35.554644, 129.093697, 450.0],
    [35.558672, 129.081661, 450.0],
    [35.578475, 129.091689, 450.0],
    [35.584372, 129.077544, 450.0],
    [35.616386, 129.061394, 450.0],
    [35.621253, 129.072544, 300.0],
    [35.611053, 129.071189, 150.0],
    [35.603336, 129.077692, 0.0]
]
emergency_sites_lla = [
    [35.620108, 129.119181, 0.0],
    [35.567822, 129.106728, 0.0],
    [35.591989, 129.075197, 0.0],
]


def save_json(logname, obj):
    path = os.path.join(LOG_DIR, f"{logname}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def post_and_log(endpoint, payload, logname, **meta):
    url = f"{SERVER}{endpoint}"
    try:
        resp = requests.post(url, json=payload, timeout=10)
        try:
            body = resp.json()
        except Exception:
            body = {"_parse_error": True, "_text": resp.text[:2000]}
        log_obj = {
            "ts": datetime.datetime.now().isoformat(),
            "endpoint": endpoint,
            "request": payload,
            "status": resp.status_code,
            "response": body,
            **meta
        }
        save_json(logname, log_obj)
        resp.raise_for_status()
        return body
    except Exception as e:
        log_obj = {
            "ts": datetime.datetime.now().isoformat(),
            "endpoint": endpoint,
            "request": payload,
            "error": str(e),
            **meta
        }
        save_json(logname, log_obj)
        return {}


transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
transformer_utm_to_lla = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)

def lla_to_utm(lat, lon, alt):
    x, y = transformer.transform(lon, lat)
    return (x, y, alt)

route_utm = [lla_to_utm(lat, lon, alt) for lat, lon, alt in route_lla]

def split_path_by_interval(points, interval=500):
    segments = []
    for i in range(len(points) - 1):
        p1 = np.array(points[i])
        p2 = np.array(points[i + 1])
        length = np.linalg.norm(p2 - p1)
        nseg = max(1, int(length // interval))
        for j in range(nseg):
            a0 = j / nseg
            a1 = (j + 1) / nseg
            pt0 = (1 - a0) * p1 + a0 * p2
            pt1 = (1 - a1) * p1 + a1 * p2
            segments.append((tuple(pt0), tuple(pt1)))
    return segments

route_segments_utm = split_path_by_interval(route_utm, interval=500)
route_segments_lla = []      
route_segment_risks = []  

path_req = {"points": route_lla}
resp_body = post_and_log("/path-risk", path_req, "path-risk")
risks = resp_body.get("segment_risks", [])

if not risks:
    route_segment_risks = [0.0] * len(route_segments_utm)
else:
    n = min(len(risks), len(route_segments_utm))
    route_segment_risks = risks[:n]
    route_segments_utm = route_segments_utm[:n]

#-------------------------------------Force------------------------------------------
plane_positions = [seg[0] for seg in route_segments_utm] + [route_segments_utm[-1][1]]
#-------------------------------------Force------------------------------------------
    
sampled_lla = []
headings = []
for i, pt in enumerate(plane_positions):
    lon, lat = transformer_utm_to_lla.transform(pt[0], pt[1])
    alt = pt[2]
    sampled_lla.append([lat, lon, alt])
    if i < len(plane_positions) - 1:
        dx = plane_positions[i+1][0] - pt[0]
        dy = plane_positions[i+1][1] - pt[1]
    else:
        dx = pt[0] - plane_positions[i-1][0]
        dy = pt[1] - plane_positions[i-1][1]
    heading = (degrees(atan2(dy, dx)) + 360) % 360
    headings.append(heading)

point_risks = []
emergency_paths = []

for idx, (plane_lla, heading) in enumerate(zip(sampled_lla, headings)):
    point_req = {"point": plane_lla, "heading": heading}
    # body1 = post_and_log("/point-risk", point_req, "point-risk", frame=idx)
    body1 = post_and_log("/point-risk", point_req, "point-risk")
    point_risks.append(body1.get("risk", 0))

    emergency_req = {"plane": plane_lla, "emergency_sites": emergency_sites_lla}
    # body2 = post_and_log("/emergency-risk", emergency_req, "emergency-risk", frame=idx)
    body2 = post_and_log("/emergency-risk", emergency_req, "emergency-risk")
    paths = body2.get("emergency_paths", [])
    print(f"Frame {idx}: returned {len(paths)} paths for {len(emergency_sites_lla)} sites")
    emergency_paths.append(paths)
        
plane_positions = []
for lla in sampled_lla:
    x, y, z = lla_to_utm(*lla)
    plane_positions.append([x, y, z])


emergency_sites = [lla_to_utm(*site) for site in emergency_sites_lla]

cmap = plt.get_cmap('seismic')
norm = plt.Normalize(0, 1)

fig = plt.figure(figsize=(11,9))
ax = fig.add_subplot(111, projection='3d')
pt_size = 40

route_lines = [[seg[0], seg[1]] for seg in route_segments_utm]
route_colors = cmap(norm(route_segment_risks))
route_line_collection = Line3DCollection(route_lines, colors=route_colors, linewidths=2)

def update(frame):
    ax.cla()
    
    if frame > 0:
        past_lines = route_lines[:frame]
        past_line_collection = Line3DCollection(
            past_lines, colors='gray', linewidths=2, zorder=0
        )
        ax.add_collection3d(past_line_collection)
        past_pts = np.array(plane_positions[:frame])
        ax.scatter(
            past_pts[:,0], past_pts[:,1], past_pts[:,2],
            color='gray', s=pt_size, marker='o', zorder=1
        )
    if frame < len(route_lines):
        future_lines = route_lines[frame:]
        future_risks = route_segment_risks[frame:]
        future_colors = cmap(norm(future_risks))
        future_line_collection = Line3DCollection(
            future_lines, colors=future_colors, linewidths=2, zorder=0
        )
        ax.add_collection3d(future_line_collection)
        future_pts = np.array(plane_positions[frame:])
        ax.scatter(
            future_pts[:,0], future_pts[:,1], future_pts[:,2],
            c=point_risks[frame:], norm=norm, cmap='seismic',
            s=pt_size, marker='o', zorder=1
        )
    ax.scatter(
        plane_positions[frame][0], plane_positions[frame][1], plane_positions[frame][2],
        color='silver', s=400, marker='o', edgecolor='none', zorder=15
    )

    for i, site in enumerate(emergency_sites):
        path_info = emergency_paths[frame][i]
        segs = []
        p_start = plane_positions[frame]
        p_end = site
        nseg = len(path_info["risks"])
        for k in range(nseg):
            pt0 = p_start + (np.array(p_end) - np.array(p_start)) * (k/nseg)
            pt1 = p_start + (np.array(p_end) - np.array(p_start)) * ((k+1)/nseg)
            segs.append([pt0, pt1])
        seg_colors = cmap(norm(np.array(path_info["risks"])))
        lc = Line3DCollection(segs, colors=seg_colors, linewidths=2, linestyles=':')
        ax.add_collection3d(lc)
        ax.scatter(site[0], site[1], site[2], color=cmap(norm(path_info["mean_risk"])), edgecolor=cmap(norm(path_info["mean_risk"])), s=600, marker='o', depthshade=False, zorder=10)
        ax.text(site[0], site[1], site[2]+60, f"{path_info['mean_risk']:.2f}", fontsize=10, color='k', ha='center', va='center', zorder=11)

    ax.set_xlabel("X (TM)")
    ax.set_ylabel("Y (TM)")
    ax.set_zlabel("Altitude")
    ax.set_title("3D Ground Risk Visualization (API)")
    ax.set_xlim(min([p[0] for p in plane_positions]), max([p[0] for p in plane_positions]))
    ax.set_ylim(min([p[1] for p in plane_positions]), max([p[1] for p in plane_positions]))
    ax.set_zlim(0, max([p[2] for p in plane_positions]) * 1.2)
    plt.tight_layout()

    ax.text2D(0.5, 0.93, f"Current Ground Risk: {point_risks[frame]:.2f}", fontsize=14, color='black', ha='center', transform=ax.transAxes)
    ax.text2D(0.5, 0.97, f"Current Position: {sampled_lla[frame][0]:.5f}, {sampled_lla[frame][1]:.5f}, {sampled_lla[frame][2]:.1f} | Heading: {headings[frame]:.1f}°", fontsize=12, color='dimgray', ha='center', transform=ax.transAxes)

ax.view_init(elev=90, azim=-90)    
sc = ax.scatter([p[0] for p in plane_positions], [p[1] for p in plane_positions], [p[2] for p in plane_positions], c=point_risks, cmap='seismic', norm=norm, s=pt_size)
plt.colorbar(sc, ax=ax, label='Ground Risk (0-1)')
anim = FuncAnimation(fig, update, frames=len(plane_positions), interval=500, blit=False)
plt.show()