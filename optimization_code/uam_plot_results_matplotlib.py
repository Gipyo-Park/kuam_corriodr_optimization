import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

# import os
# from PyQt5 import QtCore

# plugin_path = os.path.join(os.path.dirname(QtCore.__file__), 'Qt5', 'plugins', 'platforms')
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path


# def corridor_polygon_from_path(path, W_half_m):
#     if path is None or len(path) < 2:
#         return None

#     mean_lat_rad = np.deg2rad(float(np.mean(path[:, 0])))
#     meters_per_lat_deg = 111000.0
#     meters_per_lon_deg = 111000.0 * np.cos(mean_lat_rad)

#     W_half_lat = W_half_m / meters_per_lat_deg
#     W_half_lon = W_half_m / meters_per_lon_deg

#     left_pts = []
#     right_pts = []

#     for i in range(len(path)):
#         if i == 0:
#             vec = path[i + 1, :2] - path[i, :2]
#         elif i == len(path) - 1:
#             vec = path[i, :2] - path[i - 1, :2]
#         else:
#             vec = path[i + 1, :2] - path[i - 1, :2]

#         norm = float(np.linalg.norm(vec))
#         if norm < 1e-12:
#             continue

#         perp = np.array([-vec[1], vec[0]], dtype=float) / norm
#         offset_lat = perp[0] * W_half_lat
#         offset_lon = perp[1] * W_half_lon

#         left_pts.append([path[i, 1] + offset_lon, path[i, 0] + offset_lat])
#         right_pts.append([path[i, 1] - offset_lon, path[i, 0] - offset_lat])

#     if len(left_pts) < 2:
#         return None

#     poly_pts = left_pts + right_pts[::-1]
#     return np.array(poly_pts, dtype=float)


# def plot_results(pkl_path="uam_nsga3_results.pkl"):
#     with open(pkl_path, "rb") as f:
#         data = pickle.load(f)

#     objective_names = data["objective_names"]
#     reps_all = data["representative_paths_final"]
#     points = data["points"]
#     vertiport = data["vertiport"]
#     forbidden_zones = data["forbidden_zones"]
#     emergency_points = data["emergency_points"]
#     lat_lim = data["lat_lim"]
#     lon_lim = data["lon_lim"]
#     W_half = float(data["W_half"])

#     fig, ax = plt.subplots(figsize=(10, 8))

#     ax.set_xlim(lon_lim[0], lon_lim[1])
#     ax.set_ylim(lat_lim[0], lat_lim[1])

#     ax.plot(points[:, 1], points[:, 0], "co-", linewidth=1.5, label="Corridor")
#     ax.plot(vertiport[1], vertiport[0], "mp", markersize=12, label="Vertiport")
#     ax.scatter(emergency_points[:, 1], emergency_points[:, 0], s=60, c="b", marker="^", label="Emergency")

#     if forbidden_zones is not None and forbidden_zones.size > 0:
#         for rect in forbidden_zones:
#             min_lon, max_lon, min_lat, max_lat = rect
#             ax.add_patch(
#                 Rectangle(
#                     (min_lon, min_lat),
#                     max_lon - min_lon,
#                     max_lat - min_lat,
#                     facecolor="red",
#                     edgecolor="none",
#                     alpha=0.25,
#                     label="_nolegend_"
#                 )
#             )

#     styles = ["r-", "b-", "g-", "m-"]

#     num_obj = len(objective_names)
#     num_rep_types = num_obj + 1

#     final_routes = []
#     for i in range(num_rep_types):
#         seg_paths = []
#         for seg in reps_all:
#             if seg and i < len(seg) and seg[i] is not None and len(seg[i]) > 0:
#                 seg_paths.append(seg[i])
#         if seg_paths:
#             final_routes.append(np.vstack(seg_paths))
#         else:
#             final_routes.append(np.empty((0, 3)))

#     labels = [f"Overall {name} Min" for name in objective_names] + ["Overall Balanced"]

#     for i, route in enumerate(final_routes):
#         if route.size == 0:
#             continue
#         ax.plot(route[:, 1], route[:, 0], styles[i % len(styles)], linewidth=1.2, label=labels[i])

#         if i == len(final_routes) - 1:
#             poly = corridor_polygon_from_path(route, W_half)
#             if poly is not None:
#                 ax.add_patch(
#                     Polygon(poly, closed=True, facecolor="yellow", edgecolor="none", alpha=0.15, label="Corridor Width")
#                 )

#     ax.set_aspect("equal", adjustable="box")
#     ax.grid(True, linewidth=0.4, alpha=0.5)
#     ax.legend()
#     ax.set_title("Final UAM Corridor Paths")
#     plt.show()


# if __name__ == "__main__":
#     plot_results()


import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_nsga_result(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    reps = data.get("representative_paths_final", [])
    base_points = data.get("base_points", None)
    forbidden_zones = data.get("forbidden_zones", None)
    lat_lim = data.get("lat_lim", None)
    lon_lim = data.get("lon_lim", None)

    plt.figure(figsize=(8, 8))

    # NFZ
    if forbidden_zones is not None:
        for rect in forbidden_zones:
            min_lon, max_lon, min_lat, max_lat = rect
            xs = [min_lon, max_lon, max_lon, min_lon, min_lon]
            ys = [min_lat, min_lat, max_lat, max_lat, min_lat]
            plt.plot(xs, ys, color="red", linewidth=2)

    # base corridor
    if base_points is not None:
        plt.plot(
            base_points[:, 1],
            base_points[:, 0],
            "k--",
            linewidth=2,
            label="base corridor"
        )

    # representative paths
    for i, path in enumerate(reps):
        if path is None or len(path) == 0:
            continue
        p = np.array(path)
        plt.plot(
            p[:, 1],
            p[:, 0],
            linewidth=1.5,
            label=f"path {i}"
        )
        plt.scatter(p[:, 1], p[:, 0], s=15)

    if lon_lim is not None and lat_lim is not None:
        plt.xlim(lon_lim)
        plt.ylim(lat_lim)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_nsga_result("uam_nsga3_results_whole.pkl")
