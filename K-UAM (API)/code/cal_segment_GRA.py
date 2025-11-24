import numpy as np
from math import floor, atan2, degrees
from itertools import product
import sys
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors
from pyproj import Transformer
import re



def compute_heading(lat1, lon1, lat2, lon2):
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    angle_rad = atan2(d_lon, d_lat)
    return (degrees(angle_rad) + 360) % 360

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


def interpolate_risk(data, lat1, lon1, alt1, lat2, lon2, alt2):
    if (alt1 < 100 or alt1 > 600 or alt2 < 100 or alt2 > 600):
        if (alt1 == 0 or alt2 == 0):
            print("Note: z=0 allowed at final endpoint")
        else:
            print("Out of altitude range")
            sys.exit(1)

    lat_grid = data[:, 0, 0, 0]
    lon_grid = data[0, :, 0, 1]
    alt_grid = data[0, 0, :, 2]

    heading = compute_heading(lat1, lon1, lat2, lon2)

    angle_idx_low = int(floor(heading / 45)) % 8
    angle_idx_high = (angle_idx_low + 1) % 8
    angle_weight = (heading % 45) / 45

    i_lat0, i_lat1, w_lat = find_bounds_and_weights(lat1, lat_grid)
    i_lon0, i_lon1, w_lon = find_bounds_and_weights(lon1, lon_grid)
    i_alt0, i_alt1, w_alt = find_bounds_and_weights(alt1, alt_grid)

    risk_values = []
    weights = []

    print(f"\nHeading: {heading:.2f}° → angle index {angle_idx_low} and {angle_idx_high} (weight {angle_weight:.3f})\n")
    print("---- 16 Ground Risk ----\n")

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

        print(f"[{i:3}, {j:3}, {k}] | angle {angle_idx_low}-{angle_idx_high} → r1={r1:.3f}, r2={r2:.3f}, interp={r:.3f}, weight={w:.4f}")

        risk_values.append(r)
        weights.append(w)

    final_risk = sum(r * w for r, w in zip(risk_values, weights))
    print("\n------------------------")
    print(f"Final Ground Risk: {final_risk:.6f}")
    return final_risk


def generate_segments(start_x, start_y, num_segments=5, xy_range=(100, 300), alt_range=(100, 500)):
    segments = []
    x, y, z = start_x, start_y, random.uniform(*alt_range)
    for _ in range(num_segments):
        dx = random.uniform(*xy_range)
        dy = random.uniform(*xy_range)
        dz = random.uniform(0, 100)
        next_x = x + dx
        next_y = y + dy
        next_z = min(600, z + dz)
        segments.append(((x, y, z), (next_x, next_y, next_z)))
        x, y, z = next_x, next_y, next_z
    return segments

def generate_connected_segments_from_data(data, num_segments=5):
    
    lat_vals = data[:, 0, 0, 0]
    lon_vals = data[0, :, 0, 1]
    alt_vals = data[0, 0, :, 2]

    lat_min, lat_max = lat_vals.min(), lat_vals.max()
    lon_min, lon_max = lon_vals.min(), lon_vals.max()
    alt_min, alt_max = alt_vals.min(), alt_vals.max()

    x = random.uniform(lat_min, lat_max)
    y = random.uniform(lon_min, lon_max)
    z = random.uniform(alt_min, alt_max)

    segments = []
    for _ in range(num_segments):
        next_x = random.uniform(lat_min, lat_max)
        next_y = random.uniform(lon_min, lon_max)
        next_z = random.uniform(alt_min, alt_max)

        segments.append(((x, y, z), (next_x, next_y, next_z)))

        x, y, z = next_x, next_y, next_z

    return segments



if __name__ == "__main__":
    file_path = '../data/GRC.npy'
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)
        
# #--------------------------One Segment------------------------------
#     lat1, lon1, alt1 = 1137580.0, 1726680.0, 330.0
#     lat2, lon2, alt2 = 1137670.0, 1726760.0, 500.0

#     risk = interpolate_risk(data, lat1, lon1, alt1, lat2, lon2, alt2)
# #--------------------------One Segment------------------------------


#--------------------------Multi Segment------------------------------

#--------------------------Smooth Segment------------------------------

    # start_x = 1.13745e+06
    # start_y = 1.72645e+06

    # segments = generate_segments(start_x, start_y)
    
#--------------------------Smooth Segment------------------------------

#--------------------------Ulsan Corridor-----------------------------


    def dms_str_to_decimal(dms):
        parts = re.findall(r"(\d+)°(\d+)'([\d.]+)\"", dms)
        if not parts:
            raise ValueError(f"Invalid DMS format: {dms}")
        d, m, s = map(float, parts[0])
        return d + m / 60 + s / 3600

    raw_coords = [
        
        ('35°36\'12.01"N', '129°4\'39.69"E'),
        ('35°35\'4.53"N',  '129°5\'37.13"E'),
        ('35°36\'9.55"N',  '129°6\'47.04"E'),
        ('35°37\'57.65"N', '129°7\'25.89"E'),
        ('35°37\'29.85"N', '129°8\'0.79"E'),
        ('35°36\'12.51"N', '129°7\'36.55"E'),
        ('35°35\'4.33"N',  '129°6\'27.53"E'),
        ('35°34\'9.25"N',  '129°6\'30.71"E'),
        ('35°33\'16.72"N', '129°5\'37.31"E'),
        ('35°33\'31.22"N', '129°4\'53.98"E'),
        ('35°34\'42.51"N', '129°5\'30.08"E'),
        ('35°35\'3.74"N',  '129°4\'37.16"E'),
        ('35°36\'58.99"N', '129°3\'41.02"E'),
        ('35°37\'16.51"N', '129°4\'21.16"E'),
        ('35°36\'39.79"N', '129°4\'16.28"E'),
        ('35°36\'12.01"N', '129°4\'39.69"E'),
    ]

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True) # TM East Origin

#-------------------------No VertyPort--------------------------------

    # converted_points = []
    # for lat_dms, lon_dms in raw_coords:
    #     lat = dms_str_to_decimal(lat_dms)
    #     lon = dms_str_to_decimal(lon_dms)
    #     x, y = transformer.transform(lon, lat)
    #     # z = random.uniform(300, 600)
    #     z = 450.0
    #     converted_points.append((x, y, z))
    
#-------------------------No VertyPort--------------------------------
    
    start_dms = ('35°36\'12.01"N', '129°4\'39.69"E')

    altitudes = []
    for lat_dms, lon_dms in raw_coords:
        if (lat_dms, lon_dms) == start_dms:
            altitudes.append(0.0)
        else:
            altitudes.append(450.0)

    altitudes[1] = 300.0  
    altitudes[-2] = 150.0  
    altitudes[-3] = 300.0  

    converted_points = []
    for (lat_dms, lon_dms), z in zip(raw_coords, altitudes):
        lat = dms_str_to_decimal(lat_dms)
        lon = dms_str_to_decimal(lon_dms)
        x, y = transformer.transform(lon, lat)
        converted_points.append((x, y, z))
        
        print(converted_points)


    segments = []
    for i in range(len(converted_points) - 1):
        segments.append((converted_points[i], converted_points[i + 1]))

#--------------------------Ulsan Corridor-----------------------------

#--------------------------Random Segment------------------------------

    # segments = generate_connected_segments_from_data(data, num_segments=10)

#--------------------------Random Segment------------------------------

    risks = []
    xs, ys, zs = [], [], []

    for (x1, y1, z1), (x2, y2, z2) in segments:
        risk = interpolate_risk(data, x1, y1, z1, x2, y2, z2)
        risks.append(risk)
        
        xs.append(x1)
        ys.append(y1)
        zs.append(z1)

    # Normalize risk to [0, 10]
    risks = np.array(risks)
    risk_norm = 10 * (risks - np.min(risks)) / (np.max(risks) - np.min(risks) + 1e-6)

    # # 3D Visualization with colormap: purple → red
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # sc = ax.scatter(xs, ys, zs, c=risk_norm, cmap='Reds', s=100)  # 'Reds' for purple→red
    # plt.colorbar(sc, ax=ax, label='Ground Risk (0-10)')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Altitude')
    # ax.set_title('Ground Risk Along Corridor Segments')
    # plt.show()

#--------------------------Multi Segment------------------------------

#--------------------------Line Segment------------------------------

    # Create line segments [(x1,y1,z1),(x2,y2,z2)] for each segment
    lines = []
    line_colors = []

    for (x1, y1, z1), (x2, y2, z2), risk in zip([s[0] for s in segments], [s[1] for s in segments], risk_norm):
        lines.append([(x1, y1, z1), (x2, y2, z2)])
        line_colors.append(risk)

    # Set up colormap
    cmap = plt.get_cmap('Reds')
    norm = plt.Normalize(0, 10)
    line_collection = Line3DCollection(lines, colors=cmap(norm(line_colors)), linewidths=3)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(xs, ys, zs, c=risk_norm, cmap='Reds', s=80, label='Segment StartPoints')

    ax.add_collection3d(line_collection)

    # Labels
    plt.colorbar(sc, ax=ax, label='Ground Risk (0-10)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Altitude')
    ax.set_title('Ground Risk Along Corridor Segments')
    plt.legend()
    plt.show()
    
    
    #-------------------------2D Visualization----------------------------
    fig2d = plt.figure()
    ax2d = fig2d.add_subplot(111)

    for ((x1, y1, _), (x2, y2, _)), risk in zip(segments, risk_norm):
        ax2d.plot([x1, x2], [y1, y2], color=cmap(norm(risk)), linewidth=2)

    sc2d = ax2d.scatter(xs, ys, c=risk_norm, cmap='Reds', s=60, label='Segment StartPoints')

    fig2d.colorbar(sc2d, ax=ax2d, label='Ground Risk (0–10)')
    ax2d.set_xlabel('X')
    ax2d.set_ylabel('Y')
    ax2d.set_title('2D Ground Risk Map')
    plt.legend()
    plt.show()
    #-------------------------2D Visualization----------------------------

#--------------------------Line Segment------------------------------
