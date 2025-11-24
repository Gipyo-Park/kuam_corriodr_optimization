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
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation




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

    # print(f"\nHeading: {heading:.2f}° → angle index {angle_idx_low} and {angle_idx_high} (weight {angle_weight:.3f})\n")
    # print("---- 16 Ground Risk ----\n")

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

        # print(f"[{i:3}, {j:3}, {k}] | angle {angle_idx_low}-{angle_idx_high} → r1={r1:.3f}, r2={r2:.3f}, interp={r:.3f}, weight={w:.4f}")

        risk_values.append(r)
        weights.append(w)

    final_risk = sum(r * w for r, w in zip(risk_values, weights))
    # print("\n------------------------")
    # print(f"Final Ground Risk: {final_risk:.6f}")
    return final_risk

if __name__ == "__main__":
    file_path = '../data/GRC.npy'
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        # print(f"File not found: {file_path}")
        sys.exit(1)


    def dms_str_to_decimal(dms):
        parts = re.findall(r"(\d+)°(\d+)'([\d.]+)\"", dms)
        if not parts:
            raise ValueError(f"Invalid DMS format: {dms}")
        d, m, s = map(float, parts[0])
        return d + m / 60 + s / 3600

    raw_coords = [
        
        ('35°36\'12.01"N', '129°4\'39.69"E'), # Verty Port
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
        ('35°36\'12.01"N', '129°4\'39.69"E'), # Verty Port
    ]

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True) # TM East Origin
    
    start_dms = ('35°36\'12.01"N', '129°4\'39.69"E')  # Verty Port

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
        
        # print(converted_points)
        
        
#--------------------------Emergency Landing Sites------------------------------

    emergency_dms_coords = [
        ('35°37\'12.39"N', '129°7\'9.05"E'),
        ('35°34\'4.16"N',  '129°6\'24.22"E'),
        ('35°35\'31.16"N', '129°4\'30.71"E'),
    ]

    emergency_sites = []
    for lat_dms, lon_dms in emergency_dms_coords:
        lat = dms_str_to_decimal(lat_dms)
        lon = dms_str_to_decimal(lon_dms)
        x, y = transformer.transform(lon, lat)
        emergency_sites.append((x, y, 0.0))

#--------------------------Emergency Landing Sites------------------------------

#------------------------Calculate segments & Risks-----------------------------

    segments = []
    for i in range(len(converted_points) - 1):
        segments.append((converted_points[i], converted_points[i + 1]))
        
        
    
    risks = []
    xs, ys, zs = [], [], []

    for (x1, y1, z1), (x2, y2, z2) in segments:
        risk = interpolate_risk(data, x1, y1, z1, x2, y2, z2)
        risks.append(risk)
        
        xs.append(x1)
        ys.append(y1)
        zs.append(z1)

    risks = np.array(risks)
    risk_norm = 10 * (risks - np.min(risks)) / (np.max(risks) - np.min(risks) + 1e-6)
        
#------------------------Calculate segments & Risks-----------------------------

#--------------------------Refine Segments for Fine-Grained Risk------------------------------

    segment_detail_interval = 500  # meters

    refined_segments = []
    refined_risks = []
    
    refined_xs, refined_ys, refined_zs = [], [], []

    for (x1, y1, z1), (x2, y2, z2) in segments:
        p1 = np.array([x1, y1, z1])
        p2 = np.array([x2, y2, z2])
        segment_length = np.linalg.norm(p2 - p1)
        num_subsegments = max(1, int(segment_length // segment_detail_interval))

        for i in range(num_subsegments):
            alpha_start = i / num_subsegments
            alpha_end = (i + 1) / num_subsegments
            pt_start = (1 - alpha_start) * p1 + alpha_start * p2
            pt_end   = (1 - alpha_end) * p1 + alpha_end * p2

            try:
                risk = interpolate_risk(data, pt_start[0], pt_start[1], pt_start[2],
                                            pt_end[0], pt_end[1], pt_end[2])
            except SystemExit:
                continue  # skip this segment if out of altitude

            refined_segments.append((tuple(pt_start), tuple(pt_end)))
            refined_risks.append(risk)
            
            refined_xs.append(pt_start[0])
            refined_ys.append(pt_start[1])
            refined_zs.append(pt_start[2])

    refined_risks = np.array(refined_risks)
    refined_risk_norm = 10 * (refined_risks - np.min(refined_risks)) / (np.max(refined_risks) - np.min(refined_risks) + 1e-6)
    
    # print("Total refined segments:", len(refined_segments))

#--------------------------Refine Segments for Fine-Grained Risk------------------------------

#--------------------------Emergency Risk & Segments----------------------------

    cumulative_dists = [0.0]
    for (p1, p2) in segments:
        dist = np.linalg.norm(np.array(p2[:2]) - np.array(p1[:2]))  
        cumulative_dists.append(cumulative_dists[-1] + dist)

    interval = 500  # meters
    emergency_segment_interval = 500
    
    target_dists = np.arange(0, cumulative_dists[-1], interval)

    plane_positions = []
    for td in target_dists:
        for i in range(1, len(cumulative_dists)):
            if cumulative_dists[i-1] <= td <= cumulative_dists[i]:
                r = (td - cumulative_dists[i-1]) / (cumulative_dists[i] - cumulative_dists[i-1])
                p1 = np.array(segments[i-1][0])
                p2 = np.array(segments[i-1][1])
                pos = (1 - r) * p1 + r * p2  
                plane_positions.append(tuple(pos))
                break

    emergency_segments = []
    emergency_risks = []


    for plane_pos in plane_positions:
        for site in emergency_sites:
            risk = interpolate_risk(data, plane_pos[0], plane_pos[1], plane_pos[2], site[0], site[1], site[2])
            emergency_risks.append(risk)
            emergency_segments.append((plane_pos, site))

    emergency_risks = np.array(emergency_risks)
    emergency_risk_norm = 10 * (emergency_risks - np.min(emergency_risks)) / (np.max(emergency_risks) - np.min(emergency_risks) + 1e-6)
    
    
    
    
#--------------------------Emergency Risk & Segments----------------------------

#--------------------------Ulsan Corridor-----------------------------

#--------------------------Multi Segment------------------------------

#--------------------------Line Segment------------------------------

    # Create line segments [(x1,y1,z1),(x2,y2,z2)] for each segment
    
    lines = []
    line_colors = []

    for (x1, y1, z1), (x2, y2, z2), risk in zip([s[0] for s in segments], [s[1] for s in segments], risks):
        lines.append([(x1, y1, z1), (x2, y2, z2)])
        line_colors.append(risk)
    
    
    refined_lines = []
    refined_colors = []
    

    for (x1, y1, z1), (x2, y2, z2), risk in zip(
        [s[0] for s in refined_segments],
        [s[1] for s in refined_segments],
        refined_risks):
        refined_lines.append([(x1, y1, z1), (x2, y2, z2)])
        refined_colors.append(risk)

    # Set up colormap
    cmap = plt.get_cmap('seismic')
    norm = plt.Normalize(0, 10)
    line_collection = Line3DCollection(lines, colors=cmap(norm(line_colors)), linewidths=3)
    refined_line_collection = Line3DCollection(refined_lines, colors=cmap(norm(refined_colors)), linewidths=2)

    
#----------------------------------------Add Map-----------------------------------------------
    
    x_grid = data[:, 0, 0, 0]
    y_grid = data[0, :, 0, 1]
    min_x, max_x = x_grid.min(), x_grid.max()
    min_y, max_y = y_grid.min(), y_grid.max()

    bg = mpimg.imread('/home/hj/K-UAM/code/grc_bbox_cropped.png')  # shape: (H, W, 3)
    step = 5
    bg_small = bg[::step, ::step, :]
    bg_small = np.flipud(bg_small)
    h, w, _ = bg_small.shape

    xs_img = np.linspace(min_x, max_x, w)
    ys_img = np.linspace(min_y, max_y, h)
    X, Y = np.meshgrid(xs_img, ys_img)
    Z = np.zeros_like(X)  

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    ELEV, AZIM = 90, -90
    
    pt_start_marker_size = 20

    def init():
        ax.plot_surface(
            X, Y, Z,
            rstride=1, cstride=1,
            facecolors=bg_small,
            shade=False
        )
        
        # ax.add_collection3d(line_collection)  
        
        ax.add_collection3d(refined_line_collection)
        
        # ax.scatter(xs, ys, zs, c=risks, norm = norm, cmap='seismic', s=60)
        # ax.scatter(refined_xs, refined_ys, refined_zs, c=refined_risk_norm, cmap='seismic', s=pt_start_marker_size, marker='o')
        ax.scatter(refined_xs, refined_ys, refined_zs, c=refined_risks, norm = norm, cmap='seismic', s=pt_start_marker_size, marker='o')        
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(0, max(zs)*1.1)
        return []
    
    def update(frame):
        ax.cla() 

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=bg_small, shade=False)
        
        # ax.add_collection3d(line_collection)
        
        ax.add_collection3d(refined_line_collection)
        
        # ax.scatter(xs, ys, zs, c=risks, norm = norm, cmap='seismic', s=60)
        # ax.scatter(refined_xs, refined_ys, refined_zs, c=refined_risk_norm, cmap='seismic', s=pt_start_marker_size, marker='o')
        ax.scatter(refined_xs, refined_ys, refined_zs, c=refined_risks, norm = norm, cmap='seismic', s=pt_start_marker_size, marker='o')


        pos = plane_positions[frame]    
        refined_segs = []
        refined_emergnency_risks = []
        avg_risks = []

        for site in emergency_sites:
            p1 = np.array(pos)
            p2 = np.array(site)
            segment_length = np.linalg.norm(p2 - p1)
            num_subsegments = max(1, int(segment_length // segment_detail_interval))

            subsegment_risks = []
            for i in range(num_subsegments):
                alpha_start = i / num_subsegments
                alpha_end = (i + 1) / num_subsegments
                pt_start = (1 - alpha_start) * p1 + alpha_start * p2
                pt_end = (1 - alpha_end) * p1 + alpha_end * p2

                try:
                    risk = interpolate_risk(data, *pt_start, *pt_end)
                except SystemExit:
                    continue

                refined_segs.append([pt_start.tolist(), pt_end.tolist()])
                refined_emergnency_risks.append(risk)
                subsegment_risks.append(risk)

            avg_risks.append(np.mean(subsegment_risks) if subsegment_risks else 0.0)
            
        lc = Line3DCollection(refined_segs, colors=cmap(norm(refined_emergnency_risks)), linewidths=2, linestyles=':')
        ax.add_collection3d(lc)
        
        
        seg_starts = [seg[0] for seg in refined_segs]
        seg_xs = [p[0] for p in seg_starts]
        seg_ys = [p[1] for p in seg_starts]
        seg_zs = [p[2] for p in seg_starts]

        # ax.scatter(seg_xs, seg_ys, seg_zs,
        #         c=refined_emergnency_risks,
        #         norm=norm,
        #         cmap='seismic',
        #         s=10,
        #         marker='o')

        ax.scatter(
            [x for x,_,_ in emergency_sites],
            [y for _,y,_ in emergency_sites],
            [z for _,_,z in emergency_sites],
            c=cmap(norm(avg_risks)),
            edgecolor=cmap(norm(avg_risks)),
            s=600, marker='o', depthshade=False, zorder=10
        )
        
        ax.scatter(pos[0], pos[1], pos[2], color='yellow', s=100)

        # ax.view_init(elev=90, azim=-90)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(0, max(zs)*1.1)
        
        return []

    ax.plot_surface(
        X, Y, Z,
        rstride=1, cstride=1,
        facecolors=bg_small,
        shade=False
    )

    # ax.add_collection3d(line_collection)
    ax.add_collection3d(refined_line_collection)  

    # sc = ax.scatter(xs, ys, zs, c=risk_norm, cmap='seismic', s=60)
    # sc = ax.scatter(refined_xs, refined_ys, refined_zs, c=refined_risk_norm, cmap='seismic', s=pt_start_marker_size, marker='o')
    sc = ax.scatter(refined_xs, refined_ys, refined_zs, c=refined_risks, norm = norm, cmap='seismic', s=pt_start_marker_size, marker='o')



    ax.view_init(elev=90, azim=-90)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(0, max(zs)*1.1)

    plt.colorbar(sc, ax=ax, label='Ground Risk (0–10)')
    ax.set_xlabel('X (TM)')
    ax.set_ylabel('Y (TM)')
    ax.set_zlabel('Altitude')
    ax.set_title('3D Ground Risk Along Corridor Segments')
    plt.tight_layout()
    anim = FuncAnimation(fig, update, init_func=init, frames=len(plane_positions), interval=1000, blit=False)
    plt.show()
        
#----------------------------------------Add Map-----------------------------------------------
    # # 2D top-down view with background
    
    fig2d = plt.figure()
    ax2d = fig2d.add_subplot(111)

    ax2d.imshow(
        bg_small,
        extent=[min_x, max_x, min_y, max_y],
        origin='lower',
        aspect='auto',
        zorder=1
    )

    # 3D static path in 2D
    for ((x1, y1, _), (x2, y2, _)), r in zip(segments, risks):
        ax2d.plot([x1, x2], [y1, y2], color=cmap(norm(r)), linewidth=2, zorder=2)

    ax2d.scatter(xs, ys, c=risks, norm=norm, cmap='seismic', s=60, label='Segment', zorder=2)

    for ((x1, y1, _), (x2, y2, _)), r in zip(refined_segments, refined_risks):
        ax2d.plot([x1, x2], [y1, y2], color=cmap(norm(r)), linewidth=1, linestyle='--', zorder=2)

    ax2d.scatter(refined_xs, refined_ys, c=refined_risks, norm=norm, cmap='seismic', s=pt_start_marker_size, marker='o', zorder=2)

    plane_dot_2d = ax2d.scatter([], [], color='yellow', s=100, zorder=3)
    seg_dots_2d = ax2d.scatter([], [], c=[], cmap='seismic', norm=norm, s=10, marker='o', zorder=2)
    seg_lines_2d = []
    site_dots_2d = None


    def update_2d(frame):
        global seg_lines_2d, site_dots_2d
        for l in seg_lines_2d:
            l.remove()
        seg_lines_2d = []
        if site_dots_2d:
            site_dots_2d.remove()

        pos = plane_positions[frame]
        refined_segs = []
        refined_emergency_risks = []
        avg_risks = []

        for site in emergency_sites:
            p1 = np.array(pos)
            p2 = np.array(site)
            segment_length = np.linalg.norm(p2 - p1)
            num_subsegments = max(1, int(segment_length // segment_detail_interval))

            subsegment_risks = []
            for i in range(num_subsegments):
                alpha_start = i / num_subsegments
                alpha_end = (i + 1) / num_subsegments
                pt_start = (1 - alpha_start) * p1 + alpha_start * p2
                pt_end = (1 - alpha_end) * p1 + alpha_end * p2

                try:
                    risk = interpolate_risk(data, *pt_start, *pt_end)
                except SystemExit:
                    continue

                refined_segs.append((pt_start, pt_end))
                refined_emergency_risks.append(risk)
                subsegment_risks.append(risk)

            avg_risks.append(np.mean(subsegment_risks) if subsegment_risks else 0.0)

        for (start, end), risk in zip(refined_segs, refined_emergency_risks):
            (x1, y1, _), (x2, y2, _) = start, end
            line = ax2d.plot([x1, x2], [y1, y2], color=cmap(norm(risk)), linewidth=1, linestyle=':', zorder=2)[0]
            seg_lines_2d.append(line)

        plane_dot_2d.set_offsets([[pos[0], pos[1]]])

        starts = np.array([s[0] for s in refined_segs])
        if len(starts) > 0:
            seg_dots_2d.set_offsets(starts[:, :2])
            seg_dots_2d.set_array(np.array(refined_emergency_risks))

        site_dots_2d = ax2d.scatter(
            [x for x, _, _ in emergency_sites],
            [y for _, y, _ in emergency_sites],
            c=cmap(norm(avg_risks)),
            edgecolor=cmap(norm(avg_risks)),
            s=600, marker='o', zorder=4
        )
        return []

    fig2d.colorbar(seg_dots_2d, ax=ax2d, label='Ground Risk (0–10)')
    ax2d.set_xlabel('X (TM)')
    ax2d.set_ylabel('Y (TM)')
    ax2d.set_title('2D Top-Down Ground Risk View')
    plt.legend()

    anim2d = FuncAnimation(fig2d, update_2d, frames=len(plane_positions), interval=1000, blit=False)
    plt.show()