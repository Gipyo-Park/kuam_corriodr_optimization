import numpy as np
from scipy.ndimage import map_coordinates
from tqdm import tqdm

# Added matplotlib and cartopy for optional plotting
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

# Helper function files that should be in the same directory
from crossover_GP import crossover_gp
from mutation_GP import mutation_gp
from fast_non_dominated_sort import fast_non_dominated_sort
from generate_initial_population_GP import generate_initial_population_gp
from generate_reference_points_2obj import generate_reference_points_2obj
from normalize_objectives import normalize_objectives
from niching_selection import niching_selection
from evaluate_objectives_with_constraints_GP import evaluate_objectives_with_constraints_gp

# =============================================================================
# 1. Engine Configuration & Global Data Loading
# =============================================================================
# --- Parameters ---
W_GROUND, W_AIR = 0.5, 1.0
ALTITUDE_LEVELS = np.array([500])
USE_HEADING_MAP = True
W_BUF = 1000.0
NODE_GRID_RESOLUTION_M = 100.0
AIR_RISK_THRESHOLD = 0.5
MIN_SAFE_NODES_TARGET = 500
SAFE_NODE_PERCENTILE_LIST = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
MAX_INIT_ATTEMPTS = 100
CELL_SIZE, REFINE_SCALES, DELTA_Z_MAX = 100.0, np.array([1.0, 0.5, 0.2, 0.1]), 100.0
NMAX, N_POP, OFFSPRING_RATIO, H_REF_POINTS = 50, 50, 2.0, 20
FLIGHT_DIST_LIMIT = 100.0

# --- Fixed Data ---
FORBIDDEN_ZONES = np.array([[129.08, 129.10, 35.59, 35.61], [129.11, 129.12, 35.59, 35.63], [129.12, 129.13, 35.62, 35.63]])
LAT_LIM = [35.5446, 35.6427]
LON_LIM = [129.0514, 129.1436]

# --- Global Variables for Loaded Data ---
Norm_RiskTensor = None
AirRisk = None
Ny, Nx = 0, 0

def _load_risk_maps():
    """Load and preprocess risk maps once. This is a heavy operation."""
    global Norm_RiskTensor, AirRisk, Ny, Nx

    if Norm_RiskTensor is not None and AirRisk is not None:
        print("Risk maps already loaded.")
        return

    # --- Load Ground Risk ---
    print('Loading unified ground/population risk map...')
    try:
        pop_risk_raw = np.load('high_res_affected_population_GRC.npy', allow_pickle=True)
        selected_scenario = pop_risk_raw[:, :, 0, 3:]
        Ny, Nx, H = selected_scenario.shape
        A = len(ALTITUDE_LEVELS)
        RiskTensor = np.zeros((A, H, Ny, Nx))
        for ai, alt in enumerate(ALTITUDE_LEVELS):
            for hi in range(H):
                RiskTensor[ai, hi, :, :] = selected_scenario[:, :, hi]
        
        min_val = np.min(RiskTensor)
        RiskTensor_shifted = RiskTensor - min_val
        max_val = np.max(RiskTensor_shifted)
        Norm_RiskTensor = RiskTensor_shifted / max_val if max_val > 0 else RiskTensor_shifted
        print('Ground/Population risk map loaded and normalized.')
    except FileNotFoundError:
        print("FATAL ERROR: high_res_affected_population_GRC.npy not found.")
        raise

    # --- Load Air Risk ---
    print('Loading air risk map...')
    try:
        air_risk_raw = np.load('AirRisk_combined_max_risk_map.npy', allow_pickle=True).item()
        z_vec_air = air_risk_raw['z_vec']
        raw_air_risk_tensor = air_risk_raw['Risk_3d']
        
        if raw_air_risk_tensor.shape[0] == Nx and raw_air_risk_tensor.shape[1] == Ny:
            raw_air_risk_tensor = np.transpose(raw_air_risk_tensor, (1, 0, 2))
        
        AirRisk = np.zeros((Ny, Nx, len(ALTITUDE_LEVELS)))
        for i, alt in enumerate(ALTITUDE_LEVELS):
            air_alt_idx = np.argmin(np.abs(z_vec_air - alt))
            AirRisk[:, :, i] = raw_air_risk_tensor[:, :, air_alt_idx]
        print('Air risk map loaded and aligned.')
    except FileNotFoundError:
        print("FATAL ERROR: AirRisk_combined_max_risk_map.npy not found.")
        raise

# =============================================================================
# 2. Core Logic Functions (from MAIN_for_UNIST_MAP_GP.py)
# =============================================================================
# All helper functions like generate_nodes_3d_segment, run_nsga3_segment, etc.
# are placed here without modification, except for removing visualization parts.

def generate_nodes_3d_segment(p1, p2, W_buf, node_grid_resolution_m, lat_lim, lon_lim, Ny, Nx, forbidden_zones):
    minLat, maxLat = lat_lim
    minLon, maxLon = lon_lim
    dLat_deg = (maxLat - minLat) / (Ny - 1)
    dLon_deg = (maxLon - minLon) / (Nx - 1)
    j1, i1 = (p1[0] - minLat) / dLat_deg, (p1[1] - minLon) / dLon_deg
    j2, i2 = (p2[0] - minLat) / dLat_deg, (p2[1] - minLon) / dLon_deg
    p1_grid, p2_grid = np.array([i1, j1]), np.array([i2, j2])
    vec = p2_grid - p1_grid
    len_val = np.linalg.norm(vec)
    if len_val < 1e-6: return np.array([]), np.array([])
    u_vec = vec / len_val
    v_vec = np.array([-u_vec[1], u_vec[0]])
    mean_lat_rad = np.deg2rad(np.mean([p1[0], p2[0]]))
    meters_per_lon_deg = 111000 * np.cos(mean_lat_rad)
    meters_per_unit_u = np.sqrt((u_vec[0] * dLon_deg * meters_per_lon_deg)**2 + (u_vec[1] * dLat_deg * 111000)**2)
    meters_per_unit_v = np.sqrt((v_vec[0] * dLon_deg * meters_per_lon_deg)**2 + (v_vec[1] * dLat_deg * 111000)**2)
    len_m = len_val * meters_per_unit_u
    n_s = max(1, round(len_m / node_grid_resolution_m))
    s_vec_m = np.linspace(0, len_m, n_s)
    n_t = max(1, round(2 * W_buf / node_grid_resolution_m))
    t_vec_m = np.linspace(-W_buf, W_buf, n_t)
    S_m, T_m = np.meshgrid(s_vec_m, t_vec_m)
    S_idx, T_idx = S_m / meters_per_unit_u, T_m / meters_per_unit_v
    I = p1_grid[0] + S_idx * u_vec[0] + T_idx * v_vec[0]
    J = p1_grid[1] + S_idx * u_vec[1] + T_idx * v_vec[1]
    valid_mask = (I >= 0) & (I < Nx) & (J >= 0) & (J < Ny)
    Ii, Ji = I[valid_mask].flatten(), J[valid_mask].flatten()
    all_nodes_lon = minLon + Ii * dLon_deg
    all_nodes_lat = minLat + Ji * dLat_deg
    all_nodes_alt = np.full_like(all_nodes_lat, p1[2])
    all_grid_nodes = np.vstack([all_nodes_lat, all_nodes_lon, all_nodes_alt]).T
    if forbidden_zones is not None and forbidden_zones.shape[0] > 0 and all_grid_nodes.shape[0] > 0:
        lats, lons = all_grid_nodes[:, 0], all_grid_nodes[:, 1]
        is_valid_node = np.ones(all_grid_nodes.shape[0], dtype=bool)
        for rect in forbidden_zones:
            min_lon, max_lon, min_lat, max_lat = rect
            in_rect = (lons >= min_lon) & (lons <= max_lon) & (lats >= min_lat) & (lats <= max_lat)
            is_valid_node[in_rect] = False
        nodes = all_grid_nodes[is_valid_node]
    else:
        nodes = all_grid_nodes
    return nodes, all_grid_nodes

# Added plot_solutions helper function for visualization
def plot_solutions(gx, solutions, styles, width, seg_num=None, labels=None):
    """
    Plot multiple solution paths on a cartopy map axis.

    Args:
        gx: Cartopy GeoAxes object
        solutions: List of paths or dict of paths
        styles: List of line styles (e.g., ['r-', 'b--', 'g-.'])
        width: Line width
        seg_num: Segment number for labeling
        labels: List of labels for each solution type

    Returns:
        List of line handles
    """
    h = []
    if gx is None:
        return h
    if labels is None:
        labels = ['Risk', 'Dist', 'Pareto']

    data = list(solutions.values()) if isinstance(solutions, dict) else solutions
    for i, path_group in enumerate(data):
        paths = path_group if isinstance(path_group, list) else [path_group]
        for path_idx, path in enumerate(paths):
            if path is not None and len(path) > 0:
                path_arr = np.array(path)
                label = None
                if path_idx == 0 and seg_num is not None and i < len(labels):
                    label = f'Seg{seg_num} {labels[i]} Rep.'
                style = styles[i % len(styles)]
                line, = gx.plot(path_arr[:, 1], path_arr[:, 0], style,
                               linewidth=width, label=label,
                               transform=ccrs.Geodetic(), zorder=10)
                h.append(line)
    return h

def select_representative_paths(population, fvals, p1, p2):
    if not population or fvals.shape[0] == 0:
        fallback = np.vstack([p1, p2])
        return fallback, fallback, fallback
    rep_r = population[np.argmin(fvals[:, 1])]
    rep_d = population[np.argmin(fvals[:, 0])]
    F = fast_non_dominated_sort(fvals)
    if not F or not F[0]:
        rep_p = rep_r
    else:
        front1 = F[0]
        f_pareto = fvals[front1, :]
        if f_pareto.shape[0] > 1:
            norm_f = normalize_objectives(f_pareto)
            best_idx = front1[np.argmin(np.linalg.norm(norm_f, axis=1))]
        else:
            best_idx = front1[0]
        rep_p = population[best_idx]
    return rep_r, rep_d, rep_p

def selection_nsga3(population, f_vals, feasible, N, ref_points):
    Fronts = fast_non_dominated_sort(f_vals)
    next_pop_indices = []
    for front in Fronts:
        valid = [idx for idx in front if feasible[idx]]
        if not valid: continue
        if len(next_pop_indices) + len(valid) <= N:
            next_pop_indices.extend(valid)
        else:
            remaining = N - len(next_pop_indices)
            last_front = np.array(valid)
            if last_front.shape[0] > 0 and remaining > 0:
                last_front_fvals = f_vals[last_front, :]
                norm_f = normalize_objectives(last_front_fvals)
                selected = niching_selection(norm_f, ref_points, remaining)
                next_pop_indices.extend(last_front[selected])
            break
    return [population[i] for i in next_pop_indices[:N]]

def variation_nsga3(pop, nodes, ratio):
    if not pop: return []
    offspring_num = round(len(pop) * ratio)
    offspring = []
    for _ in range(offspring_num):
        p1_idx, p2_idx = np.random.choice(len(pop), 2)
        child = crossover_gp(pop[p1_idx], pop[p2_idx])
        offspring.append(mutation_gp(child, nodes))
    return offspring

def run_nsga3_segment(nodes, p1, p2):
    population = []
    for attempt in range(MAX_INIT_ATTEMPTS):
        temp_population = generate_initial_population_gp(N_POP, nodes, p1, p2)
        if not temp_population: continue
        is_any_feasible = False
        for path in temp_population:
            _, feasible = evaluate_objectives_with_constraints_gp(path, Norm_RiskTensor, AirRisk, USE_HEADING_MAP, FLIGHT_DIST_LIMIT, FORBIDDEN_ZONES, DELTA_Z_MAX, ALTITUDE_LEVELS, CELL_SIZE, REFINE_SCALES, AIR_RISK_THRESHOLD, W_GROUND, W_AIR, LAT_LIM, LON_LIM)
            if feasible:
                is_any_feasible = True
                break
        if is_any_feasible:
            print(f"Found a feasible initial population on attempt {attempt + 1}.")
            population = temp_population
            break
        else:
            print(f"Attempt {attempt + 1}: Initial population has no feasible solutions. Retrying...")
    if not population:
        print("Failed to generate a feasible initial population after all attempts.")
        return [], np.array([])

    ref_points = generate_reference_points_2obj(H_REF_POINTS)
    for gen in range(1, NMAX + 1):
        print(f'  - Generation {gen}/{NMAX}')
        Np = len(population)
        f_vals, feasible = np.zeros((Np, 2)), np.zeros(Np, dtype=bool)
        for i in range(Np):
            f_vals[i, :], feasible[i] = evaluate_objectives_with_constraints_gp(population[i], Norm_RiskTensor, AirRisk, USE_HEADING_MAP, FLIGHT_DIST_LIMIT, FORBIDDEN_ZONES, DELTA_Z_MAX, ALTITUDE_LEVELS, CELL_SIZE, REFINE_SCALES, AIR_RISK_THRESHOLD, W_GROUND, W_AIR, LAT_LIM, LON_LIM)
        
        new_pop = selection_nsga3(population, f_vals, feasible, N_POP, ref_points)
        if gen < NMAX:
            if not new_pop:
                print("Warning: New population is empty. Stopping early.")
                population = []
                break
            population = new_pop + variation_nsga3(new_pop, nodes, OFFSPRING_RATIO)
        else:
            population = new_pop
    if not population:
        return [], np.array([])

    f_vals_final = np.zeros((len(population), 2))
    for i in range(len(population)):
        f_vals_final[i,:], _ = evaluate_objectives_with_constraints_gp(population[i], Norm_RiskTensor, AirRisk, USE_HEADING_MAP, FLIGHT_DIST_LIMIT, FORBIDDEN_ZONES, DELTA_Z_MAX, ALTITUDE_LEVELS, CELL_SIZE, REFINE_SCALES, AIR_RISK_THRESHOLD, W_GROUND, W_AIR, LAT_LIM, LON_LIM)
    return population, f_vals_final

# =============================================================================
# 3. Main Engine Function
# =============================================================================
def find_optimal_path(start_point, end_point, corridor_points=None, enable_plot=False, plot_save_path=None):
    """
    Calculates the optimal path from a start point to an end point,
    optionally via corridor points.

    Args:
        start_point (list): [lat, lon, alt]
        end_point (list): [lat, lon, alt]
        corridor_points (list of lists, optional): [[lat, lon, alt], ...]
        enable_plot (bool): Enable visualization (default: False)  # 
        plot_save_path (str): Path to save plot image (default: 'optimal_path.png')  # 

    Returns:
        list: A list of waypoints for the pareto-optimal path, or an empty list if no path is found.
    """
    # Ensure risk maps are loaded before proceeding
    _load_risk_maps()

    # --- Construct points array ---
    if corridor_points:
        points = np.vstack([start_point] + corridor_points + [end_point])
    else:
        points = np.vstack([start_point, end_point])

    # Initialize plot if enabled
    fig, gx = None, None
    if enable_plot:
        try:
            plt.ion()
            fig = plt.figure('Optimal Path Finding', figsize=(14, 14))
            request = cimgt.OSM()
            gx = fig.add_subplot(1, 1, 1, projection=request.crs)
            gx.set_extent([LON_LIM[0], LON_LIM[1], LAT_LIM[0], LAT_LIM[1]], crs=ccrs.Geodetic())
            gx.add_image(request, 13)

            # Plot start and end points
            gx.plot(start_point[1], start_point[0], 'go', markersize=12,
                   label='Start', transform=ccrs.Geodetic(), zorder=15)
            gx.plot(end_point[1], end_point[0], 'ro', markersize=12,
                   label='End', transform=ccrs.Geodetic(), zorder=15)

            # Plot corridor points if any
            if corridor_points:
                for i, cp in enumerate(corridor_points):
                    gx.plot(cp[1], cp[0], 'co', markersize=10,
                           label=f'Corridor {i+1}' if i == 0 else '',
                           transform=ccrs.Geodetic(), zorder=15)

            gx.legend(loc='best', fontsize=10)
            gx.set_title('Initializing...', size=16)
            plt.pause(0.1)
            print("Plot initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize plot: {e}")
            enable_plot = False
            fig, gx = None, None

    # --- Per-Segment NSGA-III Optimization ---
    num_segments = points.shape[0] - 1
    representative_paths = [[None]*3 for _ in range(num_segments)]

    for k in range(num_segments):
        p1, p2 = points[k, :], points[k + 1, :]
        print(f'\nProcessing Segment {k+1}/{num_segments}...')
        
        # 1. Generate candidate nodes
        nodes, all_grid_nodes = generate_nodes_3d_segment(p1, p2, W_BUF, NODE_GRID_RESOLUTION_M, LAT_LIM, LON_LIM, Ny, Nx, FORBIDDEN_ZONES)
        
        # 2. Dynamically adjust and filter for safe nodes
        half_all_nodes = all_grid_nodes.shape[0] // 2
        base_target = int(max(MIN_SAFE_NODES_TARGET, half_all_nodes))
        current_min_safe_nodes_target = min(base_target, nodes.shape[0])
        print(f"Dynamic MIN_SAFE_NODES_TARGET: {current_min_safe_nodes_target}")

        node_risks = np.zeros(nodes.shape[0])
        if nodes.shape[0] > 0:
            node_lons, node_lats, node_alts = nodes[:, 1], nodes[:, 0], nodes[:, 2]
            I_nodes = ((node_lons - LON_LIM[0]) / (LON_LIM[1] - LON_LIM[0]) * (Nx - 1)).astype(int)
            J_nodes = ((node_lats - LAT_LIM[0]) / (LAT_LIM[1] - LAT_LIM[0]) * (Ny - 1)).astype(int)
            alt_idx = np.argmin(np.abs(ALTITUDE_LEVELS - node_alts[0]))
            avg_ground_risk_map = np.mean(Norm_RiskTensor[alt_idx, :, :, :], axis=0)
            ground_risks = avg_ground_risk_map[J_nodes, I_nodes]
            air_risks = AirRisk[:, :, alt_idx][J_nodes, I_nodes]
            node_risks = W_GROUND * ground_risks + W_AIR * air_risks

        safe_nodes = np.array([])
        print("Starting adaptive search for safe nodes...")
        if nodes.shape[0] > 0:
            for percentile in SAFE_NODE_PERCENTILE_LIST:
                risk_threshold = np.percentile(node_risks, percentile)
                current_mask = node_risks <= risk_threshold
                safe_nodes = nodes[current_mask]
                print(f'  - At {percentile}% percentile, found {safe_nodes.shape[0]} nodes.')
                if safe_nodes.shape[0] >= current_min_safe_nodes_target:
                    print(f'-> Target of {current_min_safe_nodes_target} nodes met.')
                    break
            else:
                print(f'-> Target not met. Using last found nodes: {safe_nodes.shape[0]} nodes.')
        
        if safe_nodes.shape[0] == 0:
            print("Warning: No safe nodes found. Using all generated nodes.")
            safe_nodes = nodes
        
        # 3. Run optimization
        print(f'Starting optimization with {safe_nodes.shape[0]} nodes...')
        population, f_vals = run_nsga3_segment(safe_nodes, p1, p2)
        
        # 4. Store representative paths for this segment
        rep_r, rep_d, rep_p = select_representative_paths(population, f_vals, p1, p2)
        representative_paths[k] = [rep_r, rep_d, rep_p]
        print(f'Segment {k+1} Complete.')

        # Plot segment results if enabled
        if enable_plot and gx is not None:
            try:
                plot_solutions(gx, [rep_r, rep_d, rep_p],
                              ['r-', 'b-', 'g-'], 2.0, k+1)
                gx.set_title(f'Segment {k+1}/{num_segments} Complete', size=16)
                gx.legend(loc='best', fontsize=10)
                plt.pause(0.5)
            except Exception as e:
                print(f"Warning: Failed to plot segment {k+1}: {e}")

    # --- Assemble final pareto path ---
    final_pareto_path = []
    for k in range(num_segments):
        pareto_segment = representative_paths[k][2] # Index 2 is for pareto path
        if pareto_segment is not None and len(pareto_segment) > 0:
            # Avoid duplicating the connection point
            if k > 0 and len(final_pareto_path) > 0:
                final_pareto_path.extend(pareto_segment[1:])
            else:
                final_pareto_path.extend(pareto_segment)
    
    # Convert numpy arrays to standard lists for JSON serialization
    if not final_pareto_path:
        print("Could not find a final path.")
        # Close plot if enabled
        if enable_plot and fig is not None:
            plt.close(fig)
        return []

    final_path_list = [list(waypoint) for waypoint in final_pareto_path]
    print(f"Final Pareto path found with {len(final_path_list)} waypoints.")

    # Save final plot if enabled
    if enable_plot and fig is not None:
        try:
            if plot_save_path is None:
                plot_save_path = 'optimal_path.png'

            gx.set_title(f'Final Optimal Path ({len(final_path_list)} waypoints)', size=16)
            plt.savefig(plot_save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plot saved to: {plot_save_path}")
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Failed to save plot: {e}")
            if fig is not None:
                plt.close(fig)

    return final_path_list

if __name__ == '__main__':
    # This is for direct testing of the engine
    print("--- Direct Engine Test ---")
    test_start = [35.5845361, 129.1076472, 500]
    test_end = [35.6249583, 129.1335528, 500]
    
    optimal_path = find_optimal_path(test_start, test_end)
    
    if optimal_path:
        print("\n--- Optimal Path Found ---")
        for point in optimal_path:
            print(point)
    else:
        print("\n--- No Optimal Path Found ---")