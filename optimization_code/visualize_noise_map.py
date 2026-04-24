"""
Visualize Noise Risk Map from noise_output_lden.csv

This script demonstrates how v14 integrates noise data:
- Reads noise CSV with receiver_id -> grid mapping
- Remaps receiver_id (flat index) to 2D grid coordinates
- Creates georeferenced visualization aligned with lat/lon
- Shows how noise risk overlays on the UAM corridor

Key insights:
- receiver_id is the flat (snake/raster) index: flat_idx = receiver_id - base
- Grid coordinates: j = flat_idx // Nx (row/lat), i = flat_idx % Nx (col/lon)
- Automatic 0/1-based indexing detection
- WGS84 coordinate system for South Korea (Ulsan area)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata


# Region of Interest (same as ground/air risk maps)
CODE_LAT_LIM = [35.5446, 35.6427]
CODE_LON_LIM = [129.0514, 129.1436]


def load_ground_risk_shape(grc_path="Modified_high_res_affected_population_GRC.npy"):
    """
    Load ground risk data to determine grid dimensions Ny, Nx.
    These are the same dimensions used by noise risk grid.
    """
    if not os.path.exists(grc_path):
        print(f"Warning: GRC file not found at {grc_path}")
        return None, None
    
    try:
        data = np.load(grc_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item() if isinstance(data.item(), dict) else data
        
        # GRC shape: [Ny, Nx, 1, num_scenarios]
        if isinstance(data, dict):
            grc_3d = data.get('Risk_3d') or data
        else:
            grc_3d = data
            
        if grc_3d.ndim >= 2:
            Ny, Nx = grc_3d.shape[0], grc_3d.shape[1]
            print(f"Grid dimensions from GRC: Ny={Ny}, Nx={Nx}")
            return Ny, Nx
    except Exception as e:
        print(f"Error loading GRC file: {e}")
    
    return None, None


def detect_grid_base_from_csv(csv_path):
    """
    Detect whether receiver_id uses 0-based or 1-based indexing.
    Returns the offset to convert receiver_id to flat_idx.
    """
    df = pd.read_csv(csv_path)
    min_id = df['receiver_id'].min()
    max_id = df['receiver_id'].max()
    count = len(df['receiver_id'].unique())
    
    print(f"Receiver ID range: {min_id} to {max_id} (count: {count})")
    
    # If min_id == 1, use 1-based indexing (subtract 1 to get flat_idx)
    # If min_id == 0, use 0-based indexing (no offset)
    if min_id == 0:
        print("Detected 0-based indexing (receiver_id starts at 0)")
        return 0
    else:
        print(f"Detected 1-based indexing (receiver_id starts at {min_id})")
        return min_id


def remap_receiver_to_grid(csv_path, Ny, Nx, grc_path=None):
    """
    Remap receiver_id (flat index) to 2D grid coordinates.
    
    Returns:
        noise_grid: 2D array [Ny, Nx] with Lden values
        receiver_metadata: dict with mapping info
    """
    df = pd.read_csv(csv_path)
    
    base = detect_grid_base_from_csv(csv_path)
    
    print(f"\nRemapping {len(df)} receivers to grid [{Ny}, {Nx}]...")
    
    # Initialize grid with NaN
    noise_grid = np.full((Ny, Nx), np.nan, dtype=float)
    
    # Count receivers
    mapped_count = 0
    out_of_bounds = 0
    
    for _, row in df.iterrows():
        receiver_id = row['receiver_id']
        lden_db = row['Lden_db']
        
        # Convert receiver_id to flat_idx
        flat_idx = int(receiver_id) - base
        
        # Convert flat_idx to 2D (row, col)
        j = flat_idx // Nx  # row / latitude
        i = flat_idx % Nx   # col / longitude
        
        # Sanity check
        if 0 <= j < Ny and 0 <= i < Nx:
            noise_grid[j, i] = lden_db
            mapped_count += 1
        else:
            out_of_bounds += 1
            if out_of_bounds <= 5:
                print(f"  Out-of-bounds: receiver_id={receiver_id}, flat_idx={flat_idx}, j={j}, i={i}")
    
    print(f"Mapped: {mapped_count} | Out-of-bounds: {out_of_bounds} | Total: {len(df)}")
    
    # Load GRC for comparison if available
    grc_shape = None
    if grc_path and os.path.exists(grc_path):
        try:
            grc_data = np.load(grc_path, allow_pickle=True)
            if isinstance(grc_data, np.ndarray) and grc_data.dtype == object:
                grc_data = grc_data.item() if isinstance(grc_data.item(), dict) else grc_data
            grc_shape = grc_data.shape if hasattr(grc_data, 'shape') else None
        except:
            pass
    
    metadata = {
        'base_indexing': base,
        'grid_shape': (Ny, Nx),
        'mapped_count': mapped_count,
        'out_of_bounds': out_of_bounds,
        'nan_ratio': np.isnan(noise_grid).sum() / (Ny * Nx),
        'grc_shape': grc_shape,
    }
    
    return noise_grid, metadata


def create_lat_lon_grids(Ny, Nx):
    """
    Create latitude and longitude grids matching CODE region.
    
    Returns:
        lat_2d, lon_2d: 2D arrays of lat/lon coordinates
    """
    lat = np.linspace(CODE_LAT_LIM[0], CODE_LAT_LIM[1], Ny)
    lon = np.linspace(CODE_LON_LIM[0], CODE_LON_LIM[1], Nx)
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    return lat_2d, lon_2d


def analyze_noise_data(noise_grid):
    """
    Print detailed statistics about noise data distribution.
    """
    finite = noise_grid[np.isfinite(noise_grid)]
    
    if finite.size == 0:
        print("\nNo valid noise data found!")
        return
    
    print("\n[Noise Data Analysis]")
    print(f"- Valid points: {finite.size}")
    print(f"- NaN points: {np.isnan(noise_grid).sum()}")
    print(f"- Min Lden: {finite.min():.2f} dB")
    print(f"- Max Lden: {finite.max():.2f} dB")
    print(f"- Mean Lden: {finite.mean():.2f} dB")
    print(f"- Median Lden: {np.median(finite):.2f} dB")
    
    p1, p25, p50, p75, p99 = np.percentile(finite, [1, 25, 50, 75, 99])
    print(f"- Percentiles (p1/p25/p50/p75/p99): {p1:.2f} / {p25:.2f} / {p50:.2f} / {p75:.2f} / {p99:.2f}")
    
    # Check floor behavior
    floor_mask = finite == 0.0
    if floor_mask.any():
        print(f"- Points at 0.0 dB floor: {floor_mask.sum()} ({floor_mask.sum()/len(finite)*100:.1f}%)")
    
    negative_mask = finite < 0.0
    if negative_mask.any():
        print(f"- Negative values (< 0 dB): {negative_mask.sum()} (expected for low-footprint areas)")


def _draw_roi_box(ax):
    """Draw the CODE region box on map."""
    rect = Rectangle(
        (CODE_LON_LIM[0], CODE_LAT_LIM[0]),
        CODE_LON_LIM[1] - CODE_LON_LIM[0],
        CODE_LAT_LIM[1] - CODE_LAT_LIM[0],
        fill=False,
        edgecolor="cyan",
        linewidth=2.0,
        linestyle="--",
        label="Main CODE extent",
    )
    ax.add_patch(rect)


def visualize_noise_heatmap(noise_grid, save_path=None, show_plot=True):
    """
    Main heatmap visualization: noise grid overlaid on lat/lon map.
    """
    Ny, Nx = noise_grid.shape
    lat_2d, lon_2d = create_lat_lon_grids(Ny, Nx)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    
    # --- Subplot 1: Grid Index View (diagnose receiver_id mapping) ---
    ax = axes[0]
    im0 = ax.imshow(noise_grid, cmap='RdYlGn_r', origin='lower', interpolation='nearest')
    ax.set_title('Noise Grid Index View\n(diagnose receiver_id → grid mapping)')
    ax.set_xlabel('Column (X index)')
    ax.set_ylabel('Row (Y index)')
    fig.colorbar(im0, ax=ax, label='Lden (dB)')
    
    # --- Subplot 2: Full Georeferenced View (entire mapped area) ---
    ax = axes[1]
    im1 = ax.pcolormesh(lon_2d, lat_2d, noise_grid, cmap='RdYlGn_r', shading='auto')
    _draw_roi_box(ax)
    ax.set_title('Noise Source Georef - Full Extent\n(full mapped area)')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    fig.colorbar(im1, ax=ax, label='Lden (dB)')
    
    # --- Subplot 3: ROI Only (CODE region focus) ---
    ax = axes[2]
    im2 = ax.imshow(
        noise_grid,
        cmap='RdYlGn_r',
        origin='lower',
        extent=[CODE_LON_LIM[0], CODE_LON_LIM[1], CODE_LAT_LIM[0], CODE_LAT_LIM[1]],
        aspect='auto',
        interpolation='nearest',
    )
    ax.set_xlim(CODE_LON_LIM)
    ax.set_ylim(CODE_LAT_LIM)
    ax.set_title('Noise Georef - ROI Only\n(CODE corridor region)')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    fig.colorbar(im2, ax=ax, label='Lden (dB)')
    
    fig.suptitle(f'Noise Risk Map (Lden, dB) - v14 Integration Example', fontsize=16, fontweight='bold')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def visualize_noise_with_interpolation(noise_grid, save_path=None, show_plot=True):
    """
    Alternative visualization using interpolation for smoother appearance.
    Shows how bilinear interpolation works (as v14 uses for path evaluation).
    """
    Ny, Nx = noise_grid.shape
    lat_2d, lon_2d = create_lat_lon_grids(Ny, Nx)
    
    # Create finer grid for interpolation
    lat_fine = np.linspace(CODE_LAT_LIM[0], CODE_LAT_LIM[1], Ny * 3)
    lon_fine = np.linspace(CODE_LON_LIM[0], CODE_LON_LIM[1], Nx * 3)
    lon_fine_2d, lat_fine_2d = np.meshgrid(lon_fine, lat_fine)
    
    # Flatten for griddata
    points = np.column_stack([lat_2d.ravel(), lon_2d.ravel()])
    values = noise_grid.ravel()
    
    # Remove NaN for interpolation
    valid_mask = np.isfinite(values)
    if valid_mask.sum() < 3:
        print("Not enough valid points for interpolation")
        return
    
    points_valid = points[valid_mask]
    values_valid = values[valid_mask]
    
    # Interpolate on fine grid
    noise_interp = griddata(points_valid, values_valid, 
                           (lat_fine_2d, lon_fine_2d), 
                           method='linear', fill_value=np.nan)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    
    # Original (discrete)
    ax = axes[0]
    im1 = ax.imshow(
        noise_grid,
        cmap='RdYlGn_r',
        origin='lower',
        extent=[CODE_LON_LIM[0], CODE_LON_LIM[1], CODE_LAT_LIM[0], CODE_LAT_LIM[1]],
        aspect='auto',
        interpolation='nearest',
    )
    ax.set_title('Original Discrete Noise Grid\n(as used in v14 for bilinear interp)')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    fig.colorbar(im1, ax=ax, label='Lden (dB)')
    
    # Interpolated (smooth)
    ax = axes[1]
    im2 = ax.imshow(
        noise_interp,
        cmap='RdYlGn_r',
        origin='lower',
        extent=[CODE_LON_LIM[0], CODE_LON_LIM[1], CODE_LAT_LIM[0], CODE_LAT_LIM[1]],
        aspect='auto',
        interpolation='bilinear',
    )
    ax.set_title('Bilinear Interpolated\n(for smooth path sampling)')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    fig.colorbar(im2, ax=ax, label='Lden (dB)')
    
    fig.suptitle('Noise Interpolation: v14 Path Evaluation Method', fontsize=16, fontweight='bold')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved interpolation visualization: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def export_noise_grid_npy(noise_grid, metadata, output_path=None):
    """
    Export noise grid in NPY format (like ground/air risk maps).
    For potential integration into main optimization pipeline.
    """
    if output_path is None:
        output_path = os.path.join('noise_data', 'noise_lden_grid.npy')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data_dict = {
        'Risk_2d': noise_grid,  # Single altitude layer (noise is ground-level)
        'lat_lim': CODE_LAT_LIM,
        'lon_lim': CODE_LON_LIM,
        'metadata': metadata,
    }
    
    np.save(output_path, data_dict, allow_pickle=True)
    print(f"Exported noise grid to: {output_path}")


def main():
    """
    Main workflow: Load CSV → Remap → Analyze → Visualize
    """
    # Paths
    csv_path = os.path.join('noise_data', 'noise_output_lden.csv')
    grc_path = 'Modified_high_res_affected_population_GRC.npy'
    
    # Determine grid dimensions from ground risk file
    Ny, Nx = load_ground_risk_shape(grc_path)
    if Ny is None:
        print("Error: Could not determine grid dimensions")
        return
    
    # Remap receiver_id to 2D grid
    if not os.path.exists(csv_path):
        print(f"Error: Noise CSV file not found at {csv_path}")
        return
    
    noise_grid, metadata = remap_receiver_to_grid(csv_path, Ny, Nx, grc_path)
    
    # Analyze distribution
    analyze_noise_data(noise_grid)
    
    # Create visualizations
    output_dir = os.path.join('figure', 'noise_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    visualize_noise_heatmap(
        noise_grid,
        save_path=os.path.join(output_dir, 'noise_lden_heatmap_3view.png'),
        show_plot=False
    )
    
    visualize_noise_with_interpolation(
        noise_grid,
        save_path=os.path.join(output_dir, 'noise_lden_interpolation.png'),
        show_plot=False
    )
    
    # Export for potential integration
    export_noise_grid_npy(noise_grid, metadata)
    
    print("\n" + "="*70)
    print("v14 NOISE INTEGRATION REFERENCE")
    print("="*70)
    print("""
    KEY INSIGHTS:
    
    1. Data Format
       - CSV: corridor_id, receiver_id, Lden_db
       - Lden_db: Day-Evening-Night weighted long-term average (dB)
       - Negative values expected where noise floor < 0 dB
    
    2. Receiver ID to Grid Mapping
       - Auto-detect base: 0-based or 1-based indexing
       - flat_idx = receiver_id - base
       - j = flat_idx // Nx  (row / latitude index)
       - i = flat_idx % Nx   (col / longitude index)
       - noise_grid[j, i] = Lden value
    
    3. Coordinate System
       - WGS84 (EPSG:4326): Latitude, Longitude
       - Latitude range: {:.4f}° - {:.4f}° N
       - Longitude range: {:.4f}° - {:.4f}° E
       - Grid: {} rows (Ny) × {} columns (Nx)
    
    4. v14 Path Evaluation
       - Path waypoints are interpolated onto noise_grid
       - Bilinear interpolation used for continuous evaluation
       - Noise risk contribution = cumulative interpolated values
       - Weighted by w_noise = 0.1 in multi-objective function
       - Min/max clamping to [0, max_noise] ensures positivity
    
    5. Multi-Objective Function
       f = [distance, ground_risk, air_risk, noise_risk]
       where each is normalized and weighted appropriately
    """.format(
        CODE_LAT_LIM[0], CODE_LAT_LIM[1],
        CODE_LON_LIM[0], CODE_LON_LIM[1],
        Ny, Nx
    ))


if __name__ == '__main__':
    main()
