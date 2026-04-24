# Noise Data Integration - Comprehensive Summary

## 1. NOISE DATA LOADING & USAGE

### CSV File Location and Format
- **File**: `noise_data/noise_output_lden.csv`
- **Columns**: `corridor_id`, `receiver_id`, `Lden_db`, `SEL_total_db`, `LAeq_day_db`
- **Currently Used Column**: `Lden_db` (default, configurable via `noise_metric_col`)
- **Data Range**: receiver_id values like 2623, 2748, 2749, 3000+, etc. (sparse, non-sequential)
- **Metric Range**: Lden_db typically -13 to -22 dB range in the CSV

### Key Function: `load_noise_risk_from_csv()` 
**Location**: [main_JS_1218_v14.py](main_JS_1218_v14.py#L325)

```
Function Signature:
load_noise_risk_from_csv(
    csv_path,
    Ny,
    Nx,
    metric_col="Lden_db",
    receiver_id_base="auto",
    noise_floor_db=0.0,
)
```

**Returns**:
- `noise_norm` (2D array Ny×Nx): Normalized noise risk [0, 1]
- `noise_active_db` (2D array Ny×Nx): Noise values > floor, else 0
- `meta` (dict): Metadata including actual detected base (0 or 1)

---

## 2. RECEIVER_ID TO GRID COORDINATE MAPPING

### How It Works
1. **Read receiver_id and Lden_db** from CSV
2. **Auto-detect base** (0-based or 1-based indexing):
   - Try base=0: count valid indices where `idx = receiver_id - 0`
   - Try base=1: count valid indices where `idx = receiver_id - 1`
   - Use the base that yields more valid mappings
   
3. **Flat index to 2D coordinates**:
   ```python
   # flat_idx = receiver_id - receiver_id_base
   j = flat_idx // Nx    # Row (latitude/y direction)
   i = flat_idx % Nx     # Column (longitude/x direction)
   noise_db[j, i] = value
   ```

### Grid Ordering (Snake/Raster Pattern)
- **Horizontal (fast)**:  Along **longitude** (X direction, size **Nx**)
- **Vertical (slow)**: Along **latitude** (Y direction, size **Ny**)
- **Pattern**: Standard C-order (row-major) - values increase left→right, top→bottom
- **Total cells**: Ny × Nx
- For example, if Ny=100, Nx=150, then receiver_id=2750 maps to:
  - `j = 2750 // 150 = 18` (row)
  - `i = 2750 % 150 = 50` (column)

### Grid Dimensions (From Ground Risk Map)
**In current v14 code**:
- Ground risk `.npy` shape: `(Ny, Nx, channels, heading)`
- The **exact Ny and Nx are derived from loading** the ground risk file
- Grid dimensions are extracted at runtime: `Ny, Nx, H_time = selected.shape`
- **No "snake grid" file exists** - receiver ordering is implicit flat-index raster

---

## 3. COORDINATE SYSTEM

### Geographic Coverage (WGS84 / Lat-Lon)
```
Latitude range  (Y/North-South):  [35.535° to 35.652°] N
Longitude range (X/East-West):    [129.020° to 129.150°] E
```
**Location**: South Korea (Ulsan area)

### Coordinate Conversion
The code uses standard lat/lon to meters conversion:
```python
meters_per_lat_deg = 111000.0
meters_per_lon_deg = 111000.0 * cos(lat_rad)

# In evaluate_objectives_GP.py (lines 55-60):
dLat_deg = (maxLat - minLat) / (Ny - 1)  # per row
dLon_deg = (maxLon - minLon) / (Nx - 1)  # per column

# Grid coordinates to lat/lon:
Jq = (yq_lat - minLat) / dLat_deg
Iq = (xq_lon - minLon) / dLon_deg
```

### Grid Spacing Calculation
- **Latitude spacing**: (35.652 - 35.535) / (Ny - 1) degrees
- **Longitude spacing**: (129.150 - 129.020) / (Nx - 1) degrees

---

## 4. NOISE RISK OBJECTIVE CALCULATION

### Noise Objective Function
**File**: [evaluate_objectives_GP.py](evaluate_objectives_GP.py#L1)

**Main calculation process**:
1. **Load path** (sequence of [lat, lon, alt] waypoints)
2. **For each segment** p1→p2:
   - Interpolate path with refinement factor (based on segment length)
   - Sample noise at each interpolated point using bilinear interpolation
   - **Apply floor threshold**: only noise > `noise_floor_db` contributes
   - **Accumulate**: `cumulative_noise_risk += sum(interp_noise)`
   
3. **Noise objective returned** as 4th component (when NoiseRisk provided):
   ```python
   return np.array([
       total_dist,
       cumulative_ground_risk,
       cumulative_air_risk,
       cumulative_noise_risk * w_noise,    # ← Noise objective
   ])
   ```

### Noise Weighting
- **Noise weight parameter**: `w_noise = 0.1` (configurable in main_JS_1218_v14.py line ~1775)
- **Applied at**: `cumulative_noise_risk * w_noise` in evaluate_objectives_GP.py line 149
- **Optimization objective count**: WITHOUT noise = 3, WITH noise = 4 objectives

### Noise Floor Logic
- **Default floor**: `noise_floor_db = 0.0` dB
- **Post-floor values**: Only noise values > floor contribute
  - Below floor: treated as 0 contribution
  - Above floor: contributes full value to objective

---

## 5. NOISE DATA INTEGRATION IN MAIN OPTIMIZATION

### Loading Sequence (main_JS_1218_v14.py lines ~1895-2100)

```python
# 1. Load CSV and convert to normalized grid (line 1902-1910)
noise_2d_norm, noise_2d_db_after_floor, noise_meta = load_noise_risk_from_csv(
    csv_path=noise_csv_path,
    Ny=Ny,
    Nx=Nx,
    metric_col=noise_metric_col,
    receiver_id_base="auto",  # Auto-detect 0-based or 1-based
    noise_floor_db=noise_floor_db,
)

# 2. Expand to 3D (match all altitude levels)
NoiseRisk = np.repeat(noise_2d_norm[:, :, np.newaxis], len(altitude_levels), axis=2)
NoiseRiskDb = np.repeat(noise_2d_db_after_floor[:, :, np.newaxis], len(altitude_levels), axis=2)

# 3. Pass to objective evaluation (line 2254)
evaluate_objectives_with_constraints_gp(
    ...
    NoiseRisk=NoiseRisk,
    noise_floor_db=noise_floor_db,
    w_noise=w_noise,
    ...
)
```

### Multi-Objective Optimization
- **Objective 1**: Total path distance (w_dist = 0.1)
- **Objective 2**: Cumulative ground risk (w_ground = 1.0)
- **Objective 3**: Cumulative air risk (w_air = 2.0)
- **Objective 4**: Cumulative noise risk × w_noise (w_noise = 0.1)
- **Strategy**: NSGA-III (niching-based multi-objective selection)

---

## 6. METADATA & CONFIGURATION

### Current Configuration (main_JS_1218_v14.py)
```python
w_dist, w_ground, w_air, w_noise = 0.1, 1.0, 2.0, 0.1
altitude_levels = np.array([600.0], dtype=float)
use_heading_map = True
noise_csv_path = Path("noise_data") / "noise_output_lden.csv"
noise_metric_col = "Lden_db"
noise_floor_db = 0.0
```

### Noise Metadata Output (from load_noise_risk_from_csv)
Each run saves to `runs/<timestamp>/params.json`:
```json
{
  "noise_csv_path": "noise_data/noise_output_lden.csv",
  "noise_metric_col": "Lden_db",
  "noise_floor_db": 0.0,
  "receiver_id_base": 1,  # Auto-detected (0 or 1)
  "rows_total": 847,      # Total rows in CSV
  "rows_mapped": 840,     # Valid rows mapped to grid
  "noise_max_db_after_floor": 12.5  # Max value after floor
}
```

---

## 7. FILES NOT FOUND - "SNAKE GRID"

### Investigation Results:
- ❌ **No dedicated "snake_grid" file exists**
- ❌ **No grid position mapping file found** (checked k_data/, l_data/)
- ✅ **Receiver ordering IS implicit**: receiver_id = flat raster index in (Ny × Nx) grid
- ✅ **Standard row-major (C-order)**: receiver_id → `[j, i]` via `j=id//Nx, i=id%Nx`

### What exists instead:
- **Ground risk**: `Modified_high_res_affected_population_GRC.npy` → defines Ny, Nx
- **Bird air risk**: `air_risk_data/bird_riskmap_springfall_3d.npy` → (Nx, Ny, Nz) or (Ny, Nx, Nz), auto-aligned
- **MOC risk**: `air_risk_data/UAM_MOC_3D_Risk_Map.npy` → binary obstacle threat

---

## 8. KEY FILES INVOLVED

| File | Role |
|------|------|
| [main_JS_1218_v14.py](main_JS_1218_v14.py#L325) | Main entry; calls `load_noise_risk_from_csv()` |
| [load_noise_risk_from_csv()](main_JS_1218_v14.py#L325) | CSV loader; receiver_id auto-base detection |
| [evaluate_objectives_GP.py](evaluate_objectives_GP.py) | Objective function computation; interpolates noise |
| [evaluate_objectives_with_constraints_GP.py](evaluate_objectives_with_constraints_GP.py#L573) | Feasibility check wrapper |
| `noise_data/noise_output_lden.csv` | Input noise data (receiver_id, Lden_db, etc.) |
| `Modified_high_res_affected_population_GRC.npy` | Ground risk; defines grid dimensions Ny, Nx |
| `runs/<timestamp>/params.json` | Output config; includes noise metadata |

---

## 9. RECEIVER_ID ORDERING EXAMPLE

Given Ny=100, Nx=150 grid:
- receiver_id 0 → grid[0, 0] (top-left)
- receiver_id 1 → grid[0, 1]
- receiver_id 149 → grid[0, 149] (top-right)
- receiver_id 150 → grid[1, 0] (first col, 2nd row)
- receiver_id 14999 → grid[99, 149] (bottom-right, last cell)

**Or 1-based (if auto-detected base=1)**:
- receiver_id 1 → grid[0, 0]
- receiver_id 2 → grid[0, 1]
- etc.

---

## 10. QUICK REFERENCE: HOW TO USE NOISE DATA

### Load and visualize noise:
```python
from pathlib import Path
import numpy as np

noise_csv = Path("noise_data/noise_output_lden.csv")
Ny, Nx = 100, 150  # from ground risk file
noise_norm, noise_db, meta = load_noise_risk_from_csv(
    noise_csv, Ny=Ny, Nx=Nx, 
    metric_col="Lden_db", 
    noise_floor_db=0.0
)
print(f"Receiver base: {meta['receiver_id_base']}")
print(f"Max noise (after floor): {meta['noise_max_db_after_floor']} dB")
```

### Access specific receiver:
```python
receiver_id = 2750
base = meta['receiver_id_base']
j = (receiver_id - base) // Nx
i = (receiver_id - base) % Nx
noise_value = noise_db[j, i]
```

### Modify weighting:
Edit `main_JS_1218_v14.py` line ~1775:
```python
w_noise = 0.1  # Increase to emphasize noise in optimization
```

---

## SUMMARY TABLE

| Item | Value/Location |
|------|---|
| **Noise CSV** | `noise_data/noise_output_lden.csv` |
| **Key Column** | `Lden_db` |
| **Grid Type** | 2D raster (Ny × Nx), row-major order |
| **Receiver Ordering** | Flat index (auto-detects 0-based or 1-based) |
| **Coord System** | WGS84 (lat/lon): 35.535-35.652°N, 129.020-129.150°E |
| **Noise Weight** | w_noise = 0.1 (configurable) |
| **Noise Floor** | 0.0 dB (values ≤ floor don't contribute) |
| **Objective 4** | cumulative_noise_risk × w_noise |
| **Snake Grid File** | **DOES NOT EXIST** — ordering is implicit |
| **Last Updated** | v14 (main_JS_1218_v14.py) |
