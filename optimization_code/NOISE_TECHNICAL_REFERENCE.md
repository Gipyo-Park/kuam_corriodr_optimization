# Noise Data Technical Reference - Code & Mapping Logic

## RECEIVER_ID TO GRID COORDINATE MAPPING - DETAILED

### The Complete Mapping Algorithm

**Source**: [main_JS_1218_v14.py](main_JS_1218_v14.py#L325-L420) in `load_noise_risk_from_csv()`

```python
def load_noise_risk_from_csv(
    csv_path,
    Ny,
    Nx,
    metric_col="Lden_db",
    receiver_id_base="auto",
    noise_floor_db=0.0,
):
    """Load receiver-based noise CSV and convert it to a normalized grid risk map."""
    
    # Step 1: Read CSV and extract receiver_id and metric values
    receiver_ids = []
    metric_vals = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rid = int(float(row["receiver_id"]))
                val = float(row[metric_col])
            except Exception:
                continue
            receiver_ids.append(rid)
            metric_vals.append(val)

    receiver_ids = np.asarray(receiver_ids, dtype=int)
    metric_vals = np.asarray(metric_vals, dtype=float)
    total_cells = int(Ny) * int(Nx)

    # Step 2: Auto-detect receiver_id base (0-based or 1-based indexing)
    if receiver_id_base == "auto":
        base_candidates = [0, 1]
    else:
        base_candidates = [int(receiver_id_base)]

    best_base = None
    best_valid = -1
    for base in base_candidates:
        idx = receiver_ids - base  # Convert receiver_id to 0-based index
        valid = int(np.sum((idx >= 0) & (idx < total_cells)))
        if valid > best_valid:
            best_valid = valid
            best_base = base

    if best_base is None or best_valid <= 0:
        raise RuntimeError("Failed to map receiver_id to grid indices (check Ny/Nx and receiver_id base)")

    # Step 3: Convert to 0-based flat indices and validate
    idx = receiver_ids - int(best_base)  # Now: idx ranges [0, total_cells)
    valid_mask = (idx >= 0) & (idx < total_cells)
    idx = idx[valid_mask]
    vals = metric_vals[valid_mask]

    # Step 4: Convert flat index to 2D (j, i) coordinates
    # This uses row-major (C-order) indexing:
    noise_db = np.full((int(Ny), int(Nx)), np.nan, dtype=float)
    for flat_idx, v in zip(idx.tolist(), vals.tolist()):
        j = int(flat_idx // int(Nx))  # Row index (latitude/Y)
        i = int(flat_idx % int(Nx))   # Column index (longitude/X)
        noise_db[j, i] = float(v)

    # Step 5: Apply noise floor threshold
    # Values <= floor don't contribute to objective
    noise_active_db = np.where(np.isfinite(noise_db) & (noise_db > float(noise_floor_db)), 
                                noise_db, 0.0)
    
    # Step 6: Normalize to [0, 1]
    vmax = float(np.max(noise_active_db)) if noise_active_db.size > 0 else 0.0
    noise_norm = (noise_active_db / vmax) if vmax > 1e-12 else np.zeros_like(noise_active_db)

    # Step 7: Return metadata
    meta = {
        "csv_path": str(csv_path),
        "metric_col": str(metric_col),
        "receiver_id_base": int(best_base),
        "rows_total": int(len(receiver_ids)),
        "rows_mapped": int(len(idx)),
        "noise_floor_db": float(noise_floor_db),
        "noise_max_db_after_floor": float(vmax),
    }
    return noise_norm.astype(float), noise_active_db.astype(float), meta
```

---

## FLAT INDEX TO 2D CONVERSION - EXAMPLES

### Formula
```
receiver_id (input)
    ↓
flat_idx = receiver_id - base  (convert to 0-based)
    ↓
j = flat_idx // Nx             (ROW: latitude/Y direction)
i = flat_idx % Nx              (COL: longitude/X direction)
    ↓
grid[j, i] = noise_value       (access grid)
```

### Example 1: 100×150 grid, 1-based receiver_id

```
Ny = 100, Nx = 150
base = 1
total_cells = 100 × 150 = 15,000

receiver_id = 1      → idx = 0    → j = 0 // 150 = 0,   i = 0 % 150 = 0      → [0, 0]
receiver_id = 2      → idx = 1    → j = 1 // 150 = 0,   i = 1 % 150 = 1      → [0, 1]
receiver_id = 150    → idx = 149  → j = 149 // 150 = 0, i = 149 % 150 = 149  → [0, 149]
receiver_id = 151    → idx = 150  → j = 150 // 150 = 1, i = 150 % 150 = 0    → [1, 0]
receiver_id = 15000  → idx = 14999 → j = 14999 // 150 = 99, i = 14999 % 150 = 149 → [99, 149]
```

### Example 2: Real data - receiver 2750, unknown base

```
Assuming Ny = 100, Nx = 150

If base = 1:
  idx = 2750 - 1 = 2749
  valid? 0 ≤ 2749 < 15000 ✓ YES
  j = 2749 // 150 = 18
  i = 2749 % 150 = 49
  → grid[18, 49]

If base = 0:
  idx = 2750 - 0 = 2750
  valid? 0 ≤ 2750 < 15000 ✓ YES
  j = 2750 // 150 = 18
  i = 2750 % 150 = 50
  → grid[18, 50]

Algorithm picks the base that maps MORE valid entries.
Typically: base = 1 if most receiver_ids match total_cells range, else base = 0.
```

---

## NOISE INTERPOLATION IN OBJECTIVE FUNCTION

**Source**: [evaluate_objectives_GP.py](evaluate_objectives_GP.py#L100-L120)

```python
# For each path segment p1 → p2:

# 1. Interpolate path with refinement
num_samples = compute_refinement(dist_2d_m, cell_size, refine_scales)
yq_lat = np.linspace(p1[0], p2[0], num_samples)
xq_lon = np.linspace(p1[1], p2[1], num_samples)

# 2. Convert lat/lon to grid indices
Iq = (xq_lon - minLon) / dLon_deg
Jq = (yq_lat - minLat) / dLat_deg
coords = np.vstack((Jq, Iq))

# 3. Interpolate noise at all points
if NoiseRisk is not None and np.size(NoiseRisk) > 0:
    if np.ndim(NoiseRisk) == 3:
        noise_map = NoiseRisk[:, :, alt_idx]  # Select altitude layer
    else:
        noise_map = NoiseRisk  # 2D only
    
    # Bilinear interpolation at sub-grid positions
    interp_noise = map_coordinates(noise_map, coords, order=1, cval=0.0)
    
    # Apply floor threshold: noise below floor contributes 0
    interp_noise = np.where(interp_noise > noise_floor_db, interp_noise, 0.0)
    
    # Accumulate
    cumulative_noise_risk += np.sum(interp_noise)
```

---

## COORDINATE TRANSFORMATION

### Lat/Lon to Grid Index

**Source**: [evaluate_objectives_GP.py](evaluate_objectives_GP.py#L50-L70)

```python
# Map setup (computed once)
minLat, maxLat = lat_lim  # [35.535, 35.652]
minLon, maxLon = lon_lim  # [129.020, 129.150]
Ny, Nx = 100, 150         # (example)

dLat_deg = (maxLat - minLat) / (Ny - 1)  # 0.117 / 99 ≈ 0.00118 deg
dLon_deg = (maxLon - minLon) / (Nx - 1)  # 0.130 / 149 ≈ 0.000872 deg

# For any lat/lon point:
lat, lon = 35.590, 129.080

J = (lat - minLat) / dLat_deg  # Grid row (continuous, 0 to Ny-1)
I = (lon - minLon) / dLon_deg  # Grid col (continuous, 0 to Nx-1)

# Integer indices for discrete grid access
j_int = int(np.round(J))
i_int = int(np.round(I))
```

### Grid Index to Lat/Lon (Inverse)

```python
# For grid cell [j, i]:
lat = minLat + j * dLat_deg
lon = minLon + i * dLon_deg
```

### Meters Conversion

```python
# Degree to meters
meters_per_lat_deg = 111000.0  # (constant)
meters_per_lon_deg = 111000.0 * np.cos(np.deg2rad(lat))  # (depends on latitude)

# Example at lat = 35.59°:
m_per_lon = 111000 * cos(35.59°) ≈ 90,800 meters/degree
m_per_lat ≈ 111,000 meters/degree (nearly constant)
```

---

## NOISE OBJECTIVE COMPUTATION

**Source**: [evaluate_objectives_GP.py](evaluate_objectives_GP.py#L135-L150)

### 3-Objective Case (No Noise)
```python
if NoiseRisk is None or np.size(NoiseRisk) == 0:
    return np.array([
        total_dist,
        cumulative_ground_risk,
        cumulative_air_risk,
    ])
```

### 4-Objective Case (With Noise)
```python
else:
    return np.array([
        total_dist,              # Objective 1: Total 3D distance
        cumulative_ground_risk,  # Objective 2: Ground/population risk
        cumulative_air_risk,     # Objective 3: Air/bird risk
        cumulative_noise_risk * w_noise,  # Objective 4: Noise (weighted)
    ])
```

### Weight Application

**Main code** [main_JS_1218_v14.py](main_JS_1218_v14.py#L1775):
```python
w_dist, w_ground, w_air, w_noise = 0.1, 1.0, 2.0, 0.1
```

Applied in `evaluate_objectives_GP.py`:
- `total_dist *= w_dist` (line 50)
- Ground/air risks use their raw weights (via `w_ground`, `w_air` in segment risk calc)
- `cumulative_noise_risk *= w_noise` (line 149)

---

## DATA FLOW DIAGRAM

```
noise_output_lden.csv
    │
    ├─ Read rows: [corridor_id, receiver_id, Lden_db, ...]
    │
    ├─ load_noise_risk_from_csv(Ny, Nx)
    │   ├─ Extract: receiver_id[], Lden_db[]
    │   ├─ Auto-detect base (0 or 1)
    │   ├─ Convert: flat_idx = receiver_id - base
    │   ├─ Map: (j, i) = (flat_idx // Nx, flat_idx % Nx)
    │   ├─ Fill: noise_db[j, i] = Lden_db
    │   ├─ Floor: values ≤ noise_floor_db → 0
    │   └─ Normalize: noise_norm = noise_db / max(noise_db)
    │
    ├─ Expand to 3D:
    │   NoiseRisk = noise_norm[:, :, np.newaxis] × len(altitudes)
    │
    ├─ Multi-objective optimization loop:
    │   For each candidate path:
    │   ├─ Sample path points with interpolation
    │   ├─ Convert (lat, lon) → (J, I) grid indices
    │   ├─ Interpolate: noise_value = map_coordinates(NoiseRisk, (J, I))
    │   ├─ Apply floor: if noise_value > noise_floor_db then count else skip
    │   └─ Accumulate: cumsum += noise_value
    │
    └─ Objective = cumsum × w_noise
         (returned as 4th objective)
```

---

## PYTHON CODE SNIPPET - ACCESSING NOISE AT A POINT

```python
import numpy as np
from scipy.ndimage import map_coordinates

# After loading:
noise_norm, noise_db, meta = load_noise_risk_from_csv(...)

# Query noise at a specific lat/lon
query_lat, query_lon = 35.59, 129.08
minLat, maxLat = 35.535, 35.652
minLon, maxLon = 129.020, 129.150
Ny, Nx = noise_norm.shape  # (e.g., 100, 150)

# Convert to grid coordinates
dLat_deg = (maxLat - minLat) / (Ny - 1) if Ny > 1 else 1.0
dLon_deg = (maxLon - minLon) / (Nx - 1) if Nx > 1 else 1.0

J = (query_lat - minLat) / dLat_deg
I = (query_lon - minLon) / dLon_deg

# Bilinear interpolation
coords = np.array([[J], [I]])
noise_normalized = map_coordinates(noise_norm, coords, order=1, cval=0.0)[0]
noise_db_value = map_coordinates(noise_db, coords, order=1, cval=0.0)[0]

print(f"Normalized noise: {noise_normalized:.4f}")
print(f"Noise (dB): {noise_db_value:.2f} dB")
```

---

## CRITICAL VALUES & THRESHOLDS

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `noise_floor_db` | 0.0 | Any | Values ≤ floor don't contribute |
| `w_noise` | 0.1 | [0, ∞) | Objective weight; higher = emphasize noise |
| `noise_metric_col` | "Lden_db" | CSV column | Currently: Lden_db, SEL_total_db, LAeq_day_db available |
| `receiver_id_base` | "auto" | 0, 1 | Auto-detects; can override |
| `altitude_levels` | [600.0] | Per app | Noise is replicated across all altitudes |

---

## GRID STORAGE ORDER

### Memory Layout (NumPy C-order)
```
noise_db: (Ny, Nx) array

Physical memory order (left-to-right, top-to-bottom):
noise_db[0, 0]   noise_db[0, 1]   ... noise_db[0, Nx-1]
noise_db[1, 0]   noise_db[1, 1]   ... noise_db[1, Nx-1]
...
noise_db[Ny-1,0] noise_db[Ny-1,1] ... noise_db[Ny-1,Nx-1]

Flat index k → [k // Nx, k % Nx]
```

### Receiver Ordering
- **Fast axis**: Longitude (X/columns, size Nx)
- **Slow axis**: Latitude (Y/rows, size Ny)
- **Order**: Row-major (C convention, contiguous rows)
- **Formula**: receiver_id ≡ j×Nx + i (where j=row, i=col)

---

## DEBUGGING TIPS

### Check auto-detected base:
```python
# Will be in meta dict
print(f"Detected receiver_id_base: {meta['receiver_id_base']}")
print(f"Rows mapped: {meta['rows_mapped']} / {meta['rows_total']}")
```

### Verify grid dimensions:
```python
# From ground risk file
ground_risk = np.load("Modified_high_res_affected_population_GRC.npy", allow_pickle=True)
selected = ground_risk[:, :, 0, 3:]
Ny, Nx, H_time = selected.shape
print(f"Grid dimensions: {Ny} rows × {Nx} columns = {Ny*Nx} cells")
print(f"Expected max receiver_id (1-based): {Ny*Nx}")
```

### Validate receiver coverage:
```python
# Check NaN regions (unmapped cells)
nan_count = np.isnan(noise_db).sum()
print(f"Unmapped cells: {nan_count} / {Ny*Nx} ({100*nan_count/(Ny*Nx):.1f}%)")
```

### Inspect specific receiver:
```python
receiver_id = 2750
base = meta['receiver_id_base']
j = (receiver_id - base) // Nx
i = (receiver_id - base) % Nx
val = noise_db[j, i]
print(f"Receiver {receiver_id} → [{j}, {i}] = {val:.2f} dB (norm: {noise_norm[j, i]:.4f})")
```

---

## KNOWN LIMITATIONS & EDGE CASES

1. **Sparse receiver_id coverage**: Not all grid cells may have noise data
   - Cells without data remain NaN initially
   - Are treated as 0 during interpolation (cval=0.0)

2. **Auto-base detection**: May fail if receiver_ids don't fit standard 0- or 1-based pattern
   - Fallback: explicitly set `receiver_id_base=0` or `receiver_id_base=1`

3. **Noise normalization**: Divides by max(noise_active_db)
   - If all noise ≤ floor: max=0, norm array becomes all zeros

4. **Interpolation at grid boundaries**:
   - Points outside lat/lon bounds are clamped to NaN/0 (see `cval=0.0`)

5. **Multiple altitudes**: Noise is replicated (not altitude-specific)
   - Current implementation doesn't model altitude-dependent noise variation

6. **Duplicate receiver_ids**: Last value wins
   - Loop: `for flat_idx, v in zip(...): noise_db[j, i] = v`

---

## RELATED FILES

- **Risk map loading**: [main_JS_1218_v14.py](main_JS_1218_v14.py#L2000-L2100)
- **Objective scaling**: [normalize_objectives.py](normalize_objectives.py)
- **Path evaluation**: [evaluate_objectives_GP.py](evaluate_objectives_GP.py)
- **Constraints check**: [evaluate_objectives_with_constraints_GP.py](evaluate_objectives_with_constraints_GP.py)
- **Multi-objective selection**: [niching_selection.py](niching_selection.py)
- **Output logging**: [main_JS_1218_v14.py](main_JS_1218_v14.py#L3900+)
