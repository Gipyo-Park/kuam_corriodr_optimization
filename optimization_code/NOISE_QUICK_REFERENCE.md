# Noise Data - Quick Reference Guide

## TL;DR

### How Noise Data is Used
1. **Load**: CSV file with `receiver_id` and `Lden_db` columns
2. **Map**: `receiver_id` → grid coordinates `[j, i]` via flat-index raster order
3. **Weight**: Noise objective gets `w_noise = 0.1` weight in multi-objective optimization
4. **Floor**: Values ≤ 0.0 dB don't contribute to risk calculation
5. **Interpolate**: Path segments are sampled at noise grid with bilinear interpolation

---

## One-Page Summary

### File Locations
```
Input:  noise_data/noise_output_lden.csv
Output: runs/<timestamp>/params.json (metadata)
Other:  NO "snake grid" file exists
```

### Key Function
```python
# Load and map CSV to grid
noise_norm, noise_db, meta = load_noise_risk_from_csv(
    csv_path="noise_data/noise_output_lden.csv",
    Ny=100, Nx=150,               # Grid dimensions (from ground risk)
    metric_col="Lden_db",         # Column to use
    receiver_id_base="auto",      # Auto-detect 0 or 1 based
    noise_floor_db=0.0            # Threshold
)
```

### Mapping Formula
```
flat_idx = receiver_id - base        (0 = base 0, 1 = base 1)
j = flat_idx // Nx                   (row)
i = flat_idx % Nx                    (column)
noise_value = grid[j, i]
```

### Example
```
receiver_id=2750, base=1, Nx=150
→ flat_idx = 2749
→ j = 2749 // 150 = 18, i = 2749 % 150 = 49
→ grid[18, 49]
```

### Coordinate System
```
Latitude:  35.535° to 35.652° N (100 rows)
Longitude: 129.020° to 129.150° E (150 cols)
Reference: WGS84 (standard lat/lon)
```

### Objective Function
```
4 objectives = [Distance, Ground Risk, Air Risk, Noise Risk]
Noise Risk = cumulative_noise_value × w_noise
w_noise = 0.1 (configurable)
```

### Grid Dimensions
```
Ny = 100 (rows, latitude)
Nx = 150 (columns, longitude)
Total cells = 15,000
Receiver_id range: 1 to 15,000 (typically 1-based)
```

---

## Key Values (For Tuning)

| Variable | Current | File | Line | Notes |
|----------|---------|------|------|-------|
| `w_noise` | 0.1 | main_JS_1218_v14.py | ~1775 | Increase to emphasize noise |
| `noise_floor_db` | 0.0 | main_JS_1218_v14.py | ~1899 | Values ≤ floor → 0 |
| `noise_metric_col` | "Lden_db" | main_JS_1218_v14.py | ~1900 | CSV column name |
| `lat_lim` | [35.535, 35.652] | main_JS_1218_v14.py | ~2153 | Map bounds |
| `lon_lim` | [129.020, 129.150] | main_JS_1218_v14.py | ~2154 | Map bounds |

---

## Common Tasks

### Q: How do I increase noise priority in optimization?
**A**: Edit line ~1775 in main_JS_1218_v14.py:
```python
w_noise = 0.5  # Increase from 0.1
```

### Q: What if noise data won't load?
**A**: Check:
1. File exists: `noise_data/noise_output_lden.csv` ✓
2. Columns match: `receiver_id`, `Lden_db` ✓
3. Grid dimensions correct: `Ny`, `Nx` match ground risk ✓
4. `receiver_id` fits range: `[1, Ny×Nx]` or `[0, Ny×Nx-1]` ✓

### Q: How do I find receiver_id for a specific lat/lon?
**A**: 
```python
lat_target, lon_target = 35.59, 129.08
j = int((lat_target - 35.535) / ((35.652-35.535)/(100-1)))
i = int((lon_target - 129.020) / ((129.150-129.020)/(150-1)))
receiver_id = j * 150 + i + 1  # +1 for 1-based
# Result: receiver_id ≈ 2750
```

### Q: How is noise converted from dB to [0, 1]?
**A**:
```python
# Step 1: Apply floor
noise_active = max(0, noise_dB - 0.0) if noise_dB > floor else 0

# Step 2: Normalize
noise_norm = noise_active / max(noise_active_values)  # Range [0, 1]

# Step 3: Use in objective
objective += noise_norm * w_noise
```

### Q: Are there multiple noise columns in CSV?
**A**: Yes, but only one is used:
- `Lden_db` ← **Currently used**
- `SEL_total_db` (available but not used)
- `LAeq_day_db` (available but not used)

To switch:
```python
noise_metric_col = "SEL_total_db"  # main line ~1900
```

### Q: Does noise vary with altitude?
**A**: No, currently replicated across all altitudes:
```python
NoiseRisk = np.repeat(noise_2d[:, :, np.newaxis], 
                      len(altitude_levels), axis=2)
# All altitude layers get same noise values
```

### Q: What if grid cells have no noise data?
**A**: 
- Stored as NaN in grid
- Interpolation treats NaN as 0 (cval=0.0)
- Doesn't affect final risk (added as 0)

---

## Verification Checklist

- [ ] CSV file exists and readable
- [ ] Has columns: `corridor_id`, `receiver_id`, `Lden_db`
- [ ] Ground risk file loaded (defines Ny, Nx)
- [ ] Receiver IDs fit expected range (1 to Ny×Nx)
- [ ] `w_noise` value is positive
- [ ] `noise_floor_db` makes sense for dB range
- [ ] Params JSON shows metadata with receiver_id_base
- [ ] Optimization produces 4-objective results

---

## Output Check

After running main_JS_1218_v14.py, check `runs/<timestamp>/params.json`:

```json
{
  "noise_csv_path": "noise_data/noise_output_lden.csv",
  "noise_metric_col": "Lden_db",
  "noise_floor_db": 0.0,
  "receiver_id_base": 1,           ← Auto-detected!
  "rows_total": 847,               ← CSV rows
  "rows_mapped": 840,              ← Mapped to grid
  "noise_max_db_after_floor": 12.5 ← Peak value
}
```

---

## Objective Breakdown

**Returned by evaluate_objectives_GP()** when noise is enabled:

```python
[
  total_dist * w_dist = 5000m * 0.1 = 500,
  cumul_ground_risk * w_ground = 100 * 1.0 = 100,
  cumul_air_risk * w_air = 50 * 2.0 = 100,
  cumul_noise_risk * w_noise = 1000 * 0.1 = 100,
]
```

NSGA-III optimizer tries to minimize all 4 simultaneously.

---

## Files to Modify for Tuning

| File | What to Change | Line |
|------|---|---|
| main_JS_1218_v14.py | `w_noise` | ~1775 |
| main_JS_1218_v14.py | `noise_floor_db` | ~1899 |
| main_JS_1218_v14.py | `noise_metric_col` | ~1900 |
| main_JS_1218_v14.py | `noise_csv_path` | ~1899 |
| evaluate_objectives_GP.py | Interpolation method | ~115 |

---

## Performance Notes

- CSV loading: < 1 second for ~850 rows
- Grid mapping: Automatic (single pass)
- Objective evaluation: ~0.1ms per path (with interpolation)
- Memory: ~100 KB per grid (negligible)

---

## Related Documentation

- Full summary: `NOISE_DATA_MAPPING_SUMMARY.md`
- Technical details: `NOISE_TECHNICAL_REFERENCE.md`
- Main script: `main_JS_1218_v14.py`
- Objective code: `evaluate_objectives_GP.py`

---

## Snake Grid - Why It Doesn't Exist

The CSV `receiver_id` column IS the snake/raster order index:
- No separate file needed
- Standard row-major indexing: receiver_id ≡ j×Nx + i
- Auto-base detection handles both 0-based and 1-based conventions

**Why not create one?**
- Redundant (computed on-the-fly)
- Takes more space
- Makes updates harder

---

## Minimal Working Example

```python
import numpy as np
from pathlib import Path
import csv
# (Simplified version of load_noise_risk_from_csv)

csv_path = Path("noise_data/noise_output_lden.csv")
Ny, Nx = 100, 150

# Read
receiver_ids, ldens = [], []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        receiver_ids.append(int(row["receiver_id"]))
        ldens.append(float(row["Lden_db"]))

# Detect base
for base in [0, 1]:
    valid = sum(1 for rid in receiver_ids if 0 <= rid - base < Ny*Nx)
    if valid > 100: break

# Map
noise_grid = np.full((Ny, Nx), np.nan)
for rid, lden in zip(receiver_ids, ldens):
    idx = rid - base
    if 0 <= idx < Ny*Nx:
        j, i = idx // Nx, idx % Nx
        noise_grid[j, i] = lden

# Use
print(f"Noise at [18, 49]: {noise_grid[18, 49]:.2f} dB")
```

---

## See Also

- Ground risk: `Modified_high_res_affected_population_GRC.npy`
- Air risk: `air_risk_data/bird_riskmap_springfall_3d.npy`
- MOC risk: `air_risk_data/UAM_MOC_3D_Risk_Map.npy`
- Multi-obj: NSGA-III (implemented in niching_selection.py)
