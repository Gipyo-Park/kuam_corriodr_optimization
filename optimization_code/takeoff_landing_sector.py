import numpy as np


SEASONS = ("annual", "spring", "summer", "autumn", "winter")

SEASONAL_TAKEOFF_MASKS = {
    "annual": np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool),
    "spring": np.array([0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool),
    "summer": np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool),
    "autumn": np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool),
    "winter": np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=bool),
}

SEASONAL_LANDING_MASKS = {
    "annual": np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0], dtype=bool),
    "spring": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=bool),
    "summer": np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=bool),
    "autumn": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0], dtype=bool),
    "winter": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=bool),
}


def normalize_season(season):
    s = str(season).strip().lower()
    if s not in SEASONS:
        raise ValueError(f"season must be one of {SEASONS}, got '{season}'")
    return s


def get_season_masks(season):
    s = normalize_season(season)
    return SEASONAL_TAKEOFF_MASKS[s].copy(), SEASONAL_LANDING_MASKS[s].copy()


def validate_sector_1based(sector_1based, label="sector"):
    idx = int(sector_1based)
    if idx < 1 or idx > 12:
        raise ValueError(f"{label} must be in [1, 12], got {sector_1based}")
    return idx


def sector_allowed(mask12, sector_1based):
    idx = validate_sector_1based(sector_1based) - 1
    m = np.asarray(mask12, dtype=bool).reshape(-1)
    if m.size != 12:
        raise ValueError("mask length must be 12")
    return bool(m[idx])

