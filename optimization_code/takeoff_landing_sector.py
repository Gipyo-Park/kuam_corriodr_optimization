import numpy as np

def as_mask(x):
    x = np.asarray(x, dtype=int).reshape(-1)
    if x.size != 12:
        raise ValueError("mask length must be 12")
    return (x > 0)

class SectorMaskDB:
    def __init__(self):
        self.takeoff = {}
        self.landing = {}
        self.climb = {}

    def set_climb(self, climb_deg, mask12):
        self.climb[float(climb_deg)] = np.array(mask12, dtype=bool)

    def set_takeoff(self, season, mask12):
        self.takeoff[str(season).lower()] = np.array(mask12, dtype=bool)

    def set_landing(self, season, mask12):
        self.landing[str(season).lower()] = np.array(mask12, dtype=bool)

    def get(self, season, climb_deg):
        s = str(season).lower()
        c = float(climb_deg)
        return (
            self.takeoff[s],
            self.landing[s],
            self.climb[c]
        )


def combine_masks(season_mask, climb_mask, mode="union"):
    if mode == "union":
        m = np.logical_or(season_mask, climb_mask)
    elif mode == "intersection":
        m = np.logical_and(season_mask, climb_mask)
    elif mode == "season_only":
        m = season_mask.copy()
    elif mode == "climb_only":
        m = climb_mask.copy()
    else:
        raise ValueError("mode must be union, intersection, season_only, climb_only")
    idx = np.where(m)[0].tolist()
    return m.astype(int), idx

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def plot_sector_mask_ax(ax, mask12, title):
    mask12 = np.asarray(mask12, dtype=int).reshape(12,)
    n = 12
    w = 2*np.pi/n

    start = np.deg2rad(90.0)   # 12시
    ax.set_aspect("equal")

    for i in range(n):
        th1 = start - i*w          # 시계방향
        th2 = start - (i+1)*w
        th = np.linspace(th1, th2, 80)

        x = np.r_[0.0, np.cos(th), 0.0]
        y = np.r_[0.0, np.sin(th), 0.0]

        if mask12[i] == 1:
            ax.fill(x, y, color="tab:blue", alpha=0.6)
        else:
            ax.fill(x, y, color="lightgray", alpha=0.25)

        ax.plot([0, np.cos(th1)], [0, np.sin(th1)], color="k", linewidth=0.5)

    ax.set_title(title)
    ax.axis("off")



def combine_union(a, b):
    return np.logical_or(a, b).astype(int)


def combine_intersection(a, b):
    return np.logical_and(a, b).astype(int)

if __name__ == "__main__":
    db = SectorMaskDB()

    # 예시 저장
    db.set_climb(5.0,  [0,0,0,0,1,1,0,0,0,0,0,1])
    db.set_climb(8.0,  [0,0,0,1,1,1,1,1,1,0,1,1])
    db.set_climb(12.0, [1,1,1,1,1,1,1,1,1,1,1,1])

    # 계절별 이륙
    db.set_takeoff("annual", [1,1,1,1,1,0,0,0,0,0,0,0])
    db.set_takeoff("spring", [0,1,1,1,1,1,1,0,0,0,0,0])
    db.set_takeoff("summer", [1,1,1,1,1,1,0,0,0,0,0,0])
    db.set_takeoff("autumn", [1,1,1,1,0,0,0,0,0,0,0,0])
    db.set_takeoff("winter", [1,1,1,1,0,0,0,0,0,0,0,1])

    # 계절별 착륙 예시 (보통 이륙 반대 방향이거나 더 보수적)
    db.set_landing("annual", [0,0,0,0,0,0,1,1,1,1,1,0])
    db.set_landing("spring", [0,0,0,0,0,0,0,0,1,1,1,1])
    db.set_landing("summer", [0,0,0,0,0,0,0,1,1,1,1,1])
    db.set_landing("autumn", [0,0,0,0,0,1,1,1,1,1,1,0])
    db.set_landing("winter", [0,0,0,0,0,1,1,1,1,1,0,0])

    # ===== 사용부 =====
    season = "winter"
    climb = 12.0

    takeoff_mask, landing_mask, climb_mask = db.get(season=season, climb_deg=climb)

    climb_only        = climb_mask.astype(int)
    takeoff_only      = takeoff_mask.astype(int)
    landing_only      = landing_mask.astype(int)
    climb_takeoff     = combine_intersection(climb_mask, takeoff_mask)
    climb_landing     = combine_intersection(climb_mask, landing_mask)

    # ===== plot =====
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    plot_sector_mask_ax(axes[0], climb_only,   "Climb only")
    plot_sector_mask_ax(axes[1], takeoff_only, "Takeoff only")
    plot_sector_mask_ax(axes[2], landing_only, "Landing only")
    plot_sector_mask_ax(axes[3], climb_takeoff,"Climb ∩ Takeoff")
    plot_sector_mask_ax(axes[4], climb_landing,"Climb ∩ Landing")

    plt.tight_layout()
    plt.show()