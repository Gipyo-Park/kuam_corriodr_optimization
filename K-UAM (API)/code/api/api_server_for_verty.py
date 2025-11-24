from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
from math import atan2, degrees
import numpy as np
import os, datetime, sys

# Import risk_utils from current directory FIRST
from risk_utils import batch_lla_to_utm, split_path_by_interval, interpolate_risk_with_heading

# Then add optimization to path for its helper modules
_current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_current_dir, 'optimization'))

from optimization.path_engine import find_optimal_path, _load_risk_maps

app = FastAPI()
data = np.load("../../data/high_res_affected_population_GRC.npy")

print("Server is starting up...")
# Change to optimization directory for .npy file loading
_orig_dir = os.getcwd()
os.chdir(os.path.join(_current_dir, 'optimization'))
_load_risk_maps()
os.chdir(_orig_dir)
print("Risk maps loaded successfully. Server is ready.")

# [Original dummy data - kept for reference]
# ROUTE_LLA = [
#     [35.603336, 129.077692, 500.0],
#     [35.584592, 129.093647, 500.0],
#     [35.602653, 129.113067, 500.0],
#     [35.632681, 129.123858, 500.0],
#     [35.624958, 129.133553, 500.0],
#     [35.603475, 129.126819, 500.0],
#     [35.584536, 129.107647, 500.0],
#     [35.569236, 129.108531, 500.0],
#     [35.554644, 129.093697, 500.0],
#     [35.558672, 129.081661, 500.0],
#     [35.578475, 129.091689, 500.0],
#     [35.584372, 129.077544, 500.0],
#     [35.616386, 129.061394, 500.0],
#     [35.621253, 129.072544, 500.0],
#     [35.611053, 129.071189, 500.0]]

class GroundRiskInput(BaseModel):
    start_point: Tuple[float, float, float]   # LLA
    end_point: Tuple[float, float, float]     # LLA

@app.post("/path-with-ground-risk")

def path_with_ground_risk(input: GroundRiskInput):

    # Use optimized path instead of dummy ROUTE_LLA
    # waypoints = [input.start_point] + ROUTE_LLA[1:-1] + [input.end_point]

    path = find_optimal_path(
            start_point=list(input.start_point),
            end_point=list(input.end_point),
            enable_plot=True
            # enable_plot=False
        )
    waypoints = path if path else [input.start_point, input.end_point]

    utm_pts = batch_lla_to_utm(waypoints)

    segments = [(utm_pts[i], utm_pts[i+1]) for i in range(len(utm_pts)-1)]

    risks = [
        interpolate_risk_with_heading(
            data,
            *seg[0],
            (degrees(atan2(seg[1][1] - seg[0][1],
                           seg[1][0] - seg[0][0])) + 360) % 360
        )
        for seg in segments
    ]

    return {
        "ts": datetime.datetime.utcnow().isoformat(),
        "endpoint": "/path-with-ground-risk",
        "request": {
            "start_point": input.start_point,
            "end_point": input.end_point,
        },
        "status": 200,
        "response": {
            "waypoints": waypoints,          
            "segment_risks": risks,          
        },
    }
