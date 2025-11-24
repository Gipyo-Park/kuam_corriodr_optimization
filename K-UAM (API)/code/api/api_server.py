from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
from math import atan2, degrees
import numpy as np

from risk_utils import batch_lla_to_utm, split_path_by_interval, interpolate_risk_with_heading, emergency_risk_along_path

app = FastAPI()
data = np.load("../../data/high_res_affected_population_GRC.npy")

class PathInput(BaseModel):
    points: List[Tuple[float, float, float]]  # LLA

class PointRiskInput(BaseModel):
    point: Tuple[float, float, float]  # LLA
    heading: float  # degrees

class EmergencyInput(BaseModel):
    plane: Tuple[float, float, float]    # LLA
    emergency_sites: List[Tuple[float, float, float]]  # LLA

@app.post("/path-risk")
def path_risk(input: PathInput):
    utm_pts = batch_lla_to_utm(input.points)
    segments = split_path_by_interval(utm_pts, interval=500)
    risks = [interpolate_risk_with_heading(data, *seg[0], 
        (degrees(atan2(seg[1][1]-seg[0][1], seg[1][0]-seg[0][0])) + 360)%360)
        for seg in segments]
    return {"segment_risks": risks}

@app.post("/point-risk")
def point_risk(input: PointRiskInput):
    utm = batch_lla_to_utm([input.point])[0]
    risk = interpolate_risk_with_heading(data, *utm, input.heading)
    return {"risk": risk}

@app.post("/emergency-risk")
def emergency_risk(input: EmergencyInput):
    plane_utm = batch_lla_to_utm([input.plane])[0]
    sites_utm = batch_lla_to_utm(input.emergency_sites)
    all_risks = emergency_risk_along_path(data, plane_utm, sites_utm)
    return {"emergency_paths": all_risks}
