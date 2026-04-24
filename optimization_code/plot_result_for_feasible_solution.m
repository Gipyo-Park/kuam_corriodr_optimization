clc; clear; close all;
T = readtable("성공한 경로 final_path_debug_20260112_030600");
type = string(T.type);
verti  = T{type=="vertiport", ["lat","lon"]};
landing = T{type=="landing",  ["lat","lon"]};
takeoff = T{type=="takeoff",  ["lat","lon"]};
core = T{~ismember(type, ["vertiport","takeoff","landing"]), ["lat","lon"]};
% 전체 path: verti -> landing -> core -> takeoff -> verti
path_plot = [verti; landing; core; takeoff; verti];
% backbone: verti -> waypoint들(엑셀에 나온 순서) -> verti
wp = T{type=="waypoint", ["lat","lon"]};
backbone = [verti; wp; verti];
figure
gx = geoaxes; hold(gx,"on")
geobasemap(gx,"streets")
% path
geoplot(gx, path_plot(:,1), path_plot(:,2), "-k", "LineWidth", 1)
% backbone
geoplot(gx, backbone(:,1), backbone(:,2), "--", "LineWidth", 1)
% 포인트
geoscatter(gx, verti(1),  verti(2), 80, "r", "filled")
geoscatter(gx, landing(1),landing(2),60, "b", "filled")
geoscatter(gx, takeoff(1),takeoff(2),60, "g", "filled")
idx_wp = type=="waypoint";
geoscatter(gx, T.lat(idx_wp), T.lon(idx_wp), 10, "k", "filled")
idx_n = type=="node";
geoscatter(gx, T.lat(idx_n), T.lon(idx_n), 10, "m", "filled")
legend("path","backbone","vertiport","landing","takeoff","waypoint","node","Location","best")
title("Path + Backbone (Geo)")