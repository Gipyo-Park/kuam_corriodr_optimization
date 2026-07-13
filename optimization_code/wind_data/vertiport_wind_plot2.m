clc, clear all, close all
% =========================================================================
% 1. 공통 환경 및 좌표/마스크 설정
% =========================================================================
base_data = load('AirRisk_Data_1.mat', 'X_2d', 'Y_2d');
X_2d = base_data.X_2d;
Y_2d = base_data.Y_2d;
lat_center = 35.60613056;
lon_center = 129.07598611;
radius_m = 2200;
crs = projcrs(5179);
[x_center, y_center] = projfwd(crs, lat_center, lon_center);
dist_2d = sqrt((X_2d - x_center).^2 + (Y_2d - y_center).^2);
roi_mask = dist_2d <= radius_m;
idx_200 = 3;
idx_300 = 4;
theta_circle = linspace(0, 2*pi, 100);
x_circle_2200 = x_center + 2200 * cos(theta_circle);
y_circle_2200 = y_center + 2200 * sin(theta_circle);
x_circle_600 = x_center + 600 * cos(theta_circle);
y_circle_600 = y_center + 600 * sin(theta_circle);

% =========================================================================
% 2. 월별 데이터 로드 및 200~300m 평균 연산
% =========================================================================
[Ny, Nx] = size(X_2d);
M_U = cell(1, 12); M_V = cell(1, 12); M_W = cell(1, 12);
M_Udir_c = cell(1, 12); M_Vdir_c = cell(1, 12);
M_WindSpeed = cell(1, 12);
M_Udir = cell(1, 12);
M_Vdir = cell(1, 12);
for m = 1:12
    filename = sprintf('AirRisk_Data_%d.mat', m);
    
    if isfile(filename)
        data = load(filename, 'U3d', 'V3d', 'W3d', 'theta3d');
        
        U2 = data.U3d(:, :, idx_200); U3 = data.U3d(:, :, idx_300);
        V2 = data.V3d(:, :, idx_200); V3 = data.V3d(:, :, idx_300);
        W2 = data.W3d(:, :, idx_200); W3 = data.W3d(:, :, idx_300);
        T2 = data.theta3d(:, :, idx_200); T3 = data.theta3d(:, :, idx_300);
        
        inv_mask = (U2 == -1) | (U2 == 0) | (U3 == -1) | (U3 == 0) | ...
                   (T2 == -1) | (T2 == 0) | (T3 == -1) | (T3 == 0);
        
        U_avg = 0.5 * U2 + 0.5 * U3;
        V_avg = 0.5 * V2 + 0.5 * V3;
        W_avg = 0.5 * W2 + 0.5 * W3;
        
        U_dir_month = 0.5 * cosd(T2) + 0.5 * cosd(T3);
        V_dir_month = 0.5 * sind(T2) + 0.5 * sind(T3);
        
        U_avg(inv_mask) = NaN; V_avg(inv_mask) = NaN; W_avg(inv_mask) = NaN;
        U_dir_month(inv_mask) = NaN; V_dir_month(inv_mask) = NaN;
        
        M_U{m} = U_avg; M_V{m} = V_avg; M_W{m} = W_avg;
        M_Udir_c{m} = U_dir_month; M_Vdir_c{m} = V_dir_month;
        
        ws = sqrt(U_avg.^2 + V_avg.^2 + W_avg.^2);
        ud = U_dir_month; vd = V_dir_month;
        
        ws(~roi_mask) = NaN; ud(~roi_mask) = NaN; vd(~roi_mask) = NaN;
        
        M_WindSpeed{m} = ws; M_Udir{m} = ud; M_Vdir{m} = vd;
    else
        warning('%s 파일이 없습니다.', filename);
    end
end

% =========================================================================
% 3. 계절별 2차 합산 연산
% =========================================================================
season_names = {'봄 (Spring, 3~5월)', '여름 (Summer, 6~8월)', '가을 (Autumn, 9~11월)', '겨울 (Winter, 12~2월)'};
season_months = { [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 1, 2] };
S_WindSpeed = cell(1, 4);
S_Udir = cell(1, 4);
S_Vdir = cell(1, 4);
for s = 1:4
    months = season_months{s};
    
    U_sum = zeros(Ny, Nx); V_sum = zeros(Ny, Nx); W_sum = zeros(Ny, Nx);
    Ud_sum = zeros(Ny, Nx); Vd_sum = zeros(Ny, Nx);
    valid_count = zeros(Ny, Nx);
    
    for k = 1:3
        m = months(k);
        u = M_U{m}; v = M_V{m}; w = M_W{m};
        ud = M_Udir_c{m}; vd = M_Vdir_c{m};
        
        if isempty(u), continue; end 
        
        valid_idx = ~isnan(u);
        U_sum(valid_idx) = U_sum(valid_idx) + u(valid_idx);
        V_sum(valid_idx) = V_sum(valid_idx) + v(valid_idx);
        W_sum(valid_idx) = W_sum(valid_idx) + w(valid_idx);
        Ud_sum(valid_idx) = Ud_sum(valid_idx) + ud(valid_idx);
        Vd_sum(valid_idx) = Vd_sum(valid_idx) + vd(valid_idx);
        valid_count(valid_idx) = valid_count(valid_idx) + 1;
    end
    
    U_s_avg = U_sum ./ valid_count;
    V_s_avg = V_sum ./ valid_count;
    W_s_avg = W_sum ./ valid_count;
    Ud_s_avg = Ud_sum ./ valid_count;
    Vd_s_avg = Vd_sum ./ valid_count;
    
    ws_season = sqrt(U_s_avg.^2 + V_s_avg.^2 + W_s_avg.^2);
    
    ws_season(~roi_mask) = NaN; Ud_s_avg(~roi_mask) = NaN; Vd_s_avg(~roi_mask) = NaN;
    
    S_WindSpeed{s} = ws_season; S_Udir{s} = Ud_s_avg; S_Vdir{s} = Vd_s_avg;
end

% =========================================================================
% 3.5 글로벌 최댓값(Global Maximum) 도출: 모든 플롯의 스케일 통일
% =========================================================================
global_max_ws = 0;
for m = 1:12
    if isempty(M_WindSpeed{m}), continue; end
    tmp = M_WindSpeed{m}(:);
    v_max = max(tmp(~isnan(tmp)));
    if ~isempty(v_max) && v_max > global_max_ws
        global_max_ws = v_max;
    end
end
for s = 1:4
    tmp = S_WindSpeed{s}(:);
    v_max = max(tmp(~isnan(tmp)));
    if ~isempty(v_max) && v_max > global_max_ws
        global_max_ws = v_max;
    end
end
if isempty(global_max_ws) || isnan(global_max_ws) || global_max_ws <= 0
    global_max_ws = 1; 
end
warning('off', 'MATLAB:contour:NonFiniteData');

% =========================================================================
% 4. 시각화 [Figure 1] - 계절별 (2x2)
% =========================================================================
figure('Position', [50, 50, 1400, 1000], 'Name', '계절별 평균 유동장 (200~300m)');
for s = 1:4
    subplot(2, 2, s); hold on; grid on;
    
    [~, h] = contourf(X_2d, Y_2d, S_WindSpeed{s}, 20, 'LineColor', 'none'); 
    colormap('jet'); 
    caxis([0 global_max_ws]); 
    
    dir_norm = sqrt(S_Udir{s}.^2 + S_Vdir{s}.^2);
    q_u = S_Udir{s} ./ dir_norm; q_v = S_Vdir{s} ./ dir_norm;
    
    q = quiver(X_2d, Y_2d, q_u, q_v, 0.4, 'k', 'LineWidth', 0.8); q.MaxHeadSize = 0.5;
    
    plot(x_circle_2200, y_circle_2200, 'r--', 'LineWidth', 1.5);
    plot(x_circle_600, y_circle_600, 'm--', 'LineWidth', 1.5);
    plot(x_center, y_center, '^', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
    
    % [추가] 원 옆에 반경 텍스트 표시 (45도 방향)
    text(x_center + 2200*cosd(45), y_center + 2200*sind(45), ' R=2.2km', 'Color', 'r', 'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'r');
    text(x_center + 600*cosd(45), y_center + 600*sind(45), ' R=600m', 'Color', 'm', 'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'm');
    
    axis equal; xlim([x_center - radius_m, x_center + radius_m]); ylim([y_center - radius_m, y_center + radius_m]);
    title(season_names{s}, 'FontSize', 12, 'FontWeight', 'bold');
    
    cb = colorbar; cb.Label.String = '평균 풍속 (m/s)'; cb.Label.FontWeight = 'bold';
    hold off;
end
sgtitle('계절별 반경 2.2km 평균 풍향 및 풍속 (해발 고도 200~300m)', 'FontSize', 16, 'FontWeight', 'bold');

% =========================================================================
% 5. 시각화 [Figure 2] - 월별 (3x4)
% =========================================================================
figure('Position', [100, 100, 1800, 1200], 'Name', '월별 평균 유동장 (200~300m)');
for m = 1:12
    if isempty(M_WindSpeed{m}), continue; end 
    
    subplot(3, 4, m); hold on; grid on;
    
    [~, h] = contourf(X_2d, Y_2d, M_WindSpeed{m}, 20, 'LineColor', 'none'); 
    colormap('jet'); 
    caxis([0 global_max_ws]); 
    
    dir_norm = sqrt(M_Udir{m}.^2 + M_Vdir{m}.^2);
    q_u = M_Udir{m} ./ dir_norm; q_v = M_Vdir{m} ./ dir_norm;
    
    q = quiver(X_2d, Y_2d, q_u, q_v, 0.4, 'k', 'LineWidth', 0.8); q.MaxHeadSize = 0.5;
    
    plot(x_circle_2200, y_circle_2200, 'r--', 'LineWidth', 1.0);
    plot(x_circle_600, y_circle_600, 'm--', 'LineWidth', 1.0);
    plot(x_center, y_center, '^', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
    
    % [추가] 원 옆에 반경 텍스트 표시 (공간 확보를 위해 크기 소폭 조절)
    text(x_center + 2200*cosd(45), y_center + 2200*sind(45), ' R=2.2km', 'Color', 'r', 'FontSize', 9, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'r');
    text(x_center + 600*cosd(45), y_center + 600*sind(45), ' R=600m', 'Color', 'm', 'FontSize', 9, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'm');
    
    axis equal; xlim([x_center - radius_m, x_center + radius_m]); ylim([y_center - radius_m, y_center + radius_m]);
    
    title(sprintf('%d월 평균', m), 'FontSize', 12, 'FontWeight', 'bold');
    
    cb = colorbar; 
    cb.Label.String = '[m/s]'; 
    hold off;
end
sgtitle('월별 반경 2.2km 평균 풍향 및 풍속 (해발 고도 200~300m)', 'FontSize', 18, 'FontWeight', 'bold');