% UAM Corridor Path Generation with NSGA-III
% v7 (Final Review & Refinement)
clear; clc; close all;

%% 0. Parameters
% --- General Parameters ---
altitude_levels    = [50 100 150 200 250 300];     % UAM이 비행할 수 있는 고도 레벨 목록 [m], value가 늘어나면 이것도 늘어난다
use_heading_map    = true;                         % 비행 방향(Heading)에 따라 다른 위험도 맵을 사용할지 여부
risk_percentile_search_list = [10, 20, 30, 40];    % 노드 생성 시, 탐색 영역 내 위험도 하위 *%에 해당하는 안전한 곳에만 노드 생성
W_buf              = 100;                          % Corridor 중심선에서 좌우로 노드를 탐색할 버퍼 폭 [m]
cell_size          = 100;                          % 위험도 맵의 기본 격자 크기 [m]
refine_scales      = [1.0 0.5 0.2 0.1];                % 두 지점이 가까울 때 격자를 얼마나 더 세분화할지 (100m, 50m, 20m, 10m)
delta_z_max        = 50;                           % 경로 내 인접 노드 간 최대 고도 변화 허용치 [m]
final_pick         = struct('risk',5,'dist',5,'pareto',5); % 최종적으로 각 카테고리별로 선택할 경로의 수
min_nodes_per_segment = 50;                        % 적응형 샘플링 시, 이 개수 이상의 노드를 찾으면 탐색을 중단




% --- NSGA-III Specific Parameters ---
Nmax               = 50;                           % 최적화를 진행할 총 세대(Generation) 수
N_pop              = 50;                           % 한 세대에서 유지할 인구(해)의 수 (N)
offspring_ratio    = 2;                            % 부모 세대 대비 생성할 자손의 비율 (e.g., 2 -> 2*N_pop 개의 자손 생성)
H_ref_points       = 10;                           % 참조점(Reference Point) 생성 시 사용할 분할 수 (다양성 확보에 사용)

%% 1. Load Risk Maps
% GRC_l*.xlsx 파일들을 읽어 4차원 위험도 텐서(RiskTensor)를 생성합니다.
% RiskTensor 차원: [고도 x 헤딩 x 위도 x 경도]
file_list = { 'GRC_l3.xlsx','GRC_l4.xlsx','GRC_l5.xlsx','GRC_l6.xlsx', ...
              'GRC_l7.xlsx','GRC_l8.xlsx','GRC_l9.xlsx','GRC_l10.xlsx' };
disp('Loading risk maps...');
RiskTensor = load_grc_maps(file_list, altitude_levels);
disp('Risk maps loaded successfully.');

%% 2. Define Vertiport and Corridor Points
vertiport = [35.6033361, 129.0776917, 150]; % [lat, lon, alt]
corridor_lat = [35.5845917 35.6026528 35.6326806 35.6249583 35.6034750 35.5845361 35.5692361 35.5546444 35.5586722 35.5784750 35.5843722 35.6163861 35.6212528 35.6109972];
corridor_lon = [129.0936472 129.1130667 129.1238583 129.1335528 129.1268194 129.1076472 129.1085306 129.0936972 129.0816611 129.0916889 129.0770000 129.0613944 129.0725444 129.0711889];
corridor_alt = 150 * ones(size(corridor_lat)); % TODO : 위험도맵이 높이변화가 어떻게 되는지 알게되면 지정해주기

% Vertiport에서 시작하여 모든 Corridor Point를 거쳐 다시 Vertiport로 돌아오는 폐쇄 루프(Closed-Loop) 경로를 생성합니다.
points = [vertiport; [corridor_lat', corridor_lon', corridor_alt']; vertiport];
flight_dist_limit = 100; % 경로 내 인접 노드 간 최대 거리 제약
forbidden_zones   = [];  % 금지 구역 설정 (현재는 비어있음)
emergency_lat = [35.6201083, 35.5678222, 35.5919889];
emergency_lon = [129.1191806, 129.106728, 129.0751972];

% 모든 좌표를 포함하여 지도의 경계를 설정합니다.
all_lat = [points(:,1)', emergency_lat];
all_lon = [points(:,2)', emergency_lon];
lat_lim = [min(all_lat)-0.01, max(all_lat)+0.01];
lon_lim = [min(all_lon)-0.01, max(all_lon)+0.01];

%% 3. Initialize Figure 1 for Process Visualization
% 실시간 진행 과정과 최종 상세 결과를 표시할 Figure 1을 생성합니다.
fig1 = figure('Name', 'Figure 1: Comprehensive Process & Results', 'NumberTitle', 'off', 'WindowState', 'maximized');
gx = geoaxes;
geobasemap(gx, 'topographic');
geolimits(gx, lat_lim, lon_lim);
hold(gx, 'on'); grid(gx, 'on');
title(gx, 'UAM Pathfinding Initialized');

% 지도에 고정된 요소들 (Corridor, Vertiport 등)을 미리 그립니다.
geoplot(gx, points(:,1), points(:,2), 'co-', 'LineWidth', 1.5, 'DisplayName', 'Corridor');
geoscatter(gx, points(2:end-1,1), points(2:end-1,2), 50, 'c', 'filled', 'DisplayName', 'Corridor Points');
geoplot(gx, vertiport(1), vertiport(2), 'mp', 'MarkerSize', 15, 'MarkerFaceColor','m', 'DisplayName','VertiPort');
geoscatter(gx, emergency_lat, emergency_lon, 80, 'b', 'filled', 'DisplayName','Emergency Landing');
legend(gx, 'Location','best');

%% 4. Per-Segment NSGA-III Optimization with Live Visualization
% 전체 경로를 한 번에 탐색하는 것이 아니라, 각 구간(segment)별로 순차적으로 탐색을 진행합니다.
num_segments = size(points,1) - 1;
full_paths = cell(num_segments, 1);
representative_paths = cell(num_segments, 3);
h_all_nodes = []; % [디버깅] 모든 노드 시각화 핸들, 생성된 노드를 화면에 표시하고 지우기 위한 핸들
h_rep_nodes = []; % 대표 노드 시각화 핸들, 생성된 노드를 화면에 표시하고 지우기 위한 핸들

for k = 1:num_segments
    p1 = points(k,:);
    p2 = points(k+1,:);
    title(gx, sprintf('Processing Segment %d of %d...', k, num_segments));
    
    % --- VISUALIZATION 1: 노드 생성 및 확인 ---
    fprintf('Segment %d: Generating nodes...\n', k);
    % 함수가 2개의 출력을 반환: 알고리즘용(nodes), 시각화용(all_safe_nodes)
    [nodes, all_safe_nodes] = generate_nodes_3d_segment(p1, p2, RiskTensor, altitude_levels, W_buf, cell_size, refine_scales, delta_z_max, lat_lim, lon_lim, min_nodes_per_segment, risk_percentile_search_list);
    
    % 2단계 시각화: 전체 후보군(회색) + 최종 대표 노드(검정) , 이전 구간의 노드 삭제
    if ishandle(h_all_nodes), delete(h_all_nodes); end
    if ishandle(h_rep_nodes), delete(h_rep_nodes); end

    if ~isempty(all_safe_nodes)
        h_all_nodes = geoscatter(gx, all_safe_nodes(:,1), all_safe_nodes(:,2), 10, '.', 'MarkerEdgeColor', [0.7 0.7 0.7], 'DisplayName', 'All Safe Nodes (Viz)');
    end
    if ~isempty(nodes)
        h_rep_nodes = geoscatter(gx, nodes(:,1), nodes(:,2), 25, 'k', '.', 'DisplayName', 'Representative Nodes (Used by GA)');
    end

    fprintf('Node generation complete. Starting optimization...\n');
    pause(1);
    
    % --- NSGA-III 알고리즘 실행 ---
    fprintf('Segment %d: Running NSGA-III...\n', k);
    [population, f_vals] = run_nsga3_segment(nodes, p1, p2, RiskTensor, ...
        use_heading_map, flight_dist_limit, forbidden_zones, ...
        Nmax, N_pop, offspring_ratio, H_ref_points, gx);
    
    % --- 최종 해 선택 및 저장 ---
    solutions = select_final_solutions(population, f_vals, final_pick, H_ref_points);
    full_paths{k} = solutions; % Figure 1의 최종 종합 표시를 위해 모든 해 저장

    % --- VISUALIZATION 3: 구간별 결과 및 대표 경로 누적 표시 ---
    title(gx, sprintf('Segment %d: Displaying 15 candidate paths', k));
    h_temp_paths = plot_solutions(gx, solutions, {'r--','b:','g-.'}, 0.5); % 15개 후보군 잠시 표시
    pause(5);
    
    % 15개 후보군 중에서 가장 대표적인 경로 3개 선정
    [rep_risk, rep_dist, rep_pareto] = select_representative_paths(population, f_vals, p1, p2);
    representative_paths(k,:) = {rep_risk, rep_dist, rep_pareto};
    
    delete(h_temp_paths); % 임시 후보군 삭제
    
    % 대표 경로 3개만 굵은 선으로 누적해서 표시
    plot_solutions(gx, {rep_risk; rep_dist; rep_pareto}, {'r-','b-','g-'}, 2.0, k);
    title(gx, sprintf('Segment %d Complete.', k));
    fprintf('Segment %d Complete.\n\n', k);
    pause(1);
end

%% 5. Final Visualization
% 두 종류의 최종 결과물을 별도의 Figure로 생성합니다.
if ishandle(h_all_nodes), delete(h_all_nodes); end % 마지막 구간의 노드 삭제
if ishandle(h_rep_nodes), delete(h_rep_nodes); end % 마지막 구간의 노드 삭제

% --- Figure 1: 모든 상세 경로 종합 표시 ---
title(gx, 'Final UAM Corridor Paths (All Candidates)');
disp('Displaying all candidate paths on Figure 1...');
for k = 1:num_segments
    plot_solutions(gx, full_paths{k}, {'r:','b:','g:'}, 0.2);
end
legend(gx, 'Location','best');
disp('Figure 1: Comprehensive results displayed.');

% --- Figure 2: 최종 추천 루트 ---
fig2 = figure('Name', 'Figure 2: Optimal End-to-End Routes', 'NumberTitle', 'off', 'WindowState', 'maximized');
gx2 = geoaxes;
geobasemap(gx2, 'topographic');
geolimits(gx2, lat_lim, lon_lim);
hold(gx2, 'on'); grid(gx2, 'on');
title(gx2, 'Optimal End-to-End Routes');
geoplot(gx2, points(:,1), points(:,2), 'co-', 'LineWidth', 1.5);
geoscatter(gx2, points(2:end-1,1), points(2:end-1,2), 50, 'c', 'filled');
geoplot(gx2, vertiport(1), vertiport(2), 'mp', 'MarkerSize', 15, 'MarkerFaceColor','m');
geoscatter(gx2, emergency_lat, emergency_lon, 80, 'b', 'filled');
disp('Displaying final recommended routes on Figure 2...');

% 각 구간별 대표 경로들을 하나로 이어 붙여 최종 루트 생성
final_route_risk = cell2mat(representative_paths(:,1));
final_route_dist = cell2mat(representative_paths(:,2));
final_route_pareto = cell2mat(representative_paths(:,3));

% 최종 합의된 색상 규칙(위험-빨강, 거리-파랑, 균형-초록)으로 플로팅
geoplot(gx2, final_route_risk(:,1), final_route_risk(:,2), 'r-', 'LineWidth', 2.5, 'DisplayName', 'Overall Minimum Risk Route');
geoplot(gx2, final_route_dist(:,1), final_route_dist(:,2), 'b-', 'LineWidth', 2.5, 'DisplayName', 'Overall Shortest Distance Route');
geoplot(gx2, final_route_pareto(:,1), final_route_pareto(:,2), 'g-', 'LineWidth', 2.5, 'DisplayName', 'Overall Balanced Route');
legend(gx2, 'Location','best');
disp('All tasks completed.');

%% --- 로컬 함수들 ---
function RiskTensor = load_grc_maps(files, alt_levels)
    % 제공된 코드와 동일
    H = numel(files);
    A = numel(alt_levels);
    
    temp_T = readtable(files{1});
    I = temp_T.i+1;
    J = temp_T.j+1;
    Ny = max(J);
    Nx = max(I);
    
    RiskTensor = zeros(A, H, Ny, Nx);

    for hi = 1:H
        T = readtable(files{hi});
        data = table2array(T(:, 3:end));
        I = T.i+1;
        J = T.j+1;
        for ai = 1:A
            layer_data = data(:, ai);
            layer_data(layer_data<0) = 0;
            M = accumarray([J I], layer_data, [Ny Nx]);
            RiskTensor(ai,hi,:,:) = M;
        end
    end
end

function [nodes, all_safe_nodes] = generate_nodes_3d_segment(p1, p2, RT, alt_levels, W, cs, scales, dz, lat_lim, lon_lim, min_nodes, pct_list)
%{
[최종 버전] 지능형 노드 생성 함수
이 함수는 다음과 같은 복합적인 전략을 사용하여 최적의 노드 그룹을 생성합니다.
  1. 적응형 샘플링: 성긴 격자(100m)부터 조밀한 격자(10m) 순으로 탐색하여 효율성 확보.
  2. 비례 샘플링: 탐색 공간의 실제 거리에 비례하여 샘플링 밀도를 동적으로 조절.
  3. 회전된 경계 상자: 비행 경로 방향에 맞는 정확한 탐색 공간을 설정하여 불필요한 노드 생성 방지.
  4. 누적/중복제거 기반 위험도 필터링: 가장 안전한 기준부터 순차적으로 탐색하며 최소 노드 개수를 만족하는 지점을 찾음.
  5. 하이브리드 대표 노드 선정: 최종적으로 찾은 모든 안전 노드 중에서 각 (x,y) 위치당 가장 안전한 고도의 노드 하나만을 선택하여 알고리즘에 제공.
%}

    %% --- 1. 초기 설정 및 그리드/좌표계 정의 ---
    % RiskTensor의 크기 및 위경도-그리드 변환 계수 계산
    [~, ~, Ny, Nx] = size(RT);
    minLat = lat_lim(1); maxLat = lat_lim(2);
    minLon = lon_lim(1); maxLon = lon_lim(2);
    dLat_deg = (maxLat - minLat) / (Ny-1);
    dLon_deg = (maxLon - minLon) / (Nx-1);

    % p1, p2의 위경도 좌표를 그리드 인덱스로 변환
    j1 = (p1(1) - minLat) / dLat_deg + 1;
    i1 = (p1(2) - minLon) / dLon_deg + 1;
    j2 = (p2(1) - minLat) / dLat_deg + 1;
    i2 = (p2(2) - minLon) / dLon_deg + 1;
    p1_grid = [i1, j1]; p2_grid = [i2, j2];
    
    % '회전된 경계 상자'를 위한 지역 좌표계 설정
    vec = p2_grid - p1_grid;
    len = norm(vec);
    if len < 1e-6, nodes = []; all_safe_nodes = []; return; end
    
    u_vec = vec / len; 
    v_vec = [-u_vec(2), u_vec(1)];
    W_grid = W / (111000 * cosd(mean([p1(1),p2(1)]))) / dLon_deg;

    %% --- 2. 적응형 & 누적 탐색 루프 ---
    nodes_idx = []; % 최종 알고리즘에 사용될 대표 노드
    all_safe_nodes_idx = []; % 시각화용 모든 안전 노드
    goto_end_node_generation = false; % 루프 탈출 플래그
    
    % Outer Loop: 격자 크기를 100m -> 50m ... 순으로 변경하며 탐색
    grid_scales = [1, scales];
    for gs = grid_scales
        fprintf('  - Searching with grid scale: %.1f (Density: %.1fm)\n', gs, cs*gs);
        
        % --- 3. 거리에 비례한 후보 노드 샘플링 ---
        i_min_wide = floor(min(i1,i2) - W_grid*2); i_max_wide = ceil(max(i1,i2) + W_grid*2);
        j_min_wide = floor(min(j1,j2) - W_grid*2); j_max_wide = ceil(max(j1,j2) + W_grid*2);
        width_m  = (i_max_wide - i_min_wide) * dLon_deg * 111000 * cosd(p1(1));
        height_m = (j_max_wide - j_min_wide) * dLat_deg * 111000;
        
        sampling_density_m = cs * gs;
        num_samples_i = max(2, round(width_m / sampling_density_m));
        num_samples_j = max(2, round(height_m / sampling_density_m));
        
        i_vec = linspace(i_min_wide, i_max_wide, num_samples_i);
        j_vec = linspace(j_min_wide, j_max_wide, num_samples_j);
        [I, J] = meshgrid(i_vec, j_vec);
        candidate_nodes_grid = [I(:), J(:)];
    
        % --- 4. 1차 필터링 (기하학적 조건) ---
        vec_candidates = candidate_nodes_grid - p1_grid;
        dist_lon = vec_candidates * u_vec';
        dist_lat = vec_candidates * v_vec';
        in_box_mask = (dist_lon >= 0) & (dist_lon <= len) & (abs(dist_lat) <= W_grid);
        filtered_nodes_grid = candidate_nodes_grid(in_box_mask, :);
        
        % Inner Loop: 위험도 퍼센트 기준을 순차적으로 높여가며 탐색
        cumulative_nodes_idx = []; 
        for pct = pct_list
            
            % --- 5. 2차 필터링 (위험도 조건) ---
            % 현재 퍼센트(pct) 기준으로 안전한 노드를 찾음
            current_pct_nodes_idx = [];
            alt_diff = abs(alt_levels - p1(3));
            valid_alt_idx = find(alt_diff <= dz);
            if isempty(valid_alt_idx), [~, valid_alt_idx] = min(alt_diff); end

            for ai = valid_alt_idx
                M = squeeze(mean(RT(ai,:,:,:),2));
                Ii = round(filtered_nodes_grid(:,1));
                Ji = round(filtered_nodes_grid(:,2));
                valid = Ii>=1 & Ii<=Nx & Ji>=1 & Ji<=Ny;
                
                risks_in_box = M(sub2ind([Ny,Nx], Ji(valid), Ii(valid)));
                if isempty(risks_in_box), continue; end
                
                thr = prctile(risks_in_box, pct);
                
                sel_mask = risks_in_box <= thr;
                
                newly_found_nodes = filtered_nodes_grid(valid,:);
                newly_found_nodes = newly_found_nodes(sel_mask, :);
                
                alt_col = alt_levels(ai) * ones(size(newly_found_nodes,1),1);
                current_pct_nodes_idx = [current_pct_nodes_idx; newly_found_nodes, alt_col];
            end
            
            % --- 6. 누적 및 중복 제거 ---
            % 현재 퍼센트 단계에서 찾은 노드를 포함하여 전체 안전 노드 목록 업데이트
            cumulative_nodes_idx = unique([cumulative_nodes_idx; current_pct_nodes_idx], 'rows');
            
            % --- 7. 하이브리드 대표 노드 선정 (루프 내부로 이동) ---
            % 현재까지 찾은 모든 안전 노드 중에서, (i,j) 위치별 가장 안전한 노드만 선택
            if ~isempty(cumulative_nodes_idx)
                unique_xy = unique(cumulative_nodes_idx(:, 1:2), 'rows');
                representative_nodes_idx = zeros(size(unique_xy,1), 3);
                for i_xy = 1:size(unique_xy,1)
                    current_xy = unique_xy(i_xy,:);
                    mask = all(cumulative_nodes_idx(:,1:2) == current_xy, 2);
                    nodes_at_xy = cumulative_nodes_idx(mask,:);
                    
                    risks_at_xy = zeros(size(nodes_at_xy,1),1);
                    for j_alt = 1:size(nodes_at_xy,1)
                        node_info = nodes_at_xy(j_alt,:);
                        [~, alt_idx] = min(abs(alt_levels - node_info(3)));
                        M = squeeze(mean(RT(alt_idx,:,:,:),2));
                        risks_at_xy(j_alt) = M(round(node_info(2)), round(node_info(1)));
                    end
                    [~, min_risk_idx] = min(risks_at_xy);
                    representative_nodes_idx(i_xy, :) = nodes_at_xy(min_risk_idx, :);
                end
            else
                representative_nodes_idx = [];
            end
            
            fprintf('    - Risk Percentile <= %d%% | Found %d representative nodes (Target: %d)\n', pct, size(representative_nodes_idx,1), min_nodes);
            
            % [의도확인 ✓] 성공 조건: '대표 노드'의 개수가 최소치를 만족하는지 확인
            if size(representative_nodes_idx, 1) >= min_nodes
                nodes_idx = representative_nodes_idx;
                all_safe_nodes_idx = cumulative_nodes_idx;
                goto_end_node_generation = true;
                break; % Inner loop 탈출
            end
        end % Inner Loop 끝
        
        if goto_end_node_generation
            break; % Outer loop 탈출
        end
    end % Outer Loop 끝
    
    % --- 8. 최종 변환 및 반환 ---
    % 루프를 모두 돌았는데도 노드를 못 찾은 경우 Fallback 처리
    if isempty(nodes_idx)
        mid_grid = (p1_grid + p2_grid)/2;
        nodes_idx = [mid_grid, p1(3)];
        all_safe_nodes_idx = nodes_idx;
    end
    
    % 시각화용 노드: 위경도로 변환
    N_all = size(all_safe_nodes_idx,1);
    all_safe_nodes = zeros(N_all,3);
    for n = 1:N_all
        ii = all_safe_nodes_idx(n,1); jj = all_safe_nodes_idx(n,2);
        lat = minLat + (jj-1)*dLat_deg;
        lon = minLon + (ii-1)*dLon_deg;
        all_safe_nodes(n,:) = [lat, lon, all_safe_nodes_idx(n,3)];
    end
    
    % 알고리즘용 대표 노드: 순서를 섞고 위경도로 변환
    nodes_idx = nodes_idx(randperm(size(nodes_idx,1)),:);
    N_rep = size(nodes_idx,1);
    nodes = zeros(N_rep,3);
    for n = 1:N_rep
        ii = nodes_idx(n,1); jj = nodes_idx(n,2);
        lat = minLat + (jj-1)*dLat_deg;
        lon = minLon + (ii-1)*dLon_deg;
        nodes(n,:) = [lat, lon, nodes_idx(n,3)];
    end
end

function h = plot_solutions(gx, solutions_cell, styles, width, seg_num)
    h = gobjects(0);
    labels     = {'Risk','Dist','Pareto'};
    for g = 1:size(solutions_cell, 1)
        for i = 1:size(solutions_cell, 2)
            path = solutions_cell{g,i};
            if isempty(path), continue; end
            
            d_name = ' ';
            if nargin > 4 
                if g == 1, current_label = labels{1};
                elseif g == 2, current_label = labels{2};
                else, current_label = labels{3};
                end
                d_name = sprintf('Seg%d %s Rep.', seg_num, current_label);
            end

            h(end+1) = geoplot(gx, path(:,1), path(:,2), styles{g}, 'LineWidth', width, 'DisplayName', d_name);
        end
    end
end

function [rep_r, rep_d, rep_p] = select_representative_paths(population, fvals, p1, p2)
    % 1. 마스터 폴백: 유효한 해가 전혀 없는 경우, 직선 경로를 반환.
    if isempty(population) || isempty(fvals)
        warning('Population is empty for this segment. Creating a straight-line fallback path for all categories.');
        fallback_path = [p1; p2]; % p1과 p2로 직선 경로 생성
        rep_r = fallback_path;
        rep_d = fallback_path;
        rep_p = fallback_path;
        return;
    end

    % 2. 기본 선택: Risk와 Distance에 대한 최적해는 항상 찾을 수 있음.
    [~, min_risk_idx] = min(fvals(:,2));
    rep_r = population{min_risk_idx};

    [~, min_dist_idx] = min(fvals(:,1));
    rep_d = population{min_dist_idx};

    % 3. Pareto 경로 선택 및 개별 폴백
    F = fast_non_dominated_sort(fvals);
    if isempty(F) || isempty(F{1})
        % Pareto Front가 없는 경우, Risk 최적해를 Balanced 대체제로 사용.
        warning('Pareto front is empty. Using the best risk path as a balanced alternative.');
        rep_p = rep_r; 
    else
        % 정상적으로 Pareto Front가 있는 경우
        front1_indices = F{1};
        f_pareto = fvals(front1_indices,:);
        
        if size(f_pareto,1) > 1
            % 2개 이상의 해가 Front에 있을 때
            range = max(f_pareto) - min(f_pareto);
            range(range==0) = 1; % 분모가 0이 되는 것을 방지
            norm_f = (f_pareto - min(f_pareto)) ./ range;
            
            dist_to_utopia = sqrt(sum(norm_f.^2, 2));
            [~, best_pareto_sub_idx] = min(dist_to_utopia);
            best_pareto_global_idx = front1_indices(best_pareto_sub_idx);
        else
            % Front에 해가 하나만 있을 때
            best_pareto_global_idx = front1_indices(1);
        end
        rep_p = population{best_pareto_global_idx};
    end
end

function new_population = selection_nsga3(population, f_vals, feasible_flags, N, ref_points)
    % 이 함수는 NSGA-III의 핵심 선택 로직을 구현합니다.
    % Front를 채우고, 마지막 Front에서는 참조점 기반 Niching으로 다양성을 확보합니다.
    new_population = {};
    count = 0;
    front_idx = 1;

    Fronts = fast_non_dominated_sort(f_vals);

    % Front 1부터 순서대로 N개가 찰 때까지 new_population에 추가
    while front_idx <= numel(Fronts)
        front_indices = Fronts{front_idx};
        valid_indices = front_indices(feasible_flags(front_indices));
        
        if isempty(valid_indices)
            front_idx = front_idx + 1;
            continue;
        end

        if count + numel(valid_indices) <= N
            new_population = [new_population; population(valid_indices)];
            count = count + numel(valid_indices);
            front_idx = front_idx + 1;
        else
            % 마지막 Front에서는 N개를 채우기 위해 일부만 선택
            remaining = N - count;
            last_front_indices = valid_indices;
            last_front_fvals = f_vals(last_front_indices, :);
            
            norm_f = normalize_objectives(last_front_fvals);
            
            % Niching을 통해 다양성이 높은 해들을 선택
            selected_idx_in_last_front = niching_selection(norm_f, ref_points, remaining);
            new_population = [new_population; population(last_front_indices(selected_idx_in_last_front))];
            break; % N개를 모두 채웠으므로 루프 종료
        end
    end
end

function [population, f_vals] = run_nsga3_segment(nodes, p1, p2, RiskTensor, ...
    use_heading_map, flight_dist_limit, forbidden_zones, ...
    Nmax, N_pop, offspring_ratio, H, gx)
    % 이 함수는 NSGA-III 알고리즘의 메인 루프를 담당합니다.

    % 초기 집단(경로) 생성
    population = generate_initial_population_GP(N_pop, nodes, p1, p2);
    
    % NSGA-III의 핵심인 참조점(Reference Point) 생성
    ref_points = generate_reference_points_2obj(H);

    h_ga_paths = gobjects(0); % 시각화 핸들

    % Nmax 세대만큼 최적화 반복
    for gen = 1:Nmax
        fprintf('  - Generation %d/%d\n', gen, Nmax);
        
        % --- 1. 평가 (Evaluation) ---
        % 현재 세대의 모든 경로(해)에 대해 거리와 위험도 계산
        Np   = numel(population);
        f_vals   = zeros(Np, 2);
        feasible_flags = false(Np, 1);
        for i = 1:Np
            [f_vals(i,:), feasible_flags(i)] = evaluate_objectives_with_constraints_GP( ...
                population{i}, RiskTensor, use_heading_map, p1, p2, ...
                flight_dist_limit, forbidden_zones);
        end

        % --- VISUALIZATION 2: 최적화 과정 실시간 모니터링 ---
        if nargin > 11 && ishandle(gx)
             delete(h_ga_paths(isgraphics(h_ga_paths)));
             h_ga_paths = gobjects(min(Np, 50), 1);
             plot_indices = randperm(Np, min(Np, 50));
             for k_plot=1:numel(plot_indices)
                 idx = plot_indices(k_plot);
                 path = population{idx};
                 h_ga_paths(k_plot) = geoplot(gx, path(:,1), path(:,2), '-', 'Color', [0.5 0.5 0.5 0.3]);
             end
             drawnow limitrate;
        end

        % --- 2. 선택 (Selection) ---
        % NSGA-III의 참조점 기반 선택 로직을 사용하여 다음 세대에 생존할 N_pop개의 해를 선택
        new_population = selection_nsga3(population, f_vals, feasible_flags, N_pop, ref_points);

        % --- 3. 변이 (Variation) ---
        % 마지막 세대가 아니면, 선택된 해들을 기반으로 Crossover와 Mutation을 통해 자손을 생성
        if gen < Nmax
            offspring = variation_nsga3(new_population, nodes, offspring_ratio);
            population = [new_population; offspring]; % 부모와 자손을 합쳐 다음 세대의 탐색 집단으로 사용
        else
            population = new_population; % 마지막 세대는 선택된 최적해 집단만 남김
        end
    end

    % --- 최종 값 계산 ---
    % 마지막 세대의 최종 해들에 대한 정확한 f_vals 값을 다시 계산
    Np = numel(population);
    f_vals = zeros(Np, 2);
    for i = 1:Np
        [f_vals(i,:), ~] = evaluate_objectives_with_constraints_GP( ...
            population{i}, RiskTensor, use_heading_map, p1, p2, ...
            flight_dist_limit, forbidden_zones);
    end
    delete(h_ga_paths(isgraphics(h_ga_paths)));
end



function cd = crowding_distance_local(fvals)
    M = size(fvals,1);
    cd = zeros(M,1);
    if M<=2, cd(:) = Inf; return; end
    for j = 1:size(fvals,2)
        [v, idx] = sort(fvals(:,j));
        cd(idx(1))   = Inf;
        cd(idx(end)) = Inf;
        range = v(end) - v(1);
        if range == 0, continue; end
        for k = 2:M-1
            cd(idx(k)) = cd(idx(k)) + (v(k+1)-v(k-1))/range;
        end
    end
end

function offspring = variation_nsga3(pop, nodes, ratio)
    % Crossover와 Mutation을 통해 자손(offspring)을 생성하는 함수
    pop_size = numel(pop);
    if pop_size == 0, offspring = {}; return; end

    offspring_num = round(pop_size * ratio);
    offspring = cell(offspring_num,1);

    for i = 1:offspring_num
        p1_idx = randi(pop_size);
        p2_idx = randi(pop_size);
        
        parent1 = pop{p1_idx};
        parent2 = pop{p2_idx};
    
        child = crossover_GP(parent1, parent2);
        mutated_child = mutation_GP(child, nodes);
        offspring{i} = mutated_child;
    end
end

function solutions = select_final_solutions(pop, fv, pick, H)
    % 최종 세대에서, 시각화를 위해 3가지 카테고리별로 *개씩의 경로를 선택
    if isempty(pop) || isempty(fv)
        solutions = cell(3, pick.risk); 
        return;
    end
    
    % 1) Risk-only: 위험도가 가장 낮은 *개
    [~, idxR] = sort(fv(:,2), 'ascend');
    idxR = idxR(1:min(pick.risk, numel(idxR)));

    % 2) Distance-only: 거리가 가장 짧은 *개
    [~, idxD] = sort(fv(:,1), 'ascend');
    idxD = idxD(1:min(pick.dist, numel(idxD)));

    % 3) Pareto-only: NSGA-III의 Niching 방식으로 다양성이 높은 *개
    F = fast_non_dominated_sort(fv);
    if isempty(F) || isempty(F{1})
        selP = idxR(1:min(pick.pareto, numel(idxR))); % 대체 로직
    else
        front1 = F{1};
        if numel(front1) <= pick.pareto
            selP = front1;
        else
            % Front1 내에서 Niching으로 *개 선택
            ref_points = generate_reference_points_2obj(H);
            norm_f1 = normalize_objectives(fv(front1,:));
            idx_in_front1 = niching_selection(norm_f1, ref_points, pick.pareto);
            selP = front1(idx_in_front1);
        end
    end

    % 4) Cell 배열에 모으기
    Pmax = max([pick.risk, pick.dist, pick.pareto]);
    solutions = cell(3, Pmax);
    for i = 1:numel(idxR), solutions{1,i} = pop{idxR(i)}; end
    for i = 1:numel(idxD), solutions{2,i} = pop{idxD(i)}; end
    for i = 1:numel(selP), solutions{3,i} = pop{selP(i)}; end
end