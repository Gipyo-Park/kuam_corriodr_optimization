function f = evaluate_objectives_GP(path, RT, use_heading_map)
    % INPUT
    %   path            : N×3 array, [x y alt] (grid coords)
    %   RT              : A×H×Ny×Nx tensor (A=고도 레벨 수, H=헤딩 수)
    %   use_heading_map : true/false
    %
    % OUTPUT
    %   f = [total_dist, cumulative_risk]

    % 0) 고도 레벨 정보 가져오기 (메인 워크스페이스)
    altitude_levels = evalin('base','altitude_levels');  % [50,100,...]

    % 1) 총 3D 거리 계산
    diffs = diff(path(:,1:3),1,1);          % [dx dy dz]
    total_dist = sum(sqrt(sum(diffs.^2,2)));

    % 2) 누적 위험도 계산
    cumulative_risk = 0;
    [A, H, Ny, Nx] = size(RT);

    for i = 1:size(path,1)-1
        p1 = path(i,   :);
        p2 = path(i+1, :);

        % -- 헤딩 인덱스 결정 --
        if use_heading_map
            vec = p2(1:2) - p1(1:2);
            theta = atan2d(vec(2), vec(1));    % [-180,180]
            if theta < 0, theta = theta + 360; end
            rounded = round(theta/45)*45;
            if rounded == 360, rounded = 0; end
            head_idx = rounded/45 + 1;         % 1..8
        else
            head_idx = 1;
        end

        % -- 고도 인덱스 결정 (가장 근접한 레벨) --
        alt = p1(3);
        [~, alt_idx] = min(abs(altitude_levels - alt));  % 1..A

        % -- 해당 2D 위험도 맵 슬라이스 --
        currentRiskMap = squeeze(RT(alt_idx, head_idx, :, :));  % Ny×Nx

        % -- 보간된 점 생성 및 위험도 누적 --
        pts = interpolate_line(p1(1:2), p2(1:2));  % [x y] pairs
        for k = 1:size(pts,1)
            x = round(pts(k,1));  y = round(pts(k,2));
            if x>=1 && x<=Nx && y>=1 && y<=Ny
                cumulative_risk = cumulative_risk + currentRiskMap(y, x);
            end
        end
    end

    f = [total_dist, cumulative_risk];
end


% 보조 함수 (2D 보간)
function pts = interpolate_line(p1, p2)
    n = ceil(norm(p2 - p1));
    if n == 0
        pts = p1;
    else
        x = linspace(p1(1), p2(1), n);
        y = linspace(p1(2), p2(2), n);
        pts = [x', y'];
    end
end
