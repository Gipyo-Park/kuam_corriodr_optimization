function [f_val, feasible] = evaluate_objectives_with_constraints_GP(path, RiskTensor, use_heading_map, p1, p2, flight_dist_limit, forbidden_zones)
    penalty_cost = 1e6; % 충분히 큰 값
    feasible = true;

    % 1) 다음 노드 간 거리 체크
    for i = 1:size(path,1)-1
        dist = norm(path(i+1,1:2) - path(i,1:2));
        if dist > flight_dist_limit
            feasible = false;
            f_val = [penalty_cost, penalty_cost];
            return;
        end
    end

    % 2) 금지 구역 체크
    % 금지 구역 체크 (선분 통과 여부 포함)
    for i = 1:size(path,1)-1
        p1 = path(i,1:2);
        p2 = path(i+1,1:2);
        for j = 1:size(forbidden_zones,1)
            rect = forbidden_zones(j,:) + [-2 2 -2 2];  % [xmin xmax ymin ymax]
            if violates_forbidden_zone(p1, p2, rect)
                feasible = false;
                f_val = [penalty_cost, penalty_cost];
                return;
            end
        end
    end

    % 3) 제약 만족 시 원래 목적 함수 계산
    f_val = evaluate_objectives_GP(path, RiskTensor, use_heading_map);
    feasible = true;
end

function violated = violates_forbidden_zone(p1, p2, rect)
    % 금지 구역 내부 통과 또는 경계 교차 여부
    violated = false;

    if is_inside_rect(p1, rect) || is_inside_rect(p2, rect)
        violated = true;
        return;
    end

    % 사각형의 네 변과의 교차 여부 확인
    [xmin, xmax, ymin, ymax] = deal(rect(1), rect(2), rect(3), rect(4));
    edges = [
        xmin, ymin; xmax, ymin;
        xmax, ymin; xmax, ymax;
        xmax, ymax; xmin, ymax;
        xmin, ymax; xmin, ymin
    ];

    for i = 1:2:size(edges,1)
        q1 = edges(i,:);
        q2 = edges(i+1,:);
        if segments_intersect(p1, p2, q1, q2)
            violated = true;
            return;
        end
    end
end

function inside = is_inside_rect(p, rect)
    inside = (rect(1) <= p(1) && p(1) <= rect(2)) && ...
             (rect(3) <= p(2) && p(2) <= rect(4));
end

function flag = segments_intersect(p1, p2, q1, q2)
    d1 = direction(q1, q2, p1);
    d2 = direction(q1, q2, p2);
    d3 = direction(p1, p2, q1);
    d4 = direction(p1, p2, q2);

    if ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ...
       ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))
        flag = true;
        return;
    end

    flag = (d1 == 0 && on_segment(q1, q2, p1)) || ...
           (d2 == 0 && on_segment(q1, q2, p2)) || ...
           (d3 == 0 && on_segment(p1, p2, q1)) || ...
           (d4 == 0 && on_segment(p1, p2, q2));
end

function d = direction(a, b, c)
    d = (b(1)-a(1))*(c(2)-a(2)) - (b(2)-a(2))*(c(1)-a(1));
end

function flag = on_segment(a, b, c)
    flag = (min(a(1),b(1)) <= c(1) && c(1) <= max(a(1),b(1))) && ...
           (min(a(2),b(2)) <= c(2) && c(2) <= max(a(2),b(2)));
end


