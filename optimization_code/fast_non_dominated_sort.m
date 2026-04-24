function Fronts = fast_non_dominated_sort(f_vals)
    % 목적 함수 값(f_vals)을 받아 비지배 정렬을 수행하고, Front들의 집합을 반환
    N = size(f_vals,1);
    S = cell(N,1);   % 지배하는 집합
    n = zeros(N,1);  % 지배 당한 횟수
    rank = zeros(N,1);
    Fronts = {};

    F1 = [];

    for p = 1:N
        S{p} = [];
        n(p) = 0;
        for q = 1:N
            % p가 q를 지배하는지 검사
            if all(f_vals(p,:) <= f_vals(q,:)) && any(f_vals(p,:) < f_vals(q,:))
                S{p} = [S{p}, q];
            % q가 p를 지배하는지 검사
            elseif all(f_vals(q,:) <= f_vals(p,:)) && any(f_vals(q,:) < f_vals(p,:))
                n(p) = n(p) + 1;
            end
        end
        % 아무에게도 지배당하지 않으면 Front 1
        if n(p) == 0
            rank(p) = 1;
            F1 = [F1, p];
        end
    end

    Fronts{1} = F1;
    i = 1;

    while ~isempty(Fronts{i})
        Q = [];
        for p = Fronts{i}
            for q = S{p}
                n(q) = n(q) - 1;
                if n(q) == 0
                    rank(q) = i + 1;
                    Q = [Q, q];
                end
            end
        end
        i = i + 1;
        Fronts{i} = Q;
    end

    % 마지막 비어 있는 front 제거
    if isempty(Fronts{end})
        Fronts(end) = [];
    end
end
