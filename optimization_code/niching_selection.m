function selected_idx = niching_selection(norm_f, ref_points, remaining)
    % 참조점과 정규화된 목적 함수 값들을 이용해 'remaining' 개수만큼 해를 선택 (다양성 확보)
    M = size(norm_f,1);
    K = size(ref_points,1);

    distances = pdist2(norm_f, ref_points);
    [min_dists, assoc_rp] = min(distances, [], 2);

    RP_to_solutions = cell(K,1);
    for i = 1:M
        RP_to_solutions{assoc_rp(i)} = [RP_to_solutions{assoc_rp(i)} i];
    end

    niche_count = zeros(K,1);
    selected = false(M,1);
    selected_idx = [];

    while length(selected_idx) < remaining
        % 가장 적게 할당된 reference point 찾기 (Inf 제외)
        finite_niche = find(niche_count < Inf);
        if isempty(finite_niche)
            break; % 선택할 후보 없음
        end
        [~, min_idx] = min(niche_count(finite_niche));
        rp_idx = finite_niche(min_idx);

        candidates = RP_to_solutions{rp_idx};
        candidates = candidates(~selected(candidates));

        if isempty(candidates)
            niche_count(rp_idx) = Inf; % 더 이상 후보가 없는 니치는 제외
            continue;
        end

        if niche_count(rp_idx) == 0
            [~, idx_min] = min(min_dists(candidates));
            chosen = candidates(idx_min);
        else
            chosen = candidates(randi(length(candidates)));
        end

        % 중복 체크 강화
        if ~selected(chosen)
            selected(chosen) = true;
            selected_idx = [selected_idx; chosen];
            niche_count(rp_idx) = niche_count(rp_idx) + 1;
        end
    end
end
