function child = crossover_GP(parent1, parent2)
    % 두 부모 경로의 중간 노드를 조합하여 자식 경로를 생성
    start_point = parent1(1, :);
    end_point = parent1(end, :);
    inter1 = parent1(2:end-1, :);
    inter2 = parent2(2:end-1, :);
    if isempty(inter1) || isempty(inter2)
        if isempty(inter1), child = parent2; else, child = parent1; end
        return;
    end
    len1 = size(inter1,1);
    child_len = len1;
    cut1 = randi(len1);
    cut2 = randi(len1);
    if cut1 > cut2, [cut1, cut2] = deal(cut2, cut1); end
    segment = inter1(cut1:cut2, :);
    remaining_nodes = setdiff(inter2, segment, 'rows', 'stable');
    child_inter = [segment; remaining_nodes];
    if size(child_inter,1) > child_len
        child_inter = child_inter(1:child_len,:);
    elseif size(child_inter,1) < child_len
        needed = child_len - size(child_inter,1);
        additional_nodes = setdiff(inter1, child_inter, 'rows', 'stable');
        child_inter = [child_inter; additional_nodes(1:min(needed, size(additional_nodes,1)),:)];
    end
    child = [start_point; child_inter; end_point];
end