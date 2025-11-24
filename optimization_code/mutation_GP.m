function mutated = mutation_GP(path, nodes)
    % 일정 확률로 경로의 중간 노드 하나를 다른 무작위 노드로 교체
    if rand < 0.2 && size(path,1) > 2 && ~isempty(nodes)
        idx_to_mutate = randi([2, size(path,1)-1]);
        new_node = nodes(randi(size(nodes,1)), :);
        path(idx_to_mutate, :) = new_node;
    end
    mutated = path;
end