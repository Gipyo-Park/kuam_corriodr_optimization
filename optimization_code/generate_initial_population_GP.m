function population = generate_initial_population_GP(pop_size, nodes, start_point, end_point)
    % 수정
    M = size(nodes,1);
    max_inter_nodes = min(8, M);
    min_inter_nodes = min(3, max_inter_nodes);  % M이 3보다 작으면 M으로 축소

    population = cell(pop_size, 1); % 각 경로를 cell로 저장

    for i = 1:pop_size
        % 경유 노드 개수 무작위로 선택
        num_inter = randi([min_inter_nodes, max_inter_nodes]);

        % 중복 없이 무작위로 노드 선택
        selected_idx = randperm(M, num_inter);

        % 경로 순서도 랜덤하게 섞기
        ordered_idx = selected_idx(randperm(num_inter));

        % 전체 경로 좌표 구성
        path = [start_point;
                nodes(ordered_idx, :);
                end_point];

        population{i} = path;
    end
end
