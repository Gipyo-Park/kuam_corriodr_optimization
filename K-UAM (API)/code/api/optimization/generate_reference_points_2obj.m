function ref_points = generate_reference_points_2obj(H)
    % H: 분할 수 (예: 10)
    % ref_points: (H+1) x 2 크기, [x,y] 좌표
    % 2-Objective 문제에 대한 참조점(Reference Point)을 생성
    ref_points = zeros(H+1, 2);
    for i = 0:H
        w1 = i / H;
        w2 = 1 - w1;
        ref_points(i+1, :) = [w1, w2];
    end
end
