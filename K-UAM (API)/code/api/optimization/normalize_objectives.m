function normalized_f = normalize_objectives(f_vals)
    % 목적 함수 값들을 0과 1 사이로 정규화
    f_min = min(f_vals, [], 1);
    f_max = max(f_vals, [], 1);
    range = f_max - f_min;
    range(range < 1e-10) = 1; % 분모가 0이 되는 것을 방지
    normalized_f = (f_vals - f_min) ./ range;
end
