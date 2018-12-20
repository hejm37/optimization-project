function accuracy = test(xs_t, ys_t, ws, bs)
% Test the accuracy with weights 'ws'&'bs'.
    hit = 0;    % The number of right prediction.
    for i = 1:length(ys_t)
        [~, idx] = max(ws'*xs_t(:, i) + bs);
        if idx == ys_t(i) + 1
            hit = hit + 1;
        end
    end
    accuracy = hit / length(ys_t);
end