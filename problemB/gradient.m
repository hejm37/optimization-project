function [delta_ws, delta_bs] = gradient(xs, ys, ws, bs)
% Calculate the gradient for log likelihood function.
    delta_ws = zeros(size(ws));
    delta_bs = zeros(size(bs));
    for k = 1:length(bs)
        [delta_w, delta_b] = gradient_k(xs, ys, ws, bs, k);
        delta_ws(:, k) = delta_w;
        delta_bs(k) = delta_b;
    end
end

% Vectorization
function [delta_w, delta_b] = gradient_k(xs, ys, ws, bs, k)
    delta_w = zeros(size(ws, 1), 1);
    delta_b = 0;
    for i = 1:length(ys)
        if ys(i) + 1 == k
            delta_w = delta_w - xs(:, i);
            delta_b = delta_b - 1;
        end
        
        softmax_i = exp(ws(:, k)'*xs(:, i) + bs(k)) / ...
                    sum(exp(ws'*xs(:, i) + bs));

        delta_w = delta_w + softmax_i*xs(:, i);
        delta_b = delta_b + softmax_i;
    end
end
