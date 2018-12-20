function z = loss_func(xs, ys, ws, bs)
% Loss function
    z = 0;
    for i = 1:length(ys)
        label = int32(ys(i)) + 1;   % Be carefult!
        z = z - (ws(:, label)'*xs(:, i) + bs(label));
        sum = 0;
        for j = 1:length(bs)
            sum = sum + exp(ws(:, j)'*xs(:, i) + bs(j));
        end
        z = z + log(sum);
    end
end