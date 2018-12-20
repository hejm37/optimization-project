
% load data
load('variables.mat');

% parameters
p = 10;
tic
x_his = sgm_solver(A, b, p, 'itr', 10000000, 'epsilon', eps);
toc
x_k = x_his(:, size(x_his, 2));
f_t = lasso(A, x_t, b, p);
f_k = lasso(A, x_k, b, p);

fprintf("The true value is %f\n", f_t)
fprintf("The optimal value is %f\n", f_k)

cnt = 0;
for i = 1:100
    if abs(x_k(i)) > 0.001
        cnt = cnt + 1;
    end
end

fprintf("Nonzero number of x %d\n", cnt)

% plotResult(x_his, x_t, "Subgradient Method");
plotLogResult(x_his, x_t, "Subgradient Method");
