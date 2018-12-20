
% load data
load('variables.mat');

% parameters
p = 10;
tic
x_his = ista_solver(A, b, p, 'epsilon', eps);
toc
x_k = x_his(:, size(x_his, 2));
f_t = lasso(A, x_t, b, p);
f_k = lasso(A, x_k, b, p);

fprintf("The true value is %f\n", f_t)
fprintf("The optimal value is %f\n", f_k)
% fprintf("Nonzero number of x %d\n", nnz(x_k))

cnt = 0;
for i = 1:100
    if abs(x_k(i)) > 0.001
        cnt = cnt + 1;
    end
end

fprintf("Nonzero number of x %d\n", cnt)


% plotResult(x_his, x_t, "Proximal Gradient Method");
plotLogResult(x_his, x_t, "Proximal Gradient Method");
