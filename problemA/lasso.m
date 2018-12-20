function z = lasso(A, x, b, p)
% Lasso problem
    z = 1/2*norm(A*x - b)^2 + p*norm(x, 1);
end
