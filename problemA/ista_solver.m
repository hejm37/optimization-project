function x_his = ista_solver(A, b, p, varargin)
% A simple implementation of Itrative Shrinkage Thresholding Algorithm for
% Lasso problem.
    defaultAlpha = 0.001;
    defaultItr = 10000;
    defaultEpsilon = 0.00001;
    defaultX0 = zeros(size(A, 2), 1);
    par = inputParser;
    addParameter(par, 'alpha', defaultAlpha);
    addParameter(par, 'itr', defaultItr);
    addParameter(par, 'epsilon', defaultEpsilon);
    addParameter(par, 'x0', defaultX0);
    parse(par, varargin{:});
    
    alpha = par.Results.alpha;
    itr = par.Results.itr;
    epsilon = par.Results.epsilon;
    x_ko = par.Results.x0;
    x_his = zeros(100, itr);

    for k = 1:itr
        x_k = x_ko - alpha*A'*(A*x_ko - b);
        for j = 1:length(x_k)
            if x_k(j) > alpha*p
                x_k(j) = x_k(j) - alpha*p;
            elseif x_k(j) < -alpha*p
                x_k(j) = x_k(j) + alpha*p;
            else
                x_k(j) = 0;
            end
        end
        x_his(:, k) = x_ko;
        if norm(x_k - x_ko) < epsilon
            break
        end
        x_ko = x_k;
    end
    x_his = x_his(:, 1:k);
end