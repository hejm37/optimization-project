function x_his = adm_solver(A, b, p, varargin)
% A simple implementation of Alternating Directions Method for Lasso
% problem.
    defaultC = 5;
    defaultAlpha = 0.001;   % step size for ista algorithm
    defaultItr = 10000;
    defaultEpsilon = 0.00001;
    defaultX0 = zeros(size(A, 2), 1);
    defaultV0 = zeros(size(A, 2), 1);
    par = inputParser;
    addParameter(par, 'c', defaultC);
    addParameter(par, 'alpha', defaultAlpha);
    addParameter(par, 'itr', defaultItr);
    addParameter(par, 'epsilon', defaultEpsilon);
    addParameter(par, 'x0', defaultX0);
    addParameter(par, 'v0', defaultV0);
    parse(par, varargin{:});
    
    c = par.Results.c;
    alpha = par.Results.alpha;
    itr = par.Results.itr;
    epsilon = par.Results.epsilon;
    x_ko = par.Results.x0;
    z_ko = par.Results.x0;
    v_ko = par.Results.v0;
    x_his = zeros(100, itr);
    
    for k = 1:1000
        x_k = (A'*A + c*eye(size(A, 2)))^(-1) * ...
            (v_ko + c*z_ko + A'*b);
        z_k_his = ista_solver(eye(size(A, 2)), ...
            x_k-v_ko/c, p/c, "alpha", alpha, ...
            "x0", x_k);
%         This is much slower
%         z_k_his = ista_solver(eye(size(A, 2)), ...
%             x_k-v_ko/c, p/c, "alpha", alpha);
        z_k = z_k_his(:, size(z_k_his, 2));
        v_k = v_ko + c*(z_k - x_k);
        
        x_his(:, k) = x_ko;
        if norm(x_k - x_ko) < epsilon
            break
        end
        x_ko = x_k;
        z_ko = z_k;
        v_ko = v_k;
    end
    x_his = x_his(:, 1:k);
end