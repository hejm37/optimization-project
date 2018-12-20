function x_his = sgm_solver(A, b, p, varargin)
% A simple implementation of Subgradient Method for Lasso problem.
    defaultRank = '1/2';
    defaultAlpha = 0.001;
    defaultItr = 10000;
    defaultEpsilon = 0.00001;
    defaultX0 = zeros(size(A, 2), 1);
    par = inputParser;
    addParameter(par, 'rank', defaultRank);
    addParameter(par, 'alpha', defaultAlpha);
    addParameter(par, 'itr', defaultItr);
    addParameter(par, 'epsilon', defaultEpsilon);
    addParameter(par, 'x0', defaultX0);
    parse(par, varargin{:});
    
    rank = par.Results.rank;
    alpha = par.Results.alpha;
    itr = par.Results.itr;
    epsilon = par.Results.epsilon;
    x_ko = par.Results.x0;
    x_his = zeros(100, itr);
    
    for k = 1:itr
        if strcmp(rank, '1/2')
            alpha_k = alpha/sqrt(k+1);
        else 
            alpha_k = alpha/(k+1);
        end
        % if x_j == 0, we set the subgradient as 1
        subGrad = ones(length(x_ko), 1);
        for j = 1:length(x_ko)
            if x_ko(j) < 0
                subGrad(j) = -p;
%             elseif x_ko(j) == 0
%                 subGrad(j) = 0;
            else
                subGrad(j) = p;
            end
        end
        
        gradient = A'*(A*x_ko - b) + subGrad;
        x_k = x_ko - alpha_k*gradient;
        
        x_his(:, k) = x_ko;
        if norm(x_k - x_ko) < epsilon
            break
        end
        x_ko = x_k;
        
%         disp(k)
%         disp(lasso(A, x_ko, b, p))
        
    end
    x_his = x_his(:, 1:k);
end