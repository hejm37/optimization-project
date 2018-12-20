function [w_his, b_his] = gd_model(xs, ys, xs_t, ys_t, varargin)
% A simple implementation of Gradient Descend model for Logistic
% Regression.

    % Reduce the use of Magic Number
    label_num = 10;
    image_length = 28*28;
    full_batch = length(ys);
    
    defaultRank = 0;
    defaultBatch = full_batch;
    defaultAlpha = 0.00001;
    defaultItr = 100000;
    defaultEpsilon = 0.00001;
    defaultWs0 = zeros(image_length, label_num);
    defaultBs0 = zeros(label_num, 1);
    defaultRestoreOpt = 'no';
    validRestoreOpt = {'restore', 'no', 'debug'};
    checkRestoreOpt = @(x) any(validatestring(x, validRestoreOpt));
    
    par = inputParser;
    addParameter(par, 'rank', defaultRank);
    addParameter(par, 'batch', defaultBatch);
    addParameter(par, 'alpha', defaultAlpha);
    addParameter(par, 'itr', defaultItr);
    addParameter(par, 'epsilon', defaultEpsilon);
    addParameter(par, 'ws0', defaultWs0);
    addParameter(par, 'bs0', defaultBs0);
    addOptional(par, 'restoreOpt', defaultRestoreOpt, checkRestoreOpt);
    parse(par, varargin{:});

    rank = par.Results.rank;
    batch = par.Results.batch;
    alpha = par.Results.alpha;
    itr = par.Results.itr;
    epsilon = par.Results.epsilon;
    ws_io = par.Results.ws0;
    bs_io = par.Results.bs0;
    restoreOpt = par.Results.restoreOpt;

    itr_per_round = ceil(full_batch/batch);
    error_cal_circle = 5 * itr_per_round;   % show error every 5 epcho
    
    supplement_num = itr_per_round*batch - full_batch;
    xs_supplement = zeros(image_length, supplement_num);
    ys_supplement = zeros(supplement_num, 1);
    
    % Be careful that the xs_supplement do not store any weight,
    % and it is only used for memory alignment.
    xs_r = reshape(cat(2, xs, xs_supplement), image_length, ...
                    batch, itr_per_round);
    ys_r = reshape(cat(1, ys, ys_supplement), batch, ...
                    itr_per_round);
                
    if strcmp(restoreOpt, 'restore')
        load('checkpoint', 'w_his', 'b_his', 'i');
        ws_io = w_his(:, :, i);
        bs_io = b_his(:, i);
        start = i + 1;
        fprintf('Restore from itr %d\n', i);
    else
        w_his = zeros(image_length, label_num, itr);
        b_his = zeros(label_num, itr);
        w_his(:, :, 1) = ws_io;
        b_his(:, 1) = bs_io;
        start = 2;      % Position 1 store the start point, x_0
    end
    
    % Main loop of algorithm.
    for i = start:itr
        alpha_k = alpha/i^rank; 
        mini_batch_idx = mod(i-2, itr_per_round)+1;
        batch_i = batch;
        if mini_batch_idx == itr_per_round
            batch_i = batch - supplement_num;
        end
        
        [delta_w, delta_b] = gradient(xs_r(:, 1:batch_i, mini_batch_idx), ...
                            ys_r(1:batch_i, mini_batch_idx), ws_io, bs_io);
        ws_i = ws_io - alpha_k*delta_w;
        bs_i = bs_io - alpha_k*delta_b;
        
        w_his(:, :, i) = ws_i;
        b_his(:, i) = bs_i;
        
        % Display target function value each step.
        if mod(i-1, error_cal_circle) == 0 || i == 2
            accuracy = test(xs_t, ys_t, ws_i, bs_i);
            fprintf('Iteration %d, loss %e, accuracy %f%%\n', i-1, ...
                loss_func(xs, ys, ws_i, bs_i), accuracy*100);
        end

        % mod(i, 50)
        if mod(i, 1000) == 0 && ~strcmp(restoreOpt, 'debug')
            save('checkpoint', 'w_his', 'b_his', 'i');
            fprintf('Checkpoint at itr %d\n', i);
        end
        
        if norm(ws_i-ws_io) + norm(bs_i-bs_io) < epsilon
            fprintf('Iteration %d, end for change smaller than eps.\n', i);
            accuracy = test(xs_t, ys_t, ws_i, bs_i);
            fprintf('Iteration %d, loss %e, accuracy %f%%\n', i-1, ...
                loss_func(xs, ys, ws_i, bs_i), accuracy*100);
            break
        end
        if accuracy > 0.90
            fprintf('Iteration %d, loss %e, accuracy %f%%\n', i, ...
                loss_func(xs, ys, ws_i, bs_i), accuracy*100);
            break
        end
        
        ws_io = ws_i;
        bs_io = bs_i;
    end
end
