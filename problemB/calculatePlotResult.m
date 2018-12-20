% Because calculating the plot result is very time consuming, we 
% first calculate the plot data, then plot.

addpath('./MNIST/');
xs = loadMNISTImages('train-images-idx3-ubyte');
ys = loadMNISTLabels('train-labels-idx1-ubyte');

xs_t = loadMNISTImages('t10k-images-idx3-ubyte');
ys_t = loadMNISTLabels('t10k-labels-idx1-ubyte');


% % PARAMETER FOR GD
% algorithm = 'Gradient Descend';
% weightsFilename = 'checkpoint10000gd';
% saveFilename = 'gdPlot.mat';

% PARAMETER FOR SGD
algorithm = 'SGD';
weightsFilename = 'checkpoint30000sgd';
saveFilename = 'sgdPlot.mat';

load(weightsFilename, 'w_his', 'b_his');

max_itr = size(w_his, 3);
iterations = (1:max_itr)';

distance = zeros(max_itr, 1);
diff = zeros(28*28+1, 10);
for i = 1:max_itr
    diff(1:784, :) = w_his(:, :, i) - w_his(:, :, max_itr);
    diff(785, :) = (b_his(:, i) - b_his(:, max_itr))';
    distance(i) = norm(diff);
end
fprintf('Distance calculate finished\n')

loss = zeros(1000, 1);
for i = 1:1000
    loss(i) = loss_func(xs, ys, w_his(:, :, i), b_his(:, i));
end
fprintf('Loss calculate finished\n')

error = zeros(max_itr, 1);
for i = 1:max_itr
    error(i) = 1-test(xs_t, ys_t, w_his(:, :, i), b_his(:, i));
end
fprintf('Error calculate finished\n')

save(saveFilename, 'iterations', 'distance', 'loss', 'error')
