addpath('./MNIST/');
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');

images_t = loadMNISTImages('t10k-images-idx3-ubyte');
labels_t = loadMNISTLabels('t10k-labels-idx1-ubyte');

tic
% % basic usage
% [w_his, b_his] = gd_model(images, labels, images_t, labels_t);

% % save weight for plotting
% [w_his, b_his] = gd_model(images, labels, images_t, labels_t, ...
%                             'restore', 'batch', 1000, 'itr', 30001);
% [w_his, b_his] = gd_model(images, labels, images_t, labels_t, ...
%                           'restore', 'itr', 10001);
% filename = "weights";
% save(filename, 'w_his', 'b_his');

% % debug
% [~, ~] = gd_model(images, labels, images_t, labels_t, ...
%                           'debug', 'itr', 3);
%

% Script for only get elapsed time
[~, ~] = gd_model(images, labels, images_t, labels_t, ...
                    'debug', 'batch', 4000, 'itr', 1000000, ...
                    'alpha', 0.01, 'epsilon', eps);

toc