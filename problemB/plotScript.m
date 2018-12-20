
itr = 50;
% plotDistance('gdPlot.mat', ' Gradient Descend', itr);
% plotError('gdPlot.mat', ' Gradient Descend', itr);
plotLoss('gdPlot.mat', ' Gradient Descend', itr);

% SGD batch size: 1000
itr = 500; 
% plotDistance('sgdPlot.mat', ' SGD', itr);
% plotError('sgdPlot.mat', ' SGD', itr);
plotLoss('sgdPlot.mat', ' SGD', itr);


%% Function definition
function plotDistance(filename, ~, itr)
% Plot result: distance to optimal point
    load(filename, 'iterations', 'distance');

    itr = min(itr, length(iterations));
    plot(iterations(1:itr), log(distance(1:itr)));
    title 'log distance to optimal point';
    xlabel 'itreration';
    ylabel 'y';
    fprintf('Distance plot finished\n')
    legend('y=||w-w_{opt}||_2')
end

function plotError(filename, algorithm, itr)
% Plot result: accuracy
    load(filename, 'iterations', 'error');
    
    itr = min(itr, length(iterations));
    plot(iterations(1:itr), error(1:itr));
    title(strcat('classification error on', algorithm));
    xlabel 'itreration';
    ylabel 'error';
    fprintf('Error plot finished\n')
        
end

function plotLoss(filename, algorithm, itr)
% Plot result: loss
    load(filename, 'iterations', 'loss');
    
    itr = min(itr, length(loss));
    plot(iterations(1:itr), loss(1:itr));
    title(strcat('loss on', algorithm));
    xlabel 'itreration';
    ylabel 'loss';
    fprintf('Loss plot finished\n')
   
end