function [optTheta Jtrain Jval] = minibatchGD(X, y, parameters, Xval, yval)
% Mini-batch Gradient Descent for softmax classifier

% Useful parameters
learningRateDecay = 0.01;
numepochs = parameters.numepochs;
batchsize = parameters.batchsize;
learningRate = parameters.learningRate;
numClasses = parameters.numClasses;
numiters = floor(size(X,2)/batchsize);

thetaSize = [numClasses size(X, 1)];
theta = reshape(0.005 * randn(thetaSize), [], 1);

% Momentum
if parameters.useMomentum
    mom = 0.5;
    velocity = zeros(size(theta));
else
    mom = 0;
end

% Initialize variables
Jtrain = zeros(numepochs, 1);
Jval = zeros(numepochs, 1);

for epoch = 1:numepochs
    
    if parameters.useDecayLearningRate
    % ====================== YOUR CODE HERE ======================
    % Calculate the learning rate using decaying learning rate
    
    learningRate = parameters.learningRate/(1+learningRateDecay*epoch);
    % ============================================================

    end
    
    k = randperm(size(X,2)); %randomized vector
    
    for iter = 1:numiters
        
        % Select the current batch
        Xbatch = X(:,k(1+(iter-1)*batchsize:iter*batchsize));
        ybatch = y(k(1+(iter-1)*batchsize:iter*batchsize));
        
        [~, grad] = costSoftmax(theta, Xbatch, ybatch, numClasses, parameters.lambda);
        
        if parameters.useMomentum
            if epoch > 5
                mom = 0.9;
            end;
            
            % ====================== YOUR CODE HERE ======================
            % Update theta without momentum
            velocity = mom * velocity + learningRate * grad;
            theta = theta - velocity;
            % ============================================================
            
        else
            % ====================== YOUR CODE HERE ======================
            % Update theta without momentum
            theta = theta - learningRate * grad;
            % ============================================================
            
        end
        
    end
    
    % ====================== YOUR CODE HERE ======================
    % Calculate the classification accuracy with the current theta for the 
    % whole training set X and store it in Jtrain(epoch) and the 
    % classification accuracy on the whole validation set Xval and store it 
    % in Jval(epoch)
    predtrain = predictSoftmax(theta, X, numClasses);
    Jtrain(epoch) = mean(predtrain == y);
    predval = predictSoftmax(theta, Xval, numClasses);
    Jval(epoch) = mean(predval == yval);
    % ============================================================
    
    fprintf('%i/%i Accuracy train: %0.2f Accuracy val: %0.2f\n', epoch, numepochs, Jtrain(epoch)*100, Jval(epoch)*100)

    % ====================== YOUR CODE HERE ======================
    % Implement an early-stopping strategy. Use the command return if the
    % condition is met (for example Jval has not improved in the last x
    % number of epochs.

    % ============================================================
    
end

optTheta = theta;


    
end