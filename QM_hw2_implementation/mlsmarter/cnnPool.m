function pooled = cnnPool(convolved, poolDim, pooling)

if nargin<3
    pooling='mean';
end

numImages = size(convolved, 4);
numFilters = size(convolved, 3);
convolvedDim = size(convolved, 1);
pooledDim = convolvedDim/poolDim;
assert(mod(pooledDim,1)==0,'pooledDim is not integer');

pooled = zeros(pooledDim, pooledDim, numFilters, numImages);

% ====================== YOUR CODE HERE ======================
% Pool the convolved features (convolved) over regions of poolDim x
% poolDim. Use mean pooling here. This can be done with 4 for-loops:
% x = 1:pooledDim
% y = 1:pooledDim
% filter = 1:numFilters
% image = 1:numImages
% The resulting pooled feature (pooled) should be a 4D matrix with dimensions
% pooledDim x pooledDim x numFilters x numImages

if strcmp(pooling, 'max')
    for x = 1:pooledDim
        for y = 1:pooledDim
            pooled(x, y, :, :) = max(max(convolved(1+(x-1)*poolDim:x*poolDim, 1+(y-1)*poolDim:y*poolDim, :, :),[],1),[],2); % Max-pooling
        end
    end
elseif strcmp(pooling, 'mean')
    
    for x = 1:pooledDim
        for y = 1:pooledDim
            pooled(x, y, :, :) = mean(mean(convolved(1+(x-1)*poolDim:x*poolDim, 1+(y-1)*poolDim:y*poolDim, :, :))); % Mean-pooling
        end
    end
end

% ============================================================


end

