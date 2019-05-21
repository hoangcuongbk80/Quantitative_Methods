function convolved = cnnConvolve(images, filters)
% images - imageDim x imageDim x numChannels x numImages
% filters - filterDim x filterDim x numChannels, numFilters

imageDim = size(images, 1);
numChannels = size(images, 3);
numImages = size(images, 4);
numFilters = size(filters, 4);
filterDim = size(filters,1);
convolvedDim = imageDim - filterDim + 1;

% Pre-allocate
convolved = zeros(convolvedDim, convolvedDim, numFilters, numImages, class(images));

% ====================== YOUR CODE HERE ======================
% Implement convolution on the input images. Use three for loops over
% numImages, numFilters, and numChannels. You can use the MATLAB function
% conv2(images(:, :, channel, image), filters(end:-1:1,end:-1:1, channel, filter), 'valid')
% The result (convolved) should have dimension convolvedDim x convolvedDim x numFilters x numImages
% The 'valid' option does not add 0's at the borders of the image.
% filters(end:-1:1,end:-1:1, channel, filter) is used to cancel out the 
% flipping that conv2 is doing.
% The convolved image is the sum of all the convolved images over all
% channels. We will only use gray-scale images in this exercise.

for image = 1:numImages
    for filter = 1:numFilters
        convolvedImage = zeros(convolvedDim, convolvedDim, class(images));
        for channel = 1:numChannels
            convolvedImage = convolvedImage + conv2(images(:, :, channel, image), filters(end:-1:1,end:-1:1, channel, filter), 'valid');            
        end
        convolved(:, :, filter, image) = convolvedImage;
    end
end

% ===========================================================


end

