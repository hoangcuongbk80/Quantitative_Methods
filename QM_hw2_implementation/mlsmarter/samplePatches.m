function patches = samplePatches(I, filterDim, numPatches)

imageHeight = size(I,1);
imageWidth = size(I,2);
imageChannels = size(I,3);
numImages = size(I,4);

patches = zeros(filterDim * filterDim * imageChannels, numPatches);
for i=1:numPatches
    x = randi(imageWidth - filterDim + 1,1);
    y = randi(imageHeight - filterDim + 1,1);
    im = randi(numImages,1);
    patches(:,i) = reshape(I(y : y + filterDim - 1, x : x + filterDim - 1, :,im),[],1);
end

end

