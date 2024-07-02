%Close all open figures
close all
%Clear the workspace
clear
%%
%Clear the command window
clc 
%%

dataFolder = "Medical-imaging-dataset";
imds = imageDatastore(dataFolder, IncludeSubfolders=true,LabelSource="foldernames");

%check the images correctly stored in imds
figure;
perm = randperm(1000,20); 
for i = 1:20
subplot(4,5,i); 
imshow(imds.Files{perm(i)});
end
%%
%Dividing the dataset into 70% training and 30% validation
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized'); 

%inputSize = [227, 227, 3];

%check the actual size of the image in the dataset
actualsize = readimage(imds,1); 
size(actualsize) 

%check the number of classes in the selected dataset
numClasses = numel(categories(imdsTrain.Labels))

net = alexnet; 

%check alexnet size of first layer
inputSize = net.Layers(1).InputSize 

%%
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
 'RandXReflection',true, ...
 'RandXTranslation',pixelRange, ...
 'RandYTranslation',pixelRange);

%convert the size of the acutal image to inputsize of firstlayer of net
augimdsTrain = augmentedImageDatastore(inputSize(1:3),imdsTrain, ...
 'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);  

augimdsValidation = augmentedImageDatastore(inputSize(1:3),imdsValidation,...
     'ColorPreprocessing','gray2rgb');

options = trainingOptions('sgdm', ...
 'MiniBatchSize',10, ...
 'MaxEpochs',6, ...
 'InitialLearnRate',1e-4, ...
 'Shuffle','every-epoch', ...
 'ValidationData',augimdsValidation, ...
 'ValidationFrequency',3, ...
 'Verbose',false, ...
 'Plots','training-progress');

%lRelu = layer('ReLU', 'Name', 'relu1');
%deepNetworkDesigner(resnet50)

layersTransfer = net.Layers(1:end-3); 

layers = [
 layersTransfer
 fullyConnectedLayer(4,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
 softmaxLayer
 classificationLayer
 ]; 

netTransfer = trainNetwork(augimdsTrain,layers,options); 


%% 
[YPred,scores] = classify(netTransfer,augimdsValidation);

%%
idx = randperm(numel(imdsValidation.Files),4);

figure
for i = 1:4
subplot(2,2,i)
I = readimage(imdsValidation,idx(i));
imshow(I);
label = YPred(idx(i));
title(string(label));
end

%%
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)
plotconfusion(YValidation,YPred)

% Extract details of the first convolutional layer (conv1).
conv1 = net.Layers(2);

% Display information about the first convolutional layer (conv1).
disp('Details of the first convolutional layer (conv1): ');
disp(['Filter size: ' num2str(conv1.FilterSize)]);
disp(['Number of filters: ' num2str(conv1.NumFilters)]);
disp(['Stride: ' num2str(conv1.Stride)]);
disp(['Padding: ' num2str(conv1.PaddingSize)]);

%%
% Assuming square input size
inputSize = net.Layers(1).InputSize(1);  
% Assuming square filter size
filterSize = conv1.FilterSize(1);    
% Padding of conv1
padding = conv1.PaddingSize;     
% Stride of conv1
stride = conv1.Stride;  
