% Support Vector Machine

% The dataset consists of a folder with face images and an attribute list
% for gender.
% 
% Note: training attributes are 0 and 1 (0-male, 1-female)
%

clc; close all; clear all;


% Import training dataset
train_foldername='C:/Users/Giorgos/Desktop/ergasies_metaptxiakwn/ergasia filippaki_(predictive analytics)/ergasia_examinou/ergasia_mfilip_2021/ergasia_mfilip_2021/meros-3-luseis-mf/train_images/';
train_images=dir([train_foldername '*.png']);
train_attr=load('C:/Users/Giorgos/Desktop/ergasies_metaptxiakwn/ergasia filippaki_(predictive analytics)/ergasia_examinou/ergasia_mfilip_2021/ergasia_mfilip_2021/meros-3-luseis-mf/train_attributes.dat'); 
num_files=size(train_images, 1)

% Initialize the final data structure
trainData = zeros(256,256,1,num_files);


% Pre-process training data
for j = 1:num_files
    currentimage = imread([train_foldername, '/',num2str(j) ,'.png']);
    image{j} = currentimage;
    image{j} = im2double(image{j});
    image{j} = rgb2gray(image{j}); 
    %images{j} = imresize(images{j},[28 28]);
    trainData(:,:,1,j) = image{j};
     
  
end

% Repeat for testing data
test_foldername='C:/Users/Giorgos/Desktop/ergasies_metaptxiakwn/ergasia filippaki_(predictive analytics)/ergasia_examinou/ergasia_mfilip_2021/ergasia_mfilip_2021/meros-3-luseis-mf/test_images';
test_attr=load('C:/Users/Giorgos/Desktop/ergasies_metaptxiakwn/ergasia filippaki_(predictive analytics)/ergasia_examinou/ergasia_mfilip_2021/ergasia_mfilip_2021/meros-3-luseis-mf/test_attributes.dat'); 
testData = zeros(256,256,1,20);


% Pre-process test data
for j = 1:20
    currentimage = imread([test_foldername, '/',num2str(j) ,'.png']);
    image{j} = currentimage;
    image{j} = im2double(image{j});
    image{j} = rgb2gray(image{j});
    testData(:,:,1,j) = image{j};
     
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 1. Define a CNN architecture and training options:
layers = [
    imageInputLayer([256 256 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];


tempdir= ''
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'Verbose', true, ...
    'Plots','training-progress', ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',30, ...
    'CheckpointPath', tempdir);
    


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 2. Use the training dataset and options you defined previously to train
% your network

net = trainNetwork(trainData,categorical(train_attr),layers,options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 3. Use test_images dataset to test your trained model:

y_prediction= classify(net, testData);
disp(y_prediction);

accuracy = sum(y_prediction == categorical(test_attr))/numel(test_attr)