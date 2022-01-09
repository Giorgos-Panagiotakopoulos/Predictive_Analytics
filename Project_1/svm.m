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
trainData = zeros(num_files, 256*256);


% Pre-process training data
for j = 1:num_files
    currentimage = imread([train_foldername, '/',num2str(j) ,'.png']);
    image{j} = currentimage;
    image{j} = im2double(image{j});
    image{j} = rgb2gray(image{j}); % from change the number of color channels
    image{j} = reshape(image{j}', 1, size(image{j},1)*size(image{j},2)); % change the data shape
    trainData(j,:) = image{j}; % change the data structure so that it fits the fitsvm function's requirements
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 1. Use 'fitcsvm' function to train an SVM:
%classifier = fitcsvm(trainData,train_attr)
%classifier = fitcsvm(trainData,train_attr,'KernelFunction','rbf','BoxConstraint',5)
%classifier = fitcsvm(trainData,train_attr,'KernelFunction','linear')
%classifier = fitcsvm(trainData,train_attr,'KernelFunction','polynomial')
classifier = fitcsvm(trainData,train_attr,'KernelFunction','gaussian')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 2. Use test_images dataset to test your trained model:

test_foldername='C:/Users/Giorgos/Desktop/ergasies_metaptxiakwn/ergasia filippaki_(predictive analytics)/ergasia_examinou/ergasia_mfilip_2021/ergasia_mfilip_2021/meros-3-luseis-mf/test_images/';
test_attr=load('C:/Users/Giorgos/Desktop/ergasies_metaptxiakwn/ergasia filippaki_(predictive analytics)/ergasia_examinou/ergasia_mfilip_2021/ergasia_mfilip_2021/meros-3-luseis-mf/test_attributes.dat'); 
testData = zeros(20, 256*256);

% Pre-process test data
for j = 1:20
    currentimage = imread([test_foldername, '/',num2str(j) ,'.png']);
    image{j} = currentimage;
    image{j} = im2double(image{j});
    image{j} = rgb2gray(image{j});
    image{j} = reshape(image{j}', 1, size(image{j},1)*size(image{j},2));
    testData(j,:) = image{j};
     
end

% Make predictions using your trained model and the processed test data
[labels,scores]= predict(classifier, testData);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 3. Visualize results using a confusion matrix
confusionchart(test_attr, labels);
for j = 1:20
	disp(labels(j));
    disp(scores(j));
end