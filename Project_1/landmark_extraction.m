% Landmark extraction using dlib

% The dataset consists of a folder with face images 
%
% 
% Note: here we do not need attributes




clc; close all; clear all;

% Dataset
foldername='C:/Users/Giorgos/Desktop/ergasies_metaptxiakwn/ergasia filippaki_(predictive analytics)/ergasia_examinou/ergasia_mfilip_2021/ergasia_mfilip_2021/meros-3-luseis-mf/train_images/';
images=dir([foldername '*.png']); 

% Import predictor
h = detector('new', 'C:/Users/Giorgos/Desktop/ergasies_metaptxiakwn/ergasia filippaki_(predictive analytics)/ergasia_examinou/ergasia_mfilip_2021/ergasia_mfilip_2021/meros-3-luseis-mf/shape_predictor_68_face_landmarks.dat');


for ii = 1:size(images, 1)
    rgb = imread(sprintf('C:/Users/Giorgos/Desktop/ergasies_metaptxiakwn/ergasia filippaki_(predictive analytics)/ergasia_examinou/ergasia_mfilip_2021/ergasia_mfilip_2021/meros-3-luseis-mf/train_images//%s.png',num2str(ii)));
    gray = rgb2gray(rgb); % grayscale image
    f = detector('detect', h, rgb); 
    m = detector('mean_shape', h);
    params = initroi(m, [64, 64], [0.25, 0.6], [0.75, 0.9], 'similarity');
    r = zeros(size(f, 1), 64, 64, 3, 'uint8');
    s = zeros(size(f, 1), 68, 2);
    for i = 1:size(f, 1)
        s(i, :, :) = detector('fit', h, rgb, f(i, :)); % extract 68 landmarks
        r(i, :, :, :) = extractroi(rgb, s(i, :, :), params);
    end
 
    % show face detections and alignments 
    figure(1); 
    imshow(rgb);
    hold on; 
    for i = 1:size(f, 1)
        x = [f(i, 1), f(i, 3), f(i, 3), f(i, 1),f(i, 1)]; 
        y = [f(i, 2), f(i, 2), f(i, 4),f(i, 4), f(i, 2)]; 
        plot(x, y, '-r');
        plot(s(i, :, 1), s(i, :, 2), 'g+'); %(x of landmarks, y of landmarks)
    end

end
