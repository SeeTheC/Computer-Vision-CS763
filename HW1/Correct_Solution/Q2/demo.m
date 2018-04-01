close all
clc
clear
im = double(rgb2gray(imread('checkerbox_sq.jpg')))./255;
k1 = 1;
k2 = 0.5;
%%
[xd, yd, imD]=radDist(im, k1, k2, 20, 0);
figure
scatter(xd,yd)
figure
imshow(imD)

%%
%imD = double(imread('imD2.jpg'))./255;
[xu, yu, imU]=radUnDist(imD, k1, k2, 20 ,111, 0);
figure
scatter(xu,yu)
figure
imshow(imU)

