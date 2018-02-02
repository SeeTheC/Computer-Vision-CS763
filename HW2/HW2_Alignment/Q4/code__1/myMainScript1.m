%% Init
file='../input/ledge/1.JPG';
img1=rgb2gray(imread(file)); 
cimg1=imread(file); 
dim1=size(img1);
format shortG

file='../input/ledge/2.JPG';
img2=rgb2gray(imread(file));
cimg2=imread(file);
dim2=size(img2);

file='../input/ledge/3.JPG';
img3=rgb2gray(imread(file)); 
cimg3=imread(file);
dim3=size(img3);

%% Finding features and matching it

%Find the SURF features.
points1 = detectSURFFeatures(img1);
points2 = detectSURFFeatures(img2);

%Extract the features.

[f1,vpts1] = extractFeatures(img1,points1);
[f2,vpts2] = extractFeatures(img2,points2);

%Retrieve the locations of matched points.
indexPairs = matchFeatures(f1,f2) ;
matchedPoints1 = vpts1(indexPairs(:,1));
matchedPoints2 = vpts2(indexPairs(:,2));

%%
% Display the matching points. 
figure; 
showMatchedFeatures(img1,img2,matchedPoints1,matchedPoints2);
legend('matched points 1','matched points 2');
%% Homomorphic projection
tic
thresh=1;
[H,inliners,cp1,cp2] = ransacHomography(matchedPoints1.Location, matchedPoints2.Location, thresh);

%% Trasformation of Image

pad=200;
[timg1] = trasformationImage(cimg1,inv(H),pad);
[timg2] = padarray(cimg2,[pad,pad]);
%%
figure('name','transformed image 1');
imshow(uint8(timg1));
impixelinfo;
title('\fontsize{10}{\color{magenta}transformed image 1}');
axis tight,axis on;
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);
%%    
figure('name','transformed image 2');
imshow(uint8(timg2));
impixelinfo;
title('\fontsize{10}{\color{magenta}transformed image 1}');
axis tight,axis on;
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);
%%
stich1=stichImage(timg1,timg2);
%%
%stich2=stichImage(stich1,timg3);
%%
figure('name','transformed image 3');
imshow(uint8(stich1));
impixelinfo;
title('\fontsize{10}{\color{magenta}transformed image 1}');
axis tight,axis on;
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);