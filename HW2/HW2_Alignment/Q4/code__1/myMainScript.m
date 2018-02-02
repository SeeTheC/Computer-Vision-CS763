%% MyMainScript
%% Assignment2-4 
% Rollno: 163059009, 16305R011, 16305R001 

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

%% Feature Extraction(extract matching keypoints for two images);

[matchA,loc1,loc2]=match(img1,img2);
%
[matchB,loc3,loc2_2]=match(img3,img2);

%% Homomorphic projection
tic
thresh=2;
[H,inliners,cp1,cp2] = ransacHomography(matchA, loc1,loc2, thresh);

[H2,inliners2,cp3,cp2_2] = ransacHomography(matchB, loc3,loc2_2, thresh);
toc;
%% 2. Trasformation of Image 1

pad=200;
[timg1] = trasformationImage(mimg1,H,pad);
[timg2] = padarray(mimg2,[pad,pad]);
%%
[timg3] = trasformationImage(cimg3,H2,pad);

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
figure('name','transformed image 3');
imshow(uint8(timg3));
impixelinfo;
title('\fontsize{10}{\color{magenta}transformed image 3}');
axis tight,axis on;
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);

%%
stich1=stichImage(timg1,timg2);
%%
stich2=stichImage(stich1,timg3);
%%
figure('name','transformed image 3');
imshow(uint8(stich2));
impixelinfo;
title('\fontsize{10}{\color{magenta}transformed image 1}');
axis tight,axis on;
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);
%%
tim
yshit
affine2d