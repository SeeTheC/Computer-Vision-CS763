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
%%
pad=200;
pimg1=padarray(img1,[pad,pad]);
pimg2=padarray(img2,[pad,pad]);
pimg3=padarray(img3,[pad,pad]);

pcimg1=padarray(cimg1,[pad,pad]);
pcimg2=padarray(cimg2,[pad,pad]);
pcimg3=padarray(cimg3,[pad,pad]);



img1=pimg1;
img2=pimg2;
img3=pimg3;
cimg1=pcimg1;
cimg2=pcimg2;
cimg3=pcimg3;
%% Finding features and matching it

%Find the SURF features.
points1 = detectSURFFeatures(img1);
points2 = detectSURFFeatures(img2);
points3 = detectSURFFeatures(img3);

%Extract the features.

[f1,vpts1] = extractFeatures(img1,points1);
[f2,vpts2] = extractFeatures(img2,points2);
[f3,vpts3] = extractFeatures(img3,points3);

%Retrieve the locations of matched points.
indexPairs = matchFeatures(f1,f2) ;
matchedPoints1_1 = vpts1(indexPairs(:,1));
matchedPoints1_2 = vpts2(indexPairs(:,2));

indexPairs = matchFeatures(f2,f3) ;
matchedPoints2_2 = vpts2(indexPairs(:,1));
matchedPoints2_3 = vpts3(indexPairs(:,2));

%%
% Display the matching points. 
figure; 
%showMatchedFeatures(img1,img2,matchedPoints1,matchedPoints2);
legend('matched points 1','matched points 2');
%% Homomorphic projection
tic
thresh=3;
[H,inliners,cp1,cp2] = ransacHomography(matchedPoints1_1.Location, matchedPoints1_2.Location, thresh);
%% 
thresh=3;
[H2,inliners2,cp3,cp2_2] = ransacHomography(matchedPoints2_3.Location, matchedPoints2_2.Location,thresh);
%% validation
val=H*cp1';
val=bsxfun(@times,val',1./val(3,:)');
%% mark point

pt=cp1;
noOfPoints=size(pt,1);        
for i=1:noOfPoints
    img1 = insertMarker(img1,[pt(i,2),pt(i,1)],'x','color','red','size',15);
end

figure('name','image 1');
imshow(uint8(img1));
impixelinfo;
title('\fontsize{10}{\color{magenta} image 1}');
axis tight,axis on;
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);


pt=cp2;
noOfPoints=size(pt,1);        
for i=1:noOfPoints
    img2 = insertMarker(img2,[pt(i,2),pt(i,1)],'x','color','red','size',15);
end

figure('name','image 2');
imshow(uint8(cimg2));
impixelinfo;
title('\fontsize{10}{\color{magenta} image 2}');
axis tight,axis on;
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);

%% Trasformation of Image

pad=0;
[timg1] = trasformationImage(cimg1,inv(H),pad);
[timg2] = padarray(cimg2,[pad,pad]);
%%
[timg3] = trasformationImage(cimg3,inv(H2),pad);

%%
figure('name','transformed image 1');
imshow(uint8(timg3));
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
stich2=stichImage(timg2,timg3);
%%
figure('name','transformed image 3');
imshow(uint8(stich2));
impixelinfo;
title('\fontsize{10}{\color{magenta}transformed image 1}');
axis tight,axis on;
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);