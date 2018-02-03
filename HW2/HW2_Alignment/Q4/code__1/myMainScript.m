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
%% Feature Extraction(extract matching keypoints for two images);

[matchA,vpts1,vpts2]=match(img1,img2);
indexPairs=find(matchA);
matchedPoints1_1 = vpts1(indexPairs,:);
matchedPoints1_2 = vpts2(matchA(indexPairs),:);


%%
[matchB,vpts3,vpts2_2]=match(img3,img2);
indexPairs=find(matchB);
matchedPoints2_3 = vpts3(indexPairs,:);
matchedPoints2_2 = vpts2_2(matchB(indexPairs),:);
%% Homomorphic projection
tic
thresh=2;
[H,inliners,cp1,cp2] = ransacHomography(matchedPoints1_1, matchedPoints1_2, thresh);
toc
%%
thresh=2;
[H2,inliners2,cp3,cp2_2] = ransacHomography(matchedPoints2_3,matchedPoints2_2,thresh);
toc;
%% validation
val=H*cp1';
val=bsxfun(@times,val',1./val(3,:)');

%% mark point

pt=cp1;
noOfPoints=size(pt,1);        
for i=1:noOfPoints
    img1 = insertMarker(img1,[pt(i,2),pt(i,1)],'x','color','red','size',15);
    img1= insertText(img1,[pt(i,2)+5,pt(i,1)-5],strcat(num2str(i)), 'FontSize',18,'BoxColor', 'yellow');

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
    img2= insertText(img2,[pt(i,2)+5,pt(i,1)-5],strcat(num2str(i)), 'FontSize',18,'BoxColor', 'yellow');
end

figure('name','image 2');
imshow(uint8(img2));
impixelinfo;
title('\fontsize{10}{\color{magenta} image 2}');
axis tight,axis on;
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);


%% 2. Trasformation of Image 1

pad=0;
[timg1] = trasformationImage(cimg1,H,pad);
%%
[timg2] = padarray(cimg2,[pad,pad]);
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

%% Stiching of Multiple Images
stichedImage=stichMultipleImage({timg1,timg2,timg3});
%%
figure('name','transformed image 3');
imshow(uint8(stichedImage));
impixelinfo;
title('\fontsize{10}{\color{magenta}transformed image 1}');
axis tight,axis on;
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);
%%
tim
yshit
affine2d