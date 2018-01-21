%% Assign1-2 Barrel Distortion


%% Init
tic;
checkerboard='../input/rad_checkerbox.jpg';
cmGray256=gray(256);
img=imread(checkerboard);
img=img(:,:,1);
dim=size(img);
row=dim(1);col=dim(2);
%%imagesc(img),daspect([1,1,1]),colormap(cmGray256),colorbar();
toc

%%
tic;
figure('name','Original Image');
imshow(img,colormap(cmGray256)),daspect([1,1,1]);
title('\fontsize{10}{\color{red}Rad Checkerbox }');
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);
axis tight,axis on;
impixelinfo();
% Size of Img:  517   519     3
toc;
%%

[ outImg,xMap,yMap,value ]=reverseBarrelDistortion(img);
%%
figure('name','out');
imshow(outImg,colormap(jet)),daspect([1,1,1]);
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);
axis tight,axis on;
impixelinfo();

