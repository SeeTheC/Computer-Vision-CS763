%% Assign1-2 Barrel Distortion


%% Init
tic;
checkerboard='../input/rad_checkerbox.jpg';
cmGray256=gray(256);
img=imread(checkerboard);
img=img(:,:,1);
dim=size(img);
row=dim(1);col=dim(2);
img1 = padarray(img,[200 200],0,'both');
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
imshow(outImg,colormap(cmGray256));
%o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);
%axis tight,axis on;
impixelinfo();
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
b=zeros(5170,5190);
rimg=imresize(img,1);
%%
for i=1:517
    for j=1:519
        x=round(xMap(i,j)*10);y=round(yMap(i,j)*10);
        %fprintf('\n (%d,%d) -> (x:%d y:%d)',i,j,x,y);
        b(x,y)=rimg(i,j);
    end
end
%%
figure('name','out1');
imshow(rimg,colormap(cmGray256));
impixelinfo();
%
figure('name','out2');
imshow(b,colormap(cmGray256));
impixelinfo();
%%

chess='../input/chess.png';
cmGray256=gray(256);
img2=imread(chess);
img2=img2(:,:,1);

tic;
figure('name','Original Image');
imshow(img2,colormap(cmGray256)),daspect([1,1,1]);
title('\fontsize{10}{\color{red}Rad Checkerbox }');
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);
axis tight,axis on;
impixelinfo()

[outImg2]=barrelDistortion(img2);
figure('name','out');

figure('name','out');
imshow(outImg2,colormap(jet)),daspect([1,1,1]);
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);
axis tight,axis on;
impixelinfo();
