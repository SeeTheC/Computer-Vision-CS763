%% Rigid Motion Aligment using Joint entropy

%% Assignment2-3 
% Rollno: 163059009, 16305R011, 16305R001 

%% 1. Barbar Img
file='../input/barbara.png';
fixImg=imread(file); 

file='../input/negative_barbara.png';
movImg=imread(file); 

%Showing Original Image
figure('name','Original Img: Barbara');
subplot(1,2,1);
imshow(fixImg);
title('\fontsize{10}{\color{red}Fixed Image of Barbara (Img1)}');
axis tight,axis on;

subplot(1,2,2);
imshow(movImg);
title('\fontsize{10}{\color{red}Moving Image of Barbara (Img2)}');
axis tight,axis on;

%% 1.1 Moving Barbara Img2 
% Rotation by: 23.5 deg
% Translation: -3
% Add noise between [0,8]
rot=23.5; tran=[-3,0]; noise=8;
movedBarbara=moveImage(movImg,rot,tran,noise);

% Showing Rotated translated noised negative Barbara Image
figure('name','Rotated translated noise barbara Image');subplot(1,2,1);
imshow(fixImg);
title('\fontsize{10}{\color{red}Fixed Image of Barbara (Img1)}');
axis tight,axis on;

subplot(1,2,2);
imshow(uint8(movedBarbara));
title('\fontsize{10}{\color{magenta}Rotated + Translated + Noise barbara Image}');
axis tight,axis on;

%% 1.2 Finding Alignment
% Finding Alignment using brute force
movImg=movedBarbara;
rotRange=[-60,60]; transRange = [-12,12]; binSize=10;
[entropyValueMatrix,minEntropyVal,minTheta,minTx] = findAlignment(movImg, fixImg,rotRange,transRange,binSize);

%% 1.3 Plotting of Joint entropy
% For barbara image theta = -24 tx=3 minValue=3.343641
% Surface Plotting of joint entropy as a function of θ and tx

figure('name','joint entropy as a function of θ and tx');
[tansG,rotG]=meshgrid([transRange(1):transRange(2)],[rotRange(1):rotRange(2)]);
surf(tansG,rotG,entropyValueMatrix);
title('\fontsize{10}{\color{magenta}Joint entropy of Barbar}');
xlabel('Translation');ylabel('Rotation');zlabel('Entropy');

% Showing joint entropy as a function of θ and tx
figure('name','joint entropy');
imagesc(transRange,rotRange,entropyValueMatrix);
colorbar;
title('\fontsize{10}{\color{magenta}Joint entropy of Barbar}');
xlabel('Translation');ylabel('Rotation');
axis tight,axis on;

fprintf('For barbara image theta = %d tx=%d minValue=%f\n',minTheta,minTx,minEntropyVal);

%%
range=256;
timg=movImg;
transIdx=0;
evm=zeros(range);
timg=moveImage(timg,45,[-10,1],0);
for i=1:range
    timg=moveImage(movImg,0,[-i,i],0);
    [entropyValue]=entropy(timg,fixImg,binSize);
    evm(i)=entropyValue;  
    if entropyValue<2.3309
        break;
    end
    transIdx=transIdx+1;  
end
fprintf('Done: ev=%f  i=%d\n',entropyValue,i);
figure
imshow(uint8(timg));

%% 2. Flash image

file='../input/flash1.jpg';
fixImg=rgb2gray(imread(file)); 

file='../input/noflash1.jpg';
movImg=rgb2gray(imread(file));

%Showing Original Image
figure('name','Original Img: Barbara');
subplot(1,2,1);
imshow(fixImg);
title('\fontsize{10}{\color{red}Fixed Image of Flash (Img1)}');
axis tight,axis on;

subplot(1,2,2);
imshow(movImg);
title('\fontsize{10}{\color{red}Moving Image of Flash (Img2)}');
axis tight,axis on;

%% 2.1 Moving Flash Img2 
% Rotation by: 23.5 deg
% Translation: -3
% Add noise between [0,8]
rot=23.5; tran=[-3,0]; noise=8;
movedFlash=moveImage(movImg,rot,tran,noise);

% Showing Rotated translated noised negative Barbara Image
figure('name','Rotated translated noise barbara Image');
subplot(1,2,1);
imshow(fixImg);
title('\fontsize{10}{\color{red}Fixed Image of Flash (Img1)}');
axis tight,axis on;

subplot(1,2,2);
imshow(uint8(movedFlash));
title('\fontsize{10}{\color{magenta}Rotated + Translated + Noise Flash Image}');
axis tight,axis on;

%% 2.2 Finding Alignment
% Finding Alignment using brute force
% Total computation time: 437.173013 seconds.
tic
movImg=movedFlash;
rotRange=[-60,60]; transRange = [-12,12]; binSize=10;
[entropyValueMatrixs,minEntropyVal,minTheta,minTx] = findAlignment(movImg, fixImg,rotRange,transRange,binSize);
toc

%% 2.3 Plotting of Joint entropy

% Surface Plotting of joint entropy as a function of θ and tx
figure('name','Joint entropy as a function of θ and tx');
[tansG,rotG]=meshgrid([transRange(1):transRange(2)],[rotRange(1):rotRange(2)]);
surf(tansG,rotG,entropyValueMatrix);
title('\fontsize{10}{\color{magenta}Joint entropy of Flash}');
xlabel('Translation');ylabel('Rotation');zlabel('Entropy');

% Showing joint entropy as a function of θ and tx
figure('name','joint entropy');
%imagesc(entropyValueMatrix);
imagesc(transRange,rotRange,entropyValueMatrix);
colorbar;
title('\fontsize{10}{\color{magenta}Joint entropy of Flash}');
xlabel('Translation');ylabel('Rotation');
axis tight,axis on;

fprintf('For Flash image theta = %d tx=%d minValue=%f\n',minTheta,minTx,minEntropyVal);

