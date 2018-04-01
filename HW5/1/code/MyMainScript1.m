%% KLT
% Roll no: 163059009, 16305R011
%% Init
clear;
close all;

%% (a) Read the frames from input folder
numOfFrames=247;
imgDim=[480,640];
Frames=zeros(imgDim(1),imgDim(2),numOfFrames-1);
for i=1:numOfFrames
    if i ~= 60
        Frames(:,:,i)=imread(['../input/' num2str(i) '.jpg']);
    end
end
%% 2. (b) & (c) Display major features points overlay-ed on the first frame, for feature 
% point detection you can use Harris corner detector or SURF; 
% both are inbuilt in MATLAB.
% parameter N for max num of features

N=50;
%C=zeros(N,2);
[r,c]=size(Frames(:,:,1));
img = Frames(:,:,1);
C = corner(img,N);
%C=detectHarrisFeatures(img);
windowSize=7;
sigma=1.3;
blurSigma=0.01;
k=0.16;
%% 2. Part.b) Finding and Ploting all Feature point using Haris corner dectector
img = Frames(:,:,1);
[H,W]=size(img);
C = corner(img,N);
C=[C(:,2),C(:,1)];
% Plotting
figure('name','Corners for First Frame');
imshow(img,[]);
hold on
%plot(C(49,1),C(49,2),'r*');
%plot(goodFeaturePoint(49,1),goodFeaturePoint(49,2),'r*');
plot(C(:,2),C(:,1),'m*');

title('\fontsize{10}{\color{magenta}Corners for First Frame}');
axis tight,axis on;
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);
saveas(gcf,strcat('../output/',num2str(1),'.jpg'));
impixelinfo;
%% 3. Part.c) Finding Good Feature Point
patchSize=41;thershold=5e5;
[goodFeaturePoint] = getGoodFeaturePoints(img,C,patchSize,thershold);
goodFeaturePoint= filterPtMovingArea(goodFeaturePoint,[1,480,1,268]);

%%
k=[15:20];
% Plotting
figure('name','Corners for First Frame');
imshow(img,[]);
hold on
plot(goodFeaturePoint(k,2),goodFeaturePoint(k,1),'m*');

title('\fontsize{10}{\color{magenta}Corners for First Frame}');
axis tight,axis on;
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);
impixelinfo;
%% 4. Part. d)
patchSize=41;
numOfFrames=10;% Fo testing

frameNumFromRef=1;%used to change the reference frame every 8th frame
currentRefFrameNum=1;
newCorners=zeros(numOfFrames,N,2);
newCorners(1,:,:)=C;
%for cornerNum=49:49
        %fpt=[C(cornerNum,2),C(cornerNum,1)];        
        %fprintf('###Corner Point:%d###',cornerNum);
        fpt=[304,165];
        fpt=[goodFeaturePoint(13,1),goodFeaturePoint(13,2)];                        
        [outputCoord]=KLT(fpt,patchSize,Frames);          
%end

%% Save all the trajectories frame by frame
noOfPoints = 1;
for i=1:20
    if i==60 % frame doesnot exits
        continue;
    end        
    NextFrame = Frames(:,:,i);
    imshow(uint8(NextFrame),[]); hold on;
    for j = 1:i
        plot(outputCoord(j,2), outputCoord(j,1),'m*')
    end
    hold off;
    saveas(gcf,strcat('../output/trajectory/',num2str(i),'.jpg'));
    close all;
    noOfPoints = noOfPoints + 1;
end 
   

