%% KLT
% Roll no: 163059009, 16305R011
%% Init
clear;
close all;
clc;

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
C = corner(img,N);

% Plotting
figure('name','Corners for First Frame');
imshow(img,[]);
hold on
%plot(C(49,1),C(49,2),'r*');
%plot(goodFeaturePoint(49,1),goodFeaturePoint(49,2),'r*');
plot(C(:,1),C(:,2),'m*');

title('\fontsize{10}{\color{magenta}Corners for First Frame}');
axis tight,axis on;
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);
saveas(gcf,strcat('../output/',num2str(1),'.jpg'));
impixelinfo;
%% 3. Part.c) Finding Good Feature Point
%patchSize=41;thershold=700;
%[goodFeaturePoint] = getGoodFeaturePoints(img,C,patchSize,thershold);

%% 4. Part. d)
patchSize=41;
numOfFrames=10;% Fo testing

frameNumFromRef=1;%used to change the reference frame every 8th frame
currentRefFrameNum=1;
newCorners=zeros(numOfFrames,N,2);
newCorners(1,:,:)=C;
for cornerNum=49:49
    
    %assume rotation to be zero initially
    p=[0 0 C(cornerNum,1) 0 0 C(cornerNum,2)];

    % make a template
    centerCoordinate=[C(cornerNum,1) C(cornerNum,2)];
    [x1t,y1t,x2t,y2t]=getWindowCoordinate(patchSize,centerCoordinate,[r c]);
    img = Frames(:,:,i);
    template=img(x1t:x2t,y1t:y2t);
    [x,y]=ndgrid(0:size(template,1)-1,0:size(template,2)-1);
    
    %Calculate center of the template image
    TemplateCenter=floor(size(template)/2);
    
    %Make center of the template image coordinates 0,0
    x=x-TemplateCenter(1); y=y-TemplateCenter(2);

    for i=2:numOfFrames
        if i==60
            continue;
        end
        %newFrame
        nextFrame =Frames(:,:,i);
        
        blurSigma=0.01;
        blurMask=fspecial('gaussian',[patchSize patchSize],blurSigma);
        blurImg=imfilter(nextFrame,blurMask);
        
        % taking derivative of image
        xDervMask=[-1,0,1];
        yDervMask=xDervMask';  
        IxGrad = imfilter(blurImg,xDervMask,'conv');
        IyGrad = imfilter(blurImg,yDervMask,'conv');

        frameNumFromRef=frameNumFromRef+1;
        counter=0;% used for convergence criterion
        %Threshold
        Threshold = 0.5;
        changeInP=10;
        
        while ( norm(changeInP) > Threshold)%add one more condition if error greater than previous error
            counter= counter + 1;
            %Break if it is not convergence for more than 80 loop, and consider it as convergence
            if(counter > 40)
                break;
            end
            
            Wp=[1+p(1) p(2) p(3);p(4) 1+p(5) p(6)];
            % Warp img with w
            imgWarped = warpping(nextFrame,x,y,Wp);
            imgError= double(template) - double(imgWarped);
            Ix =  warpping(IxGrad,x,y,Wp);   
            Iy = warpping(IyGrad,x,y,Wp);
            
            WJacobianx=[x(:) y(:) ones(size(x(:))) zeros(size(x(:)))  zeros(size(x(:)))  zeros(size(x(:)))];
            WJacobiany=[zeros(size(x(:)))  zeros(size(x(:))) zeros(size(x(:))) x(:) y(:)  ones(size(x(:)))];
            
            % Compute steepest descent
            imgSteepest=zeros(numel(x),6);
            for j1=1:numel(x),
                WJacobian=[WJacobianx(j1,:); WJacobiany(j1,:)];
                Gradient=[Ix(j1) Iy(j1)];
                imgSteepest(j1,1:6)=double(Gradient)*double(WJacobian);
            end
            
            %6 Compute Hessian
            H=zeros(6,6);
            for j2=1:numel(x)
                H=H+ imgSteepest(j2,:)'*imgSteepest(j2,:); 
            end
            %7 Multiply steepest descend with error
            total=zeros(6,1);
            for j3=1:numel(x)
                total=total+imgSteepest(j3,:)'*double(imgError(j3)); 
            end
            %8 Computer delta_p
            changeInP=H\total;
            %9 Update the parameters p <- p + delta_p
            if norm(changeInP)>3
                break;
            end
            %%
            p = p + changeInP';             
            fprintf('error=%f\tchangeInP=%f \n',norm(double(imgError)),norm(changeInP));            
            %break;
        end
        %to change the reference frame every 8 frame
        if(frameNumFromRef>8)
            frameNumFromRef=1;
            currentRefFrameNum=i;
            p=[0 0 p(3) 0 0 p(6)];
            centerCoordinate=[p(3) p(6)];
            [x1t,y1t,x2t,y2t]=getWindowCoordinate(patchSize,centerCoordinate,[r c]);
            img = reshape(Frames(currentRefFrameNum,:,:),r,c);
            template=img(floor(x1t):floor(x2t),floor(y1t):floor(y2t));
            [x,y]=ndgrid(0:size(template,1)-1,0:size(template,2)-1);
            %Calculate center of the template image
            TemplateCenter=floor(size(template)/2);
            %Make center of the template image coordinates 0,0
            x=x-TemplateCenter(1); y=y-TemplateCenter(2);
        end
        % to store corners of new frames
        newCorners(i,cornerNum,1)=p(3);
        newCorners(i,cornerNum,2)=p(6);
        Display(nextFrame,newCorners,i,cornerNum)
    end
end

%% Save all the trajectories frame by frame
% variable trackedPoints assumes that you have an array of size 
% No of frames * 2(x, y) * No Of Features
% noOfFeatures is the number of features you are tracking
% Frames is array of all the frames(assumes grayscale)
noOfPoints = 1;
for i=1:N
    NextFrame = Frames(i,:,:);
    imshow(uint8(NextFrame)); hold on;
    for nF = 1:noOfFeatures
        plot(trackedPoints(1:noOfPoints, 1, nF), trackedPoints(1:noOfPoints, 2, nF),'*')
    end
    hold off;
    saveas(gcf,strcat('output/',num2str(i),'.jpg'));
    close all;
    noOfPoints = noOfPoints + 1;
end 
   


k