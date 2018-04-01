function [ goodFeaturePoint ] = getGoodFeaturePoints(img,featurePoint,windowSize,thershold)
    xDervMask=[-1,0,1];
    yDervMask=xDervMask';  
    Ix=	imfilter(img,xDervMask,'conv');
    Iy= imfilter(img,yDervMask,'conv');
    goodFeaturePoint=[];
    noOfPoint=size(featurePoint,1);
    [H,W]=size(img);
    for i=1:noOfPoint
        cx=featurePoint(i,1);cy=featurePoint(i,2);
        [x1,x2,y1,y2] = getPatchCoordinate(cx,cy,[windowSize,windowSize]);
        if x1<1 || x2 >H || y1<1 || y2 >W
            continue;
        end
        patchIx=Ix(x1:x2,y1:y2);
        patchIy=Iy(x1:x2,y1:y2);        
        [lamda1,lamda2]=structureTensor(patchIx,patchIy);
        %fprintf('point:%d\tx:%d\ty:%d \tl2:%f\n',i,cx,cy,lamda2);                  
        if lamda2>thershold
            goodFeaturePoint=vertcat(goodFeaturePoint,[cx,cy,lamda2]);
        end
    end
    [~,idx]=sort(goodFeaturePoint(:,3),'descend');
    goodFeaturePoint=goodFeaturePoint(idx,:);
    
end

%Corner-ness and Eigen Valus of Structure Tensor Matrix
function [lamda1,lamda2]=structureTensor(Ix,Iy)

    % Finding second moment matrix Or Structure []
    sumIxSq= sum ( sum ( (Ix.^2) ) );
    sumIySq= sum ( sum ( (Iy.^2) ) );
    sumIxIy= sum ( sum (  Ix.*Iy ) );
    % structure tensor
    A = [sumIxSq, sumIxIy; sumIxIy, sumIySq];
    % Finding Eigen values of the Tensor Matrix
    [V,D]=eig(A);
    diagD=sort(diag(D),'descend');
    lamda1=diagD(1);lamda2=diagD(2);
end

