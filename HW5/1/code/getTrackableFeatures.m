function [trackableFeature,allFeatures] = getTrackableFeatures(img,patchSize,noTrackFeatures )
    %% Init
    thershold=5e5;
    N=50;
    %% Haris corner
    C = corner(img,N);
    C=[C(:,2),C(:,1)];
    % Finding good feature
    [goodFeaturePoint] = getGoodFeaturePoints(img,C,patchSize,thershold);
    goodFeaturePoint= filterPtMovingArea(goodFeaturePoint,[1,480,1,268]);    
    n=max(noTrackFeatures,size(goodFeaturePoint,1));
    trackableFeature=goodFeaturePoint(1:n,:);
    allFeatures=C;
end

