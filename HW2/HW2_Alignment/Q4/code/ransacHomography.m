function [ bestHomography,bestInliners,cp1,cp2] = ransacHomography(matchA, loc1,loc2,thresh )
%RANSACHOMOGRAPHY Summary of this function goes here
%   Detailed explanation goes here
    
    iter = 6000;    
    matchIndexs=find(matchA); %contains indeces of elementss with non-zero value of match(key points of image1)
    matchSize=numel(matchIndexs);
    k=6;
    p1=zeros(4,3);
    p2=zeros(4,3);
    cp1=p1;cp2=p2;
    bestInliners=0;
    bestHomography=zeros(3,3);
    for i=1:iter
        index=randperm(matchSize,k);
        inliners=0;
        matchRandIndex=matchIndexs(index); %contains random 4 non zero index of match(also means index of image 1 keypoints)
        for j=1:k
            p1(j,1)=loc1(matchRandIndex(j),1);
            p1(j,2)=loc1(matchRandIndex(j),2);
            p1(j,3)=1;
            p2(j,1)=loc2(matchA(matchRandIndex(j)),1);
            p2(j,2)=loc2(matchA(matchRandIndex(j)),2);
            p2(j,3)=1;
        end
        
        H=homography(p1,p2);
        % calculating inliners        
        for j=1:matchSize
            testPoint1=[loc1(matchIndexs(j),1) loc1(matchIndexs(j),2) 1];
            testPoint2=[loc2(matchA(matchIndexs(j)),1) loc2(matchA(matchIndexs(j)),2) 1];
            temp=(H*testPoint1')';
            temp(:,1)=temp(:,1)./temp(:,3);
            temp(:,2)=temp(:,2)./temp(:,3);
            temp(:,3)=1;
            dist=sqrt(sum((temp-testPoint2).^2));
            %if(abs(temp(1)-testPoint2(1))<thresh && abs(temp(2)==testPoint2(2))<thresh)
            %    inliners=inliners+1;
            %end
            if(dist<=thresh)
                inliners=inliners+1;
            end
        end
        if(inliners>bestInliners)
            bestInliners=inliners;
            bestHomography=H;
            cp1=p1;cp2=p2;
        end             
    end    
end

