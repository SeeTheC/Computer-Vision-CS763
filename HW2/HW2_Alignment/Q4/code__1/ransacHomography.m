function [ bestHomography,bestInliners,cp1,cp2] = ransacHomography(matchedPoints1, matchedPoints2,thresh )
%   RANSACHOMOGRAPHY Summary of this function goes here
%   Detailed explanation goes here
    
    iter = 4000;    
    matchSize=size(matchedPoints1,1);
    k=4;
    p1=zeros(4,3);
    p2=zeros(4,3);
    cp1=p1;cp2=p2;
    bestInliners=0;
    bestHomography=zeros(3,3);
    for i=1:iter
        index=randperm(matchSize,k);
        inliners=0;
        for j=1:k
            p1(j,1)=matchedPoints1(index(j),1);
            p1(j,2)=matchedPoints1(index(j),2);
            p1(j,3)=1;
            p2(j,1)=matchedPoints2(index(j),1);
            p2(j,2)=matchedPoints2(index(j),2);
            p2(j,3)=1;
        end
        
        H=homography(p1,p2);
        % calculating inliners        
        for j=1:matchSize
            testPoint1=[matchedPoints1(j,1),matchedPoints1(j,2), 1];
            testPoint2=[matchedPoints2(j,1),matchedPoints2(j,2), 1];
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

