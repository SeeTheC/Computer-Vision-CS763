function [ H ] = homography( p1, p2 )
%HOMOGRAPHY Summary of this function goes here
%Detailed explanation goes here
    %% 1) Creating M matrix
    % Creating the M matrix of dimension 2Ix9 = 8x9 as I is 4
    % I2=H*I1
    % 
    % <<Mequation.png>>
    % 
    noOfPoints=size(p1,1);
    M=zeros(2*noOfPoints,9);

    for i=1:noOfPoints
        M(2*(i-1)+1,1:2)=-1.*p1(i,1:2);
        M(2*(i-1)+1,3:6)=[-1,0,0,0];
        M(2*(i-1)+1,7:8)=p2(i,1).*p1(i,1:2);
        M(2*(i-1)+1,9)=p2(i,1);

        M(2*i,1:3)=[0,0,0];
        M(2*i,4:5)=-1.*p1(i,1:2);
        M(2*i,6)=-1;
        M(2*i,7:8)=p2(i,2).*p1(i,1:2);
        M(2*i,9)=p2(i,2);
    end
    
    %% 2) Finding projection ~H
    [U,S,V]=svd(M);
    psol=V(:,9);
    H=reshape(psol,[3,3])';
    H=H./H(3,3);   

end

