function [ outImg ] = reverseBarrelDistortion(img)
%   Radial Distortion:  xd = xu(1 + q1.r + q2.r^2 )
%   q1 and q2 are given. q1 = 1 and q2 = 0.5 and r = ||xu|| (L2-Norm)    
    outImg=0;
    dim=size(img);
    row=dim(1);col=dim(2);   
    x=22,y=22;
    [nX,nY]=normalizeCooridate(x,y,dim);
    %[x,y]=inverseNormalizeCooridate(nX,nY,dim);
    xc =inverseMapping([nX;nY;1]);
    
    
end

function newX = inverseMapping(dX)
    tillConverge=1;
    xi=dX;    
    q1=1;q2=0.5;
    thershold=0;
    while tillConverge
        r=norm([xi(1),xi(2)],2);
        delta=(1+(q1*r)+(q2*r^2));
        deltaX=xi(1)*delta;deltaY=xi(2)*delta;        
        H=[1,0,deltaX;0,1,deltaY;0,0,1];
        newX=inv(H)*dX;   
        errorDiff=newX-xi;
        mError=norm(errorDiff,2);
        if mError<=thershold
            tillConverge=0;
        end
        xi=newX;
    end
end


% Convert from sensor c.s to center as origin and X and Y axis of unit length
% in all direction i.e up,down,top and bottom
function [nX,nY]=normalizeCooridate(x,y,dim)
    row=dim(1);col=dim(2);
    oX=ceil(row/2);oY=ceil(col/2);    
    xPerCellDist=1/(oX-1);yPerCellDist=1/(oY-1);
    nX=(x-oX) * xPerCellDist;
    nY=(y-oY) * yPerCellDist;    
end

function [x,y]=inverseNormalizeCooridate(nX,nY,dim)
    row=dim(1);col=dim(2);
    oX=ceil(row/2);oY=ceil(col/2);    
    xPerCellDist=1/(oX-1);yPerCellDist=1/(oY-1);
    x = (nX/xPerCellDist)+oX;
    y = (nY/yPerCellDist)+oY;   
end


