function [ outImg ] = reverseBarrelDistortion(img)
%   Radial Distortion:  xd = xu(1 + q1.r + q2.r^2 )
%   q1 and q2 are given. q1 = 1 and q2 = 0.5 and r = ||xu|| (L2-Norm)    
    outImg=0;
    dim=size(img);
    row=dim(1);col=dim(2);   
    x=22,y=22;    
    [nX,nY]=normalizeCooridate(x,y,dim);
    [x,y]=inverseNormalizeCooridate(nX,nY,dim);
    %xc=inverseMapping([3.412;3.412;1]);
    xc=inverseMapping([nX;nY;1]);        
end

function newX = inverseMapping(dX)
    tillConverge=1;
    xi=dX;       
    thershold=0;
    q1=1;q2=0.5;

%{   
    while tillConverge     
        predX=inv(H(xi))*dX;   
        errorDiff=H(predX)*predX-dX;
        mError=norm(errorDiff,2);
        if mError<=thershold
            tillConverge=0;
        end
        xi=predX;
    end
 
%}
    
%  
    ai=[xi(1);xi(2)];
    dx=[dX(1);dX(2)];
    i=0;
    while tillConverge     
        r=norm(ai,2);
        delta=(1+(q1*r)+(q2*r^2));
        predX=dx./delta;
        
        %finding error
        rPred=norm(predX,2);
        deltaPred=(1+(q1*rPred)+(q2*rPred^2));        
        errorDiff=(predX.*deltaPred)-dx;
        mError=norm(errorDiff,2);
        disp(mError);
        if mError<=0.005
            tillConverge=0;
        end
        ai=predX;
        i=i+1;
    end
    fprintf('#of iteration:%d',i);
    
end
% return Distorted projection matrix
function [h]=H(xi)
        q1=1;q2=0.5;
        r=norm([xi(1),xi(2)],2);
        delta=((q1*r)+(q2*r^2));
        deltaX=xi(1)*delta;deltaY=xi(1)*delta;        
        h=[1,0,deltaX;0,1,deltaY;0,0,1];
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


