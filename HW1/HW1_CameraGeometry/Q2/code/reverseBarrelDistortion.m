function [ outImg ] = reverseBarrelDistortion(img)
%   Radial Distortion:  xd = xu(1 + q1.r + q2.r^2 )
%   q1 and q2 are given. q1 = 1 and q2 = 0.5 and r = ||xu|| (L2-Norm)    
    
    dim=size(img);
    row=dim(1);col=dim(2);
    
    img= [ zeros(row,200) img zeros(row,200)];
    img= [ zeros(200,col+(2*200));img;zeros(200,col+(2*200))]; 
    
    dim=size(img);
    row=dim(1);col=dim(2);            
    
    hrow=ceil(row/2);
    hcol=ceil(col/2);
    xMap=zeros(hrow,hcol); 
    yMap=zeros(hrow,hcol); 
    
    for r=1:hrow
        for c=1:hcol
            [nx,ny]=normalizeCooridate(r,c,dim);
            %fprintf('**Processing (%d,%d) -> (%f,%f):\n',r,c,nx,ny);            
            predictX=inverseMapping([nx;ny;1]);
            px=predictX(1);py=predictX(2);
            
            % Quadrant: left-top            
            [unx,uny]=inverseNormalizeCooridate(px,py,dim);
            xMap(r,c)=unx;yMap(r,c)=uny;
            
            % Quadrant: right-top
            [unx,uny]=inverseNormalizeCooridate(px,-1*py,dim);
            xMap(r,col-(c-1) )=unx;yMap(r,col-(c-1))=uny;
            
            % Quadrant: left-bottom
            [unx,uny]=inverseNormalizeCooridate(-1*px,py,dim);
            xMap(row-(r-1) ,c )=unx;yMap(row-(r-1) ,c)=uny;            
            
            % Quadrant: right-bottom
            [unx,uny]=inverseNormalizeCooridate(-1*px,-1*py,dim);
            xMap(row-(r-1) ,col-(c-1) )=unx;yMap(row-(r-1) ,col-(c-1))=uny;
            
        end
        %fprintf('\n r:%d',r);
    end          
    
    %Mappung Xd->Xu using Interpolation
    outImg=interp2(im2double(img),xMap,yMap)*256;
end

% Inversion mapping using Interative method
function newX = inverseMapping(dX)
    tillConverge=1;
    xi=dX;       
    q1=1;q2=0.5;

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
        %errorDiff=H(predX)*[predX;1]-[dx;1];
        mError=norm(errorDiff,2);
        %disp(mError);
        if mError<=0.00000000001
        %if sum(predX-ai)==0
            tillConverge=0;
        end
        ai=predX;
        i=i+1;
    end
    newX=predX;   
    
end

% Convert from sensor c.s to center as origin and X and Y axis of unit length
% in all direction i.e up,down,top and bottom
function [nX,nY]=normalizeCooridate(x,y,dim)
    row=dim(1);col=dim(2);
    oX=ceil(row/2);oY=ceil(col/2); 
    mag=norm([row/2;col/2],2);
    xPerCellDist=1/(oX-1);yPerCellDist=1/(oY-1);
    %xPerCellDist=1;yPerCellDist=1;
    nX=(x-oX) / mag;
    nY=(y-oY) / mag;    
end

function [x,y]=inverseNormalizeCooridate(nX,nY,dim)
    row=dim(1);col=dim(2);
    oX=ceil(row/2);oY=ceil(col/2);    
    xPerCellDist=1/(oX-1);yPerCellDist=1/(oY-1);
    mag=norm([row/2;col/2],2);
    %xPerCellDist=1;yPerCellDist=1;
    x = (nX * mag)+oX;
    y = (nY * mag)+oY;   
end