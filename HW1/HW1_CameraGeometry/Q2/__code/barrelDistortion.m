function [ outImg ] = barrelDistortion( img )
    dim=size(img);
    row=dim(1);col=dim(2);
    xc=ceil(row/2);yc=ceil(col/2);    
    outImg=zeros(floor(1.5*row),floor(1.5*col));    
    dim=size(outImg);    
    xdc=ceil(dim(1)/2);ydc=ceil(dim(1)/2); 
    
    xMap=zeros(row*2,col*2); 
    yMap=zeros(row,col);   
    value=zeros(row,col);         
    for r=1:row
        for c=1:col
            [xd,yd]=barrelDistort(r,c,xc,yc,xdc,ydc);
            %xMap(r,c)=unx;yMap(r,c)=uny;value(r,c)=img(r,c);             
            outImg(round(xd),round(yd))=img(r,c);
        end
    end

end

function [xd,yd]=barrelDistort(xu,yu,xc,yc,xdc,ydc)
    r=norm([(xu-xc),(yu-yc)],2);
    q1=1;q2=0.5;
    delta=1+(q1*r)+(q2*r^2);
    xd=((xu-xc)*delta)/r+xdc;
    yd=((yu-yc)*delta)/r+ydc;
end
