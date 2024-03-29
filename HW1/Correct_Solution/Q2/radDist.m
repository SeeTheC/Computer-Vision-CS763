function [xd, yd, imOut] = radDist(imIn, k1, k2, gridSize, norm)
    xd=[];
    yd=[];
    [m, n] = size(imIn);
    [x, y]=meshgrid(1:n, 1:m);
    k=1;
    cx = m/2;
    cy = n/2;
    
    x = x - cx;
    y = y - cy;
    x = x/cx;
    y = y/cy;

    x_orig = x;
    y_orig = y;

    r2 = sqrt(x.^2 + y.^2);
    dr =  1 + k1*r2 + k2*r2.^2;   
    x =  x_orig.*dr;
    y =  y_orig.*dr;
    if norm ~= 0 
        x = x/max(max(x));
        y = y/max(max(y));
    end
    x = x*cx + cx;
    y = y*cy + cy;
    
    for i=1:gridSize:m 
        for j=1:gridSize:n
            xd(k,1)=x(i,j);
            yd(k,1)=y(i,j);
            k = k + 1;
        end
    end
    
    imOut = interp2(imIn, x, y, 'cubic');
end