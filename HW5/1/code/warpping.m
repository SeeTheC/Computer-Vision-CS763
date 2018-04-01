function imgOut = warpping(Iin,x,y,W)
    % Affine transformation
    Tlocalx =  W(1,1) * x + W(1,2) *y + W(1,3);
    Tlocaly =  W(2,1) * x + W(2,2) *y + W(2,3);
    xBas0=floor(Tlocalx);yBas0=floor(Tlocaly);    
    %imgOut = interp2(Iin,Tlocaly,Tlocalx);   
    imgOut = interp2(Iin,yBas0,xBas0);   
    
    %imgOut=double(Iin(1+xBas0+yBas0*size(Iin,1)));
end


