function imgOut = warpping(Iin,x,y,W)
    % Affine transformation
    predX =  W(1,1) * x + W(1,2) *y + W(1,3);
    predY =  W(2,1) * x + W(2,2) *y + W(2,3);
    fPredX=floor(predX);fPredY=floor(predY);    
    %imgOut = interp2(Iin,predY,predX,'cubic');   
    %imgOut = interp2(Iin,yBas0,xBas0);       
    imgOut=double(Iin(1+fPredX+fPredY*size(Iin,1)));
end


