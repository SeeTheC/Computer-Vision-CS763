% It will Rotate + Traslate the passed image.
% Also will add the noise
function [ outImg ] = moveImage(movImg,rot,tran,noise)
    % Rotation
    if rot ~=0
        movImg = imrotate(movImg,rot,'bilinear','crop');
    end 
    
    % translating img by -3 pixels in X direction
    if tran(1) ~=0 || tran(2) ~=0
        movImg=imtranslate(movImg,tran,'FillValues',0);
    end
    
    % Adding noise
    if noise ~=0
        movImg = double(movImg) + rand(size(movImg))*noise;
    end 
    
    outImg=movImg;
end

