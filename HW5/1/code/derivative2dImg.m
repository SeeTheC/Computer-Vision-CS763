% Taking derivative of image 
function [IxGrad,IyGrad] = derivative2dImg(img,blursigma)
    
    % Derivative are susptable to noise. So bluring ith with gaussaion nosie 
    % defore taking it.
    %{
    blurMask=fspecial('gaussian',[patchSize patchSize],blurSigma);
    blurImg=imfilter(nextFrame,blurMask);
    % taking derivative of image
    xDervMask=[-1,0,1];        
    yDervMask=xDervMask';      
    [xder,yder]=ndgrid(floor(-3*blurSigma):ceil(3*blurSigma),floor(-3*blurSigma):ceil(3*blurSigma));        
    IxGrad1 = imfilter(blurImg,xder,'conv');
    IyGrad1 = imfilter(blurImg,yder,'conv');                       
    %}
    
    %Make derivatives kernels
    [xder,yder]=ndgrid(floor(-3*blursigma):ceil(3*blursigma),...
                       floor(-3*blursigma):ceil(3*blursigma));      
    % gaussain blur kernel + derivative kernel    
    derGx=-(xder./(2*pi*blursigma^4)).*exp(-(xder.^2+yder.^2)/(2*blursigma^2));
    derGy=-(yder./(2*pi*blursigma^4)).*exp(-(xder.^2+yder.^2)/(2*blursigma^2));
    
    % Filter the images to get the derivatives
    IxGrad = imfilter(img,derGx,'conv');
    IyGrad = imfilter(img,derGy,'conv');

end

