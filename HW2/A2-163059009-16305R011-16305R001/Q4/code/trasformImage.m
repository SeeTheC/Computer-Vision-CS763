function [ timg ] = trasformImage(img,H)
dim=size(img);
colorImg=numel(dim)==3;
if colorImg
    timg = zeros (size(img,1),size(img,2),3);
else
    timg = zeros (size(img,1),size(img,2));          
end 
tdim=size(timg);
Hinv=inv(H); 
% Inverse mapping for avoiding "holes" in transformed image
for tr=1:tdim(1)
    for tc=1:tdim(2)
        inverseCoord=Hinv*[tr;tc;1];
        r=round(inverseCoord(1)/inverseCoord(3));
        c=round(inverseCoord(2)/inverseCoord(3));
        if (r>0 && r <=dim(1) && c> 0 && c<=dim(2))
             timg(tr,tc,:)=img(r,c,:);
        end
    end
end

end

