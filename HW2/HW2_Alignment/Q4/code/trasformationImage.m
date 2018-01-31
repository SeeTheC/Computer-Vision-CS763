function [ timg ] = trasformationImage(img,H,pad)
dim=size(img);
pimg = padarray(img,[pad pad]);
pdim=size(pimg);
colorImg=numel(pdim)==3;
if colorImg
    timg = zeros (size(pimg,1),size(pimg,2),3);
else
    timg = zeros (size(pimg,1),size(pimg,2));          
end 
%pimg=img;
tdim=size(timg);
Hinv=inv(H); 

%inverse mapping
for tr=1:tdim(1)
    for tc=1:tdim(2)
        inverseCoord=Hinv*[tr;tc;1];
        r=round(inverseCoord(1)/inverseCoord(3));
        c=round(inverseCoord(2)/inverseCoord(3));
        if (r>0 && r <=pdim(1) && c> 0 && c<=pdim(2))
             timg(tr,tc,:)=pimg(r,c,:);
        end
    end
end

end

