% Add to image of same dimension
function [ outImg ] = stichImage(img1,img2)
        dim=size(img1);  
        colorImg=numel(dim)==3;
        if colorImg
            outImg=zeros(dim(1),dim(2),3);
        else
            outImg=zeros(dim(1),dim(2));           
        end        
        for r=1:dim(1)
            for c=1:dim(2)
                v1=sum(img1(r,c,:))/3;v2=sum(img2(r,c,:))/3;
                if(v1==0 && v2 == 0)
                    continue;               
                elseif(v1==0)
                    v=img2(r,c,:);
                elseif(v2==0)
                    v=img1(r,c,:);
                else
                    if colorImg
                        %v(:,:,1)=(img2(r,c,1)+img1(r,c,1))./2;
                        %v(:,:,2)=(img2(r,c,2)+img1(r,c,2))./2;
                        %v(:,:,3)=(img2(r,c,3)+img1(r,c,3))./2;                                             
                    end
                    v=img1(r,c,:);   
                end
                outImg(r,c,:)=v;
            end
        end
end

