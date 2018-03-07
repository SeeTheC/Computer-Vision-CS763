% Stich mutiple images of same dimension into one image
function [ outImg ] = stichMultipleImage(imagesSet)
        noOfImg=numel(imagesSet);
        dim=size(imagesSet{1});  
        colorImg=numel(dim)==3;
        if colorImg
            outImg=zeros(dim(1),dim(2),3);
        else
            outImg=zeros(dim(1),dim(2));           
        end        
        for r=1:dim(1)
            for c=1:dim(2)
                for j=1:noOfImg
                    jImg=imagesSet{j};
                    if colorImg
                        v=sum(jImg(r,c,:))/3;
                    else
                        v=jImg(r,c);
                    end 
                    if(v ~=0 )
                        break;               
                    end
                end
                img=imagesSet{j};                
                outImg(r,c,:)=img(r,c,:);
            end
        end
end

