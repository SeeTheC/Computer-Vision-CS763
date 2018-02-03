% It will create Panaroma for given set of images
% All images size must be same
function [ outImg ] = panaroma(imageSet,pad,ransacThreshold,ransacIteration)
    %% Init
    noOfImage=numel(imageSet);
    dim=size(imageSet{1});  
    colorImg=numel(dim)==3;
    %% Converting Images to Gray Scale
    if colorImg
        for i=1:noOfImage
            workingImgSet{i}=rgb2gray(imageSet{i}); 
        end
    else
        workingImgSet=imageSet;
    end


    %% Padding zeros for avoiding cropping of images
    for i=1:noOfImage
        workingImgSet{i}=padarray(workingImgSet{i},[pad,pad]); 
        imageSet{i}=padarray(imageSet{i},[pad,pad]); 
    end
    
    %% Feature Extraction(extract matching keypoints for two adjecent images);
   
    for i=1:noOfImage-1
        [mapping,vpts1,vpts2]=match(workingImgSet{i},workingImgSet{i+1});
        indexPairs=find(mapping);
        matchedPoints{i,1} = vpts1(indexPairs,:);
        matchedPoints{i,2} = vpts2(mapping(indexPairs),:);        
    end

    
    %% Finding Homomorphic projection 
    if (noOfImage == 3)
        [H1,inliners1] = ransacHomography(matchedPoints{1,1}, matchedPoints{1,2}, ransacThreshold,ransacIteration);
        [H2,inliners2] = ransacHomography(matchedPoints{2,2}, matchedPoints{2,1},ransacThreshold,ransacIteration);
    elseif noOfImage == 2
        [H1,inliners1] = ransacHomography(matchedPoints{1,1}, matchedPoints{1,2}, ransacThreshold,ransacIteration);
    end
    
    %%  Trasformation of Image For Striching
    if (noOfImage == 3)
        timg{1} = trasformImage(imageSet{1},H1);
        timg{2}= imageSet{2};
        timg{3}= trasformImage(imageSet{3},H2);
    elseif (noOfImage == 2)
        timg{1} = trasformImage(imageSet{1},H1);
        timg{2} = imageSet{2};       
    end

    %% Stiching of Multiple Images
    stichedImage=stichMultipleImage(timg);
    
    %% Output
    outImg= stichedImage;

end

