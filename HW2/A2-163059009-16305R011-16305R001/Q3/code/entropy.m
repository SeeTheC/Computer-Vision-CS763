function [entropyValue] = entropy(movImg,fixImg,binSize)
    % Finding total number of bins
    noOfBin=ceil(255/binSize);
    movImg(movImg<0)=0;movImg(movImg>255)=255;
    fixImg(fixImg<0)=0;fixImg(fixImg>255)=255;
    
    % Converting pixel intensity to the "bin label"
    % Bin label starts at 1
    binLabledMovI=ceil(movImg/binSize);
    binLabledfixI=ceil(fixImg/binSize);
    
    % Removing All zeros which are at same location in both images
    % i.e finding Region of overlap
    dim=size(binLabledMovI);
    totalNoOfPixels=dim(1)*dim(2);   
    binLabledMI=reshape(binLabledMovI,[dim(1)*dim(2),1]);
    binLabledfI=reshape(binLabledfixI,[dim(1)*dim(2),1]);
    % Joint Count Matrix
    jointMtx=zeros(noOfBin,noOfBin);
    totalOverlapPixel=0;
    
    % Finding Count of the combination
    for i=1:1:totalNoOfPixels
        %fprintf('%d\n',i);        
        x=binLabledMI(i);
        y=binLabledfI(i);
        if ( x ~=0 && y ~=0)
            % Removing All zeros which are at same location in both images
            % i.e finding entropy for "Region of overlap"
            jointMtx(x,y)=jointMtx(x,y)+1;
            totalOverlapPixel=totalOverlapPixel+1;
        end        
    end
    
    % Finding Probability    
    jointProbMtx=jointMtx./totalOverlapPixel;    
    
    % Finding Entropy H= -1*  Sigma p.log p
    % Removing zero element, as log at 0 is not define
    jointProbMtx=jointProbMtx(jointProbMtx>0);
    entropyValue=-1 * sum(jointProbMtx.*log(jointProbMtx));    
end