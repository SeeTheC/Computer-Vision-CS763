function [entropyValue] = entropy(movImg,fixImg,binSize)
    % Finding total number of bins
    noOfBin=ceil(255/binSize);
    
    % Correcting the Data
    movImg(movImg<0)=0;
    movImg(movImg>255)=255;
    fixImg(fixImg<0)=0;
    fixImg(fixImg>255)=255;
    
    % Converting pixel intensity to the "bin label"
    binLabledMovI=ceil(movImg/noOfBin);
    binLabledfixI=ceil(fixImg/noOfBin);
    
    % Removing All zeros which are at same location in both images
    binLabledMI = binLabledMovI(binLabledMovI ~=0 & binLabledfixI ~=0);
    binLabledfI = binLabledfixI(binLabledMovI ~=0 & binLabledfixI ~=0);
    
    totalNoOfPixels=numel(binLabledMI);
   
    % Joint Count Matrix
    jointMtx=zeros(noOfBin,noOfBin);
   
    % Finding Count of the combination
    for i=1:1:totalNoOfPixels
        %fprintf('%d\n',i);
        x=binLabledMI(i);y=binLabledfI(i);
        jointMtx(x,y)=jointMtx(x,y)+1;
    end
    
    % Finding Probability    
    jointProbMtx=jointMtx./totalNoOfPixels;    
    
    % Finding Entropy H= -1*  Sigma p.log p
    % Removing zero element as log in not define
    jointProbMtx=jointProbMtx(jointProbMtx>0);
    entropyValue=-1 * sum(jointProbMtx.*log(jointProbMtx));    
end