% Uses the Brute force method fo finding the alignment

function [entropyValueMatrix,minEntropyVal,minTheta,minTx] = findAlignment(movedImg, fixedImg,rotRange,transRange,binSize)

    thetaIdx=1;
    rotRangeLen=rotRange(2)-rotRange(1) + 1;
    transRangeLen=transRange(2)-transRange(1) + 1;
    entropyValueMatrix=zeros(rotRangeLen,transRangeLen);
    for theta=rotRange(1):1:rotRange(2)
        rImg=moveImage(movedImg,theta,[0,0],0);
        transIdx=1;
        for tx=transRange(1):1:transRange(2);
            mImg=moveImage(rImg,0,[tx,0],0);
            [entropyValue]=entropy(mImg,fixedImg,binSize);
            entropyValueMatrix(thetaIdx,transIdx)=entropyValue;
            transIdx=transIdx+1;           
        end
        thetaIdx=thetaIdx+1;        
    end

    % Finding rotation and translation for which min entroy occured
    [minEntropyVal,index]=min(entropyValueMatrix(:));
    % Finding the theta and translation value
    minTheta=mod(index,rotRangeLen)-ceil(rotRangeLen/2);
    minTx=ceil(index/rotRangeLen)-ceil(transRangeLen/2);

end

