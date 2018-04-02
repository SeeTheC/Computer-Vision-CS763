function [ outputCoord ] = KLTSinglePointTracking(fpt,patchSize,Frames)
    %% INIT
    
    numOfFrames=size(Frames,3);
    %numOfFrames=20;
    maxConvgIter=50;   
    minDeltaPNorm=0.02;
    epsilon=3e-4;
    outputCoord=zeros(numOfFrames,2);    
    resetTemplate=20;
    
    %% Tracking Feature Point        
    [x,y]=ndgrid(0:(patchSize-1),0:(patchSize-1));    
    %center=floor(patchSize/2);   
    center=patchSize/2;    
    x=x-center ; y=y-center;
    %x=x+fptX; y=y+fptY;
    noOfPoints=numel(x);    
    
    prvFrameIndx=1;
    outputCoord(prvFrameIndx,:)=fpt;
    %DisplaySinglePoint(Frames(:,:,prvFrameIndx),fpt,'1');   
    for i=2:numOfFrames%numOfFrames
        if i==60 % frame doesnot exits
            continue;
        end
        %% Init           
        if (mod((i-2),resetTemplate)==0)
            % Init P
            img = Frames(:,:,prvFrameIndx); 
            p=[1,0,fpt(1),0,1,fpt(2)];                                
            [x1t,y1t,x2t,y2t]=getWindowCoordinate(patchSize,fpt,size(img));    
            template=img(x1t:x2t,y1t:y2t);    
        end
        %% Finding delta P        
        %newFrame
        nextFrame =Frames(:,:,i);
        % taking derivative
        [IxGrad,IyGrad] = derivative2dImg(nextFrame,3); 
         
        %Threshold
        changeInP=10;change=Inf;oldError=Inf;        
        lcount=1;% used for convergence criterion      
        fprintf('****Frame:%d********\n',i);
        while ( lcount<maxConvgIter && norm(changeInP) > minDeltaPNorm && abs(change)>epsilon)
            
            %% Part f.1) Creating W and Warping
            W=[p(1),p(2),p(3);...
               p(4),p(5),p(6)];            
            % Warp img with w
            imgWarped = warpping(nextFrame,x,y,W);            
            Ix =  warpping(IxGrad,x,y,W);   
            Iy =  warpping(IyGrad,x,y,W);         
            imgError= double(template) - double(imgWarped);
            %Ix = IxGrad(x1t:x2t,y1t:y2t);Iy = IyGrad(x1t:x2t,y1t:y2t);
            
            %% Checking for convergence
            error=rmse(imgWarped,template);   
            change=oldError-error;
            fprintf('%d) error=%f\tchangeInP=%f \n',lcount,error,norm(changeInP));            
            
            if (error>oldError) 
                if change>0.5
                    fprintf('****Break:  error > oldError*****\n');
                    %break;     
                end
            else
                oldError=error;
            end
                                                                
            %% Part f iv. Finding Jacobian, Gradient , gradI * Jacobian(W), Hession and Total Change
            %gradI_jacW=zeros(noOfPoints,6);                
            H=zeros(6,6);
            totalChange=zeros(1,6);                        
            for j=1:noOfPoints,
                jacW=[x(j),y(j),1,0,0,0;0,0,0,x(j),y(j),1];
                gradI=[Ix(j) Iy(j)];
                gradI_jacW=gradI*jacW;            
                H=H+gradI_jacW'*gradI_jacW; 
                totalChange=totalChange + (gradI_jacW*imgError(j)); 
            end            
            
            %% Part f. Finding delta p
            changeInP=totalChange/H;            
            %% Updating P
            p = p + changeInP;   
            lcount= lcount + 1;
        end                          
        W=[p(1),p(2),p(3);p(4),p(5),p(6)];
        fpt=round(W*[0,0,1]')';
        outputCoord(i,:)=fpt;
        prvFrameIndx=i;
        DisplaySinglePoint(Frames(:,:,i),fpt,num2str(i));        
    end

end

