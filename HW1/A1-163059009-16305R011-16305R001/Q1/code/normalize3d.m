function [newpts, U,c] = normalize3d(pts)

    indices = find(abs(pts(:,4)) > eps);
    

    pts(indices,1) = pts(indices,1)./pts(indices,4);
    pts(indices,2) = pts(indices,2)./pts(indices,4);
    pts(indices,3) = pts(indices,3)./pts(indices,4);
    pts(indices,4) = 1;
    
    % Centroid of points
    c = mean(pts);           
    % Shift origin to centroid.
    ptsWithc0(indices,1) = pts(indices,1)-c(1); 
    ptsWithc0(indices,2) = pts(indices,2)-c(2);
    ptsWithc0(indices,3) = pts(indices,2)-c(3);
    
    dist = sqrt(ptsWithc0(indices,1).^2 + ptsWithc0(indices,2).^2 +ptsWithc0(indices,3).^2);
    meandist = mean(dist(:));  
    
    scale = sqrt(3)/meandist;
    
    U = [scale   0     0     -scale*c(1)
         0     scale   0     -scale*c(2)
         0       0   scale   -scale*c(3)      
         0       0     0         1      ];
    
    newpts = (U*pts')';
    