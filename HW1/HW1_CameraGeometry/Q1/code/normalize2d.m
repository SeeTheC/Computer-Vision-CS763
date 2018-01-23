function [newpts, T] = normalize2d(pts)

    indices = find(abs(pts(:,3)) > eps);
    

    pts(indices,1) = pts(indices,1)./pts(indices,3);
    pts(indices,2) = pts(indices,2)./pts(indices,3);
    pts(indices,3) = 1;
    
    c = mean(pts);            % Centroid of points
    ptsWithc0(indices,1) = pts(indices,1)-c(1); % Shift origin to centroid.
    ptsWithc0(indices,2) = pts(indices,2)-c(2);
    
    dist = sqrt(ptsWithc0(indices,1).^2 + ptsWithc0(indices,2).^2);
    meandist = mean(dist(:));  
    
    scale = sqrt(2)/meandist;
    
    T = [scale   0   -scale*c(1)
         0     scale -scale*c(2)
         0       0      1      ];
    
    newpts = (T*pts')';
    