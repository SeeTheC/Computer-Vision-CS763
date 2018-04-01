% Returns points in area of interest
function [ filterPoints ] = filterPtMovingArea(pt,movingArea)
       n=size(pt,1);
       x1=movingArea(1);x2=movingArea(2);
       y1=movingArea(3);y2=movingArea(4);
       filterPoints=[];
       for i=1:n
           x=pt(i,1);y=pt(i,2);
           if x>=x1 && x<=x2 && y>=y1 && y<=y2
            filterPoints=vertcat(filterPoints,pt(i,:));
           end
       end    
end

