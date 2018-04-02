% draw marker for specific tickness
function [ img ] = drawMarker(img,pt,marker,tickness,color,size)
    img = insertMarker(img,[pt(2),pt(1)],marker,'color',color,'size',size);        
    for j = 1:tickness-1            
        img = insertMarker(img,[pt(2)-j,pt(1)],marker,'color',color,'size',size);
        img = insertMarker(img,[pt(2)+j,pt(1)],marker,'color',color,'size',size);
        img = insertMarker(img,[pt(2),pt(1)-j],marker,'color',color,'size',size);
        img = insertMarker(img,[pt(2),pt(1)+j],marker,'color',color,'size',size);        
    end
end

