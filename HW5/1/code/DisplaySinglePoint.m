function DisplaySinglePoint(frame,newCorners,title)
    figure('name',title);
    imshow(frame,[]);
    hold on;
    plot(newCorners(2),newCorners(1),'m*'),impixelinfo;   
    hold off;
    saveas(gcf,strcat('../output/',title,'.jpg'));
    close all;
end