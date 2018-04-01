function Display(nextFrame,newCorners,i,cornerNum)
    figure('name','testing');
    imshow(nextFrame);
    hold on;
    plot(newCorners(i,cornerNum,1),newCorners(i,cornerNum,2),'r*');
    hold off;
    saveas(gcf,strcat('../output/',num2str(i),'.jpg'));
    close all;
end