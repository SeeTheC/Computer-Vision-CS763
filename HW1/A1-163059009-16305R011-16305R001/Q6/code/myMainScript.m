%% Assignmen1-6
% Roll no: 163059009, 16305R011, 16305R001

%% Init

file='../input/Painting.jpg';
img=imread(file);

figure('name','Original Image: painting');
imshow(img);
impixelinfo;
title('\fontsize{10}{\color{red}Original Image: painting}');
axis tight,axis on;
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);


%% Drawing lines
tic;

% Blue Line
A=[750 426];
B=[800 688];
slope = (B(1)-B(2))/(A(1)-A(2));
xLeft = 0; % Whatever x value you want.
yLeft = slope * (xLeft - A(1)) + B(1);
xRight = 750; % Whatever x value you want.
yRight = slope * (xRight - A(1)) + B(1);
img1=insertText(img,[745,795], strcat('b (800,745)'), 'FontSize',18,'BoxColor', 'magenta');
img1=insertText(img1,[A(2),B(2)], strcat('b0'), 'FontSize',18,'BoxColor', 'magenta');
img1=insertText(img1,[428,435], strcat('t0'), 'FontSize',18,'BoxColor', 'magenta');

% vanishing point
vanishingPt=[116,581];
img1=insertText(img1,[vanishingPt(1)+3,vanishingPt(2)+3], strcat('V'), 'FontSize',18,'BoxColor', 'magenta');

% Redline
C=[vanishingPt(1),435];
D=[vanishingPt(2), 425];
slope = (D(1)-D(2))/(C(1)-C(2));
xLeft1 = 0; % Whatever x value you want.
yLeft1 = slope * (xLeft1 - C(1)) + D(1);
xRight1 = 745; % Whatever x value you want.
yRight1 = floor(slope * (xRight1 - C(1)) + D(1));
img1=insertText(img1,[xRight1,yRight1], strcat('t (', num2str(yRight1),',745)'), 'FontSize',18,'BoxColor', 'magenta');

% magenta
E=[vanishingPt(1),745];
F=[vanishingPt(2),219];
img1=insertText(img1,[745,219], strcat('R (219,755)'), 'FontSize',18,'BoxColor', 'magenta');

img1=insertText(img1,[665,832], strcat('Vz at infinity'), 'FontSize',18,'BoxColor', 'red');

figure('name','Original Image: painting');%line([0 1],[0 1]);
imshow(img1);
line([xLeft, xRight], [yLeft, yRight], 'Color', 'b', 'LineWidth', 3);
line([xLeft1, xRight1], [yLeft1, yRight1], 'Color', 'r', 'LineWidth', 3);
line(E,F, 'Color', 'm', 'LineWidth', 3);
impixelinfo;
title('\fontsize{10}{\color{red}Original Image: painting}');
axis tight,axis on;
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);



toc;
