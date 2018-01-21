%% Assignmen1-6


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


%% Your code here
tic;


%line([750,426],[800,688]);




% Blue Line
A=[750 426];
B=[800 688];

slope = (B(1)-B(2))/(A(1)-A(2));
xLeft = 0; % Whatever x value you want.
yLeft = slope * (xLeft - A(1)) + B(1);
xRight = 750; % Whatever x value you want.
yRight = slope * (xRight - A(1)) + B(1);

% vanishing point
vanishingPt=[116,581];

% Redline
C=[vanishingPt(1),435];
D=[vanishingPt(2), 425];
slope = (D(1)-D(2))/(C(1)-C(2));
xLeft1 = 0; % Whatever x value you want.
yLeft1 = slope * (xLeft1 - C(1)) + D(1);
xRight1 = 745; % Whatever x value you want.
yRight1 = slope * (xRight1 - C(1)) + D(1);


E=[vanishingPt(1),743];
F=[vanishingPt(2),219];


figure('name','Original Image: painting');%line([0 1],[0 1]);
imshow(img);
line([xLeft, xRight], [yLeft, yRight], 'Color', 'b', 'LineWidth', 3);
line([xLeft1, xRight1], [yLeft1, yRight1], 'Color', 'r', 'LineWidth', 3);
line(E,F, 'Color', 'm', 'LineWidth', 3);
impixelinfo;
title('\fontsize{10}{\color{red}Original Image: painting}');
axis tight,axis on;
o1 = get(gca, 'Position');
colorbar(),set(gca, 'Position', o1);



toc;
