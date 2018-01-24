%% Assignment1-1 
% Rollno: 163059009, 16305R011, 16305R001 

%% Init
file='../data/camR.jpg';
img=imread(file); 
dim=size(img);

%% Points
tic;
%datatset 1: 6 points
points3D_1=[1,0,1,1 ; 2,0,2,1 ; 1,0,3,1 ; 0,1,1,1 ; 0,2,2,1 ; 0,1,3,1];
points2D_1=[737,898,1 ; 640,811,1 ; 541,898,1 ; 742,1034,1 ; 653,1105,1 ; 546,1041,1];

%datatset 2: 12 points
points3D_2=[1,0,1,1 ; %1
            2,0,2,1 ; %2
            1,0,3,1 ; %3
            0,1,1,1 ; %4
            0,2,2,1 ; %5
            0,1,3,1 ; %6          
            2,0,1,1 ; %7 
            0,2,1,1 ; %8
            4,0,2,1 ; %9
            0,4,2,1 ; %10
            3,0,3,1 ; %11
            0,3,3,1 ; %12
            0,0,4,1 ; %13
            0,0,0,1 ; %14
            ];
        
points2D_2=[737,898,1 ; %1
            640,811,1 ; %2
            541,898,1 ; %3
            742,1034,1 ; %4
            653,1105,1 ; %5
            546,1041,1 ; %6
            741,810,1  ; %7
            756,1101,1 ; %8
            644,625,1  ; %9
            673,1253,1 ; %10
            537,724,1  ; %11
            555,1179,1 ; %12
            447, 981,1 ; %13
            825,973, 1 ; % 14
            ];
 
points3D=points3D_2;
points2D=points2D_2;
noOfPoints=size(points2D,1);


%% 1) Normalization
[newpts2D, T2,c2d]=normalize2d(points2D);
[newpts3D, T3,c3d]=normalize3d(points3D);
%% 2D point Marking
img1=img;
for i=1:noOfPoints
    img1 = insertMarker(img1,[points2D(i,2),points2D(i,1)],'x','color','red','size',15);
    img1= insertText(img1,[points2D(i,2)+3,points2D(i,1)+3], num2str(i), 'FontSize',18,'BoxColor', 'magenta');
end

% centoride
img1 = insertMarker(img1,[c2d(2),c2d(1)],'x','color','green','size',15);

% Adding Axis-label
img1=insertText(img1,[993+5,69], 'z-axis', 'FontSize',18,'BoxColor', 'red');
img1=insertText(img1,[36,913 + 5], 'x-axis', 'FontSize',18,'BoxColor', 'red');
img1=insertText(img1,[1570,989 + 5], 'y-axis', 'FontSize',18,'BoxColor', 'red');

% showing image
figure('name','Original:Point Marked image');
imshow(img1);
impixelinfo;
title('\fontsize{10}{\color{red}Original: Point Marked image}');
axis tight,axis on;

% Drawing 3d axis
line([973,993], [825,69], 'Color', 'red', 'LineWidth', 3);
line([973,36], [825,913], 'Color', 'red', 'LineWidth', 3);
line([973,1570], [825,989], 'Color', 'red', 'LineWidth', 3);


%% 2) Creating M matrix

M=zeros(2*noOfPoints,12);

for i=1:noOfPoints
    M(2*(i-1)+1,1:3)=-1.*newpts3D(i,1:3);
    M(2*(i-1)+1,4:8)=[-1,0,0,0,0];
    M(2*(i-1)+1,9:11)=newpts2D(i,1).*newpts3D(i,1:3);
    M(2*(i-1)+1,12)=newpts2D(i,1);
    
    M(2*i,1:4)=[0,0,0,0];
    M(2*i,5:7)=-1.*newpts3D(i,1:3);
    M(2*i,8)=-1;
    M(2*i,9:11)=newpts2D(i,2).*newpts3D(i,1:3);
    M(2*i,12)=newpts2D(i,2);
end

%% 3) Finding projection ~P
[U,S,V]=svd(M);
psol=V(:,12);
p=reshape(psol,[4,3])';
P=(inv(T2))*p*T3;

projectedPtn=(P*points3D')';
projectedPtn=bsxfun(@times,projectedPtn,projectedPtn(:,3).^-1);
RMSE=points2D-points2DEstimated;

%% Plotting projection point

img1=img;
for i=1:noOfPoints   
    x=round(projectedPtn(i,1));
    y=round(projectedPtn(i,2));
    fprintf('(x:%d,y:%d)',x,y);
    img1 = insertMarker(img1,[y,x],'x','color','yellow','size',20);
    img1 = insertMarker(img1,[points2D(i,2),points2D(i,1)],'o','color','red','size',15);
    img1=insertText(img1,[y+5,x+5], num2str(i), 'FontSize',18,'BoxColor', 'yellow');
    
end

% Legend
img1=insertText(img1,[10,10], 'Legend', 'FontSize',22,'BoxColor', 'green');
img1=insertText(img1,[10,60], 'X : Projection (x=P.X) of 3D point ', 'FontSize',22,'BoxColor', 'yellow');
img1=insertText(img1,[10,110], 'O : Marked Point', 'FontSize',22,'BoxColor', 'red');

% Adding Axis-label
img1=insertText(img1,[993+5,69], 'z-axis', 'FontSize',18,'BoxColor', 'red');
img1=insertText(img1,[36,913 + 5], 'x-axis', 'FontSize',18,'BoxColor', 'red');
img1=insertText(img1,[1570,989 + 5], 'y-axis', 'FontSize',18,'BoxColor', 'red');

figure('name','Projection');
imshow(img1);
impixelinfo;
title('\fontsize{10}{\color{magenta}Projection}');
axis tight,axis on;

% Drawing 3d axis
line([973,993], [825,69], 'Color', 'red', 'LineWidth', 3);
line([973,36], [825,913], 'Color', 'red', 'LineWidth', 3);
line([973,1570], [825,989], 'Color', 'red', 'LineWidth', 3);


%% 4) Finding Xo
HInfi=P(:,1:3); h=P(:,4);
HInfiInv=inv(HInfi);
%k=HInfi*HInfiInv;
Xo=-HInfiInv*h;


%% 5) Finding Rotation and Caliberation
[Q,R] = qr(HInfiInv,0);
%k=Q*Q';
k=-inv(R)*Q'*Xo;

%% 6) Test


toc;
