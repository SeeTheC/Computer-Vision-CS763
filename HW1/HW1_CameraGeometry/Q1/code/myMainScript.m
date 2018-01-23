%% Assignment1-1 
% Rollno: 163059009, 16305R011, 16305R001 

%% Init
 file='../data/camR.jpg';
 img=imread(file);
 
figure('name','Original Image: Chessboard');%line([0 1],[0 1]);
imshow(img);
impixelinfo;
title('\fontsize{10}{\color{red}Original Image: Chessboard}');
axis tight,axis on;

%% Points
tic;
points3D=[1,1,0,1 ; 3,1,0,1 ; 9,1,1,1 ; 9,2,1,1 ; 9,3,3,1 ; 2,2,0,1];
points2D=[68,30,1 ; 430,200,1 ; 1050,255,1 ; 1050,349,1 ; 1185,340,1 ; 316,303,1];
%% 2D point Marking
truePoints2D=[30,68; 200,430 ; 255,1050; 349,1050; 340,1185; 303,316];

img1 = insertMarker(img,[truePoints2D(1,2),truePoints2D(1,1)],'x','color','red','size',10);
img1 = insertMarker(img1,[truePoints2D(2,2),truePoints2D(2,1)],'x','color','red','size',10);
img1 = insertMarker(img1,[truePoints2D(3,2),truePoints2D(3,1)],'x','color','red','size',10);
img1 = insertMarker(img1,[truePoints2D(4,2),truePoints2D(4,1)],'x','color','red','size',10);
img1 = insertMarker(img1,[truePoints2D(5,2),truePoints2D(5,1)],'x','color','red','size',10);
img1 = insertMarker(img1,[truePoints2D(6,2),truePoints2D(6,1)],'x','color','red','size',10);

figure('name','Point Marked image');%line([0 1],[0 1]);
imshow(img1);
impixelinfo;
title('\fontsize{10}{\color{red}Point Marked image}');
axis tight,axis on;

%% 1) Normalization
[newpts2D, T2]=normalize2d(points2D);
[newpts3D, T3]=normalize3d(points3D);

%%

% 2) Creating M matrix

M=zeros(12,12);

for i=1:6
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
p=reshape(V(:,12),[3 4]);
P=(inv(T2))*p*T3;

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
points2DEstimated=(P*points3D')';
indices = find(abs(points2DEstimated(:,3)) > eps);
points2DEstimated(indices,1) = points2DEstimated(indices,1)./points2DEstimated(indices,3);
points2DEstimated(indices,2) = points2DEstimated(indices,2)./points2DEstimated(indices,3);
points2DEstimated(indices,3) = 1;
RMSE=points2D-points2DEstimated;

toc;
