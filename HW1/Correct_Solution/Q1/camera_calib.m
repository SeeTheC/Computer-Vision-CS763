%% Camera Calibration Demo
%% Standard Data
% Loading Calibration Data

im = imread('checkerbox_2.jpeg');
% Constructing the A matrix in Am=0
f3D = [0 1 1 1; 0 2 1 1; 0 1 3 1; 0 0 2 1; 1 0 2 1; 2 0 1 1; 3 0 0 1; 2 0 3 1; 0 0 1 1; 2 0 2 1]';
% im = imresize(im,0.2);
% imshow(im);
% [x y] = ginput(10);
% f2D = [x y]';
% f2D = [f2D; 1 1 1 1 1 1 1 1 1 1];
load('f2D.mat')
l =size(f3D,2);

%% Step 1: Normalizing f3D and f2D
mean2D = mean(f2D')
mean3D = mean(f3D')

norm_sum2d = 0;
norm_sum3d = 0;
for i = 1:l
   norm_sum2d = sqrt(f2D(1,i)^2 + f2D(2,i)^2) + norm_sum2d;
   norm_sum3d = sqrt(f3D(1,i)^2 + f3D(2,i)^2 + f3D(3,i)^2) + norm_sum3d;
end
norm2d = norm_sum2d/l;
norm3d = norm_sum3d/l;
s2D = sqrt(2)/norm2d;
s3D = sqrt(3)/norm3d;
% T for 2D
trans2D = eye(3);
trans2D(1:2,3) = -mean2D(1:2)';
scal2D = eye(3);
scal2D(1,1) = s2D; scal2D(2,2) = s2D;
T = scal2D*trans2D;
% U for 3D
trans3D = eye(4);
trans3D(1:3,4) = -mean3D(1:3)';
scal3D = eye(4);
scal3D(1,1) = s3D; scal3D(2,2) = s3D; scal3D(3,3) = s3D;
U = scal3D*trans3D;

%%
% nf2D = f2D;
% nf3D = f3D;
nf2D = T*f2D;
nf3D = U*f3D;
a1 = nf3D';
a2 = zeros(l,4);
a3x = bsxfun(@times,a1,nf2D(1,:)');
a3y = bsxfun(@times,a1,nf2D(2,:)');
A = [a1,a2,-a3x];
A = [A; a2, a1, -a3y];

% Calcultating Calibration Matrix SVD method 
[u, sigma, v] =svd(A,'econ');

% Extracting Calibration Matrix
P = reshape(v(:,12),[4,3])';
%% Extracting the intrinsic and extrinsic matrices
M = P(1:3,1:3);
M_inv = inv(M);
[R, K] = qr(M_inv);
R = inv(R)
K = inv(K)
C = null(P);
C = C./C(4);
Ext = [R -R*C(1:3)]
%%
% Verification: P1 = MP
P_hat = inv(T)*P*U
% P_hat = P;
Pver = P_hat*f3D;
Pver = [Pver(1,:)./Pver(3,:); Pver(2,:)./Pver(3,:); ones(1,l)];
diff = Pver - f2D;



% Calibration Matrix
P
% Norm Of Error
NormError=norm(diff)

