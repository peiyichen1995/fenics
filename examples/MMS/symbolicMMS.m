% clear
% 
mu1 = 4;
mu2 = 3;
mu3 = 10;

beta3 = 4;

% 
% syms x y z;
% ux = 0.1*exp(y+z);
% uy = 0;
% uz = 0;
% 
% grad_u = [diff(ux,x), diff(ux,y), diff(ux,z);
%     diff(uy,x), diff(uy,y), diff(uy,z);
%     diff(uz,x), diff(uz,y), diff(uz,z)];
% 
% I2 = eye(3);


syms F11 F12 F13 F21 F22 F23 F31 F32 F33 p d;
F = [F11, F12, F13;
    F21, F22, F23;
    F31, F32, F33];

J = det(F);

C = transpose(F)*F;

C_bar = J^(-2/3)*C;

C_bar_inv = inv(C_bar);

C_bar_cof = det(C_bar)*C_bar_inv;

W_bar = mu1*trace(C_bar) + mu2*(trace(C_bar_cof))^(3/2);

W_J = mu3*(J^beta3 + J^(-beta3));

W = W_bar + W_J +p*(J-d);

PK1 = [diff(W,F(1,1)), diff(W,F(1,2)), diff(W,F(1,3));
    diff(W,F(2,1)), diff(W,F(2,2)), diff(W,F(2,3));
    diff(W,F(3,1)), diff(W,F(3,2)), diff(W,F(3,3));];

syms x y z;

PK1 = subs(PK1,F11,1);
PK1 = subs(PK1,F12,exp(y + z)/10);
PK1 = subs(PK1,F13,exp(y + z)/10);
PK1 = subs(PK1,F21,0);
PK1 = subs(PK1,F22,1);
PK1 = subs(PK1,F23,0);
PK1 = subs(PK1,F31,0);
PK1 = subs(PK1,F32,0);
PK1 = subs(PK1,F33,1);
PK1 = simplify(PK1);

bx = -simplify(diff(PK1(1,1),x)+diff(PK1(1,2),y)+diff(PK1(1,3),z));
by = -simplify(diff(PK1(2,1),x)+diff(PK1(2,2),y)+diff(PK1(2,3),z));
bz = -simplify(diff(PK1(3,1),x)+diff(PK1(3,2),y)+diff(PK1(3,3),z));