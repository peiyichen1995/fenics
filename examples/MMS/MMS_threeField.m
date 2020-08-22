clear

mu1 = 4;
mu2 = 3;
mu3 = 10;
mu4 = 19;

beta3 = 4;
beta4 = 1;

syms x y z;
ux = 0.01*exp(z);
uy = 0;
uz = 0;

grad_u = [diff(ux,x), diff(ux,y), diff(ux,z);
    diff(uy,x), diff(uy,y), diff(uy,z);
    diff(uz,x), diff(uz,y), diff(uz,z)];

I2 = eye(3);

F = I2 +grad_u;

J = det(F);

C = transpose(F)*F;

C_bar = J^(-2/3)*C;

C_bar_inv = inv(C_bar);

C_bar_cof = det(C_bar)*C_bar_inv;

S_bar = 2*mu1*I2 + 3*mu2*sqrt(trace(C_bar_cof))*(C_bar_cof*trace(C_bar_inv) - C_bar_cof*C_bar_inv);

I4symm = (IK_JL(I2,I2)+IL_JK(I2,I2))/2;

I4vol = 1/3*IJ_KL_sym(inv(C),C);

P = J^(-2/3)*(I4symm-I4vol);

S_isc = simplify(IJKL_KL_sym(P,S_bar));

p = simplify(mu3*beta3*(J^(beta3-1)-J^(-beta3-1)));

S_vol = simplify(p*J*inv(C));

% PK2 = S_isc+S_vol;
% 
% PK1 = simplify(F*PK2);
% 
% bx = -simplify(diff(PK1(1,1),x)+diff(PK1(1,2),y)+diff(PK1(1,3),z));
% by = -simplify(diff(PK1(2,1),x)+diff(PK1(2,2),y)+diff(PK1(2,3),z));
% bz = -simplify(diff(PK1(3,1),x)+diff(PK1(3,2),y)+diff(PK1(3,3),z));

%% tissue

A_1 = [sqrt(2)/2, -sqrt(2)/2, 0];
A_2 = [sqrt(2)/2, sqrt(2)/2, 0];

M_1 = zeros(3,3);

for x=1:3
    for y=1:3
        M_1(x,y) = A_1(x)*A_1(y);
    end
end

M_2 = zeros(3,3);

for x=1:3
    for y=1:3
        M_2(x,y) = A_2(x)*A_2(y);
    end
end

e1pos = trace(C*M_1)-1;
e2pos = trace(C*M_2)-1;

S_ti_1 = 4*mu4*e1pos*exp(beta4*e1pos*e1pos)*M_1;
S_ti_2 = 4*mu4*e2pos*exp(beta4*e2pos*e2pos)*M_2;

PK2 = S_isc + S_vol + S_ti_1;

PK1 = simplify(F*PK2);

bx = -simplify(diff(PK1(1,1),x)+diff(PK1(1,2),y)+diff(PK1(1,3),z));
by = -simplify(diff(PK1(2,1),x)+diff(PK1(2,2),y)+diff(PK1(2,3),z));
bz = -simplify(diff(PK1(3,1),x)+diff(PK1(3,2),y)+diff(PK1(3,3),z));
