clear
clc
format short g
format compact
output_precision(3)
disable_diagonal_matrix(true)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ktemp(1,:) = [1 0 0 1 1];
Ktemp(2,:) = [1 1 0 1 1];
Ktemp(3,:) = [0 0 1 1 1];
Kstemp = Ktemp.'*Ktemp;
Ks = eye(5,5)-(0.25/2)*Kstemp;
Ks = zeros(5,5);
Ks(1,:) = [-0.05+0.06*1i, -0.-0.13*1i, -0.07-0.15*1i,  0.11+0.28*1i, -0.05-0.18*1i];
Ks(2,:) = [-0.1-0.19*1i, -0.3-0.05*1i, -0.28+0.07*1i, -0.25+0.28*1i, -0.11-0.29*1i];
Ks(3,:) = [0.21-0.18*1i, -0.08-0.14*1i,  0.03+0.2*1i , -0.23+0.24*1i, -0.06+0.32*1i];
Ks(4,:) = [-0.29-0.31*1i,  0.12+0.09*1i,  0.08-0.02*1i,  0.31+0.12*1i, -0.22-0.18*1i];
Ks(5,:) = [-0.18-0.06*1i,  0.08-0.21*1i,  0.25-0.18*1i, -0.26-0.1*1i ,  0.13+0.1*1i ];

disp("Ks:"); disp(Ks); disp("");
[V,S,U] = svd(Ks);  %Why not [U, S, V]?
% returns [V, S, U] such that A = V*S*U'
disp("V,S,U");
disp(V);
disp(S);
disp(U);

disp("svd equivalence")
disp(norm(V*S*U' - Ks))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate phase shifts of MZIs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
psi_u = U;
% psi_u = V';
Min = length(psi_u(:,1));
lp_in = Min;
Mch = length(psi_u(1,:));

Thetas = zeros(Mch, Mch);
Phis = zeros(Mch, Mch);

psi_d = psi_u(:,1);
% figure
for ich=1:Mch
    disp("ich:"); disp(ich);
    if ich~=1
       psi_d = Calculates_PsiD(psi_u(:,ich), ich, Thetas, Phis);
    end
    for i_in=1:lp_in
       disp("i_in:"); disp(i_in);
       disp("psi_d:"); disp(psi_d);
       a = psi_d(i_in);
       disp("a:"); disp(a);
       [theta, phi] = Calculates_R_theta_phi(a, i_in, ich, Thetas, Phis);
       Thetas(ich, i_in) = theta;
       Phis(ich, i_in) = phi;
    end
    disp("Thetas:"); disp(real(Thetas));
    disp("Phis:"); disp(real(Phis));
    lp_in = lp_in-1;
    
    disp("");disp("");
end
