function [Psi_D] = Calculates_PsiD(Psi_U, ich, Thetas, Phis)
disp("Calculates_PsiD");
C_tot = 1;
for u=1:(ich-1)  % when ich = 2 -> [1]
    disp("u:");
    disp(u);
    C = Calculates_C(Psi_U, Thetas, Phis, u);
    disp("C:");
    disp(C);
    %     C_tot = C_tot*C;
    if u==1
        Psi_D = C*Psi_U;
    else
        Psi_D = C*Psi_D;
    end
end

% Psi_D = C_tot*Psi_U;