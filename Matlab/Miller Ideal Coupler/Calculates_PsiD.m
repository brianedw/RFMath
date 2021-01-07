function [Psi_D] = Calculates_PsiD(Psi_U,ich,s)

C_tot = 1;
for u=1:(ich-1)
    C = Calculates_C(Psi_U,s,u);
    %     C_tot = C_tot*C;
    if u==1
        Psi_D = C*Psi_U;
    else
        Psi_D = C*Psi_D;
    end
end

% Psi_D = C_tot*Psi_U;