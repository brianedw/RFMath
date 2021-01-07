function [C] = Calculates_C(Psi_U, Thetas, Phis, u)
disp("Calculates_C");
Mi = length(Psi_U);
C = zeros(Mi-u,Mi-u+1);
disp("C shape");
disp(size(C));

%%% diagonal elements
md = min(Mi-u,Mi-u+1);
for imd=1:md
    theta = Thetas(u, imd);
    phi = Phis(u, imd);
    t = -1i*cos(theta)*exp(1i*phi);
    printf("    (imd, imd): (%s, %s)\n", num2str(imd), num2str(imd));
    C(imd, imd) = t;
end

%%% off diagonal elements
up_tri = 2;
t_temp = 1.0;
phi_temp = 0.0;
for ir=1:(Mi-u)
    for ic=up_tri:(Mi-u+1)
        printf("    (ir, ic): (%s, %s)\n", num2str(ir), num2str(ic));
        if ic==(ir+1)
            theta = Thetas(u, ic-1);
            phi = Phis(u, ic-1);
            r1 = 1i*sin(theta)*exp(1i*phi);
            theta = Thetas(u, ic);
            phi = Phis(u, ic);
            r2 = -1i*sin(theta)*exp(1i*phi);
            printf("      (ir, ic) (%s, %s) --> %s\n", num2str(ir), num2str(ic), num2str(r1*r2));
            C(ir,ic) = r1*r2;
            
        else
            theta = Thetas(u, ir);
            phi = Phis(u, ir);
            r_first = 1i*sin(theta)*exp(1i*phi);
            theta = Thetas(u, ic);
            phi = Phis(u, ic);
            r_last = -1i*sin(theta)*exp(1i*phi);
            
            for p=up_tri:(ic-1)
                theta = Thetas(u, p);
                phi = Phis(u, p);
                t = -1i*cos(theta)*exp(1i*phi);
                t_temp = t_temp*t;
                printf("        (u, p, t): (%s, %s, %s)\n", num2str(u), num2str(p), num2str(t));
            end
            printf("      (r_first, t_temp, r_last): (%s, %s, %s)\n", num2str(r_first), num2str(t_temp), num2str(r_last))
            printf("      (ir, ic) (%s, %s) ==> %s\n", num2str(ir), num2str(ic), num2str(r_first*t_temp*r_last));
            C(ir,ic) = r_first*t_temp*r_last;
            t_temp = 1;
        end
    end
    up_tri = up_tri+1;
end