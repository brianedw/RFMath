function [C] = Calculates_C(Psi_U,s,u)
Mi = length(Psi_U);
C = zeros(Mi-u,Mi-u+1);

%%% diagonal elements
md = min(Mi-u,Mi-u+1);
for imd=1:md
    fldnm = cat(3,'B',num2str(u),num2str(imd));
    theta = s(1).(fldnm);
    phi = s(2).(fldnm);
    t = -1i*cos(theta)*exp(1i*phi);
    C(imd,imd) = t;
end

%%% off diagonal elements
up_tri = 2;
t_temp = 1;
phi_temp = 0;
for ir=1:(Mi-u)
    for ic=up_tri:(Mi-u+1)
        if ic==(ir+1)
            fldnm = cat(3,'B',num2str(u),num2str(ic-1));
            theta = s(1).(fldnm);
            phi = s(2).(fldnm);
            r1 = 1i*sin(theta)*exp(1i*phi);
            fldnm = cat(3,'B',num2str(u),num2str(ic));
            theta = s(1).(fldnm);
            phi = s(2).(fldnm);
            r2 = -1i*sin(theta)*exp(1i*phi);
            C(ir,ic) = r1*r2;
            
        else
            fldnm = cat(3,'B',num2str(u),num2str(ir));
            theta = s(1).(fldnm);
            phi = s(2).(fldnm);
            r_first = 1i*sin(theta)*exp(1i*phi);
            
            fldnm = cat(3,'B',num2str(u),num2str(ic));
            theta = s(1).(fldnm);
            phi = s(2).(fldnm);
            r_last = -1i*sin(theta)*exp(1i*phi);
            
            for p=up_tri:(ic-1)
                fldnm = cat(3,'B',num2str(u),num2str(p));
                theta = s(1).(fldnm);
                phi = s(2).(fldnm);
                t = -1i*cos(theta)*exp(1i*phi);
                t_temp = t_temp*t;
            end
            C(ir,ic) = r_first*t_temp*r_last;
            t_temp = 1;
        end
    end
    up_tri = up_tri+1;
end