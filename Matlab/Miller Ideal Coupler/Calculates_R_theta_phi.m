function [theta,phi] = Calculates_R_theta_phi(a,i_in,ich,s)

if i_in==1
    d = 1i*a';
    theta = asin(abs(d));
    phi = atan2(imag(d),real(d));
else
    phi_temp = 0;
    t_temp = 1;
    for p=1:(i_in-1)
        fldnm = cat(3,'B',num2str(ich),num2str(p));
        phi_temp = phi_temp+s(2).(fldnm);
        t_temp = t_temp*cos(s(1).(fldnm));
    end
    d = (-1i)^(-i_in)*a'*exp(-1i*phi_temp)/t_temp;
    theta = asin(abs(d));
    phi = atan2(imag(d),real(d));
end