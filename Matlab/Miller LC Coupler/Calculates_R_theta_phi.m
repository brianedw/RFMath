function [theta, phi] = Calculates_R_theta_phi(a, i_in, ich, Thetas, Phis)
disp("  Calculate_R_theta_phi");
%disp(cstrcat("    (a, i_in, ich):", num2str(a), num2str(i_in), num2str(ich)));
printf("    (a, i_in, ich): (%s, %s, %s)\n", num2str(a), num2str(i_in), num2str(ich));
if i_in==1
    printf("    i_in == 1\n");
    d = 1i*a';
    theta = asin(abs(d));
    phi = atan2(imag(d),real(d));
    printf("    (d, theta, phi): (%s, %s, %s)\n", num2str(d), num2str(theta), num2str(phi));
else
    printf("    i_in != 1\n");
    phi_temp = 0;
    t_temp = 1;
    for p=1:(i_in-1)
        phi_temp = phi_temp+Phis(ich, p);
        t_temp = t_temp*cos(Thetas(ich, p));
    end
    printf("    (phi_temp, t_temp): (%s, %s)\n", num2str(phi_temp), num2str(t_temp));
    %d = (-1i)^(-i_in)*a'*exp(-1i*phi_temp)/t_temp;  
    d = (1i)*(-1i)^(i_in - 1)*a'*exp(-1i*phi_temp)/t_temp;
    theta = asin(abs(d));
    phi = atan2(imag(d),real(d));
    printf("    (d, theta, phi): (%s, %s, %s)\n", num2str(d), num2str(theta), num2str(phi));
end