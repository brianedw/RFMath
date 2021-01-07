clear
format long

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



[V,S,U] = svd(Ks);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate phase shifts of MZIs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% psi_u = U;
psi_u = V';
Min = length(psi_u(:,1));
lp_in = Min;
Mch = length(psi_u(1,:));


field = 'block'; value = {1;2};
s = struct(field,value);
psi_d = psi_u(:,1);
% figure
for ich=1:Mch
    if ich~=1
       psi_d = Calculates_PsiD(psi_u(:,ich),ich,s);
    end
    for i_in=1:lp_in
        a = psi_d(i_in);
        [theta,phi] = Calculates_R_theta_phi(a,i_in,ich,s);
        fldnm = cat(3,'B',num2str(ich),num2str(i_in));
        tmp=cell(size(s)); [s(:).(fldnm)]=deal(tmp{:});
        s(1).(fldnm) = theta;
        s(2).(fldnm) = phi;
    end
    lp_in = lp_in-1;
end


%%% Make fig %%%
field = 'block'; value = {1;2};
sch = struct(field,value);
lx = 0.1;
ly = 0.3;
xr = [-lx lx lx -lx -lx];
yr = [-ly -ly ly ly -ly];
xp = 1;
for i_in=1:Min
    if i_in==1
        x = linspace(0,1,Min);
        y = x;
        fldnm = cat(2,'ch',num2str(i_in));
        tmp=cell(size(sch)); [sch(:).(fldnm)]=deal(tmp{:});
        sch(1).(fldnm) = x;
        sch(2).(fldnm) = y;
    elseif i_in==Min
        x = 3;
        y = 0;
        fldnm = cat(2,'ch',num2str(i_in));
        tmp=cell(size(sch)); [sch(:).(fldnm)]=deal(tmp{:});
        sch(1).(fldnm) = x;
        sch(2).(fldnm) = y;
    else
        x = linspace(xp,xp+0.7,Min-1);
        y = x-xp;
        fldnm = cat(2,'ch',num2str(i_in));
        tmp=cell(size(sch)); [sch(:).(fldnm)]=deal(tmp{:});
        sch(1).(fldnm) = x;
        sch(2).(fldnm) = y;
        xp = xp + 1;
    end
end

for ic=1:Mch
    for i_in=1:Min
        fldnm = cat(2,'ch',num2str(ic));
        xc = sch(1).(fldnm)(Min+1-i_in);
        yc = sch(2).(fldnm)(Min+1-i_in);
        plot(xr+xc,yr+yc)
        hold on
        fldnm = cat(3,'B',num2str(ic),num2str(i_in));
        txt1 = {(fldnm)};
        txt2 = {num2str(real(s(1).(fldnm))*180/pi)};
        txt3 = {num2str(real(s(2).(fldnm))*180/pi)};
        text(xc,yc+0.15,txt1,'Color','red')
        text(xc,yc,txt2,'Color','blue')
        text(xc,yc-0.15,txt3,'Color','green')
    end
    Min = Min - 1;
end