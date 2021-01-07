#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from math import *


# In[ ]:


from Logger import Logger


# # Miller

# 
# The MZI consists of two 3dB couplers with two phase shifters.  There are several possible designs and conventions.  
# 
# Here we use the following:
#         
# The 3dB couplers have the SParams
# 
#     [[0,0,a,b],
#      [0,0,b,a],
#      [a,b,0,0],
#      [b,a,0,0]]
# and the number scheme:
# 
#     1 |---| 3
#         X    
#     2 |---| 4
# 
# In option `couplerConv = 'LC'`, these are `a = -1j/sqrt(2)`, `b = -1/sqrt(2)`.  In option  `couplerConv = 'ideal'` these are `a = -1j/sqrt(2)`, `b = 1/sqrt(2)`.
# 
# The two phase shifters have the transmissions:
# 
#     T1 = exp(i*(phi + theta))
#     T2 = exp(i*(phi - theta))
# Together, they yield a device with the following number scheme.
# 
#         1 |---| - [T1] - |---| 3
#             X              X    
#         2 |---| - [T2] - |---| 4
# 
# In the case of `couplerConv = 'LC'`, the SParams for the device are:
# 
#     S31 = -1j * np.sin(theta) * np.exp(1j*phi)
#     S41 =  1j * np.cos(theta) * np.exp(1j*phi)
#     S32 =  1j * np.cos(theta) * np.exp(1j*phi)
#     S42 =  1j * np.sin(theta) * np.exp(1j*phi)
# While in the case of `couplerConv = 'ideal'`, the SParams for the device are:
# 
#     S31 = -1j * np.sin(theta) * np.exp(1j*phi)
#     S41 = -1j * np.cos(theta) * np.exp(1j*phi)
#     S32 = -1j * np.cos(theta) * np.exp(1j*phi)
#     S42 =  1j * np.sin(theta) * np.exp(1j*phi)

# In[ ]:


np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
np.set_printoptions(precision=3)


# In[ ]:


class MillerBuilder:
    pass


# In[ ]:


def __init__(self, couplerConv="LC", verbose=False):
    self.log = Logger(indentStep=4, printQ=verbose)
    self.couplerConv = couplerConv
setattr(MillerBuilder, "__init__", __init__)


# In[ ]:


def Calculate_R_theta_phi(self, a, i_in, ich, Phis, Thetas, verbose=False):
    self.log.open("Calculate_R_theta_phi")
    v = verbose
    self.log.print("(a, i_in, ich):", a, i_in, ich)
    self.log.openContext("If-Else")
    if i_in == 0:
        self.log.printVarX("i_in", locals())
        d = 1j*a.conj().T
        theta = np.arcsin(np.clip(abs(d), 0, 1));
        phi = np.arctan2(np.imag(d), np.real(d))
        self.log.print("(d, theta, phi):", d, theta, phi)
    else:
        self.log.printVarX("i_in", locals())
        phi_temp = 0.0
        t_temp = 1.0
        for p in range(0, i_in):
            phi_temp = phi_temp + Phis[ich, p] 
            t_temp = t_temp*cos(Thetas[ich, p])
        self.log.print("(phi_temp, t_temp):", phi_temp, t_temp)
        if self.couplerConv == 'LC':
            d = (1j)*(-1j)**(i_in) * a.conj().T * np.exp(-1j*phi_temp) / t_temp
        elif self.couplerConv == 'ideal':
            d = (-1j)**(-(i_in - 1)) * a.conj().T * np.exp(-1j*phi_temp) / t_temp
        else:
            raise TypeError("couplerConv should be either 'ideal' or 'LC'")
        theta = np.arcsin(np.clip(abs(d), 0, 1))
        phi = np.arctan2(np.imag(d), np.real(d))
        self.log.print("(d, theta, phi):", d, theta, phi)
    self.log.closeContext()
    self.log.close()
    return (theta, phi)
setattr(MillerBuilder, "Calculate_R_theta_phi", Calculate_R_theta_phi)


# In[ ]:


def Calculate_PsiD(self, Psi_U, ich, Phis, Thetas, verbose=False):
    self.log.open("Calculate_PsiD")
    self.log.printVarX("ich", locals())
    C_tot = 1
    self.log.openContext("u loop")
    for u in range(0, ich-1 + 1):
        self.log.printVarX("u", locals())
        C = self.Calculate_C(Psi_U, Phis, Thetas, u, True);
        self.log.printNPArray("C", C)
        self.log.openContext("u If-Else")
        if u == 0:
            self.log.print("u == 0")
            Psi_D = np.matmul(C, Psi_U)
        else:
            self.log.print("u != 0")
            Psi_D = np.matmul(C, Psi_D)
        self.log.closeContext()
    self.log.closeContext()
    self.log.close()
    return Psi_D
setattr(MillerBuilder, "Calculate_PsiD", Calculate_PsiD)


# In[ ]:


def Calculate_C(self, Psi_U, Phis, Thetas, u, verbose = False):
    """
    initial C should be 4x5.
    """
    self.log.open("Calculate_C")
    Mi = len(Psi_U)
    C = np.zeros(shape=(Mi-u-1, Mi-u+1-1), dtype=np.complex)
    self.log.print("C.shape:", C.shape)
    md = min(Mi-u, Mi-u+1) - 1;
    self.log.printVarX("md", locals())
    self.log.openContext("diagonal loop")
    for imd in range(0, md):
        theta = Thetas[u, imd]
        phi = Phis[u, imd]
        if self.couplerConv == 'LC':
            t = 1j*np.cos(theta)*np.exp(1j*phi)
        elif self.couplerConv == 'ideal':
            t = -1j*np.cos(theta)*np.exp(1j*phi)
        else:
            raise TypeError("couplerConv should be either 'ideal' or 'LC'")
        self.log.print("(imd, imd)", (imd, imd))
        C[imd,imd] = t
    self.log.closeContext()

    up_tri = 2
    t_temp = 1.0
    phi_temp = 0.0
    self.log.openContext("ir loop")
    for ir in range(0, (Mi-u)):
        self.log.openContext("ic loop")
        for ic in range(up_tri-1, (Mi-u+1)-1):
            self.log.openContext("(ir, ic) = "+str((ir, ic)))
            if ic==(ir+1):
                theta = Thetas[u, ic-1]
                phi = Phis[u, ic-1]
                r1 = 1j*np.sin(theta)*np.exp(1j*phi)
                theta = Thetas[u, ic]
                phi = Phis[u, ic]
                r2 = -1j*np.sin(theta)*np.exp(1j*phi)
                self.log.print("(ir, ic)", (ir, ic), " --> ", np.round(r1*r2,3))
                C[ir,ic] = r1*r2;
            else:
                theta = Thetas[u, ir]
                phi = Phis[u, ir]
                r_first = 1j*np.sin(theta)*np.exp(1j*phi)
                theta = Thetas[u, ic]
                phi = Phis[u, ic]
                r_last = -1j*np.sin(theta)*np.exp(1j*phi)
                self.log.openContext("p loop ["+str(up_tri-1)+","+str(ic-1+1)+")")
                for p in range(up_tri-1, (ic-1+1)):
                    theta = Thetas[u, p]
                    phi = Phis[u, p]
                    if self.couplerConv == 'LC':
                        t = 1j*np.cos(theta)*np.exp(1j*phi)
                    elif self.couplerConv == 'ideal':
                        t = -1j*np.cos(theta)*np.exp(1j*phi)
                    else:
                        raise TypeError("couplerConv should be either 'ideal' or 'LC'")
                    t_temp = t_temp*t
                    self.log.print("(u, p, t)", (u, p, t))
                self.log.closeContext()
                self.log.print("(ir, ic)", (ir, ic), " ==> ", np.round(r_first*t_temp*r_last,3))
                self.log.print("(r_first, t_temp, r_last)", (np.round(r_first,3), np.round(t_temp,3), np.round(r_last,3)))
                C[ir,ic] = r_first*t_temp*r_last
                t_temp = 1
            self.log.closeContext()
        up_tri = up_tri+1;
        self.log.closeContext()
    self.log.closeContext()
    self.log.close()
    return C

setattr(MillerBuilder, "Calculate_C", Calculate_C)


# In[ ]:


def ConvertUnitaryToMZITriangle(self, psi_u):
    self.log.open("ConvertUnitaryToMZITriangle", True)
    Min = len(psi_u)
    lp_in = Min
    Mch = len(psi_u[0])
    psi_d = psi_u[:,0]
    Phis = np.zeros(shape=(Min, Mch))
    Thetas = np.zeros(shape=(Min, Mch))
    self.log.openContext("ich loop")
    for ich in range(0, Mch):
        self.log.printVarX("ich", locals())
        if ich != 0:
            psi_d = self.Calculate_PsiD(psi_u[:,ich], ich, Phis, Thetas, verbose=True)

        self.log.openContext("i_in loop")
        for i_in in range(0, lp_in):
            self.log.printVarX("i_in", locals())
            self.log.printNPArray("psi_d", psi_d)
            a = psi_d[i_in]
            self.log.printVarX("a", locals())
            [theta,phi] = self.Calculate_R_theta_phi(a, i_in, ich, Phis, Thetas, verbose=True)
            Thetas[ich, i_in] = theta;
            Phis[ich, i_in] = phi;
        self.log.closeContext()

        self.log.printNPArray("Thetas", Thetas)
        self.log.printNPArray("Phis", Phis)
        lp_in = lp_in-1;
    self.log.closeContext()
    self.log.close()
    return (Thetas, Phis)

setattr(MillerBuilder, "ConvertUnitaryToMZITriangle", ConvertUnitaryToMZITriangle)


# In[ ]:


def ConvertKToMZI(self, Ks):
    self.log.open("MAIN", True)
    self.log.printNPArray("Ks", Ks)
    V, S, Uh = np.linalg.svd(Ks)
    self.log.printNPArray("V", V)
    self.log.printNPArray("S", S)
    self.log.printNPArray("U", Uh.conj().T)
    self.log.print("K == V*S*U' is", np.allclose(Ks, (V*S)@(Uh)))
    self.log.print("")
    psi_u = Uh.conj().T
    (Thetas1, Phis1) = self.ConvertUnitaryToMZITriangle(psi_u)
    psi_u = V.conj().T
    (Thetas2, Phis2) = self.ConvertUnitaryToMZITriangle(psi_u)
    self.log.close()
    leftTriangle = np.dstack((Thetas1, Phis1))
    rightTriangle = np.dstack((Thetas2, Phis2))
    return (leftTriangle, S, rightTriangle)
setattr(MillerBuilder, "ConvertKToMZI", ConvertKToMZI)


# ## Test

# In[ ]:


Ks = np.array([[-0.05+0.06j, -0.  -0.13j, -0.07-0.15j,  0.11+0.28j, -0.05-0.18j],
               [-0.1 -0.19j, -0.3 -0.05j, -0.28+0.07j, -0.25+0.28j, -0.11-0.29j],
               [ 0.21-0.18j, -0.08-0.14j,  0.03+0.2j , -0.23+0.24j, -0.06+0.32j],
               [-0.29-0.31j,  0.12+0.09j,  0.08-0.02j,  0.31+0.12j, -0.22-0.18j],
               [-0.18-0.06j,  0.08-0.21j,  0.25-0.18j, -0.26-0.1j ,  0.13+0.1j ]])


# For entry into Matlab (Octave):
# ```
# Ks = zeros(5,5);
# Ks(1,:) = [-0.05+0.06*1i, -0.-0.13*1i, -0.07-0.15*1i,  0.11+0.28*1i, -0.05-0.18*1i];
# Ks(2,:) = [-0.1-0.19*1i, -0.3-0.05*1i, -0.28+0.07*1i, -0.25+0.28*1i, -0.11-0.29*1i];
# Ks(3,:) = [0.21-0.18*1i, -0.08-0.14*1i,  0.03+0.2*1i , -0.23+0.24*1i, -0.06+0.32*1i];
# Ks(4,:) = [-0.29-0.31*1i,  0.12+0.09*1i,  0.08-0.02*1i,  0.31+0.12*1i, -0.22-0.18*1i];
# Ks(5,:) = [-0.18-0.06*1i,  0.08-0.21*1i,  0.25-0.18*1i, -0.26-0.1*1i ,  0.13+0.1*1i ];
# ```
# 
# 

# In[ ]:


miller = MillerBuilder(couplerConv='LC', verbose=False)
t1, s, t2 = miller.ConvertKToMZI(Ks)
theta1, phi1 = np.rollaxis(t1, 2)
theta2, phi2 = np.rollaxis(t2, 2)
# print("theta1:")
# print(theta1*180/np.pi)
# print("phi1:")
# print(phi1*180/np.pi)
# print("theta2:")
# print(theta2*180/np.pi)
# print("phi2:")
# print(phi2*180/np.pi)

