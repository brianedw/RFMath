#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from math import *


# In[ ]:


from IPython.core.debugger import set_trace


# # Matrix Errors

# In[ ]:


def MatrixSqError(m, mTarget):
    """
    Computes ||m - mTarget||^2 where ||m|| is the Frobenius norm of M.
    """
    errorSq = (np.abs(m - mTarget)**2).sum()
    return errorSq


# In[ ]:


def MatrixError(m, mTarget):
    """
    Computes ||m - mTarget|| where ||m|| is the Frobenius norm of M.
    """
    errorSq = MatrixSqError(m, mTarget)
    error = errorSq**0.5
    return error


# In[ ]:


def MatrixErrorNormalized(m, mTarget):
    """
    Computes ||m - mTarget||/a where ||m|| is the Frobenius norm of M
    and 'a' is maximum transmissive entropy normalization factor of 
    (1/n)^0.5.

    The purpose of this function is to give a sense of the error within
    a matrix compared to the average values for a matrix of that order.
    """
    errorSq = MatrixSqError(m, mTarget)
    error = errorSq**0.5
    (n,n) = m.shape
    a = (1/n)**0.5
    return error/a


# In[ ]:


a = np.array([[0, 1], [0,   1]])
b = np.array([[0, 1], [0, 0.5]])


# In[ ]:


MatrixSqError(a,b)


# In[ ]:


MatrixError(a,b)


# In[ ]:


MatrixErrorNormalized(a,b)


# # Complex Mapping Inversion and Interpolation

# The challenge here is to create a function that will allow for the numerical inversion of an experimentally determined complex function.

# To understand this problem, let us first consider an experimentally determined function based on only real numbers.  We begin with a set of measurements
# $[x_1 \rightarrow y_1, x_2 \rightarrow y_2, ..., x_n \rightarrow y_n]$.
# Let us assume that both $x$ and $y$ are monotonic (ie constantly increasing).  We can "invert" the data rather simply to yield: $[y_1 \rightarrow x_1, y_2 \rightarrow x_2, ..., y_n \rightarrow x_n]$.  We can then determine an interpolating function $X(y)$ that will yield $x$ given a $y$.  Because $y$ is increasing, it is easy to determine the bounding pair $(y_i, y_j)$ and then perform simple linear interpolation.

# Things become much more complicated when we consider complex values.  Here we have a set of measurements in which we alter two control variables, $a$ and $b$ to yield a data set:
# $[(a_1,b_1) \rightarrow (\mathbb{r}_1, \mathbb{i}_1), (a_2,b_2) \rightarrow (\mathbb{r}_2, \mathbb{i}_2), ...,(a_n,b_n) \rightarrow (\mathbb{r}_n, \mathbb{i}_n)]$.  We needn't necessarily chose $r$ and $i$ to represent the complex output.  We could use $a$ and $p$ if they are more convenient.
# 
# Inverting the data is easy, but determining an interpolationg function won't be.  In our case the results may not be monotonic for either $\mathbb{r}$ or $\mathbb{i}$.  Indeed, we know the phase will wrap beyond 360 degrees.  Additionally, when mapping 2D space on to 2D space, we will have the output points will be an irregular grid.

# # Logging

# In[ ]:


class MyLogger:

    def __init__(self, indentStep, printQ):
        self.indentStep = indentStep
        self.indentLevel = 0
        self.indentTxt = " ."+" "*(indentStep-2)
        self.printQ = printQ

    def open(self, name, reset=False):
        """
        Increases indent level at function level.
        Applied at the beginning of a function definition
        """
        if reset:
            self.indentLevel = 0
        if self.printQ:
            print(self.indentTxt*self.indentLevel, "==", name, "==")
        self.indentLevel += 1

    def openContext(self, message):
        """
        Increases indent level with message.
        Applied just prior to a local context such as "if" or "for"
        """
        if self.printQ:
            print(self.indentTxt*self.indentLevel, message)
        self.indentLevel += 1

    def print(self, *msg):
        """
        Generic message.  Unpacks the list similar to native 'print("Hello", "World")'
        """
        if self.printQ:
            print(self.indentTxt*self.indentLevel, *msg)

    def printVar(self, name, var):
        """
        Prints "name: var"
        """
        if self.printQ:
            print(self.indentTxt*self.indentLevel+" " + name + ":", var)

    def printVarX(self, name, scope):
        """
        Prints "name: eval(name)"
        """
        if self.printQ:
            print(self.indentTxt*self.indentLevel+" "+ name + ":", eval(name, scope))

    def printNPArray(self, name, array, *options):
        """
        Prints
        name:
          [[1,2]
           [3,4]]
        """
        padding = self.indentTxt*self.indentLevel+"   "
        txt = padding + np.array2string(array, *options)
        txt2 = txt.replace("\n", "\n"+padding)
        if self.printQ:
            print(self.indentTxt*self.indentLevel+" "+ name + ":")
            print(txt2)

    def close(self):
        """
        Closes function level indent.
        """
        self.indentLevel -= 1
        if self.printQ:
            print(self.indentTxt*self.indentLevel, "====")
    
    def closeContext(self):
        """
        Closes context level indent.
        """
        self.indentLevel -= 1
        if self.printQ:
            print(self.indentTxt*self.indentLevel, "----")


# In[ ]:


log = MyLogger(4, True)


# In[ ]:


def foo():
    log.open("foo", reset=True)
    log.print("I'm in foo.")
    a = 2
    b = 4
    log.printVar("a", a)
    log.printVarX("a", locals())
    log.print("The variable 'a' is", a, "and 'b' is", b)
    bar()
    log.openContext("in loop")
    for i in range(2):
        j = i**2
        log.printVar("j", j)
    log.closeContext()
    log.close()

def bar():
    log.open("bar")
    log.print("in bar")
    a = np.array([[1,2],[3,4]])
    log.printNPArray("a", a)
    log.close()


# In[ ]:


foo()


# In[ ]:


log.printQ = False
foo()


# # Random Complex Matrices

# In[ ]:


randComplexMat = np.random.uniform(-1, 1, size=(5,5)) + 1j*np.random.uniform(-1, 1, size=(5,5))
u, s, v = np.linalg.svd(randComplexMat)
maxS = max(s)
randComplexMat = randComplexMat/(1.1 * maxS)
randComplexMat = randComplexMat.round(2)
randComplexMat


# # Miller Technique

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
    self.log = MyLogger(indentStep=4, printQ=verbose)
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
        log.printVarX("i_in", locals())
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
            log.openContext("(ir, ic) = "+str((ir, ic)))
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
        log.printVarX("ich", locals())
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
print("theta1:")
print(theta1*180/np.pi)
print("phi1:")
print(phi1*180/np.pi)
print("theta2:")
print(theta2*180/np.pi)
print("phi2:")
print(phi2*180/np.pi)


# # Scikit-RF

# I'm investigating whether we can do RF Circuit simulation here in Python.  There is a ready-made package called Scikit-RF (https://scikit-rf.readthedocs.io/) which has many ready made functions.  I have figured out how to build SParameter elements and connect them together using both the rf.connect and using the Circuit object.  However, in both, I am having trouble termining the order of the port names in the resulting SMatrix.

# ## Device Definitions

# In[ ]:


get_ipython().system('pip install scikit-rf')


# In[ ]:


import skrf as rf
import numpy as np
from math import *


# In[ ]:


deg = 2*np.pi/360


# In[ ]:


def BuildMZISParams(thetaP, phi, freq, couplerConv="LC", reciprocal=False):
    """
        This generates the scalar SParameters (not dispersive) for an MZI.  By
        default it is "forward only," however, if reciprocal=True, then it will work
        in reverse as well.

        The MZI consists of two 3dB couplers with two phase shifters.  There are
        many possible designs and conventions.  Here we use the following:
        
        The 3dB couplers have the SParams
        [[0,0,a,b],
         [0,0,b,a],
         [a,b,0,0],
         [b,a,0,0]]
        and the number scheme:
        1 |---| 3
            X    
        2 |---| 4
        In option `couplerConv = 'LC'`, these are `a = -1j/sqrt(2)`, `b = -1/sqrt(2)`.  
        In option  `couplerConv = 'ideal'` these are `a = -1j/sqrt(2)`, `b = 1/sqrt(2)`.

        The two phase shifters have the transmissions
        T1 = exp(i*(phi + theta))
        T2 = exp(i*(phi - theta))

        Together, they yield a device with the following number scheme.

        1 |---| - [T1] - |---| 3
            X              X    
        2 |---| - [T2] - |---| 4
    """
    theta = thetaP
    nFreqs = len(freq)
    if couplerConv == 'LC':
        S31 = -1j * np.sin(theta) * np.exp(1j*phi)
        S41 =  1j * np.cos(theta) * np.exp(1j*phi)
        S32 =  1j * np.cos(theta) * np.exp(1j*phi)
        S42 =  1j * np.sin(theta) * np.exp(1j*phi)
    elif couplerConv == 'ideal':
        S31 = -1j * np.sin(theta) * np.exp(1j*phi)
        S41 = -1j * np.cos(theta) * np.exp(1j*phi)
        S32 = -1j * np.cos(theta) * np.exp(1j*phi)
        S42 =  1j * np.sin(theta) * np.exp(1j*phi)
    else:
        raise TypeError("couplerConv should be either 'ideal' or 'LC'")

    S = np.zeros((4,4, nFreqs), dtype=np.complex)
    S[2,0] = S31
    S[3,0] = S41
    S[2,1] = S32
    S[3,1] = S42
    if reciprocal:
        S[0,2] = S31
        S[1,2] = S41
        S[0,3] = S32
        S[1,3] = S42
    return np.rollaxis(S, 2)


# In[ ]:


def BuildAttenSParams(T, freq, reciprocal=False):
    """
    Simple Attenuator model.  Transmitted amplitude is scaled by 'T'.
    """
    nFreqs = len(freq)
    S21 = T
    S = np.zeros((4,4, nFreqs), dtype=np.complex)
    S[1,0] = S21
    if reciprocal:
        S[0,1] = S21
    return np.rollaxis(S, 2)


# ## Circuit Builders

# In[ ]:


def BuildTriangleCircuit(theta_phi, freq, MZIBuilder, label=""):
    """
    Builds a triangular network of MZIs as a SciKit-RF Circuit.

    Inputs:
        theta_phi:  Of the form such that (theta, phi) = theta_phi[i_ch, i_in] is the theta, phi value 
                    for the MZI at (i_ch, i_in).
        freq:       SciKit-RF frequency object.
        MZIBuilder: A function such that S = MZIBuilder(theta, phi, freqValues, loc=(i_ch, i_in)).
                    See notes below.
        label:      Every element in a SciKit-RF circuit must have a unique name.  If the circuit
                    consists of multiple triangles, this can be used to separate them.

    It is assumed an MZI's S-Params can be obtained using
    S = MZIBuilder(theta, phi, freqValues, loc=(i_ch, i_in))
    where theta,phi are in radians, freqValues are [f1, f2, ...] in Hz and the
    returned S is of the shape (nFreqs, NPorts, NPorts).
    """
    Z_0 = 50.
    if label != "":
        labelX = label+"_"
    else:
        labelX = ""

    "Extract the size of the network from the input.  Assumed to be (NN, NN, 2)"
    NN = theta_phi.shape[0]
    "Input Ports"
    inPorts = [rf.Circuit.Port(freq, 'portIn_'+labelX+str(i), z0=Z_0) for i in range(NN)]
    "Output Ports"
    outPorts = [rf.Circuit.Port(freq, 'portOut_'+labelX+str(i), z0=Z_0) for i in range(NN)]
    "Absorbers on Input Side of Lower Row"
    trashPortsA = [rf.Circuit.Port(freq, 'trashOutA_'+labelX+str(i), z0=Z_0) for i in range(NN)]
    "Absorbers on Output Side of Lower Row"
    trashPortsB = [rf.Circuit.Port(freq, 'trashOutB_'+labelX+str(i), z0=Z_0) for i in range(NN)]
    "All of the MZIs"
    MZITri1 = np.empty(shape=(NN, NN), dtype=object)
    for i_ch in range(NN):
        for i_in in range(NN - i_ch):
            (theta, phi) = theta_phi[i_ch, i_in]
            s = MZIBuilder(theta, phi, freq.f, loc=(i_ch, i_in))
            mzi = rf.Network(name="MZI_"+labelX+str(i_ch)+str(i_in), frequency=freq, z0=Z_0, s=s)
            MZITri1[i_ch, i_in] = mzi

    "Simple Connections"
    portInConnections = [ [(inPorts[i_in], 0), (MZITri1[0, i_in], 0)] for i_in in range(NN)]
    portOutConnections = [ [(MZITri1[i_ch, 0], 2), (outPorts[i_ch], 0)] for i_ch in range(NN)]
    trashConnectionsA  = [ [(MZITri1[i, 4-i], 1), (trashPortsA[i], 0)] for i in range(NN)]
    trashConnectionsB  = [ [(MZITri1[i, 4-i], 3), (trashPortsB[i], 0)] for i in range(NN)]

    "Intra MZI Connections"
    mziConnectionsU = []
    for i_ch in range(0, NN-1):
        for i_in in range(1, NN-i_ch):
            c = [(MZITri1[i_ch, i_in], 2), (MZITri1[i_ch, i_in-1], 1)]
            mziConnectionsU.append(c)
    mziConnectionsL = []
    for i_ch in range(0, NN-1):
        for i_in in range(0, NN-i_ch-1):
            c = [(MZITri1[i_ch, i_in], 3), (MZITri1[i_ch+1, i_in], 0)]
            mziConnectionsU.append(c)
    cnx = [*portInConnections, *portOutConnections, *trashConnectionsA, *trashConnectionsB, *mziConnectionsU, *mziConnectionsL]

    "Build the Circuit"
    cir = rf.Circuit(cnx)
    return cir


# In[ ]:


def BuildVectorAttenuatorCircuit(T, freq, AttBuilder, label=""):
    """
    Builds a linear network of multipliers as a SciKit-RF Circuit.

    Inputs:
        T:          A vector of transmission coefficients.
        freq:       SciKit-RF frequency object.
        MZIBuilder: A function such that S = MZIBuilder(theta, phi, freqValues, loc=(i_ch, i_in)).
                    See notes below.
        label:      Every element in a SciKit-RF circuit must have a unique name.  If the circuit
                    consists of multiple triangles, this can be used to separate them.

    It is assumed an MZI's S-Params can be obtained using
    S = AttBuilder(t, freqValues, loc=i)
    where 't' is in linear scale, freqValues are [f1, f2, ...] in Hz and the
    returned S is of the shape (nFreqs, NPorts, NPorts).
    """
    Z_0 = 50.
    if label != "":
        labelX = label+"_"
    else:
        labelX = ""

    "Extract the size of the network from the input.  Assumed to be (NN, NN, 2)"
    NN = len(T)
    "Input Ports"
    inPorts = [rf.Circuit.Port(freq, 'portIn_'+labelX+str(i), z0=Z_0) for i in range(NN)]
    "Output Ports"
    outPorts = [rf.Circuit.Port(freq, 'portOut_'+labelX+str(i), z0=Z_0) for i in range(NN)]
    "All of the Attenuators"
    attVec = np.empty(shape=(NN,), dtype=object)
    for i in range(NN):
        t = T[i]
        s = AttBuilder(t, freq.f, loc=(i))
        att = rf.Network(name="Att_"+labelX+str(i), frequency=freq, z0=Z_0, s=s)
        attVec[i] = att

    "Simple Connections"
    portInConnections = [ [(inPorts[i], 0), (attVec[i], 0)] for i in range(NN)]
    portOutConnections = [ [(attVec[i], 1), (outPorts[i], 0)] for i in range(NN)]

    "Intra MZI Connections"
    cnx = [*portInConnections, *portOutConnections]

    "Build the Circuit"
    cir = rf.Circuit(cnx)
    return cir


# In[ ]:


def CascadeCircuits(circuits):
    """
    Given several nxn circuits where the first n/2 ports are "input" and the
    last n/2 are "output", it will stack these circuits in the order presented.
    """
    NN = len(cirAtt.port_indexes)//2
    Z_0 = 50.

    networks = [c.network for c in circuits]
    for i in range(len(networks)):
        networks[i].name = str(i)

    "Input Ports"
    inPorts = [rf.Circuit.Port(freq, 'portIn_'+str(i), z0=Z_0) for i in range(NN)]
    "Output Ports"
    outPorts = [rf.Circuit.Port(freq, 'portOut_'+str(i), z0=Z_0) for i in range(NN)]

    "Simple Connections"
    portInConnections = [ [(inPorts[i], 0), (networks[0], i)] for i in range(NN)]
    portOutConnections = [ [(networks[-1], NN+i), (outPorts[i], 0)] for i in range(NN)]

    intraConnections = []
    for j in range(len(networks)-1):
        conns = [ [(networks[j], NN+i), (networks[j+1], i)] for i in range(NN)]
        intraConnections.extend(conns)

    "Intra MZI Connections"
    cnx = [*portInConnections, *intraConnections, *portOutConnections]

    "Build the Circuit"
    cir = rf.Circuit(cnx)
    return cir


# ## Miller Trial

# ![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABcEAAAGHCAYAAAB4Y5obAAAgAElEQVR4AezdW28bV7bo+/3Vzufwk5781AE2lt/iAzQaCBAgQNAIsLoROA0FykIu3ugIrRXYhtvH7pyknZ30ltvHjmQ5ki3Zlq37XRRF8TIOJHGQRbJYrCrOqppz1j9AIMkki3P+5hijZg1R5P8Q/kMAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAwFOB/+HpvJgWAggggAACCCCAAAIIIIAAAggggAACCCCAAAJCE5wgQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEPBWgCa4t0vLxBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQRoghMDCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAt4K0AT3dmmZGAIIIIAAAggggAACCCCAAAIIIIAAAggggABNcGIAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAwFsBmuDeLi0TQwABBBBAAAEEEEAAAQQQQAABBBBAAAEEEKAJTgwggAACCCCAAAIIIIAAAggggAACCCCAAAIIeCtAE9zbpWViCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAjTBiQEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABbwVognu7tEwMAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAgCY4MYAAAggggAACCCCAAAIIIIAAAggggAACCCDgrQBNcG+XlokhgAACCCCAAAIIIIAAAggggAACCCCAAAII0AQnBhBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQS8FaAJ7u3SMjEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABmuDEAAIIIIAAAggggAACCCCAAAIIIIAAAggggIC3AjTBvV1aJoYAAggggAACCCCAAAIIIIAAAggggAACCCBAE5wYQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEPBWgCa4t0vLxBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQRoghMDCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAt4K0AT3dmmZGAIIIIAAAggggAACCCCAAAIIIIAAAggggABNcGIAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAwFsBmuDeLi0TQwABBBBAAAEEEEAAAQQQQAABBBBAAAEEEKAJTgwggAACCCCAAAIIIIAAAggggAACCCCAAAIIeCtAE9zbpWViCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAjTBiQEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABbwVognu7tEwMAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAgCY4MYAAAggggAACCCCAAAIIIIAAAggggAACCCDgrQBNcG+XlokhgAACCCCAAAIIIIAAAggggAACCCCAAAII0AQnBhBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQS8FaAJ7u3SMjEEEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABmuDEAAIIIIAAAggggAACCCCAAAIIIIAAAggggIC3AjTBvV1aJoYAAggggAACCCCAAAIIIIAAAggggAACCCBAE5wYQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEPBWgCa4t0vLxBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQRoghMDCCCAAAIIIIAAAggggAACCCCAAAIIIIAAAt4K0AT3dmmZGAIIIIAAAggggAACCCCAAAIIIIAAAggggABNcGIAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAwFsBmuDeLi0TQwABBBBAAAEEEEAAAQQQQAABBBBAAAEEEKAJTgwggAACCCCAAAI5C/xf//Mz4X8MiAF3YiDnEsHTIYAAAggggAACCBgWoAluGJTDIYAAAggggAACowRofrrT/GStWKvzGOA/BBBAAAEEEEAAAbcFaIK7vX6MHgEEEEAAAQQcFNDG6pUbc8L/GBAD9saA5qqDZYYhI4AAAggggAACCAQEaIIHMPgWAQQQQAABBBDIQ0AbazQ/7W1+sjaszXkMaK7mURd4DgQQQAABBBBAAIHsBGiCZ2fLkRFAAAEEEEAAgVABbazRaKXRSgzYHQOaq6GJzD8igAACCCCAAAIIOCNAE9yZpWKgCCCAAAIIIOCLgDbWaIDa3QBlfVgfzVVfag/zQAABBBBAAAEEyipAE7ysK8+8EUAAAQQQQKAwAW2s0WSlyUoM2B0DmquFFQueGAEEEEAAAQQQQMCIAE1wI4wcBAEEEEAAAQQQiC+gjTUaoHY3QFkf1kdzNX52c08EEEAAAQQQQAABGwVogtu4KowJAQQQQAABBLwW0MYaTVaarMSA3TGguep1QWJyCCCAAAIIIIBACQRogpdgkZkiAggggAACCNgloI01GqB2N0BZH9ZHc9WuCsJoEEAAAQQQQAABBJIK0ARPKsb9EUAAAQQQQACBMQW0sUaTlSYrMWB3DGiujpnyPBwBBBBAAAEEEECgYAGa4AUvAE+PAAIIIIAAAuUT0MYaDVC7G6CsD+ujuVq+KsWMEUAAAQQQQAABvwRogvu1nswGAQQQQAABBBwQ0MYaTVaarMSA3TGguepAWWGICCCAAAIIIIAAAhECNMEjcLgJAQQQQAABBBDIQkAbazRA7W6Asj6sj+ZqFnWAYyKAAAIIIIAAAgjkJ0ATPD9rngkBBBBAAAEEELgQ0MYaTVaarMSA3TGguUrpQgABBBBAAAEEEHBbgCa42+vH6BFAAAEEEEDAQQFtrNEAtbsByvqwPpqrDpYZhowAAggggAACCCAQEKAJHsDgWwQQQAABBBBAIA8BbazRZKXJSgzYHQOaq3nUBZ4DAQQQQAABBBBAIDsBmuDZ2XJkBBBAAAEEEEAgVEAbazRA7W6Asj6sj+ZqaCLzjwgggAACCCCAAALOCNAEd2apGCgCCCCAAAII+CKgjTWarDRZiQG7Y0Bz1ZfawzwQQAABBBBAAIGyCtAEL+vKM28EEEAAAQQQKExAG2s0QO1ugLI+rI/mamHFgidGAAEEEEAAAQQQMCJAE9wIIwdBAAEEEEAAAQTiC2hjjSYrTVZiwO4Y0FyNn93cEwEEEEAAAQQQQMBGAZrgNq4KY0IAAQQQQAABrwW0sUYD1O4GKOvD+miuel2QmBwCCCCAAAIIIFACAZrgJVhkpogAAggggAACdgloY40mK01WYsDuGNBctauCMBoEEEAAAQQQQACBpAI0wZOKcX8EEEAAAQQQQGBMAW2s0QC1uwHK+rA+mqtjpjwPRwABBBBAAAEEEChYgCZ4wQvA0yOAAAIIIIBA+QS0sUaTlSYrMWB3DGiulq9KMWMEEEAAAQQQQMAvAZrgfq0ns0EAAQQQQAABBwS0sUYD1O4GKOvD+miuOlBWGCICCCCAAAIIIIBAhABN8AgcbkIAAQQQQAABBLIQ0MYaTVaarMSA3TGguZpFHeCYCCCAAAIIIIAAAvkJ0ATPz5pnQgABBBBAAAEELgS0sUYD1O4GKOvD+miuUroQQAABBBBAAAEE3BagCe72+jF6BBBAAAEEEHBQQBtrNFkjmqxTr+Trpzty+9E7+f1UxP1upLltTZ6dXQZOo3YsU6mOkeZ5eYxrMa+56mCZYcgIIIAAAggggAACAQGa4AEMvkUAAQQQQAABBPIQ0Maaaw3B/Mb7m9zda3WWorG1KWafe0uWO0evyozPTfCpV/JgpyGNi/m2ZOn/0IhPEkuaq51w4RsEEEAAAQQQQAABJwVogju5bAwaAQQQQAABBFwW0MZakmZcue77WmYrgRU+3JfrRhvV5WiCX/vnobyrdX+ZcC66/JgmeJJc0lwNRCPfIoAAAggggAACCDgoQBPcwUVjyAgggAACCCDgtoA21pI048p2349fnEn9fJmbLdl88ZpXgif5JcDkC/mv1bZfX6rQBE/2SwDN1T5GfkQAAQQQQAABBBBwTIAmuGMLxnARQAABBBBAwH0BbayVrbFtz3z9fSX4xHfb8rIaePV380y2A6+qpwlOE9z9CsoMEEAAAQQQQACB5AI0wZOb8QgEEEAAAQQQQGAsAZrgyRqR5pvnnjbBJ7dkuamh2ZLK/oH8aWpOZrb133g7lKSxpLnaFeQ7BBBAAAEEEEAAARcFaIK7uGqMGQEEEEAAAQScFtDGWtKGHPc31Tz3tAl+91D2zjOjWZdnsy9kov0WKjTB08eN5qrTBYfBI4AAAggggAACCAhNcIIAAQQQQAABBBDIWUAbazS1hzcnrz+rXb4nuLTk3bOXEe8JPi+/+/uOPNk7k+MzfRuQltRqddlY25O/zCyEPDasCb4gH/1akY2TZvt5RepnZ7KysiUfTQ0fp11r+Fb+37UD+fxm73hpgvd6JFkzzdWcSwRPhwACCCCAAAIIIGBYgCa4YVAOhwACCCCAAAIIjBLQxlqSZlzZ7vvxysXHYl5Q7q0M+WDMyRfy1/WGNKLAmw15t/im86roS8e+JvjNNXlypA30wYM1aicy09dYdmk9aILTBB+Mav4FAQQQQAABBBAolwBN8HKtN7NFAAEEEEAAAQsEaIKPbkqOboIvysx25w2wRZot2d07kodPd+T20wN5sleXmt7cPJHp9luDDDbBm1I9awdFsy5vVw/k9tMdebh2JhV9vIg0DvflDz3HGD2H8+e6dn/z4njnx0zy//T9xZBXsMd7zv4GPU3wdG7njpqrFpQNhoAAAggggAACCCAwhgBN8DHweCgCCCCAAAIIIJBGQBtr/c1Kfu42K0c1wSd+rkhV8Zt1+feDwbc9mfh2vf0K76rM9DSwg68EvzxI9eDyQySDazBx7+jyPbYv7tKQJ//oji94v+Hfv5bZig4y4dfDfbneM+akz929P03wrsXwtQq/j+ZqwtXj7ggggAACCCCAAAKWCdAEt2xBGA4CCCCAAAII+C+gjbWkDbky3T+6Cf5SfjzUOGnJ2/mIV01PvpBPv1+JeDsUkerernwwGdYEnZdvNrpvk7K7/Crhq7NfyvcH3cfriON8rW9t9o05bHzx/o0meDynsPzSXI2zZtwHAQQQQAABBBBAwF4BmuD2rg0jQwABBBBAAAFPBbSxFtZ0498uG5aRTfCvduWtxkazKjOhDeyoxmfwleBVuRXx+IknndebS219PWETPGoM+d1GEzy9teaqhhtfEUAAAQQQQAABBNwUoAnu5roxagQQQAABBBBwWEAbazS8hzcnI5vgP1ekpuuf6m1DepvgvW+V0jemx90muGxv0QQ39BYtrsS+5qqGG18RQAABBBBAAAEE3BSgCe7mujFqBBBAAAEEEHBYQBtrrjQCixhnZBN87MY0TfAi1tTF59RcdbjcMHQEEEAAAQQQQAABEaEJThgggAACCCCAAAI5C2hjzcWmYF5jjmqCf7hc765Yqldn0wTPax1dfx7N1W7A8R0CCCCAAAIIIICAiwI0wV1cNcaMAAIIIIAAAk4LaGPN9QZhluOPaoIH36c73VuU5NUE54Mxs4yRPI6tuep0wWHwCCCAAAIIIIAAArwSnBhAAAEEEEAAAQTyFtDGWh5NPFefI6oJfiX4dihWvyf4a5mtpIyuVPPqez/z9vt388GY4S5xckNzNeUq8jAEEEAAAQQQQAABSwR4JbglC8EwEEAAAQQQQKA8AtpYi9OEK+t9Ipvgtw5kV8OlWZWZyaRNzrxeCT4n1+5vyu2nO4n///rOgrEP4aQJnjQ+uvfXXNVw4ysCCCCAAAIIIICAmwI0wd1cN0aNAAIIIIAAAg4LaGOtrA3uOPOObILfWJUnpxoALXk7vxjRMF6Qj35YlWvtV0VfPnd+TfA4c836PjTBu03tpNaaqxptfEUAAQQQQAABBBBwU4AmuJvrxqgRQAABBBBAwGEBbawlbciV6f7RTfA5+cNiTRoaA826zN6bH2iET3y7LrMHTRGpygxN8Aut5cfpG8Jlij+dq+aqhhpfEUAAAQQQQAABBNwUoAnu5roxagQQQAABBBBwWEAba9po4+tgY3ZUE/zK5BuZPW51o6DZkt29I3l48dYjB/Jkry618/73xX9laYLPywc/Db71ytyROohsveu7/VH/q+QH16LM8am52hXkOwQQQAABBBBAAAEXBWiCu7hqjBkBBBBAAAEEnBbQxlqZm4uj5j6yCX7+yu6bG/K8GmiEh0VFsyHvFt/IRCleCR58m5cwjLB/q8svt2h8D4tHzdUwOf4NAQQQQAABBBBAwB0BmuDurBUjRQABBBBAAAFPBLSxNqzxxr/PyfVnNalfrHdL3j17OfBWJx2jyd/kj78eydvjZuCV3y2pntRkaWVHPv128G1SrtzYkKX2q8QbtWOZ6mmQ9zWEHx5LpX3f6tra8HFEHSO329bk2VnCJGmeyt2v+uac23jtf17N1YSq3B0BBBBAAAEEEEDAMgGa4JYtCMNBAAEEEEAAAf8FtLHWaeTSdLS8uWx/s5ZYymaNNFf9r0rMEAEEEEAAAQQQ8FuAJrjf68vsEEAAAQQQQMBCAW2s0bjMpnGJK66mYkBz1cIywpAQQAABBBBAAAEEEgjQBE+AxV0RQAABBBBAAAETAtpYM9Wo4zg0fYmBbGJAc9VE3nMMBBBAAAEEEEAAgeIEaIIXZ88zI4AAAggggEBJBbSxRuMym8YlrriaigHN1ZKWKqaNAAIIIIAAAgh4I0AT3JulZCIIIIAAAggg4IqANtZMNeo4Dk1fYiCbGNBcdaW2ME4EEEAAAQQQQACBcAGa4OEu/CsCCCCAAAIIIJCZgDbWaFxm07jEFVdTMaC5mlkx4MAIIIAAAggggAACuQjQBM+FmSdBAAEEEEAAAQS6AtpYM9Wo4zg0fYmBbGJAc7WbvXyHAAIIIIAAAggg4KIATXAXV40xI4AAAggggIDTAtpYo3GZTeMSV1xNxYDmqtMFh8EjgAACCCCAAAIICE1wggABBBBAAAEEEMhZQBtrphp1HIemLzGQTQxoruZcIng6BBBAAAEEEEAAAcMCNMENg3I4BBBAAAEEEEBglIA21mhcZtO4xBVXUzGguToqp7kdAQQQQAABBBBAwG4BmuB2rw+jQwABBBBAAAEPBbSxZqpRx3Fo+hID2cSA5qqHZYgpIYAAAggggAACpRKgCV6q5WayCCCAAAIIIGCDgDbWaFxm07jEFVdTMaC5akPdYAwIIIAAAggggAAC6QVogqe345EIIIAAAggggEAqAW2smWrUcRyavsRANjGguZoq0XkQAggggAACCCCAgDUCNMGtWQoGggACCCCAAAJlEdDGGo3LbBqXuOJqKgY0V8tSm5gnAggggAACCCDgqwBNcF9XlnkhgAACCCCAgLUC2lgz1ajjODR9iYFsYkBz1dpiwsAQQAABBBBAAAEEYgnQBI/FxJ0QQAABBBBAAAFzAtpYo3GZTeMSV1xNxYDmqrns50gIIIAAAggggAACRQjQBC9CnedEAAEEEEAAgVILaGPNVKOO49D0JQayiQHN1VIXLCaPAAIIIIAAAgh4IEAT3INFZAoIIIAAAggg4JaANtZoXGbTuMQVV1MxoLnqVoVhtAgggAACCCCAAAL9AjTB+0X4GQEEEEAAAQQQyFhAG2umGnUch6YvMZBNDGiuZlwSODwCCCCAAAIIIIBAxgI0wTMG5vAIIIAAAggggEC/gDbWaFzGbVzOy/vfb8rtpzvy9d9/k4kbg4+7emfj4vbpH5blasjtWA+aYTLaRHO1P4f5GQEEEEAAAQQQQMAtAZrgbq0Xo0UAAQQQQAABDwS0sUYTcnQT8sLoH0dy1Fn3uvxyq+9xX+3IalPv0JKXj/puL1VTfF7e/+lIttsejaMD+bBU8ze79pqrGl18RQABBBBAAAEEEHBTgCa4m+vGqBFAAAEEEEDAYQFtrNEEj9mwfFwNrHZLnv/c97i7h7IXuMfG4ksppe3kitzaqEs9YCGVQ/mYJnjqeNBcDZLyPQIIIIAAAggggIB7AjTB3VszRowAAggggAACjgtoY62Ujdo0DdmvNmWp1rpY9fppRaa/6muC33gtPx1d3t6on8lPd/tv9//nq3f25HXbqCc9aIKnboCf56fmao8pPyCAAAIIIIAAAgg4J0AT3LklY8AIIIAAAggg4LqANtZogvvfnM5+jRfl85WadF8r35LKUV0qmiQ0wWmCayzwFQEEEEAAAQQQKLEATfASLz5TRwABBBBAAIFiBGiC0/w21Rz/eCXw5ifNujybfSETwbeHoQlOE7yYMsezIoAAAggggAACVgnQBLdqORgMAggggAACCJRBgCY4TXBTTfCZ7fOMaUll/0D+NNV2pQk+VuM7uDaaq2WoS8wRAQQQQAABBBDwWYAmuM+ry9wQQAABBBBAwEoBbawFm218H9UYX5O59vtdN2rHMjXwvuIv5f7+5XuCS7Mm96cHj/XhizNpnEdDsybft2+/emdLZrfPpFLXx7bk8LAiD75/IRMDzzF4TBvW7PqjI3nyZKV3vDTBaYJbWfkYFAIIIIAAAgggUJwATfDi7HlmBBBAAAEEECipAE3wpA3lLVnuxEpVZgYa1K9ltvMm2HWZDflgzO7bhpzfviifr55J4I1EOke//KYl28tvehvLA8+ZdA453p8mOE3wvojmRwQQQAABBBBAoOwCNMHLHgHMHwEEEEAAAQRyF6AJnrQhbLIJ3pDd42Z7zc8/RPJEZhd25PbCoayctF8RfnFrU57/PJ+8mTq5Il883ZHbif9fkw8mk7oMuT9N8OTrNuSXHJqruRcJnhABBBBAAAEEEEDAqABNcKOcHAwBBBBAAAEEEBgtoI01G95Ow40xmGyCt9en2ZDn5x8i2dP8XJRbO91GeGNvR97ruX1I0zl4n8fV0QEQeo+WPP85xvGDzzXse5rgNMFDY4x/RAABBBBAAAEEyitAE7y8a8/MEUAAAQQQQKAgAZrgSZu9hpvgzbr8+s/F8Ebpfx/IxWdNnsdG80SmhzWah/37w4qkaoM36/JLyNu4pPolBU3w8LUdtmYR/665WlCp4GkRQAABBBBAAAEEDAnQBDcEyWEQQAABBBBAAIG4AtpYS9XgjGjY+Xs8s03wg9dvIpqkb+TfJ7qSNfkx5EM2rXemCR6xvsl+AaO5qhHBVwQQQAABBBBAAAE3BWiCu7lujBoBBBBAAAEEHBbQxpr1zVRrGu5mm+B7K68jmqSjP2TT+nWjCR6xvjTBHS6dDB0BBBBAAAEEEEgtQBM8NR0PRAABBBBAAAEE0gnQBE/WiLxygyZ4osY7TXCa4OlKE49CAAEEEEAAAQS8FaAJ7u3SMjEEEEAAAQQQsFWAJjhN8ERN7aSvyKcJThPc1uLHuBBAAAEEEEAAgYIEaIIXBM/TIoAAAggggEB5BWiCe9wE54MxjTWgM/1FQcxfLGiulrdaMXMEEEAAAQQQQMAPAZrgfqwjs0AAAQQQQAABhwS0sWZDk8+NMTj0diiPqykjsSXPf076y4Eh9+eV4MYa8ZqrKReVhyGAAAIIIIAAAghYIkAT3JKFYBgIIIAAAgggUB4Bbay50YAe0miN+UpaM3N0qAk+uSJfPN2R20n/f/RW3p80ZE0TnCZ4ecopM0UAAQQQQAABBGIJ0ASPxcSdEEAAAQQQQAABcwI0wZM2ex1qguf6y4EhjjTBaYKbK1ccCQEEEEAAAQQQ8EKAJrgXy8gkEEAAAQQQQMAlAZrgQ5q3QxvINMETvaKeJjhNcJcKImNFAAEEEEAAAQRyEKAJngMyT4EAAggggAACCAQFaILTBE/U1B76y4E5uXJzVab733rlRVUqGnC1qvzvvtu/+G7eWJPY2Dyi5ljgbZqryslXBBBAAAEEEEAAATcFaIK7uW6MGgEEEEAAAQQcFtDGmu8NRHPz45Xgwyw/XqknzoTG1iZN8JiNdc3VxMg8AAEEEEAAAQQQQMAqAZrgVi0Hg0EAAQQQQACBMghoY21YY5N/73+l+JrM1VoXodGoHcvUQAPzpdzfv7xdmjW5P93/+Dm5/qwml+3ilrx79jKiCdx7rO9DjmXT+lx/XpNGwqQ5Wl2NmP+gnU3zzXssmqsJibk7AggggAACCCCAgGUCNMEtWxCGgwACCCCAAAL+C2hjLe+GHs9Hg5cYSBYDmqv+VyVmiAACCCCAAAII+C1AE9zv9WV2CCCAAAIIIGChgDbWaEgma0jihVfeMaC5amEZYUgIIIAAAggggAACCQRogifA4q4IIOCOgF608vUzwQADV2LAnQoz/kh1TfJu6PF8NJGJgWQxoLk6fta7cwSdM1/ZPxADbsSAO9WFkSKAAALFCtAEL9afZ0cAgYwE2LS7sWlnnVinYAxkVA6sPKzOm4ZksoYkXnjlHQOaq1YWkowGpXPmK+doYsCNGMioFHBYBBBAwDsBmuDeLSkTQgCBcwHdtOd9sczz0aAhBpLHgOZrmaqXzpl4SR4vmGGWZwxorlKfiLs8447nIt7ixEAZ61OZajFzRQAB8wI0wc2bckQEELBAQDeFcTaQ3IcLDWKg2BjQfLWgdOQ2BJ0zsVds7OGP/6gY0FzNrThY8EQ651E23E7+EAPFxoDmqgVlgyEggAACTgjQBHdimRgkAggkFdBNIZvzYjfn+OMfJwY0X5Pmucv31znH8eE+5BExUFwMaK66XG+Sjl3nTNwVF3fYYx8nBjRXk+Y490cAAQTKKkATvKwrz7wR8FxAN4VxNpDchwsNYqDYGNB89bws9UxP50zsFRt7+OM/KgY0V3sS2PMfdM6jbLid/CEGio0BzVXPSxLTQwABBIwJ0AQ3RsmBEEDAJgHdFLI5L3Zzjj/+cWJA89WmGpL1WHTOcXy4D3lEDBQXA5qrWdcEm46vcybuios77LGPEwOaqzbVD8aCAAII2CxAE9zm1WFsCCCQWkA3hXE2kNyHCw1ioNgY0HxNnfAOPlDnTOwVG3v44z8qBjRXHSwzqYescx5lw+3kDw2lyFQAACAASURBVDFQbAxorqZOdh6IAAIIlEyAJnjJFpzpIlAWAd0UsjkvdnOOP/5xYkDztSz16XyeOuc4PtyHPCIGiosBzVXqU3FrQPxjTwyEx0AZ61OZajFzRQAB8wI0wc2bckQEELBAQDeFbJrDN8244GJTDGi+WlA6chuCztmmdWAs1AViYDAGNFdzKw4WPJHOmXgYjAdMMLEpBjRXLSgbDAEBBBBwQoAmuBPLxCARQCCpgG4KbdqoMhYunIiB8BjQfE2a5y7fX+dMTITHBC642BIDmqsu15ukY9c527IGjIN6QAyEx4DmatIc5/4IIIBAWQVogpd15Zk3Ap4L6KaQTXP4phkXXGyKAc1Xz8tSz/R0zjatA2OhLhADgzGgudqTwJ7/oHMmHgbjARNMbIoBzVXPSxLTQwABBIwJ0AQ3RsmBEEDAJgHdFNq0UWUsXDgRA+ExoPlqUw3Jeiw6Z2IiPCZwwcWWGNBczbom2HR8nbMta8A4qAfEQHgMaK7aVD8YCwIIIGCzAE1wm1eHsSGAQGoB3RSyaQ7fNOOCi00xoPmaOuEdfKDO2aZ1YCzUBWJgMAY0Vx0sM6mHrHMmHgbjARNMbIoBzdXUyc4DEUAAgZIJ0AQv2YIzXQTKIqCbQps2qoyFCydiIDwGNF/LUp/O56lzJibCYwIXXGyJAc1V6hMxaUtMMg5iUWOgjPWpTLWYuSKAgHkBmuDmTTkiAghYIKCbQt0k8pULBmLA3hjQfLWgdOQ2BJ0zcWlvXLI2rM15DGiu5lYcLHginTM5QA4QA3bHgOaqBWWDISCAAAJOCNAEd2KZGCQCCCQV0E0hm3e7N++sD+tzHgOar0nz3OX765zJAXKAGLA7BjRXXa43SceucyY27Y5N1of10VxNmuPcHwEEECirAE3wsq4880bAcwHdFHKBwAUCMWB/DGi+el6WeqancyY+7Y9P1qjca6S52pPAnv+gcyb2yx37rL/966+56nlJYnoIIICAMQGa4MYoORACCNgkoJtCNvD2b+BZI9ZI89WmGpL1WHTOxD/xTwzYHQOaq1nXBJuOr3MmNu2OTdaH9dFctal+MBYEEEDAZgGa4DavDmNDAIHUArop5AKBCwRiwP4Y0HxNnfAOPlDnTHzaH5+sUbnXSHPVwTKTesg6Z2K/3LHP+tu//pqrqZOdByKAAAIlE6AJXrIFZ7oIlEVAN4Vs4O3fwLNGrJHma1nq0/k8dc7EP/FPDNgdA5qr1Ce714k8Yn3KGANlrE9lqsXMFQEEzAvQBDdvyhERQMACAd0UlnFDzJy5EHQtBjRfLSgduQ1B5+zaWjFe6kvZYkBzNbfiYMET6ZzLttbMl/rmWgxorlpQNhgCAggg4IQATXAnlolBIoBAUgHdFLq2mWW8XICVMQY0X5Pmucv31zmXcb2ZM3XOpRjQXHW53iQdu87ZpXVirNSVMsaA5mrSHOf+CCCAQFkFaIKXdeWZNwKeC+imsIwbYubMhaBrMaD56nlZ6pmezpmvn3XeGgYLLGyOgZ4E9vwHXQfXziWMl/1P2WJAc9XzksT0EEAAAWMCNMGNUXIgBBCwSUA3hWXbDDNfLgBdjAHNV5tqSNZj0TnzlcYvMeBGDGRdE2w6vsaki+cTxsw+qEwxoLlqU/1gLAgggIDNAjTBbV4dxoYAAqkFdFNYpo0wc+XCz9UY0HxNnfA8EAEEEEDAmIDWZFfPKYyb/VBZYkBz1VjycyAEEEDAcwGa4J4vMNNDoKwCuiksyyaYeXLB53IMaL6WtV4xbwQQQMAmAa3JLp9XGDv7ojLEgOaqTfWDsSCAAAI2C9AEt3l1GBsCCKQW0E1hGTbAzJELPddjQPM1dcLzQAQQQAABYwJak10/tzB+9ke+x4DmqrHk50AIIICA5wI0wT1fYKaHQFkFdFPo++Z3vPnNy/vfb8rtpzvy9d9/k4kbZi+Wrj+vSeMiAFvy7tlLGW+sZsfGWOzy1Hwta71i3ggggIBNAlqTOVdGnSsX5KNfduT20035y8yC8T3O1FrzMiSaDZl7GDUObitznGqu2lQ/GAsCCCBgswBNcJtXh7EhgEBqAd0UlnljPHLu/ziSo45wXX65ZfZC6uOVeufoeyuvjV8gjpyf4ab+8Oebl/d/OpLt9vVq4+hAPsztuc2u2fA5Zvs8mq+dgOEbBBBAAIHCBLQmF3VOcOF535s7bf+iX0SaJzJt+Lw/s91d/uXH2Z6D8/a+enNV/rZSkbfHDam2904iLanV6rKxeSh/u2P+lwp5zzGv59Nc7UYL3yGAAAIIRAnQBI/S4TYEEHBWQDeFeW1CnXyex9XA+rbk+c9mL7JK0QSfXJFbG3XptvtFpHIoHxu+GHYyvhIYaL4GApJvEUAAAQQKEtCa7Pu5Z5z5Bfc4IjX5cdrsHsrLJnjYnik0xpuyvvjG+F8ojrPetj5WczWUkX9EAAEEEBgQoAk+QMI/IICADwK6KbR102rFuL7alKVa62K566cVmf7K7AVc8ALRx1eCX72zJ6/bfj05QxM88av+NV97HPkBAQQQQKAQAa3JVuxVEvxCNdfx3j1o/wVYS2qHB8Z/+e1lE7zvxRfVk5q82jyRpc2qbFRb3VfWX0R9U5YfLybeT+QaAxbEpuZqIYWCJ0UAAQQcFKAJ7uCiMWQEEBgtoJvCsm2GbZqvv03wRfl8pSbd19G3pHJUl4qGJU3wxBetmq9KyFcEEEAAgeIEtCbbtKco21i8bII/OpF67UyeLbyT308NvvDi2j+PZbfz9igiclaRKQsazTbHnuZqcdWCZ0YAAQTcEqAJ7tZ6MVoEEIgpoJtCmzeuvo/N1yZ4cF7SrMuz2RcycfdQ9jQ2aYLTBNdY4CsCCCDgoAB7qMEGbd57Ni+b4DEa2n9YOgtkTEN+/aH4tch77ZM8n+ZqAI1vEUAAAQQiBGiCR+BwEwIIuCugm8IkG0nua/ZCI9gs9untUC4vTFtS2T+QP+krmWiCJ258B/NN89XdisPIEUAAAX8EtCYH6zTfm90jjfIsaxP8yvS+bARSybcPBR217klv11wNkPEtAggggECEAE3wCBxuQgABdwV0U5h0M1mu+6/JXPs9rRu14+g/OZ38Tf7465G8PW5IVf9UtdmSyklt6J+1hjXBJ75dl4fbZ1KpX74XuTRbcnhYkQffv3DmA5CuPzqSJ09WesdLE5wmuLvlkpEjgAACPQLsoWI0vKd35V17P1Tf35XrUa9ynlqWr5dOZOOkKbXOHqopx8dVmf33G/nd5ODzhTXBr97ZkWeH9cA+rCnb24fytzsLY52D7dr7bslyIBppgg/GRnC9NFcDZHyLAAIIIBAhQBM8AoebEEDAXQHdFAY3inzfv5EOXmhUZWbIBdzEt5vyvNpuWg8JiUatJj/em++5COttgr+RD+ZPpaIXfwPHacn28pvexvKQ8Vi5jjTBe9Y+6Rppvg6EBf+AAAIIIJC7gNbkpLW8VPePed6/+uBA1oPv8BGymvWTinxzs3eP1tsEX5TPV8+kHvLYi39qNuSlLx8i2fNK8JY8/7nXpVQxFmMfrLk6LDT4dwQQQACBXgGa4L0e/IQAAp4I6KaQzXLUxUOMJvjNLVkOXLw16nV5u3ogt5/uyO2lI3lbaUmjHTONrc2eRmiwCV47a7bvd/4hkicyu7AjtxcOZeUk2FxvyvOfexvp8dZvUf7z0c7lmM7HFfv/TfnPvovOeM8XYhrzYjj18WNcCLl8bM1XT8oP00AAAQScFtCa7PJ5JfOxxzjvT9w77Pmgx/ppTZZW9i72KQ9WTmTjtLsH2l1+1bOHCjbBq2f6CoKW7O8dycOLPVhFttp/zXcRbM2a/PjfIfuTUfuHyRX5Iva+KbjHWpMPQl7BPq77xONqZ18pcir3vkoxp1Fz9uh2zVWnCw6DRwABBHIUoAmeIzZPhQAC+QnopnDczbjfjx/VBJ+XL9f1wkukcXzUfQ/szgXEvLz/05Fsn99te6vnAi7YBL9Y+WZDnp9/iGTnsecXNotya6d7EdjY25H3em6PcfETvBBNGGIbiy97xpx6vYNj4IMxE5tqviZcPu6OAAIIIJCBgNbk1OfEpOdxF+8/8ry/LD8edvc31e1teX+gabwgHy1c/pVc/2enBJvgF0t8VpOHD/re9mTyjfy70g2Ao9XVxOffK4+r3QMk+i6LV2n/Jnf3umap9oQuxtIYY9ZcTbR03BkBBBAosQBN8BIvPlNHwGcB3RRyARfVRB7RBA/+SWrzVO5FvGp64tu38ul3va/i7mmCN+vy6z8Xwy/O/vtAtjUYmycynfRiYHpPAr16PVKMry15+ah3zKnjZeTFcNQ6cJvma4xF4y4IIIAAAhkLaE1OfU5Meh538f6jzvs/V6Sm63R6LJ8NNMC75/6rd94N/GVaTxP87FTu9e2xdG0m/nXSfZ7jA/kwqeXDiqRqgzfr8svd7hx0PON8nfjhWI7UTNL+daDZMY0znzweq7naYeMbBBBAAIFIAZrgkTzciAACrgropjCPDai7zxHdBH9v/rSz/I3trb5XcI++yAg2wQ9evwlvgF9crL2Rf5/oU9Xkx+nRx7bOfNTFcNKL0pLdX/NVo4CvCCCAAALFCWhNtu5ca9O5ccR5/7O3+mZxIv2v8o7jGmyCv3oS8Qv7yW15paFSr8iXNhklGcvkG5k97r4KvLq9JdeSPL6k99Vc1RDgKwIIIIBAtABN8GgfbkUAAUcFdFMY50KjvPeJboJ/ud69GEnztiHBJnj0BeBrme38OW9dZg2/siiX9R1xMZzLGBy+ANR8dbTcMGwEEEDAKwGtyZy7In4pH3nefyk/HmpIpHvbkGATfPlxxDhuRO/l3FjDeQn+0kDOqjIT8deHbswpas3M3aa5qtHGVwQQQACBaAGa4NE+3IoAAo4K6KaQjXLURjv6win+BVj4c9AED3chJgddNF8dLTcMGwEEEPBKQGsy56vB81XHJLIJPv4v9+PvwaL3cp3xWvuL8nl5/0m1+5YszbrM3ot45bu184iIlQzHrLnqVQFiMggggECGAjTBM8Tl0AggUJyAbgrt3/wXs2m+dIm6cHolvxx31y/6VUjhc6AJHu5CTA66aL52I47vEEAAAQSKEtCazPlq8HzVMYlsgm/Ky87niqf7C7eyNMGvPQ40wKUpy4+HfH5Mho3kzpo6+Byaq0XVCp4XAQQQcE2AJrhrK8Z4EUAgloBuCl3e2GY/9qgm+Lzc2ulSW90E54MxI95vPeIC3qKLPc3XbsTxHQIIIIBAUQJak7Pfh7hxjgp1iGyCBz/rxPImeIEfjHntXxU56PyyoCnri28Sf/5M6NpYtL/Jenyaq0XVCp4XAQQQcE2AJrhrK8Z4EUAgloBuCrPefLp9/Kgm+JwEX4Vk9XuCBy9EY0VH905p5hW65sExVA7l4xJdgIV6JJy/5mt3ZfgOAQQQQKAoAa3JJuq7t8eIPO8H3w7F8vcEf1xNGWbp5qXxMHHvUHY7DfCW7K7QAFebJF81V1MuIg9DAAEESidAE7x0S86EESiHgG4Kk2wky3ff6Cb4h8v1TrA0trcSvzonv7dDWZT/fLQjt58m/X9DPpoy9Cq0yIthQ8+RsLHsUjxrvnYCjm8QQAABBAoT0Jrs0nkk97GOOO9Pb3U/XDz6w8HD9wjBFyJE/zVe9F5upMvkinyReP+0I7cfvZX3J8PHPuo5J+7ty/qZhjcN8FFeUbdrrqomXxFAAAEEogVogkf7cCsCCDgqoJvCqI0jt424cPrHkRzp+jdP5d7NiIudqVfyl/u97+OYXxM8Ylx5NY5HXAwTa9FrpPmq4cZXBBBAAIHiBLQmc+6KOHeNOO+/N3cqDV3C02P5LKJhPPHtW/n0u94Pg8ytCZ7XPkmf5+aWLAca4AfrG3JNb+Nr4re301zVUOMrAggggEC0AE3waB9uRQABRwV0U8gFXMQF3I0RTfAby/LjYfeVTI3jQ/lk4CJuXt7/6fDyFT3bWz2bd5rgUfbcFsxNzVdHyw3DRgABBLwS0JocrNN833feHtEEvzK5Ls87zV6R6vZWSLN3QT769eTifbH7Xy3uZRP85pr8etzdV4ab9DnTGO/ZW/fnoeaqVwWIySCAAAIZCtAEzxCXQyOAQHECuins3yzyc/DiYlQTfE5637NRpFGvy9vVg8u3Hlk6kreVVveVTmVpgt9clen+Px1+UZWKhnutKv+77/Yv+l7hRRwG43BONF+VkK8IIIAAAsUJaE3mXNV7rurxGNUEvzEn1x5XJfiO2/XTmiyt7F3soR6snMjGabch7H8TfF6+2ejOV6Qpm1snsrQ56v9D+SbqLxFL3iTXXC2uWvDMCCCAgFsCNMHdWi9GiwACMQV0U9hzwVLyjfKgxegm+Pljrv2rcvEqpSj6Rq0mP97r/VNeX18JHpxXlEnwtsbWZuQreQbXJuLC28M41nwNmvE9AggggEAxAlqTy35uipx/jCb4lRvz8sliracRHrai9ZPKQKPXv1eCBz8sNExh+L9Fvyd6ufZL/TGpuTpcj1sQQAABBIICNMGDGnyPAALeCOimsH+zyM/Bi4U1matdviqnUTuWqYjm6sSXb+TO6qls1wKv/G42Zf/wRGafhn840ocvztqvEm/Ju2cvI5rAL+X+fvvVQc2afD8dHKN9319/Xuu++j1mxhytrkbM37455p0nmq8xObkbAggggECGAlqT8z4XOPV807vyrnm5CPX9XbkesYe6OrMhDzdrsn/WfSV0o96U7b0jefjLK7ka8tj/pa+abjZk7mHUPqG7l5OzSuRerljf3+TWTnf+8cO3Kc8i5x9l4/9tmqvxPbknAgggUG4BmuDlXn9mj4C3AropLHbD7//mG1/W2EQMaL56W5CYGAIIIOCQgNZkE/WdY7BPIAayiwHNVYfKC0NFAAEEChWgCV4oP0+OAAJZCeimkI13dhtvbLE1FQOar1nVA46LAAIIIBBfQGuyqRrPcdgvEAPZxIDmavzs5p4IIIBAuQVogpd7/Zk9At4K6KaQTXc2m25ccTUZA5qv3hYkJoYAAgg4JKA12WSd51jsG4gB8zGguepQeWGoCCCAQKECNMEL5efJEUAgKwHdFLLhNr/hxhRT0zGg+ZpVPeC4CCCAAALxBbQmm671HI/9AzFgNgY0V+NnN/dEAAEEyi1AE7zc68/sEfBWQDeFbLbNbrbxxDOLGNB89bYgMTEEEEDAIQGtyVnUe47JPoIYMBcDmqsOlReGigACCBQqQBO8UH6eHAEEshLQTSEbbXMbbSyxzCoGNF+zqgccFwEEEEAgvoDW5KxqPsdlP0EMmIkBzdX42c09EUAAgXIL0AQv9/ozewS8FdBNIZtsM5tsHHHMMgY0X70tSEwMAQQQcEhAa3KWdZ9js68gBsaPAc1Vh8oLQ0UAAQQKFaAJXig/T44AAlkJ6KaQDfb4G2wMMcw6BjRfs6oHHBcBBBBAIL6A1uSsaz/HZ39BDIwXA5qr8bObeyKAAALlFqAJXu71Z/YIeCugm0I21+NtrvHDL48Y0Hz1tiAxMQQQQMAhAa3JedR/noN9BjGQPgY0Vx0qLwwVAQQQKFSAJnih/Dw5AghkJaCbQjbW6TfW2GGXVwxovmZVDzguAggggEB8Aa3JeZ0DeB72G8RAuhjQXI2f3dwTAQQQKLcATfByrz+zR8BbAd0UsqlOt6nGDbc8Y0Dz1duCxMQQQAABhwS0Jud5HuC52HcQA8ljQHPVofLCUBFAAIFCBWiCF8rPkyOAQFYCuilkQ518Q40ZZnnHgOZrVvWA4yKAAAIIxBfQmpz3uYDnY/9BDCSLAc3V+NnNPRFAAIFyC9AEL/f6M3sEvBXQTSGb6WSbabzwKiIGNF+9LUhMDAEEEHBIQGtyEecDnpN9CDEQPwY0Vx0qLwwVAQQQKFSAJnih/Dw5AghkJaCbQjbS8TfSWGFVVAxovmZVDzguAggggEB8Aa3JRZ0TeF72I8RAvBjQXI2f3dwTAQQQKLcATfByrz+zR8BbAd0UsomOt4nGCaciY0Dz1duCxMQQQAABhwS0Jhd5XuC52ZcQA6NjQHPVofLCUBFAAIFCBWiCF8rPkyOAQFYCuilkAz16A40RRkXHgOZrVvWA4yKAAAIIxBfQmlz0uYHnZ39CDETHgOZq/OzmnggggEC5BWiCl3v9mT0C3groppDNc/TmGR98bIgBzVdvCxITQwABBBwS0Jpsw/mBMbBPIQaGx4DmqkPlhaEigAAChQrQBC+UnydHAIGsBHRTyMZ5+MYZG2xsiQHN16zqAcdFAAEEEIgvoDXZlnME42C/QgyEx4Dmavzs5p4IIIBAuQVogpd7/Zk9At4K6KaQTXP4phkXXGyKAc1XbwsSE0MAAQQcEtCabNN5grGwbyEGBmNAc9Wh8sJQEUAAgUIFaIIXys+TI4BAVgK6KWTDPLhhHmoy9Uq+frojtx+9k99PhT1uQT76ZUduP92Uv8wsyNDj3Ah7LP+G1/AY0HzNqh5wXAQQQACB+AJakzlvDT9v9dtMfPtWps/3UL+8kd9Nhjxu8oV8+uh8D7Uhf/xynj0Ue0UjMaC5Gj+7uScCCCBQbgGa4OVef2aPgLcCuinsv0jh55ALs4sLkd/k7l6rEw+Nrc2Bzfl7c6fS0Hs0T2S6VBcwC/If97dkdu1Utk+bUu84tKRyUpOllS35KPQXB8O8+fdgLmq+KitfEUAAAQSKE9CaHKzTfB913l6VJ6fd9dpdfjWwh/rzamcHJXJ8IB+Wag/Vazfx7bo8OWrvOZtn8tOt3tuJtfgemqvd6OM7BBBAAIEoAZrgUTrchgACzgroppCNdNyN9GuZrQSW+3BfrvddoH280mn9ikhNfpyOe2y37zfx3bYsnXR/QRBQ6v32rCY/3uPVXWlyTvO1F5SfEEAAAQSKENCanKael/MxW7IcWKja+vpAE3xmO3CHekW+7NtjlcNtXj54ciL7zYCF1GX2rtv7xCLXTnM1KMr3CCCAAALDBWiCD7fhFgQQcFhAN4VFbkxde+6PX5xdvsK52ZLNF68HLuCu3D2Q7YsLl5bUDg/k45JcwPVcuDabsn9YlaXNE1naqclhva85flaVmZtczCWNfc1Xh0sOQ0cAAQS8EdCanLSWl/f+v8n0RuPyr+WaDfnt0W8De6j3Hp1Ipb2HOl7flPdKsofqxMTUK3mw0zbqyRSa4B2jFDGhudpDyg8IIIAAAkMFaIIPpeEGBBBwWUA3heNsLHkszdzzGJjeaknlqCIPfliWqwMXKIvy+dt6921iRKS6tjZw8UssRceS5qvLNYexI4AAAr4IaE3m3BV97sInns+1fx7K+lk3Oxq1M9mt6c80wceJI81V1eQrAggggEC0AE3waB9uRQABRwV0UzjOxpLHxru4wWlZHh4FEqV2LJ8NNMuxjIoTzdeAIt8igAACCBQkoDU5qm5zG+f1WDFw91D2OnHckv21TXl/MvgWfDTBYzkO2VdqrnaI+QYBBBBAIFKAJngkDzcigICrAropHGdjyWO5wIsbA9cXOy9pOn8tuMwMuViJe7yy3U/z1dV6w7gRQAABnwS0JpftXMR8M9j3Pa5epEajVpOHDxbafylHE9xUrGmu+lR/mAsCCCCQpQBN8Cx1OTYCCBQmoJtCU5vMMhzn+rPa5XuCS0vePXs5+JYe07vyrv1hRvX93YEPzrxyY0OW2rd33hJk8oV8ulCRjZNm+9gi9bMzWVnZko+mMrjYKqr53L7Iuwx4muBJ80XztbCCwRMjgAACCHQEtCYnreVlvv/Uu+57gs89DNnfPDzuvCd45V3I26bd6n7uyvrz9h5s6pX8baUq27VW+23XWlKr1uTZwlt5fzLkOYraA0U97/Sm/H+r2/JBz3hpgpvKFc3VTvLyDQIIIIBApABN8EgebkQAAVcFdFNoapNZhuN8vFLvLPfeStgHYwb+pLVyGPLBmFuyrEfY3pKJ73blda3vgyP1dhFp1E68+RDJnleC1yvyZdQFIbcN/IJF8zUQHnyLAAIIIFCQgNbkMux9TM0x+CHay49DGtTBX5Zvbw2cB68E3jbkfA927Z9H7Q8jDw+C+tGBfNLTWA55Tmv3GzTBTcWd5mp4lPCvCCCAAAL9AjTB+0X4GQEEvBDQTaGpTWYZjmO0CV6py277VeHSrMvb1QO5/XRHHq6dtV8JdRlmjcN9+UOKi7SJ79Yujnd+zET//7QiEymeL3r952VmO9Ds39uV94w/h0sXt8nHqvnqRfFhEggggIDjAlqTo899yWu9z8cz2QSvHNfl8k1Ezl8wcCZLK3ty++mezG7XO39Vdx5i1fX1VHuaa/c3k+2d2nut6fuLg837VPsdmuCmckFz1fGSw/ARQACB3ARogudGzRMhgECeAropNLXJLMNxjDbB24tdPTiQP/W97cnEvaPAhyQ15Mk/kl9IBy82E8VVFq/S/mpHVrXhLy1ZnfvN0EVichdX41TzNdFacmcEEEAAgUwEtCa7ek4pYtzBfcm4rwS/XNSWHKyff4hk717g2pOqdD6FpHkqd7/qvX303IMN6IThc7gf8lZ4SZ///P7BMfDBmKPXbLix5mrCleTuCCCAQGkFaIKXdumZOAJ+C+imcJyNZdkea7oJXt3b7XsPSN3Ez8s3G91XTu8uv0rcNJ5a63SdEwVy4/gg5G1cdFxpvs7LZ28b3TGcVeTLvgvWssVRmvlqvnYh+Q4BBBBAoCgBrclp6nlZH2O2Cd6S3bdrci30VdbL8vBII6MlLx8l3bu8lO8PunswPVKcr/WtzVSvPB+MCZrggyZJ1/Hy/pqrcdaP+yCAAAIIiNAEJwoQQMBLAd0UmtpkluE4ZpvgVbkV0QyeeKJ/6CtSW19P3AS3ZT0m7h123/ZFmrL82NSfCqe7GLLFJek4NF+9LEZMCgEEEHBMQGty0lpe5vsbbYKfHMknoQ3wy73BJ6+7n+GysRjy3QQV0AAAIABJREFUQeYRj7VjjWiCm1oHzVXHSgzDRQABBAoToAleGD1PjAACWQroptDUJrMMxzHdBJ+Juggb9QFRUY+15bbJd/LraTeKq9tbQ161Va6Gdppc0XztavIdAggggEBRAlqT09Tzsj7GaBM89MPHu3uJkfs1W/ZJQ8dBE9xUnmiuFlUreF4EEEDANQGa4K6tGONFAIFYAropNLXJLMNxRl5U3T3svpd36AXalix3VqcqXjfBJ1/Ire3uW7I0jg/lk4hXvpchfsaZo+ZrJ3z4BgEEEECgMAGtyePU9bI9liZ4t0k/eu1pgo82iuepuVpYseCJEUAAAccEaII7tmAMFwEE4gnoptDUJrMMx6EJHu+C48qNRZkJNMDlrCozN+M+lvuF5ZLma7zs5l4IIIAAAlkKaE0Oq9f8W/h5nCZ4uEt4vNAED3dJYnh5X83VLOsBx0YAAQR8EqAJ7tNqMhcEEOgI6KbQ1CazDMdxqQle3AdjLso36w3pfBTmWU1+vDfv7Hua2xLXmq+dBOYbBBBAAIHCBLQm23KOcGEc7jTB+WBMF+Ip7hg1VwsrFjwxAggg4JgATXDHFozhIoBAPAHdFMbdRHK/OXGpCR682IwXEe171Svy5dD3qBz1Cpx5+WSl3m2AN+sySwPcyC8ANF8TrSV3RgABBBDIREBrMnujUfuC7u3Bfcny4+6/dwxHfRbKyLec6x5z5H4tcp8TfBV2wvA53JfrkcfujrEz79D7B8dQl9m7cR/H/fpdNVcTriR3RwABBEorQBO8tEvPxBHwW0A3hf2bRX4efgEx8qJq5AVafu8JPvHdmtx+upP4/+nvX8hE6AXZcJfLmJmXTxZrUtW0oQFupPmt+aj5qrx8RQABBBAoTkBrstZovo7aI8yJO03wObl2fzPx/ul8z/X1nQVD536a4KZySnO1uGrBMyOAAAJuCdAEd2u9GC0CCMQU0E2hqU1mGY7jUhM87/W49rgaaIA35Pm/Fg1dCI6+sM57rkU8n+ZrzPTmbggggAACGQpoTS7ifODqc7rUBC/emCa4qTXQXM2wHHBoBBBAwCsBmuBeLSeTQQABFdBNoalNZhmOQxM8vCF97Z/HstvUyGrK8mMa4KbzQfNVlfmKAAIIIFCcgNZk07Xe5+PRBA/fQ4WvOU3wcJckhpf31VwtrlrwzAgggIBbAjTB3VovRosAAjEFdFNoapNZhuPQBA+5+JjckMV6IOhqZ7K8eSJLo/5f2ZBrqd52JWQMJTiO5mtAmm8RQAABBAoS0Jpchr2PqTnSBB+2f1mU/3zU//Z1B/KqpsHdkFcv+m7/aSXlW9cNG4O//665qpp8RQABBBCIFqAJHu3DrQgg4KiAbgpNXdyU4Tg0wUMukoLvg54oF6oyU4Lmtam80HxNRMydEUAAAQQyEdCabKrGl+E4NMFD9lDn+6A0+6jmiUyzh4r1tnuaq5kUAg6KAAIIeChAE9zDRWVKCCAgopvCMlx4mZrj9Wc1uXzRc0vePXs5uPme3pV37bcFqe/vyvWBC5Q1mau1LsKvUTuWqYHbAxdID4+l0j5WdW1t8LmiHpvnbV9ty+vOW6EkyKyzSvT885yDA8+l+ZpAmLsigAACCGQkoDXZ1P6iDMeZeteQxvl6NBsy9zCw39FzcGff05LKu5B9z8g9VveYwf3a+vOQ/Zo+pw1fp/dkPek+6vRI/mzD2B0Yg+ZqRqWAwyKAAALeCdAE925JmRACCJwL6KawDBdezLF7YYiFmxaar1QvBBBAAIHiBbQmc05185zKupVn3TRXi68ajAABBBBwQ4AmuBvrxCgRQCChgG4KuRAoz4UAa+3uWmu+Jkxz7o4AAgggkIGA1mTOq+6eV1m7cqyd5moGZYBDIoAAAl4K0AT3clmZFAII6KaQi4ByXASwzm6vs+YrlQsBBBBAoHgBrcmcW90+t7J+/q+f5mrxVYMRIIAAAm4I0AR3Y50YJQIIJBTQTSEXAP5fALDG7q+x5mvCNOfuCCCAAAIZCGhN5vzq/vmVNfR7DTVXMygDHBIBBBDwUoAmuJfLyqQQQEA3hWz+/d78s75+rK/mK5ULAQQQQKB4Aa3JnGP9OMeyjv6uo+Zq8VWDESCAAAJuCNAEd2OdGCUCCCQU0E0hG39/N/6srT9rq/maMM25OwIIIIBABgJakznP+nOeZS39XEvN1QzKAIdEAAEEvBTIpQmuxZmvnwkGGLgQAz5UO3Vm0+/npp919WtdNV99qD1x56Bz5iv7AmLAjRiIm9s+3E9jknOtX+da1tO/9dRc9aHuJJmDzpuvbpw/WSfWKUl+Z31fmuD/k4CkKBED/TGQdeHJ4/g6Jzb8/m34WVP/1lTzNY/aYMtz6Jz5yjmYGHAjBmypHXmMQ2OS861/51vW1K811VzNoy7Y9Bw6b766cf5knVgnm+pHrk1wTrp+nXRZT//WU09QNhWptGPRuRCn/sUpa+rfmmq+ps13Fx+ncyae/Ytn1tSvNdVcdbHOpB2zzplY9iuWWU//1lNzNW2uu/o4nTcx7V9Ms6Z+ranmqk21hib4Db+CjKLBeo4TAzYWqbQFU+cyjgePJZ+IgXxiQPM1bb67+DidMzGWT4zhjHPaGNBcdbHOpB2zzjmtGY8j34iBfGJAczVtrrv6OJ03cZZPnOGMc9oY0Fy1qdbQBKcJLmkDmsf5VwxtLFJpC6bOhTj1L05ZU//WVPM1bb67+DidM/HsXzyzpn6tqeaqi3Um7Zh1zsSyX7HMevq3npqraXPd1cfpvIlp/2KaNfVrTTVXbao1NMFpgtMEJwY6MWBjkUpbMHUunEj9OpGynn6up+Zr2nx38XE6Z2Laz5hmXf1ZV81VF+tM2jHrnIljf+KYtfRzLTVX0+a6q4/TeRPXfsY16+rPumqu2lRraILTAO00QCk2/hSbtGtpY5FKWzB1LmkteBz5QAzkFwOar2nz3cXH6ZyJs/ziDGus08SA5qqLdSbtmHXOabx4DHlGDOQXA5qraXPd1cfpvIm1/GINa6zTxIDmqk21hiY4TXCa4MRAJwZsLFJpC6bOJU2x5jGc5ImBfGNA8zVtvrv4OJ0zsZZvrOGNd9IY0Fx1sc6kHbPOOakV9ye/iIF8Y0BzNW2uu/o4nTfxlm+84Y130hjQXLWp1tAEpwHaaYAmDWju718RtLFIpS2YOhfi1L84ZU39W1PN17T57uLjdM7Es3/xzJr6taaaqy7WmbRj1jkTy37FMuvp33pqrqbNdVcfp/Mmpv2LadbUrzXVXLWp1tAEpwlOE5wY6MSAjUUqbcHUuXAi9etEynr6uZ6ar2nz3cXH6ZyJaT9jmnX1Z101V12sM2nHrHMmjv2JY9bSz7XUXE2b664+TudNXPsZ16yrP+uquWpTraEJTgO00wCl2PhTbNKupY1FKm3B1LmkteBx5AMxkF8MaL6mzXcXH6dzJs7yizOssU4TA5qrLtaZtGPWOafx4jHkGTGQXwxorqbNdVcfp/Mm1vKLNayxThMDmqs21Rqa4DTBaYITA50YsLFIpS2YOpc0xZrHcJInBvKNAc3XtPnu4uN0zsRavrGGN95JY0Bz1cU6k3bMOuekVtyf/CIG8o0BzdW0ue7q43TexFu+8YY33kljQHPVplpDE5wGaKcBmjSgub9/RdDGIpW2YOpciFP/4pQ19W9NNV/T5ruLj9M5E8/+xTNr6teaaq66WGfSjlnnTCz7Fcusp3/rqbmaNtddfZzOm5j2L6ZZU7/WVHPVplpDE5wmOE1wYqATAzYWqbQFU+fCidSvEynr6ed6ar6mzXcXH6dzJqb9jGnW1Z911Vx1sc6kHbPOmTj2J45ZSz/XUnM1ba67+jidN3HtZ1yzrv6sq+aqTbWGJjgN0E4DlGLjT7FJu5Y2Fqm0BVPnktaCx5EPxEB+MaD5mjbfXXyczpk4yy/OsMY6TQxorrpYZ9KOWeecxovHkGfEQH4xoLmaNtddfZzOm1jLL9awxjpNDGiu2lRraILTBKcJTgx0YsDGIpW2YOpc0hRrHsNJnhjINwY0X9Pmu4uP0zkTa/nGGt54J40BzVUX60zaMeuck1pxf/KLGMg3BjRX0+a6q4/TeRNv+cYb3ngnjQHNVZtqDU1wGqCdBmjSgOb+/hVBG4tU2oKpcyFO/YtT1tS/NdV8TZvvLj5O50w8+xfPrKlfa6q56mKdSTtmnTOx7Fcss57+rafmatpcd/VxOm9i2r+YZk39WlPNVZtqDU1wmuA0wYmBTgzYWKTSFkydCydSv06krKef66n5mjbfXXyczpmY9jOmWVd/1lVz1cU6k3bMOmfi2J84Zi39XEvN1bS57urjdN7EtZ9xzbr6s66aqzbVGprgNEA7DVCKjT/FJu1a2lik0hZMnUtaCx5HPhAD+cWA5mvafHfxcTpn4iy/OMMa6zQxoLnqYp1JO2adcxovHkOeEQP5xYDmatpcd/VxOm9iLb9YwxrrNDGguWpTraEJThOcJjgx0IkBG4tU2oKpc+HrZ4IBBq7EQNp8d/FxuiZpNpQ8hgsRYiC/GNBcdbHOpB2zzpmv7B+IATdiIG2uu/o4jUvOhfmdC7HGOk0MaK7aVGtogtMA7TRA0wQ1j/GrGNpYpNIWTJ0LX93YvLNOrNN5DJTpP415zqN+nUdZT//WU3O1jPVJ585XztHEgN0xUKb6dD5XjUfOuf6dc1lTv9ZUc9WmGkUTnCY4TXBioBMDNhYpmwomY0EAAQRMCWi9ZbPv12af9fRvPTVXTeU+x0EAAQQQGE9A6zLnXP/OuaypX2uquTpexpt9NE1wGqCdBigFx6+Ck2Y9bSxSZkseR0MAAQTsENB6m6ZW8xjO18RAfjGguWpH5WAUCCCAAAJalzkX5ncuxBrrNDGguWpT1aIJThOcJjgx0IkBG4uUTQWTsSCAAAKmBLTeptlQ8hguRIiB/GJAc9VU7nMcBBBAAIHxBLQucy7M71yINdZpYkBzdbyMN/tomuA0QDsN0DRBzWP8KoY2FimzJY+jIYAAAnYIaL3lPOrXeZT19G89NVftqByMAgEEEEBA6zLnXP/OuaypX2uquWpT1aIJThOcJjgx0IkBG4uUTQWTsSCAAAKmBLTestn3a7PPevq3npqrpnKf4yCAAAIIjCegdZlzrn/nXNbUrzXVXB0v480+miY4DdBOA5SC41fBSbOeNhYpsyWPoyGAAAJ2CGi9TVOreQzna2IgvxjQXLWjcjAKBBBAAAGty5wL8zsXYo11mhjQXLWpatEEpwlOE5wY6MSAjUXKpoLJWBBAAAFTAlpv02woeQwXIsRAfjGguWoq9zkOAggggMB4AlqXORfmdy7EGus0MaC5Ol7Gm300TXBLGqBX72zI7ac7Mv3Dslw1PaaHFalexE1LKu/WOg3PNEHMY/wufjYWKbMlj6MhgAACdghoveW8GnFenXolXz/dkduP3snvpyLul2rftCbPzi5joVE7lqlUxzA9Jo5nYz5ortpRORgFAggggIDWZRvPGbaMif4SeyobYlFz1aaqRRPchouer3Zktalh0ZKXjwwn7OPLFvjFM2xved0Ev3pnT17XWpeYzRP5Xzasr0NjsLFIaWbwFQEEEPBJQOutDRtUO8fwm9zda5/PRaSxtWl4/7Ily52AqsqMQ+fq0eu1IP9xf0tm105l+7QpdZ1nsyWVk5osrWzJR8Z/qWB472rRemiuKiNfEUAAAQSKFdC6PPp86O+5KXLu9JfS7xmnluUvTw9l6bAux2fdfag0m3J8fCpPfn0jv5ssaVyl2JtprhZbMXqfnSZ4ioWMLDhpjnf3UPYC67Kx+DJ90oY9fyma4Ivy+UpNKp1fJpyD+nZRm32xtbFIBVKDbxFAAAFvBLTeGt9ThO0DnPy31zJbCSz34b5cNzoPP5vgE99ty9JJ4KItQNjz7VlNfrw3b3a/aXR9st/zxM09zdUeP35AAAEEEChMQOty3DpeuvvRX0qxv5mXD56cyH5PPyk8xBvVinxz0559is3xrbkaLlnMv9IEt2LD/lp+Orq8YGnUz+Snu4YTyvMm+MS36/Kk7debRjTBkxZEG4tU75ryEwIIIOCHgNbbpHW6TPf/+MXZ5auYmy3ZfPE6xQVN1H7Kzyb4zHYgP5pN2T+sytLmiSzt1OSw3tccP6vKDBdxI+NKczUgy7cIIIAAAgUKaF0u054o2VzpLyXzOt8vBveFLTk+rMqTpZ2Ltyy+vXQkb/teYNA43Jc/WNFLjNrrFn+b5mqB5WLgqWmClyFwvW2CX/627iDw27r60VngVfU0wZMWfxuL1EDV4h8QQAABDwS03iat09zf1IY+eLHjz35heqsllaOKPAj9jJlF+fxtXRqB/Kmu8Vkxo3JKczXAxrcIIIAAAgUKaF0eVb+53dSeqe84XvaXNuVlsyHvVoe9ZdyC/Ol1cA/Vkuc/97mUobeYcI6aqwWWi4GnpgmecBGdLKReFqk5mXhc7V7InRespVW52vMbPH8uavOKOxuL1EDV4h8QQAABDwS03uZV33me/gsVP5vgo9d5WR4eBRKodiyflWEvPMYcNVcDanyLAAIIIFCggNbl0ee8/nM/Pxsx87S/NNrmnfxa6wb+7vKrkX9NNvqYfsek5mpXrfjvaIKPsSl2JqA9LVIfr1x+1FP9pCJ//Vbf17KsF7VmiqeNRar4MskIEEAAAfMCWm+d2Ut4t18q737h+mLgCo7PTxl5Aau5ar4KcEQEEEAAgTQCWpfZQ5npASR29LS/NNphXm7tdCN2b8X0W/UVtJ4Z7vE1V7tqxX9HEzzDBR+dRBrkL+X+fvt9Gps1uT+t/x72dUF+/8vBxafVVjrv7diS6umZrKxsyR+/1GZw4LFhRWrqldxZq8l+5xNvW1I9rsrsv1/JVStMAuMfNp5/7MmzpTW51nN7eS9q48fbcFsbi1TxZZIRIIAAAuYFtN6aqN2+HuP6s9rle4JLS949i/rQ8Hn53d935MnemRwH9jW1Wl021vbkLzMLIY3OsP3Cgnz0a0U2Tprt5xWpn13urz6aGn7udM4/uC+kCR4SG71rrblqvgpwRAQQQACBNAJal507//b0LXrPNWbnQn/JrKeu1Sv55bgbsW/nfxu5h8hmHDoe+79qrnbViv+OJnhuhSgqQF/LbEWDoS6zwz4Yc2pVHgbfAFsfEvzarMuzfy32JmPwYmd7S67980i2A++jHXy4SEsqW1t9jeWosdt2W9hFrW1jtHc8Nhap3vjkJwQQQMAPAa23Zd8cR81f/+LrfMWHvtpm8oX8db3RfXu0sPA4f8u0xTcy0bPn69sv3Fwb8iHblwds1E68+RDJnleC1yvyZY+LvXuUqFjJ8jbN1bDQ4t8QQAABBPIX0LqcZe13+9j0l7JYv4l7R4HPnzuTh//NnmmUs+Zq/lVi+DPSBLdi4x+jSE2+kdnj9qvFz9ez2ZCNzUN58HRHbi8cytJho/OqJTk+kA+D8wo2wc+aUm3HQ/3ktP2Jtwfy7Ch4AdmSjcXl3kZ68HhDv5+XD35qf4Lu+bgS/P/FdyGvYB/6PFHFpu+iNtUxoo7v9202Fqnh5YtbEEAAAXcFtN6O2jyW+fbRTfBFmQn+Vr/Zkt29I3l4sf84kCd7danpL/2bJzLdsycI7heaUj1rx1KzLm9XDy72MA/XzqSijxeRxuG+/KHnGPH2BNfubybaE+n+afp+34saUjz3YPzMy8x2YD+5tyvvGTluPIvB8dj/OM1Vd6sNI0cAAQT8EtC67OI5JZ8x018y11+63KdcvbMnq7pXlJbsrvS/uML+/Uw+sdfroLlqUwWiCW7Fxn90kfrDYq37KqezqtzqvAd2N8jOE/N1rSVSOZSPg/MKNsEvoq8p6xcfItl97JUb8/LJ68v32L64y+mR/Dl4jFjfBy8ok4V5bX09RdM9OH79PjgGPhgzaaGzsUgliyTujQACCLghoPU2aZ0u0/1HNcEnfq50frEvzbr8+8Hg255MfLvefoV3/54guF+4jJnqwYH8qe9tT3pf9dOQJ//Q/Ubcr8E9XsLYPNyX67H2X3HHMidXvtqR1U5jvyWrc/wp76ic0lxNuHrcHQEEEEAgIwGty6Pqd3lvD+49wt9pgP7SkL3T5Ip8EXgx54OVirwKvuBUWrK92v+WvEOOZXoP5+DxNFczKgWpDksT3IpAGlWk1uV5pz/dkF9/iHjV9NQr+Uv/K4d6muBNWZ1f6fuT4HbSTm7IYud56vLLraTJvCbPOr8dSxKPLdldNvWhAsGL2v4L3qTzKd/9bSxSSSKJ+yKAAAKuCGi9Le8F2uhzbHQT/KX8eKir3ZK38xGvmp58IZ9+37/3Ce4XRKp7u/LBZNiY5uWbje4rp3eXXyX8pf1L+f6g+3gdcZyv9a3N8P1a6r3rvHz2ttF96rOKfBk65zCH8v6b5moXju8QQAABBIoU0LrMHmrYuZn+Uur+Uk/vLBjlLTk4rMiD/2fR8N5s2Br68e+aq0HJor+nCZ76QsJkUI4oUj8cS+ctw/tf5R1n/MFE3tmOSNrgp9225PnPJueY17GCF7U0wZNuDGwsUkUXSZ4fAQQQyEJA623SOl2m+0c2wb/albe6MM2qzCRu5vbuF25FPH7iib6RnIi5v1zLa1/UfZ6Je4ey23kVeFOWH0f84iDO/rIk99Fc1XDjKwIIIIBAsQJal8u0J0o2V/pLyby6e6UrDwN/ZTgkzOsnJ3LnzuBfH6Z+To/3U5qrQygL+Wea4FYEXHSRCn6AUaqLr2ATfHsr8hVMM9vdOFx+HCgGVjjFGU/vRe2MM+OOM7fs72NjkepGJN8hgAAC/ghovWXDPPzcFtkE/7kiNQ2HVG8bkmC/kGAfZe16Tr6TX08VTKR6/kHp7JEi98S6lpqrXT2+QwABBBAoUkDrstZpvvbvpegvmYuJBfmPO+/kzuqpHHdeSHD+GX11mb0X8Q4N7LEu9liaq0XWi/7npgluRXBGF6nIi8A4409w8UYTvP8EUq6fbSxS/UWLnxFAAAEfBLTemtuk+3e+itz/JNjbhBuXqAk++UJuBT5AtHF8KJ9EvPI93Mu/+Io7T81VH+oOc0AAAQR8ENC6HLeOl+9+9JeyWPOJb7dlOfj2v7yt3MgXE2iu2lR3aILHaSJnfp/oIjW91X0vyb2VFO+dneBCkSZ4eS/yzk8UNhYpmwomY0EAAQRMCWi9zWKT7ssxo5rgHy53PsREZMRfuYV7lKUJvigzgQa4nFVl5ma59zrh8TDcRHPVVO5zHAQQQACB8QS0Liet5+W5P/2lrNZ64odjOeqEb0uWH/Nq8ChrzdUOmQXf0ATPvME9fFPdDZboIvXJ6+6Fnt1NcD4Ys7umcdbdvvvYWKQsqJMMAQEEEDAuoPXW9fNGluOPaoIH36fb7iZ4kR+MuSjfrDek81GYZzX5kT/dHfmqpf6Y1lw1XgQ4IAIIIIBAKgGty/31mp+1v0B/KfUHY47sD76TXzvvxyeSqj838jl0Hd3/qrmaKtEzehBNcCsCMLpIBS8C7X5P8OCrqpJFbKp5ha5dcAx8MGbSjYCNRSpZJHFvBBBAwA0BrbdJ63SZ7h/c/wxcZAT/ys3q9wQP7vESxmaqeekF07x8slLvNsB578rEzW/NNc3VhKvH3RFAAAEEMhLQuqx1mq967tevwb1HXWbv6r9ffg3ur1L1YYJ7sBF/jTfeOw0EezvJginVvEL7S712V24EbWmCj8o9zdVkq5ftvWmCxwr0/sA3/XMwkQaL1JVHJ92LmMqhfJx0zLkVqXn54Kcduf006f+b8um3pv6MJFgoaYKPKkr9t9tYpLItgRwdAQQQKEZA621/Hebn7h4reJE20AS/dSC7unTNqswkfo/rBPuFBPuosPW7dn8zxd5oR76+s5CycTsvnyzWpNrx4cObwtYl7r9prionXxFAAAEEihXQuhy3jpfvfvSXzPWXuvvSyzhal+fdN2qQjcWXKfdq/cf182fN1WIrRu+z0wRP2lDO5P4jitRXO7La+STahvz6Q0TDePKFfPr9ikwEx5ng4m2839TZkLgJLmqDRnx/UbxtLFK9JYufEEAAAT8EtN6W78Is/l4hsgl+Y1WenGostOTt/GLERciCfPTDqlzrOdcn2C8k2EfZsJ7XHlcDDfCGPP9XlE389bBhbkWMQXNVo42vCCCAAALFCmhdLuKc4MZz0l/Kap3ee1yV7ruhhLyAtWevyR5Lc7XYitH77DTBrQjSEUXqxrx8ud7pgsuwDzW6emdHXlZbIv2vFk9w8UYTvNyFysYi1Vuy+AkBBBDwQ0DrbVabdB+OG90En5M/LNa6fyk35O0+Jr5dl9mD8z1U/1+H+dkEv/bPY9ntbBmbsvyYBvi4uaC56kflYRYIIICA+wJal8et7/4+nv5S0rWd3mpJZf9I/vb333pfUBroF159cCDrZ938aezt9r3Aoty9pDBzzdWuWvHf0QQPBHXYouXzb6OK1Jxcubkly4GEk2ZDNjYP5cH5W48sHMrSYUM6f5VRlib45Ip8MfDWK8ey1cmrM5nru336PheDUTFtY5HqLCffIIAAAh4JaL2Nqsllv21UE/zK5BuZPW51o6LZkt29I3l4ce4/kCd7dal1GsIlaIJPbshiZzMoIrUzWd48kaVR/69scBEXcT2gudoNNL5DAAEEEChSQOty2fdJw+dPf2m4TXijOvhi0Ea9IRs7FZldaL/N79KRvAr2286D/6wqMzfDj5X0uX2+v+ZqkfWi/7lpgkdsevMLxhhF6sacTNzb7/nNU/9iXvzcrMuz/j979fWV4MF5hWKE/OPxgXxoxZrbWTBtLFIhq8g/IYAAAs4LaL3Nb69h53knav4jm+Dn5/ObG/L8/K/gov5rNuTd4pu+V/Z4+Erwu4eyF+Uw9Lb+XxC4FytRcTTubZqrQ/m4AQEEEEAgVwGty+PWd38fT38p6dq+9/Ox7HdeOBEdzvXfDUAfAAAgAElEQVSTivzV2Gfa+b3n0lyNFs33VprgVjREX8r9/fYFXLMm96cjEmFqWb5eOpGNk2b3ld/NllSOT+XJ0oZ8NBXy2P9z0r5vSyrv1iLeM3NOpt412n9a3JRnD0OOZYVXe1wPK933vIyZN429HXnPpjlYNhYbi1TMpeVuCCCAgFMCWm+TbtLLdP/rz2qd/cu7ZxEfPDT5m/zx1yN5e9wMvPK7JdWTmiyt7Az58O0NWWpf7DRqxzIVdT5+eCyV9n2ra9H7qELX56tteR3zAq4nWc4q0fOPsinBbZqrPWb8gAACCCBQmIDW5ULPuVaf/+gvpYqN8/3kLwfyZKcm+7VW9y33pCW1WkO2947kwQ/LctXqtberh6e5WlixCHlimuAEcGRTPFXxwNRZUxuLVEjd4p8QQAAB5wW03nKetWuzznqwHv0xoLnqfNFhAggggIAnAlqX++s1P3MOJwbsigHNVZtKD01wGrbONmwpcOYLnI1FyqaCyVgQQAABUwJabzmXmT+XYYqpyRjQXDWV+xwHAQQQQGA8Aa3LJms9x2LvQAyYjwHN1fEy3uyjaYLTBKcJTgx0YsDGImW25HE0BBBAwA4BrbdsuM1vuDHF1GQMaK7aUTkYBQIIIICA1mWTtZ5jsXcgBszHgOaqTVWLJjgN0E4DlKQ3n/SumdpYpGwqmIwFAQQQMCWg9da18wTjZa9QthjQXDWV+xwHAQQQQGA8Aa3LZTsfMV/2YK7FgObqeBlv9tE0wWmC0wQnBjoxYGORMlvyOBoCCCBgh4DWW9c2s4yXC7CyxYDmqh2Vg1EggAACCGhdLtv5iPmyB3MtBjRXbapaNMFpgHYaoK4lFOM1fxKwsUjZVDAZCwIIIGBKQOst5zLz5zJMMTUZA5qrpnKf4yCAAAIIjCegddlkredY7B2IAfMxoLk6XsabfTRNcJrgNMGJgU4M2FikzJY8joYAAgjYIaD1lg23+Q03ppiajAHNVTsqB6NAAAEEENC6bLLWcyz2DsSA+RjQXLWpatEEpwHaaYCS9OaT3jVTG4uUTQWTsSCAAAKmBLTeunaeYLzsFcoWA5qrpnKf4yCAAAIIjCegdbls5yPmyx7MtRjQXB0v480+miY4TXCa4MRAJwZsLFJmSx5HQwABBOwQ0Hrr2maW8XIBVrYY0Fy1o3IwCgQQQAABrctlOx8xX/ZgrsWA5qpNVYsmOA3QTgPUtYRivOZPAjYWKZsKJmNBAAEETAloveVcZv5chimmJmNAc9VU7nMcBBBAAIHxBLQum6z1HIu9AzFgPgY0V8fLeLOPpglOE5wmODHQiQEbi5TZksfREEAAATsEtN6y4Ta/4cYUU5MxoLlqR+VgFAgggAACWpdN1nqOxd6BGDAfA5qrNlUtmuA0QDsNUJLefNK7ZmpjkbKpYDIWBBBAwJSA1lvXzhOMl71C2WJAc9VU7nMcBBBAAIHxBLQul+18xHzZg7kWA5qr42W82UfTBKcJThOcGOjEgI1FymzJ42gIIICAHQJab13bzDJeLsDKFgOaq3ZUDkaBAAIIIKB1uWznI+bLHsy1GNBctalq0QSnAdppgLqWUIzX/EnAxiJlU8FkLAgggIApAa23nMvMn8swxdRkDGiumsp9joMAAgggMJ6A1mWTtZ5jsXcgBszHgObqeBlv9tE0wWmC0wQnBjoxYGORMlvyOBoCCCBgh4DWWzbc5jfcmGJqMgY0V+2oHIwCAQQQQEDrsslaz7HYOxAD5mNAc9WmqkUTnAZopwFK0ptPetdMbSxSNhVMxoIAAgiYEtB669p5gvGyVyhbDGiumsp9joMAAgggMJ6A1uWynY+YL3sw12JAc3W8jDf7aJrgNMFpghMDnRiwsUiZLXkcDQEEELBDQOuta5tZxssFWNliQHPVjsrBKBBAAAEEtC6X7XzEfNmDuRYDmqs2VS2a4DRAOw1Q1xKK8Zo/CdhYpGwqmIwFAQQQMCWg9ZZzmflzGaaYmowBzVVTuc9xEEAAAQTGE9C6bLLWcyz2DsSA+RjQXB0v480+miY4TXCa4MRAJwZsLFJmSx5HQwABBOwQ0HrLhtv8hhtTTE3GgOaqHZWDUSCAAAIIaF02Wes5FnsHYsB8DGiu2lS1aILb2ACdfCGfPtqR20835I9fzncalN2knJf3v9+U20935Ou//yYTNs6BMYWsm/mi0o0JM8e2sUjZVDAZCwIIIGBKQOut6Tru7/FG732u3tm42BtN/7AsV9mHOLkPsTF+NVdN5T7HQQABBBAYT0Drso3nDCvHRH+JPVFB+2LN1fEy3uyjaYIXFAxRxfHPq43uKh8fyIf9Y/zHkRx17lGXX26ZaYBGjcme2+bld3/fkIerp7J92pRaUyFaUqvVZWPzQL6eWaDI9cdMzJ9tLFK6wnxFAAEEfBLQemvP+dXyvcSovc9XO7Ia2BO8fGT5fGKel+PGx8SXK/KXhSN5dViXSr3VSZX6WUO2947kwfcveNFESnPN1Q4q3yCAAAIIFCqgdTnuObLs96O/lGBPOPlC/mv1TOoXEd6S7Rev6C2l3D+d553maqEFo+/JaYKPsaBZFdOZ7cAq1SvyZf8YH1cDd2jJ858TJHX/sVz6+eaazO43JPArgoBD8NumrC++4WIvxdraWKSCK8v3CCCAgC8CWm+z2kt4d9xRe5+7h7IXCI6NxZcluWhZlM9XalLp/AIggNDzbUsO1jfkWoq9gXexlNBAc7WHkx8QQAABBAoT0Lpc9vNT3PnTX4rXL5v4bluWTrovJDgP8L2V1yXZT8Yzihtzej/N1cKKRcgT0wRPuBHWxczy63uPTtoXMy05Xt+U9/rH+NWmLNUuk7N+WpHpr7IJ2CznmObYH69c/j7uPI4b9bq8XTuUB0/P3zZmT2Y3z/ouAJvy/Oewt5Iph1Ua3/PH2FikQuoW/4QAAgg4L6D1Nm29Lt3jRu59XstPR5d7o0b9TH66W5LzfU/z//yv4s7k7eaJLG2eyNvjZvuVTJouLdld4UUCSXNHc1UV+YoAAgggUKyA1uWk9bys96e/NGpPuCAfLZz29ZMuY5wm+Ci76Ns1V4utGL3PThO8v8HMz9b+puvD5brUT06G/knvxLfbsnwWCPDDfbnOeiZaTxuLVGBF+RYBBBDwRkDrbVkvyJh39EVDbJ9bB7LbrMvKylbo58hMfLcrq8G9UbMm308beu6S7LE0V70pPkwEAQQQcFxA63Lsc2VJzld4pNjfnL/bwEHgz+maDdk+7r73AE3wFKaBfNNctank0AQPLBBFY7wAt8HvvfnTbn41T2Sa9aUJ3o0IvkMAAQSsEdBNoQ3nTsbg/v4nag0n/nUitUDkv53/LdHeIOrYZbhNczVAyLcIIIAAAgUKaF0uwzmIOWa5R3sts5VuINdPKvLXb+cl+A4ENMHH89dc7SoX/x1NcJqkfl0I3TuSg05eVWWG9U20vjYWqc5y8g0CCCDgkYDWWy5uxttc4xfHb12ed99Rjve3TLg31Fz1qPwwFQQQQMBpAa3L7AHi7AG4z/A42ZLl80xoNuTd0qpcbe8PaIKbixnNVZsKDk3whBvh4QlkLlCuPDzuvCd45d1aSBNzTeba7wneqB3L1MAcXnXfF/Ngr/2WIAvy0b8PZeW4KbX2X3s06g3ZWNuTT7/16L2zH510PzizdiyfDdgYXCcPj21jkbKpYDIWBBBAwJSA1ttc9hVenK9G7X1eyv399ocZNWtyP+QtPz58cXa5Rwi8JcjVO1syu30mlbo+tiWHh5Whb73m5nr1vtKJVzUl2wtqrprKfY6DAAIIIDCegNZlN8/Jyc5BRuZIfymkp3a+Di9l+s2R3Pqutx9GE9xcjGqujpfxZh9NE9zGC8PH1e4qb2+FJGz7N1YX9wp7tXPgYqdyKB/fXJMn7Q+L6h448F3zTH6515v4Ropt7raLcmunfRErIrWNDZnIfQzmCkYRa2BjkQpEKt8igAAC3ghovS2i1rv5nAn2PlKX2ZAPxuxe1Jzfviifr571fXBkMLxasr3sy4dI9r4SfGPxZcje0u39S5YxrbkajA6+RwABBBAoTkDrcpa136tj019KtO/p7heFv54bs5+muVpctRh8ZprgYy5qJsXRZJGq1WVXPxCp2ZCtzUN58HRHHqxUZT/w/v9yVpEvJ1NcAN1clemnO3I76f+PVuWaSfvJF/Jfb+vdV4GfVWXmZor5mByTg8eysUgNli3+BQEEEHBfQOttJvsIB88/ox1MNsEbsnusm6CWVI5OZHZhR24vHMrKSfeX6SJNef5zihcJTK7IF0n3RRf3X5MP0uzFRq333UPZ66RMQ379gf3R6HjrGmmudgj5BgEEEECgUAGty0lqeanvS3+JJviovWJGt2uuFlow+p6cJnhGiz1WkTVZpNoL3qhevsl/z7hubsurzntEtmR1LvkHJQV/S9YXWyN+rMmPIX+q3DO+IWtz7f5moOl+IE82T2VX/4xZRBq1U7nX9yctcY7LfebExiI1IpC4GQEEEHBSQOst555uszHawmQTvB0yzYY8n33R91djvX9V1tjbkfeG7EeGjje4j0sUnS15/nNcj/j3+/NqozuK0yP5c9L5lPz+mqtdRL5DAAEEEChSQOvy0PNwyc9bAy7Bfcm47zTQXnif+kv9XsEeF28hF3+/2e94/rPmapH1ov+5aYLbWCANF6nG8bF8PuRV0X9Y0peJizS2NhP9huw8qK8/r3Vffd0fXVE/N09kOtWrnQJv9dJ//FpdXr3alP/7/2/vfl7cuq44gP+hWWWVXaGQXbIohe5LKO2mYEghC0MJuNB24aY1paENJCltfmLHcePEdmLH9jh2bOOZucWJzxuNR0/z9KR3dKX7WZixLc1cvc+953uvDhrp9dUKdV7xtvJ/NYbU89Ps3wQIENgFgcjbVvaX1a9zzU3wgyflg39cnH/u+cNeuRWL7Ol5Zdmz4tsPyswb28VPOv3rwZPyzpy3cVnJ7uzt8lW86L0clq8vfj7/mpe9xobuH7V6+gS6BwECBAhkCEQur7Q/NrSPvaC/tNTZRxN8ff20qNWMXBg6hiZ4jeG31pB6Ut4/v2ARn/+u7MVquXf32YdoLrj/xr0ulwt7s7+qHA9+5uvTT/e9cqO8MqrJXvO1T//YagypmZn1VwIECOyMQOStJ3BD97b1NsH3vri64AnR1fL+97HUxv/m2ubn9mI5d6vrgJf9+/fKa85GC+Z9/lqMWo0V4SsBAgQIbFYgcnnz++z8faO6x6W/tNTerwm+vnUdtbrZxDg+uib4xpu6cxbYmkNq3odDdcE8+z6RTz9Es0aPUx7Ti298Vn71zl659ODw2KvSH966ud73HT/lcXSmW3y/GkPqeGT5FwECBHZDIPJ2F/aOnGtYbxN88a+3zv7W2fwP2cy55jlnxCXOGC9/9OjoXHTwpLy7Ex+CvprJmHmLWt2N5HEVBAgQ2H6ByOUxmd7k9+gvaYIvcX5cZ41ErdaUOprgG1oMCxeWkFoqpI4sPym//mLmwzH92u/SjjWGVE2B6bEQIEBgXQKRt0d7WH5zb7vG1gRfZr5e+ut35Xb3IvCD8vm/e976pcZzcGWPKWp1XbXv5xAgQIDAagKRy8vsi03fV39pqb6IV4Kv7zlJ1OpqFb/e79YEr+yg/UM4C6mlQur4hnaxnL8z83YpW/rq9uPXtL4QOu3n1hhS6408P40AAQJ1CETenpbLbo89UBN86Fp48fy9mQb4Ybl95epzH/4Zpr4OMY1arSM5PAoCBAgQiFwekuHu82FZ73uCn/IbcjvwTgOa4Os7H0at1pRamuCa4OVOrMgRDeP8D8Y8vSB/8tGjuKJSysNyrsY5rvQx1RhSM5PprwQIENgZgchbT85O39d/NNqiJvgGPxjzxfN3y43uM88Py96Nr7013IpnrqjVnQkfF0KAAIEtF4hcdoYaeIbaohdZ1tBf0gQfuK4GnK+iVmuKHE3wAROXHq5bFFKzAbHcwp7wg6Zm/TTBl3pVfY0htdy6cm8CBAhsh0DkbfoZo8Zzz6DHtEVN8GPnkGXW42H59F8rPPE4e7N83jXAS/HZKCtYzqzJqNVlZtJ9CRAgQGA6gchlZ6iB+9zsueTWzTn9gdXPWN1crPhK8Br6S7OPYfFnyAz0nzlTdE6N/F/U6nRpsPxP1gSvcfFtUUi9cPbL8uZ/vy1/WvbPO/8rL01k/+rFx0eV8ORBeWOicXYxwGoMqaPJ9DcCBAjsjkDk7S7uJdNc0+pP0IY/qVnxgzHPXCm/W/Zc9PT+731VXjkz8gnV2a/Lpw+P3g5OA3yk45wzY9Tq7qSPKyFAgMB2C0QuT3PeWN/+Uc3j01+a0/jvn+fh58X+n1HN3M8512Q+tqjVmhJHE3zDi2LuAtymkKrOb/aJayn7t256H8wl5qjGkKopMD0WAgQIrEsg8nbuOWCJ3G7n+7eoCZ49f2eulnfvzzTA9+6W18Y207Mf+xaMF7W6rtr3cwgQIEBgNYHI5XbOQCs2W/WXNME3dN6KWl2t4tf73ZrgG1oMCwNbSJ0MqT/uldsHT8qlT66Vn73eswmc+az8/sZ+2e9qZL988PePTv6sGue8ksdUY0h10+kvBAgQ2CGByNuF54FK9oY6HqMm+Nx5OHOlnL9z0FXG/v17GuBrrpuo1Q7ZXwgQIEBgowKRy3P3xTXvATsxhv7SUj0hrwTv6beNqK2o1Y0GxnODa4KPmMjJg1BInQyp2feWKofl4f1H5dKXe8/ehuVOeff6o3L7ydGroEo5LLevXPUq8CXXd40h9Vxm+ScBAgR2QiDydvIzxZL7QL2PRxN83tz8/NLMm4CXw3L3zvfl0jen/XlQ/nLeiwTmec77v6jVnQgeF0GAAIEdEIhcnpfZ/m9OA1N/6WR/6dn5+OW3vjnx1r7//PboZZUPvo2eU7wF8PXyC79t1+v5fP1FrdYUO5rgNT45FFJziuqLcuHu7Ku8F5TRwX65dunLyd5z/PnC3qV/1xhSC2baTQQIENhagcjbXdpDpr0WTfB5vudujSsBH/Q0p0nQ85wganWctO8iQIAAgXULRC7P2xf935z9TX9pTn/pqdPxt9Idtk4Py+X35hj3nCFaX49Rq8Nsc+6lCV7jYn37fnnww2+2HpYH167PKdjr5cPHP77qef/x/fL6iWu4XN66++xV0QePy1tvLijSN2+Xa89+i3Z/70559cTPWvC9G7jvS+eulT9feVC+ur9fHh799m8pBwfl/vePy6UrN8sv3/DqprFhW2NI5UShUQgQIJArEHk7Nq/b+77Vzz6vfvy4PPlhmg/LtY8vzzlfxZnn+DnqwqJz1AbOQrNz/5svBr5A4NjyPiw3Pl10/eHg61PrqNVjhP5BgAABAhsTiFye3Q/9fcGerb/Uc+a7XC7szb6bwJAlvV/+87cF1hs+F9ZWB1GrQ2Sz7qMJbpH2BILCri1AMh5PjSGVFYbGIUCAQKZA5G1GthvDmcYaGL8GolYz88FYBAgQINAvELlsbxu/t7Fjl7EGolb7qzn/Fk1wTXBNcGugWwM1hlR+LBqRAAEC0wtE3mYcQI3hiY41MH4NRK1OnwpGIECAAIEhApHL9rbxexs7dhlrIGp1SF1n3UcTXAO0a4BmFIEx6g7bGkMqKwyNQ4AAgUyByFv7Yt37ovkxP1GrmflgLAIECBDoF4hctkfbo62ButdA1Gp/NeffogmuCa4Jbg10a6DGkMqPRSMSIEBgeoHIW4f3ug/v5sf8RK1OnwpGIECAAIEhApHL9mh7tDVQ9xqIWh1S11n30QTXAO0aoAKk7gDJmJ8aQyorDI1DgACBTIHI24xsN4b93RoYvwaiVjPzwVgECBAg0C8QuWxvG7+3sWOXsQaiVvurOf8WTXBNcE1wa6BbAzWGVH4sGpEAAQLTC0TeZhxAjeGJjjUwfg1ErU6fCkYgQIAAgSECkcv2tvF7Gzt2GWsganVIXWfdRxNcA7RrgGYUgTHqDtsaQyorDI1DgACBTIHIW/ti3fui+TE/UauZ+WAsAgQIEOgXiFy2R9ujrYG610DUan8159+iCa4JrgluDXRroMaQyo9FIxIgQGB6gchbh/e6D+/mx/xErU6fCkYgQIAAgSECkcv2aHu0NVD3GohaHVLXWffRBNcA7RqgAqTuAMmYnxpDKisMjUOAAIFMgcjbjGw3hv3dGhi/BqJWM/PBWAQIECDQLxC5bG8bv7exY5exBqJW+6s5/xZNcE1wTXBroFsDNYZUfiwakQABAtMLRN5mHECN4YmONTB+DUStTp8KRiBAgACBIQKRy/a28XsbO3YZayBqdUhdZ91HE1wDtGuAZhSBMeoO2xpDKisMjUOAAIFMgchb+2Ld+6L5MT9Rq5n5YCwCBAgQ6BeIXLZH26OtgbrXQNRqfzXn36IJrgmuCW4NdGugxpDKj0UjEiBAYHqByFuH97oP7+bH/EStTp8KRiBAgACBIQKRy/Zoe7Q1UPcaiFodUtdZ99EE1wDtGqACpO4AyZifGkMqKwyNQ4AAgUyByNuMbDeG/d0aGL8GolYz88FYBAgQINAvELlsbxu/t7Fjl7EGolb7qzn/Fk1wTXBNcGugWwM1hlR+LBqRAAEC0wtE3mYcQI3hiY41MH4NRK1OnwpGIECAAIEhApHL9rbxexs7dhlrIGp1SF1n3UcTXAO0a4BmFIEx6g7bGkMqKwyNQ4AAgUyByFv7Yt37ovkxP1GrmflgLAIECBDoF4hctkfbo62ButdA1Gp/NeffogmuCa4Jbg10a6DGkMqPRSMSIEBgeoHIW4f3ug/v5sf8RK1OnwpGIECAAIEhApHL9mh7tDVQ9xqIWh1S11n30QTXAO0aoAKk7gDJmJ8aQyorDI1DgACBTIHI24xsN4b93RoYvwaiVjPzwVgECBAg0C8QuWxvG7+3sWOXsQaiVvurOf8WTXBNcE1wa6BbAzWGVH4sGpEAAQLTC0TeZhxAjeGJjjUwfg1ErU6fCkYgQIAAgSECkcv2tvF7Gzt2GWsganVIXWfdRxNcA7RrgGYUgTHqDtsaQyorDI1DgACBTIHIW/ti3fui+TE/UauZ+WAsAgQIEOgXiFy2R9ujrYG610DUan8159+iCa4JrgluDXRroMaQyo9FIxIgQGB6gchbh/e6D+/mx/xErU6fCkYgQIAAgSECkcv2aHu0NVD3GohaHVLXWffRBNcA7RqgAqTuAMmYnxpDKisMjUOAAIFMgcjbjGw3hv3dGhi/BqJWM/PBWAQIECDQLxC5bG8bv7exY5exBqJW+6s5/xZNcE1wTXBroFsDNYZUfiwakQABAtMLRN5mHECN4YmONTB+DUStTp8KRiBAgACBIQKRy/a28XsbO3YZayBqdUhdZ91HE1wDtGuAZhSBMeoO2xpDKisMjUOAAIFMgchb+2Ld+6L5MT9Rq5n5YCwCBAgQ6BeIXLZH26OtgbrXQNRqfzXn36IJrgmuCW4NdGugxpDKj0UjEiBAYHqByFuH97oP7+bH/EStTp8KRiBAgACBIQKRy/Zoe7Q1UPcaiFodUtdZ99EE1wDtGqACpO4AyZifGkMqKwyNQ4AAgUyByNuMbDeG/d0aGL8GolYz88FYBAgQINAvELlsbxu/t7Fjl7EGolb7qzn/ltQmeAD4+tvCgEHNayA/ioxIgACBtgRq3gM8NmcUa+DkGmgroVwtAQIE6hWwR53co5gwqXkN1JQmmuA/VSw1F4vHtpn1WVNIeSwECBDYRQH722b2N+7cx66BXcwh10SAAIFtFBib477PGcAa2MwaqClnUprgNV2wx0KAAAECBAgQIECAAAECBAgQIECAAAEC7Qhogrcz166UAAECBAgQIECAAAECBAgQIECAAAECzQlogjc35S6YAAECBAgQIECAAAECBAgQIECAAAEC7Qhogrcz166UAAECBAgQIECAAAECBAgQIECAAAECzQlogjc35S6YAAECBAgQIECAAAECBAgQIECAAAEC7Qhogrcz166UAAECBAgQIECAAAECBAgQIECAAAECzQlogjc35S6YAAECBAgQIECAAAECBAgQIECAAAEC7Qhogrcz166UAAECBAgQIECAAAECBAgQIECAAAECzQlogjc35S6YAAECBAgQIECAAAECBAgQIECAAAEC7Qhogrcz166UAAECBAgQIECAAAECBAgQIECAAAECzQlogjc35S6YAAECBAgQIECAAAECBAgQIECAAAEC7Qhogrcz166UAAECBAgQIECAAAECBAgQIECAAAECzQlogjc35S6YAAECBAgQIECAAAECBAgQIECAAAEC7Qhogrcz166UAAECBAgQIECAAAECBAgQIECAAAECzQlogjc35S6YAAECBAgQIECAAAECBAgQIECAAAEC7Qhogrcz166UAAECBAgQIECAAAECBAgQIECAAAECzQlogjc35S6YAAECBAgQIECAAAECBAgQIECAAAEC7Qhogrcz166UAAECBAgQIECAAAECBAgQIECAAAECzQlogjc35S6YAAECBAgQIECAAAECBAgQIECAAAEC7Qhogrcz166UAAECBAgQIECAAAECBAgQIECAAAECzQlogjc35S6YAAECBAgQIECAAAECBAgQIECAAAEC7Qhogrcz166UAAECBAgQIECAAAECBAgQIECAAAECzQlogjc35S6YAAECBAgQIECAAAECBAgQIECAAAEC7Qhogrcz166UAAECBAgQIECAAAECBAgQIECAAAECzQlogjc35S6YAAECBAgQIECAAAECBAgQIECAAAEC7Qhogrcz166UAAECBAgQIECAAAECBAgQIECAAAECzQlogjc35S6YAAECBAgQIECAAAECBAgQIECAAAEC7Qhogrcz166UAAECBAgQIECAAAECBAgQIECAAAECzQlogjc35S6YAAECBAgQIECAAAECBAgQIECAAAEC7U7c6D0AAABDSURBVAhogrcz166UAAECBAgQIECAAAECBAgQIECAAAECzQlogjc35S6YAAECBAgQIECAAAECBAgQIECAAAEC7Qj8H01SMkgZbelTAAAAAElFTkSuQmCC)

# In[ ]:


Ks = np.array([[-0.05+0.06j, -0.  -0.13j, -0.07-0.15j,  0.11+0.28j, -0.05-0.18j],
               [-0.1 -0.19j, -0.3 -0.05j, -0.28+0.07j, -0.25+0.28j, -0.11-0.29j],
               [ 0.21-0.18j, -0.08-0.14j,  0.03+0.2j , -0.23+0.24j, -0.06+0.32j],
               [-0.29-0.31j,  0.12+0.09j,  0.08-0.02j,  0.31+0.12j, -0.22-0.18j],
               [-0.18-0.06j,  0.08-0.21j,  0.25-0.18j, -0.26-0.1j ,  0.13+0.1j ]])


# In[ ]:


V, S, Uh = np.linalg.svd(Ks)
print(V)
print(S)
print(Uh)


# In[ ]:


miller = MillerBuilder(couplerConv='LC', verbose=False)
theta_phi1, sTerms, theta_phi2 = miller.ConvertKToMZI(Ks)


# In[ ]:


freq = rf.Frequency(start=40, stop=50, npoints=51, unit='mhz', sweep_type='lin')


# In[ ]:


def MZIBuilder(theta, phi, freq, loc):
    return BuildMZISParams(theta, phi, freq)
def AttBuilder(t, freq, loc):
    return BuildAttenSParams(t, freq)


# In[ ]:


cirTria1 = BuildTriangleCircuit(theta_phi1, freq, MZIBuilder=MZIBuilder, label="1")


# In[ ]:


netTria1 = cirTria1.network
TTria1 = netTria1.s[0, 5:, :5]
np.allclose(TTria1, Uh)


# In[ ]:


cirTria2 = BuildTriangleCircuit(theta_phi2, freq, MZIBuilder=MZIBuilder, label="2")


# In[ ]:


netTria2 = cirTria2.network
TTria2 = netTria2.s[0, 5:, :5]
np.allclose(TTria2, V)


# In[ ]:


cirAtt = BuildVectorAttenuatorCircuit(S, freq, AttBuilder=AttBuilder)


# In[ ]:


netAtt = cirAtt.network
TAtt = netAtt.s[0, 5:, :5]
np.allclose(TAtt, np.diag(S))


# In[ ]:


millerCir = CascadeCircuits([cirTria1, cirAtt, cirTria2])


# In[ ]:


T = millerCir.network.s[0, 5:, :5]


# In[ ]:


np.allclose(Ks, T)


# # Experimental Conversions

# In[ ]:


def ConvertThetaPhiToPT(theta_phi):
    """
    Converts an MZI theta_phi array with shape (NN, NN, 2) to multiplier 
    PT (ie Phase-Transmission) array with shape (NN, NN, 2, 2)  where 
    PT[2,3,0] = (p, t) is the phase and transmission of the top multiplier
    at position (i_ch, i_in) = (2, 3).  PT[2,3,1] would correspond to the bottom
    multiplier.
    """
    NN = theta_phi.shape[0]
    PT = np.zeros(shape=(NN, NN, 2, 2))
    for i in range(NN):
        for j in range(NN):
            theta, phi = theta_phi[i,j]
            P1 = phi + theta
            T1 = 1.
            P2 = phi - theta
            T2 = 1.
            PT[i, 0] = (P1, P2)
            PT[i, 1] = (T1, T2)
    return PT


# In[ ]:


def ConvertSToPT(sTerms):
    """
    Converts an MZI S vector with shape (NN) to multiplier 
    PT (ie Phase-Transmission) array with shape (NN, 2)  where 
    PT[2] = (p, t) is the phase and transmission of the third multiplier.
    """
    NN = sTerms.shape[0]
    PT = np.zeros(shape=(NN, 2))
    for i in range(NN):
        P = 0.
        T = sTerms[i]
        PT[i, 0] = P
        PT[i, 1] = T
    return PT


# In[ ]:


miller = MillerBuilder(couplerConv='LC', verbose=False)
theta_phi1, sTerms, theta_phi2 = miller.ConvertKToMZI(Ks)


# In[ ]:


PT1 = np.zeros(shape=(NN, NN, 2, 2))


# In[ ]:


ConvertThetaPhiToPT(theta_phi1);


# In[ ]:


ConvertSToPT(sTerms)

