#!/usr/bin/env python
# coding: utf-8

# # Four Point Inverse Interpolation

# In[ ]:


import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
from scipy.optimize import root


# In[ ]:


def ReIm(z):
    return (np.real(z), np.imag(z))


# In[ ]:


ReIm(np.array(0.4+0.2j))


# In[ ]:


ReIm(np.array([0.1+0.2j, 0.3+0.4j]))


# Suppose we have a function $f(u,v) \rightarrow z$, which is expensive to evaluate.  We wish to find, $(u_t, v_t)$ such that $f(u_t, v_t) = z_t$.  We have evaluated this function to yield
# $$
# \begin{pmatrix}
# z_{00} & z_{01}\\
# z_{10} & z_{11}
# \end{pmatrix}
# $$
# which were obtained by examining points $(u,v)$:
# $$
# \begin{pmatrix}
# (0, 0) & (0, 1)\\
# (1, 0) & (1, 1)
# \end{pmatrix}
# $$
# It is assumed that $0 < u_t < 1$ and $0 < v_t < 1$.

# There are many interpolation schemes which can be used on this data.  We will employ that which is commonly used in rectangular finite elements with a linear basis.  Supporting documentation can be found here (https://www.geophysik.uni-muenchen.de/~igel/Lectures/NMG/08_finite_elements_basisfunctions.pdf).  I will summarize below.  The basis functions are
# $$
# N_1(u,v) = (1-u)(1-v)\\
# N_2(u,v) = (1-u)\ v\\
# N_3(u,v) = u\ (1-v)\\
# N_4(u,v) = u\ v
# $$
# Note that each of these functions is zero at three out of the four points $(u,v) = [ (0,0), (1,0), (0,1), (1,1)]$, while the remaining non-zero term value is one.  An interpolation scheme can be devised as
#     $$F(u,v) :=  z_{0,0}N_1 + z_{0,1}N_2 + z_{1,0}N_3 + z_{1,1}N_4$$
# where the value of $F$ will approach the measured value at of the three corners.  At the center at $(u,v) = (1/2, 1/2)$, the value of $F$ will be the average of the four measured points.

# We will assume that $f(u,v) \approx F(u,v)$ and find $(u,v)$ such that $F(u,v) = z_t$.  The stategy will be to separate the equation into its real and imaginary parts, and solve them together.  In other words, solve:
# $$
# \mathrm{re}(F(u,v)-z_t) = 0\\
# \mathrm{im}(F(u,v)-z_t) = 0
# $$

# In[ ]:


u, v, Z, R, I = sp.symbols('u, v, Z, R, I')
z00, z01, z10, z11 = sp.symbols('z00, z01, z10, z11')
r00, r01, r10, r11 = sp.symbols('r00, r01, r10, r11')
i00, i01, i10, i11 = sp.symbols('i00, i01, i10, i11')


# In[ ]:


def N1(u,v):
    return (1-u)*(1-v)
def N2(u,v):
    return (1-u)*(v)
def N3(u,v):
    return (u)*(1-v)
def N4(u,v):
    return (u)*(v)


# In[ ]:


eq1 = z00*N1(u,v) + z01*N2(u,v) + z10*N3(u,v) + z11*N4(u,v) - Z


# In[ ]:


eq1r = r00*N1(u,v) + r01*N2(u,v) + r10*N3(u,v) + r11*N4(u,v) - R
eq1i = i00*N1(u,v) + i01*N2(u,v) + i10*N3(u,v) + i11*N4(u,v) - I


# In[ ]:


solns = sp.solve([eq1r, eq1i], [u, v], dict=True)


# In[ ]:


len(solns)


# In[ ]:


sol1, sol2 = solns


# The resulting equation provides two solutions, both containing roots to possibly generate complex numbers.  This makes sense.  Suppose that we were looking for $z_t = 1/2 + i/2$ with the following sampled points: $[(0, 1, i, 0)]$.  There would be a "circle" surrounding the second point where $\mathrm{re}(F(u,v)) = 1/2$ and there would be a circle surrounding the third point such that $\mathrm{im}(F(u,v)) = 1/2$.  These two circles will either intersect at two locations are not at all.  In the case that they don't intersect, we will likely find that $(u, v)$ are complex.

# Below we can see that just one of these four expressions is complicated and that there are divide by zero conditions that could occur.

# Using SymPy's lmabdify, we can turn these expressions into functions.

# In[ ]:


invFSol1 = lambdify([[[r00, r01, r10, r11], [i00, i01, i10, i11]], [R, I]], [sol1[u],sol1[v]])
invFSol2 = lambdify([[[r00, r01, r10, r11], [i00, i01, i10, i11]], [R, I]], [sol2[u],sol2[v]])


# In[ ]:


zPoints = [0, 1.1, -.1+1.3j, 1+1j]
zTarg = 0.51+0.51j


# In[ ]:


invFSol1(ReIm(zPoints), ReIm(zTarg))


# In[ ]:


invFSol2(ReIm(zPoints), ReIm(zTarg))


# Given that we have two solutions, it follows that we must choose one or none of them to return.  This is done with the function below.

# In[ ]:


def GetInterSolnAnal(zs, ZTarg):
    (u1, v1) = invFSol1(ReIm(zPoints), ReIm(zTarg))
    (u2, v2) = invFSol2(ReIm(zPoints), ReIm(zTarg))
    good1 = good2 = True
    if abs(np.imag(u1)) + abs(np.imag(v1)) > 0:
        good1 = False
    if abs(np.imag(u2)) + abs(np.imag(v2)) > 0:
        good2 = False
    if not (0 <= u1 <= 1 and 0 <= v1 <= 1):
        good1 = False
    if not (0 <= u2 <= 1 and 0 <= v2 <= 1):
        good2 = False
    if good1:
        return (u1, v1)
    elif good2:
        return (u2, v2)
    else:
        raise Exception('No good solution found')


# In[ ]:


GetInterSolnAnal(zPoints, zTarg)


# Given the expense of evaluating the analytical inverse of the function above, it begs the question whether this should all just be done numerically.  Below I do just that.
# 
# SciPy requires that you provide provide a function which takes only the optimization parameters as input in vector format.  First we define a function which generates such a function

# In[ ]:


def GetRootFunc(zs, ZTarg):
    z00, z01, z10, z11 = zs
    def F(uv):
        u, v = uv
        r = np.real(z00*N1(u,v) + z01*N2(u,v) + z10*N3(u,v) + z11*N4(u,v) - ZTarg)
        i = np.imag(z00*N1(u,v) + z01*N2(u,v) + z10*N3(u,v) + z11*N4(u,v) - ZTarg)
        return (r,i)
    return F


# In[ ]:


FRoot = GetRootFunc(zPoints, zTarg)


# Then we use `root` to search for the target value beginning at the center of the range.

# In[ ]:


soln = root(FRoot, [0.5, 0.5])
soln


# Similar to the analytical technique above, we must check that a solution was found at that it is in the range.

# In[ ]:


def GetInterSolnNum(zs, ZTarg):
    FRoot = GetRootFunc(zPoints, zTarg)
    soln = root(FRoot, [0.5, 0.5])
    good = True
    if not soln['success']:
        raise Exception('No good solution found')
    (u, v) = soln['x']
    if not (0 <= u <= 1 and 0 <= v <= 1):
        raise Exception('No good solution found')
    return (u, v)


# In[ ]:


GetInterSolnNum(zPoints, zTarg)


# In[ ]:




