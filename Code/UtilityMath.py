#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# # Random Complex Matrices

# In[ ]:


def RandomComplexCircularMatrix(r, size):
    """
    Generates a matrix random complex values where each value is
    within a circle of radius `r`.  Values are evenly distributed
    by area.
    """
    rMat = np.random.uniform(0, 1, size=size)**0.5 * r
    pMat = np.random.uniform(0, 2*np.pi, size=size)
    cMat = rMat*(np.cos(pMat) + 1j*np.sin(pMat))
    return cMat


# In[ ]:


RandomComplexCircularMatrix(0.3, (2,2))


# In[ ]:


def RandomComplexGaussianMatrix(sigma, size):
    """
    Generates a matrix random complex values where each value is
    within a circle of radius `r`.  Values are evenly distributed
    by area.
    """
    reMat = np.random.normal(0, sigma, size=size)
    imMat = np.random.normal(0, sigma, size=size)
    cMat = reMat + 1j*imMat
    return cMat


# In[ ]:


RandomComplexGaussianMatrix(0.3, (2,2))


# # Passivity

# In[ ]:


def RescaleToUnitary(arr):
    u, s, v = np.linalg.svd(arr)
    maxS = max(s)
    rescaledArr = arr/maxS
    return rescaledArr


# In[ ]:


arr = RandomComplexCircularMatrix(10, size=(5,5))


# In[ ]:


RescaleToUnitary(arr)


# # ReIm

# The purpose of `ReIm(z)` is to provide similar functionality to Mathematica's `ReIm[z]` function.  It breaks `z` into a real and imaginary component and returns both.  It operates on both scalars and numpy arrays.

# In[ ]:


def ReIm(z):
    return (np.real(z), np.imag(z))


# In[ ]:


ReIm(np.array(0.4+0.2j))


# In[ ]:


ReIm(np.array([0.1+0.2j, 0.3+0.4j]))


# In[ ]:


ReIm(np.array(100))


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

