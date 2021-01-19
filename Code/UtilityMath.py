#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


from scipy.optimize import root
from scipy.interpolate import interp2d


# In[ ]:


from colorize import colorizeComplexArray


# In[ ]:


import bokeh
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
output_notebook()
bokeh.io.curdoc().theme = 'dark_minimal'


# # Plot Complex Array

# In[ ]:


def plotComplexArray(array, maxRad=10):
    pixArray = colorizeComplexArray(array+0.00001j, centerColor='black', maxRad=maxRad)
    (h,w) = array.shape
    img = np.zeros((h,w), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape(h,w,4)
    view[:,:,0] = pixArray[:,:,0]
    view[:,:,1] = pixArray[:,:,1]
    view[:,:,2] = pixArray[:,:,2]
    view[:,:,3] = 255
    p = figure(x_range=(0,w), y_range=(0,h), plot_width=800, plot_height=800)
    p = figure()
    p.image_rgba(image=[img], x=0, y=0, dw=w, dh=h)
    show(p)


# In[ ]:


data = np.random.uniform(low=-10, high=10, size=(10,15)) + 1j*np.random.uniform(low=-10, high=10, size=(10,15))
data[:3,:3] = 0
plotComplexArray(data, maxRad=10)


# In[ ]:


nx, ny = (1000, 1000)
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
xv, yv = np.meshgrid(x, y)
data = xv + 1j*yv
data = np.where(np.abs(data) > 1, 0, data)
plotComplexArray(data, maxRad=1)


# In[ ]:





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


data = RandomComplexCircularMatrix(10, (100,100))
plotComplexArray(data, maxRad=10)


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


# In[ ]:


data = RandomComplexGaussianMatrix(10, (100,100))
plotComplexArray(data, maxRad=3*10)


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


# # Complex Gaussian Filter

# In[ ]:


def gaussian_filter_complex(b, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0):
    """
    See scipy.ndimage.gaussian_filter for documentation.
    """
    r = gaussian_filter(np.real(b), sigma, order, output, mode, cval, truncate)
    i = gaussian_filter(np.imag(b), sigma, order, output, mode, cval, truncate)
    return r + 1j*i


# # Complex Valued 2D PCA

# Adapted from https://datascience.stackexchange.com/questions/75733/pca-for-complex-valued-data

# In[ ]:


def complex2DPCA(dataSet, nComps):
    nSamples = len(dataSet)
    dataShape = dataSet[0].shape
    compShape = (nComps, ) + dataShape
    matrix = dataSet.reshape(nSamples, -1)
    _, s, vh = np.linalg.svd(matrix, full_matrices=False)
    compsFlat = vh[:nComps]
    comps = compsFlat.reshape(compShape)
    FAve = np.mean(dataSet, axis=0)
    baseWeights = complex2Dlstsq(comps, FAve)
    return comps, baseWeights


# In[ ]:


def complex2Dlstsq(comps, field):
    nComps, nyC, nxC = comps.shape
    fieldFlat = field.flatten()
    compsFlat = comps.reshape(nComps, nyC*nxC)
    weights = np.linalg.lstsq(compsFlat.T, fieldFlat, rcond=None)[0]
    weights = weights.reshape(-1,1,1)
    return weights


# Below we create a data set consisting of 4 terms, each with their own weighting.

# In[ ]:


ns, nx, ny = (10, 150, 100)
xs = np.linspace(-1, 1, nx)
ys = np.linspace(-1, 1, ny)
xg, yg = np.meshgrid(xs, ys)
kx1, ky1 = [2,-4]
kx2, ky2 = [1, 3]
dataSet = np.zeros(shape=(ns, ny, nx), dtype=np.complex)
for i in range(ns):
    c1 = 1j + RandomComplexGaussianMatrix(0.1, size=1)
    c2 = RandomComplexGaussianMatrix(0.1, size=1)
    F1 = np.exp(1j*(kx1*xg + ky1*yg))
    F2 = np.exp(1j*(kx2*xg + ky2*yg))
    F3 = RandomComplexGaussianMatrix(0.1, size=F1.shape)
    F4 = np.full_like(F1, fill_value=(0.3+0.2j))
    dataSet[i] = c1*F1 + c2*F2 + F3 + 0*F4


# In[ ]:


plotComplexArray(dataSet[0], maxRad=1.5)


# In[ ]:


comps, baseWeights = complex2DPCA(dataSet, nComps=4)


# In[ ]:


prototype = np.sum(comps*baseWeights, axis=0)


# In[ ]:


plotComplexArray(prototype, maxRad=1.5)


# In[ ]:


aveF = np.average(dataSet, axis=0)


# In[ ]:


plotComplexArray(prototype-aveF, maxRad=0.1)


# In[ ]:


target = dataSet[0]
weights = complex2Dlstsq(comps, target)
fit = np.sum(comps*weights, axis=0)
plotComplexArray(fit, maxRad=1.5)


# In[ ]:


plotComplexArray(fit-target, maxRad=.5)


# # Interpolated Root Finding

# In[ ]:


nx, ny = (93, 93)
xs = np.linspace(0, 1023, nx+1, endpoint=True)
ys = np.linspace(0, 1023, ny+1, endpoint=True)
xg, yg = np.meshgrid(xs, ys)
kx1, ky1 = [2*(2*np.pi)/1023, 0.3*(2*np.pi)/1023]
F1 = np.exp(1j*(kx1*xg + ky1*yg))
F2 = yg/1023
F3 = 0.7*np.exp(-((xg-500)**2 / 300**2))
F4 = RandomComplexGaussianMatrix(0.1, size=F1.shape)
dataSet = 6*F1*F2*(1-F3)+F4


# In[ ]:


plotComplexArray(dataSet, maxRad=5.5)


# In[ ]:


rF = interp2d(xs, ys, np.real(dataSet), kind='linear', bounds_error=True)
iF = interp2d(xs, ys, np.imag(dataSet), kind='linear', bounds_error=True)

def FRoot(xv, *args):
    x, y = xv
    targ, = args
    rO, = rF(x, y, assume_sorted=True) - np.real(targ)
    iO, = iF(x, y, assume_sorted=True) - np.imag(targ)
    return np.array([rO, iO])


# In[ ]:


startPos = [0, 512]
target = 0.9+0.2j
try:
    soln = root(FRoot, startPos, args=target)
except ValueError:
    print("starting over at [512, 512]")
    soln = root(FRoot, [512, 512], args=target)
np.round(soln['x']).astype(np.int)


# In[ ]:




