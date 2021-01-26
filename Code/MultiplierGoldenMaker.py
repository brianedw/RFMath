#!/usr/bin/env python
# coding: utf-8

# # Multiplier Golden Sample Maker

# #### Stock Imports

# In[ ]:


import os, sys


# In[ ]:


import numpy as np
import scipy as sp
import itertools


# In[ ]:


import PIL


# In[ ]:


from scipy.ndimage import gaussian_filter
from scipy import interpolate


# In[ ]:


import bokeh
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
output_notebook()
from bokeh.palettes import Dark2
bokeh.io.curdoc().theme = 'dark_minimal'
palette = Dark2[8]*10


# In[ ]:


palette = Dark2[8]*10
colors = itertools.cycle(palette)


# #### Custom Imports

# In[ ]:


from colorize import colorizeComplexArray


# ## Library

# In[ ]:





# Stolen from https://datascience.stackexchange.com/questions/75733/pca-for-complex-valued-data

# In[ ]:





# In[ ]:





# ## Work

# ### Data Import

# In[ ]:


os.getcwd()


# In[ ]:


def pathData(i):
    return"../GoldenSamples/MultiplierSamples/LMC6492_64by64_"+str(i)+".txt"


# In[ ]:


nSamples = 5


# In[ ]:


dataSet = np.array([np.loadtxt(pathData(i+1), dtype=np.complex) 
                    for i in range(nSamples)])


# In[ ]:


np.max(abs(dataSet[0]))


# In[ ]:


dataSet[0]


# In[ ]:


plotComplexArray(dataSet[0], maxRad=6)


# In[ ]:


golden = np.average(dataSet, axis=0)


# In[ ]:


plotComplexArray(golden, maxRad=10)


# ### PCA

# In[ ]:


trainingData = dataSet[0:-1]
trainingData = [gaussian_filter_complex(d, sigma=1, mode='nearest') for d in trainingData]
trainingData = np.array(trainingData)
nSamples = len(trainingData)


# In[ ]:


dataFlat = trainingData.reshape((len(trainingData), -1))


# In[ ]:


nCors = 4


# In[ ]:


pca = ComplexPCA(n_components=nCors)
pca.fit(dataFlat)
pcaComps = pca.components_.reshape(nSamples,64,64)[:nCors]
constComp = np.full_like(pcaComps[0], 1+0j)
basisRough = np.concatenate([ pcaComps, [constComp]])
basis = [b/np.average(np.abs(b)) for b in basisRough]
# basis = [gaussian_filter_complex(b, sigma=1, mode='nearest') for b in basis]
basis = np.array(basis)


# In[ ]:


plotComplexArray(basis[3], maxRad=4)


# In[ ]:


deviceID = 4
device = dataSet[deviceID]


# In[ ]:


weights = np.linalg.lstsq(basis.reshape(len(basis),-1).T, 
                          device.flat, 
                          rcond=None)[0]
weights


# In[ ]:


fit = (basis.T @ weights).reshape((64,64)).T


# In[ ]:


plotComplexArray(fit, maxRad=6)


# In[ ]:


plotComplexArray(fit - device, maxRad=0.1)


# In[ ]:


np.max(np.abs(fit - device))


# In[ ]:


errors = []
weightsList = []
for device in dataSet:
    weights = np.linalg.lstsq(basis.reshape(len(basis),-1).T, 
                          device.flat, 
                          rcond=None)[0]
    weightsList.append(weights)
    fit = (basis.T @ weights).reshape((64,64)).T
    aveLinError = np.average(np.abs(fit - device))
    errors.append(aveLinError)
errors


# In[ ]:


print(np.round(np.abs(weightsList),3))


# In[ ]:


def increaseResolution(inputArray, xs, ys, xsNew, ysNew):
    zR = np.real(inputArray)
    zI = np.imag(inputArray)
    fR = interpolate.interp2d(xs, ys, zR, kind='cubic')
    fI = interpolate.interp2d(xs, ys, zI, kind='cubic')
    znew = fR(xsNew, ysNew) + 1j*fI(xsNew, ysNew)
    return znew


# In[ ]:


increaseResolution(fit)


# In[ ]:


np.arange(0, 1024, 11)


# In[ ]:


def intDATA(Tinput):
    """
    Tinput: the required input data to be interpolated
    returns the interpolated data and the point new mesh
    """
    zR = np.real(Tinput)
    zI = np.imag(Tinput)
    st = int(1024/np.sqrt(np.size(Tinput)))
    x = np.arange(start=0, stop=1023, step=st)
    y = np.arange(start=0, stop=1023, step=st)
    fR = interpolate.interp2d(x, y, zR, kind='cubic')
    fI = interpolate.interp2d(x, y, zI, kind='cubic')
    (xnew, ynew, xxN, yyN) = setmesh()
    znew = fR(xnew, ynew) + 1j * fI(xnew, ynew)
    return (znew, xxN)

