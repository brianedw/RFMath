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


# In[ ]:


from sklearn.decomposition import PCA


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


# Stolen from https://datascience.stackexchange.com/questions/75733/pca-for-complex-valued-data

# In[ ]:


class ComplexPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.u = self.s = self.components_ = None
        self.mean_ = None

    @property
    def explained_variance_ratio_(self):
        return self.s

    def fit(self, matrix, use_gpu=False):
        self.mean_ = matrix.mean(axis=0)
        if use_gpu:
            import tensorflow as tf  # torch doesn't handle complex values.
            tensor = tf.convert_to_tensor(matrix)
            u, s, vh = tf.linalg.svd(tensor, full_matrices=False)  # full=False ==> num_pc = min(N, M)
            # It would be faster if the SVD was truncated to only n_components instead of min(M, N)
        else:
            _, self.s, vh = np.linalg.svd(matrix, full_matrices=False)  # full=False ==> num_pc = min(N, M)
            # It would be faster if the SVD was truncated to only n_components instead of min(M, N)
        self.components_ = vh  # already conjugated.
        # Leave those components as rows of matrix so that it is compatible with Sklearn PCA.

    def transform(self, matrix):
        data = matrix - self.mean_
        result = data @ self.components_.T
        return result

    def inverse_transform(self, matrix):
        result = matrix @ np.conj(self.components_)
        return self.mean_ + result


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
nSamples = len(trainingData)


# In[ ]:


dataFlat = trainingData.reshape((len(trainingData), -1))


# In[ ]:


nCors = 3


# In[ ]:


pca = ComplexPCA(n_components=nCors)
pca.fit(dataFlat)
pcaComps = pca.components_.reshape(nSamples,64,64)[:nCors]
constComp = np.full_like(pcaComps[0], 1+0j)
basisRough = np.concatenate([ pcaComps, [constComp]])
basis = np.array([b/np.average(np.abs(b)) for b in basisRough])


# In[ ]:


plotComplexArray(basis[0], maxRad=4)


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
    weights = np.linalg.lstsq(basisRough.reshape(len(basisRough),-1).T, 
                          (device-golden).flat, 
                          rcond=None)[0]
    weightsList.append(weights)
    fit = golden + (basisRough.T @ weights).reshape((64,64)).T
    aveLinError = np.average(np.abs(fit - device))
    errors.append(aveLinError)
errors


# In[ ]:


print(np.round(np.abs(weightsList),3))

