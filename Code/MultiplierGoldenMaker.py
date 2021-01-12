#!/usr/bin/env python
# coding: utf-8

# # Multiplier Golden Sample Maker

# #### Stock Imports

# In[ ]:


import os, sys


# In[ ]:


import numpy as np
import itertools


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
    np.empty_like(pixArray, dtype=np.uint32)
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


np.loadtxt(path)


# In[ ]:


np.max(abs(dataSet[0]))


# In[ ]:


plotComplexArray(dataSet[0], maxRad=10)


# In[ ]:


golden = np.average(dataSet, axis=0)


# In[ ]:


plotComplexArray(golden, maxRad=10)


# In[ ]:


deviations = dataSet - golden


# In[ ]:


deviationsSTD = np.zeros_like(deviations)
for i, d in enumerate(deviations):
    ave = np.average(d)
    deviationsSTD[i] = d-ave


# In[ ]:


plotComplexArray(deviationsSTD[4], maxRad=1)


# In[ ]:


deviationsSTDFlat = deviationsSTD.reshape(5,-1)


# In[ ]:


nCors = 2


# In[ ]:


pca = ComplexPCA(n_components=nCors)
pca.fit(deviationsSTDFlat)
pcaComps = pca.components_.reshape(nSamples,64,64)[:nCors]
basisRough = np.insert(pcaComps, 0, np.full_like(pcaComps[0], 1+0j), axis=0)


# In[ ]:


plotComplexArray(basisRough[1], maxRad=.1)


# In[ ]:


deviceID = 4
device = dataSet[deviceID]


# In[ ]:


weights = np.linalg.lstsq(basisRough.reshape(len(basisRough),-1).T, 
                          (device-golden).flat, 
                          rcond=None)[0]


# In[ ]:


fit = golden + (basisRough.T @ weights).reshape((64,64)).T


# In[ ]:


plotComplexArray(fit - device, maxRad=0.05)


# In[ ]:


np.max(np.abs(fit - device))


# In[ ]:


errors = []
for device in dataSet:
    weights = np.linalg.lstsq(basisRough.reshape(len(basisRough),-1).T, 
                          (device-golden).flat, 
                          rcond=None)[0]
    fit = golden + (basisRough.T @ weights).reshape((64,64)).T
    aveLinError = np.average(np.abs(fit - device))
    errors.append(aveLinError)
errors


# In[ ]:





# In[ ]:




