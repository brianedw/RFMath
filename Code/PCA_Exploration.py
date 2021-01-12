#!/usr/bin/env python
# coding: utf-8

# # PCA Exploration

# #### Imports

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


# ## Work

# ### Device Definition

# In[ ]:


def DeviceResponse(x):
    response = (1/2)*(0.5 + np.tanh((x-0.5)*10)/2) + (1/2)*x
    return response


# In[ ]:


xs=np.linspace(0, 1, num=1001)
ys=DeviceResponse(xs)


# In[ ]:


fig = figure(plot_height=500, plot_width=800, x_range=(0, 1), y_range=(0, 1))
fig.title.text = "My Plot"
fig.title.align = "center"
fig.xaxis.axis_label = "input"
fig.yaxis.axis_label = "response"
fig.scatter(x=xs, y=ys, legend_label="Measured Data", 
            size=1, fill_color='red', line_color='red', fill_alpha=.5)
show(fig)


# In[ ]:


def RealResponse():
    xs = np.linspace(0, 1, num=1001)
    a, b, c, d = np.random.normal(size=4)
    noise =  np.random.normal(size=len(xs))
    response = (1 + 0.1*a)*DeviceResponse(xs - 0.02*b) + (0.1)*c*xs**2 + 0.05*d + 0.01*noise
    return response


# In[ ]:


dataSet = [RealResponse() for i in range(25)]


# In[ ]:


fig = figure(plot_height=500, plot_width=800, x_range=(0, 1), y_range=(0, 1))
fig.title.text = "My Plot"
fig.title.align = "center"
fig.xaxis.axis_label = "input"
fig.yaxis.axis_label = "response"
for ys in dataSet:
    color = next(colors)
    fig.scatter(x=xs, y=ys, size=1, fill_color=color, line_color=color, fill_alpha=1)
show(fig)


# In[ ]:


goldenRough = np.average(dataSet, axis=0)
golden = gaussian_filter(goldenRough, 5, mode='reflect')
golden.shape


# In[ ]:


fig = figure(plot_height=500, plot_width=800, x_range=(0, 1), y_range=(0, 1))
fig.title.text = "My Plot"
fig.title.align = "center"
fig.xaxis.axis_label = "input"
fig.yaxis.axis_label = "response"
fig.scatter(x=xs, y=golden, size=1, fill_color=color, line_color=color, fill_alpha=1)
show(fig)


# In[ ]:


deviations = dataSet - golden


# In[ ]:


deviationsSTD = np.zeros_like(deviations)
for i, d in enumerate(deviations):
    ave = np.average(d)
    deviationsSTD[i] = d-ave


# In[ ]:


fig = figure(plot_height=500, plot_width=800, x_range=(0, 1), y_range=(-0.3, 0.3))
fig.title.text = "My Plot"
fig.title.align = "center"
fig.xaxis.axis_label = "input"
fig.yaxis.axis_label = "response"
for ys in deviationsSTD:
    color = next(colors)
    fig.scatter(x=xs, y=ys, size=1, fill_color=color, line_color=color, fill_alpha=1)
show(fig)


# ### Determining Correction Factors

# In[ ]:


pca = PCA(n_components=4)
pca.fit(deviations)


# In[ ]:


basisRough = np.insert(pca.components_, 0, np.full_like(xs, fill_value=1), axis=0)
basisRough = pca.components_
basisRough.shape


# In[ ]:


basisSmooth = np.zeros_like(basisRough)


# In[ ]:


for i in range(len(basisSmooth)):
    basisSmooth[i] = gaussian_filter(basisRough[i], 15)


# In[ ]:


fig = figure(plot_height=500, plot_width=800, x_range=(0, 1), y_range=(-0.1, 0.1))
fig.title.text = "My Plot"
fig.title.align = "center"
fig.xaxis.axis_label = "input"
fig.yaxis.axis_label = "response"
for ys in basisSmooth:
    color = next(colors)
    fig.scatter(x=xs, y=ys, size=1, fill_color=color, line_color=color, fill_alpha=1)
show(fig)


# ### Device Fitting

# In[ ]:


deviceID = np.random.randint(0, len(dataSet)+1)
device = dataSet[deviceID]


# In[ ]:


corrWeights = np.linalg.lstsq(basisSmooth.T, device-golden, rcond=None)[0]


# In[ ]:


basisSmooth.T @ corrWeights


# In[ ]:


fit = golden + basisSmooth.T @ corrWeights


# In[ ]:


fig = figure(plot_height=500, plot_width=800, x_range=(0, 1), y_range=(0, 1))
fig.title.text = "My Plot"
fig.title.align = "center"
fig.xaxis.axis_label = "input"
fig.yaxis.axis_label = "response"
fig.scatter(x=xs, y=device, size=1, fill_color='red', line_color='red', fill_alpha=1)
fig.scatter(x=xs, y=fit, size=1, fill_color='white', line_color='white', fill_alpha=1)
show(fig)


# In[ ]:





# In[ ]:





# In[ ]:




