#!/usr/bin/env python
# coding: utf-8

# ## Imports

# ### Stock Imports

# In[ ]:


import os, sys


# In[ ]:


import importlib


# In[ ]:


import numpy as np
import scipy as sp
from scipy.optimize import root
from scipy.interpolate import interp2d
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


# In[ ]:


import skrf as rf


# ### Custom Imports

# In[ ]:


from NetworkBuilding import (BuildMillerNetwork, MillerMultLocsX, 
                             ConvertThetaPhiToTcX, Build3dBCoupler)


# In[ ]:


from ExpComponents import (Multiplier, MultiplierBank, Build3dBCouplerSim)


# In[ ]:


from Miller import (MillerBuilder)


# In[ ]:


from UtilityMath import (convertArrayToDict)


# # Library

# # Work

# We begin by defining the kernel we want to emulate:

# In[ ]:


freq45 = rf.Frequency(45 ,45, npoints=1, unit='mhz')


# In[ ]:


Ks = np.array([[-0.05+0.06j, -0.  -0.13j, -0.07-0.15j,  0.11+0.28j, -0.05-0.18j],
               [-0.1 -0.19j, -0.3 -0.05j, -0.28+0.07j, -0.25+0.28j, -0.11-0.29j],
               [ 0.21-0.18j, -0.08-0.14j,  0.03+0.20j, -0.23+0.24j, -0.06+0.32j],
               [-0.29-0.31j,  0.12+0.09j,  0.08-0.02j,  0.31+0.12j, -0.22-0.18j],
               [-0.18-0.06j,  0.08-0.21j,  0.25-0.18j, -0.26-0.10j,  0.13+0.10j]])


# We perform an SVD decomposition on it.  This is only so that we can compare the resulting sub structures.  The MillerBuilder will also do this internally when computing MZI values.

# In[ ]:


V, S, Uh = np.linalg.svd(Ks)


# Next we use the MillerBuilder to compute MZI Theta-Phi data.  Then we convert this to Multiplier Complex Transmission Data.  We use the second varient of this function which takes into account the single multipliers on the bottom rows. 

# In[ ]:


miller = MillerBuilder(couplerConv='LC', verbose=False)
theta_phi1, sTerms, theta_phi2 = miller.ConvertKToMZI(Ks)
Tc1 = ConvertThetaPhiToTcX(theta_phi1)
Tc2 = ConvertThetaPhiToTcX(theta_phi2)


# For many applications such as optimization, it will be convenient to have the devices as 1D objects such as dictionaries as opposed to nDim arrays.

# In[ ]:


Tc1Dict = convertArrayToDict(Tc1, preSpec=("M", "Uh"))
TcSDict = convertArrayToDict(sTerms, preSpec=("M", "S"))
Tc2Dict = convertArrayToDict(Tc2, preSpec=("M", "V"))
TcDict = {**Tc1Dict, **TcSDict, **Tc2Dict}


# Next we build a bank of Multipliers.  These correspond to experimental components and have a much more complicated functionality than an elementary network object.

# In[ ]:


allMultLocs = MillerMultLocsX(5, labels=('Uh', 'S', 'V'))
multBank = MultiplierBank()
for i, loc in enumerate(allMultLocs):
    mult = Multiplier(physNumber=i, loc=loc, freq=freq45)
    multBank.addMult(mult)


# and then we set them to the values required by the kernel.  Note that this is actually using inverse functions to find the [0-1023] vga and ps input values taking into account the personality of each device.

# In[ ]:


for loc, Tc in TcDict.items():
    mult = multBank.getMultByLoc(loc)
    mult.setT(Tc)


# Next we define our MultBuilder and AttBuilder, which simply reach into the bank and grab the appropriate network object.

# In[ ]:


def MultBuilder(loc):
    return multBank.getRFNetwork(loc)


# In[ ]:


def AttBuilder(loc):
    return MultBuilder(loc)


# In[ ]:


MultBuilder(loc=("M", "Uh", 0, 0, 0))


# To make things a little more interesting, we'll define two different CouplerBuilders.  The first makes use of an ideal coupler while the second make use of one determined through simulation of the components which includes parasitic losses, and was found to be fairly close to measured devices.

# In[ ]:


def CouplerBuilderIdeal(loc):
    return Build3dBCoupler(freq45, loc=loc)


# In[ ]:


CouplerBuilderIdeal(("M", "Uh", 0, 0, 0)).s[0]


# In[ ]:


def CouplerBuilderSim(loc):
    return Build3dBCouplerSim(freq45, loc=loc)


# In[ ]:


CouplerBuilderSim(("M", "Uh", 0, 0, 0)).s[0]


# Finally, we assemble the simulation and determine that its behavior is pretty close to expected.  Deviations are likely due to the integer values used in the multiplier settings.

# In[ ]:


millerNet = BuildMillerNetwork(CouplerBuilderIdeal, MultBuilder, 
                               AttBuilder, n=5, labels=('Uh', 'S', 'V'))
T = millerNet.s[0, 5:, :5]
T


# In[ ]:


np.abs(T-Ks)


# And then we assembled a second simuation that utilizes the realistic 3dB couplers.  The deviations are much greater.

# In[ ]:


millerNet2 = BuildMillerNetwork(CouplerBuilderSim, MultBuilder, 
                               AttBuilder, n=5, labels=('Uh', 'S', 'V'))
T = millerNet2.s[0, 5:, :5]
T


# In[ ]:


np.abs(T-Ks)


# In[ ]:




