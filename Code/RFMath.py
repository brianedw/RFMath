#!/usr/bin/env python
# coding: utf-8

# ## Imports

# ### Stock Imports

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


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
import time


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


# In[ ]:


from scipy.optimize import minimize


# ### Custom Imports

# In[ ]:


from NetworkBuilding import (BuildMillerNetwork, BuildNewNetwork,
                             MillerMultLocsX, MillerCoupLocsX, NewMultLocs,
                             ConvertThetaPhiToTcX, 
                             Build3dBCoupler, Build5PortSplitter)


# In[ ]:


from ExpComponents import (Multiplier, MultiplierBank, Build3dBCouplerSim)


# In[ ]:


from Miller import (MillerBuilder)


# In[ ]:


from UtilityMath import (convertArrayToDict, MatrixError, MatrixSqError, makePolarPlot, addMatrixDiff, PolarPlot, ReIm)


# In[ ]:


from HardwareComms import (MultBankComm, SwitchComm, VNAComm, ExperimentalSetup)


# # New Architecture

# This analysis has two steps: Tuning and Performance 
# 
# In the first step, we will tune the model of the devices to account for various idiosynchrasies of the physical network (device irregularities, cable lengths, etc).  
# 
# In order to do this tuning, we will build a simulation of the network.  The key element in the network is the Multiplier.  This element is represented by a Python Object that is based on a PCA analysis of physical measurements of large set of Multipliers.  Each Multiplier's representation has its own PCA weights that can be adjusted.
# 
# In order to tune these PCA weights, we will apply a series of test settings (PS value [0-1023] and VGA value [0-1023] to the Multipliers.  Upon performing a physical measurement, this will yield a series of $n \times n$ scattering matrices for the entire network.  Following that, we can use optimization to adjust the PCA weights of each Multiplier until the network simulations of the same test settings match the physical reasults.
# 
# Once the devices have been tuned, we can specifiy a desired target network response.  This network can be transformed into Multiplier complex transmission values, $T$.  The algorithm for this step can be quite complicated depending on the network topology (Miller vs New).  By using inverse functions on the PCA weights, we can find the required digital inputs (PS and VGA value) to each physical multiplier.
# 
# Finally, we apply these digital inputs both in simulation and experiment.  We take a physical measurement of the network and compare the target, simulation, and physical network responses.

# ## Definitions

# First we define the various devices.

# In[ ]:


# switchCommIn = SwitchComm(comValue='COM1', {1:6, 2:5, 3:4, 4:3, 5:2})
# switchCommOut = SwitchComm(comValue='COM2', {1:2, 2:3, 3:4, 4:5, 5:6})
# vnaComm = VNAComm()
# multBankCom = MultBankComm(comValue='COM3')


# For convenience, higher level scripts that require coordination between the various devices can be accessed using an `ExperimentalSetup`.

# In[ ]:


# exp = ExperimentalSetup(switchCommIn, switchCommOut, vnaComm, multBankCom)


# In[ ]:


freq45 = rf.Frequency(start=45, stop=45, npoints=1, unit='mhz', sweep_type='lin')


# First we need to generate labels for the Multipliers.  For the New architecture, this is a simple square grid.  The format is
# 
# `('M', 'N', inputLine, outputLine)` 
# 
# where `'M'` is for "Multiplier", `'N'` is for "New" and `inputLine` and `outputLine` are integers in the range [0,4].

# In[ ]:


allMultLocs = NewMultLocs(5,'N')
allMultLocs[7]


# Every device has a "Physical Number" that is used for addressing to allow the computer to specify to which device a command is intended.  These are enumarated below.  Similar to SParams, the rows denote output lines while the columns denote input lines.

# In[ ]:


multPhysNumberBank = [[  1,  2,  3,  4,   5],
                      [  6,  7,  8,  9,  10],
                      [ 11, 12, 13, 14,  15],
                      [ 16, 17, 18, 19,  20],
                      [ 21, 22, 23, 24,  25]]
multPhysNumberBank = np.array(multPhysNumberBank)


# And just a quick spot check to make sure we have accidently applied a transpose.

# In[ ]:


inputLine = 5
outputLine = 1
multPhysNumberBank[outputLine - 1, inputLine - 1]


# Next we build a MultiplierBank.  This is a collection of Multipliers.  This allows a Multiplier to be retreived by either its `loc` or by its `physNumber`, allowing the MultiplierBank to function both to interact with the physical experiment or a network simulation.

# In[ ]:


multBank = MultiplierBank()
for loc in allMultLocs:
    (_, _, inputLine, outputLine) = loc
    physNumber = multPhysNumberBank[outputLine, inputLine]
    mult = Multiplier(physNumber=physNumber, loc=loc, freq=freq45)
    multBank.addMult(mult)


# Note that passive devices such as 5:1 Splitters are not modeled to the same degree and do not require controlling.  Therefore, we will generate generic elements as we need them.

# ## Tuning

# ### Physical Measurement

# Next we define a series of multiplier set points that we'll use to ascertain the multiplier's PCA weights.

# In[ ]:


tuningPSVals = np.linspace(0, 1023, 5, dtype=np.int)
tuningVGAVals = np.linspace(0, 1023, 5, dtype=np.int)


# In[ ]:


tuningVals = [(ps, vga) for vga in tuningVGAVals for ps in tuningPSVals]


# For each PS, VGA pair, the multipliers are uniformly set and the scattering matrix of the network is measured.

# In[ ]:


tuningMatricesM = []
for (psVal, vgaVal) in tuningVals:
    exp.setMults(psVal, vgaVal, multBank.getPhysNums())
    time.sleep(1)
    m = exp.measureSMatrix(delay=2)
    tuningMatricesM.append(m)
tuningMatricesM = np.array(tuningMatricesM)


# In[ ]:


np.save("tuningVals", tuningVals)
np.save("tuningMatricesM", tuningMatricesM)


# ### Fake Measurements

# In[ ]:


def MultBuilder(loc):
    return multBank.getRFNetwork(loc)


# In[ ]:


def SplitterBuilder(loc):
    return Build5PortSplitter(freq45, loc=loc)


# In[ ]:


X0 = multBank.getPersonalityVectors()


# In[ ]:


XSet = X0*np.random.normal(1, 0.1, size=len(X0))


# In[ ]:


multBank.setPersonalityVectors(XSet)


# In[ ]:


tuningMatricesM = []
for (psVal, vgaVal) in tuningVals:
    multBank.setAllMults(psVal, vgaVal)
    newNet = BuildNewNetwork(SplitterBuilder, MultBuilder, loc="N", n=5)
    m = newNet.s[0, 5:, :5]
    tuningMatricesM.append(m)
tuningMatricesM = np.array(tuningMatricesM)


# In[ ]:


multBank.setPersonalityVectors(X0)


# In[ ]:


np.save("tuningVals", tuningVals)
np.save("tuningMatricesM", tuningMatricesM)


# ### Fitting

# In[ ]:


tuningVals = np.load("tuningVals.npy")
tuningMatricesM = np.load("tuningMatricesM.npy")


# The simulation builder `BuildNewNetwork` requires that we supply it with two functions, one which creates an RF network object from of a 5-way splitter, and another which creates one of the Multiplier.  We will assume that the splitter is generic and employ a simple theoretical model for that which was imported from our `NetworkBuilding` theoretical simulation notebook.  However, for the Multiplier, we will use the `MultiplierBank` and the `loc` code to extract the model for a multiplier assigned to that specific location in the network. 

# In[ ]:


def MultBuilder(loc):
    return multBank.getRFNetwork(loc)


# In[ ]:


def SplitterBuilder(loc):
    return Build5PortSplitter(freq45, loc=loc)


# As a quick example of a simulation, we set all the multipliers to the same setting, build a network, and examine the transmissive properties of it.

# In[ ]:


multBank.setAllMults(psVal=512, vgaVal=512)


# In[ ]:


newNet = BuildNewNetwork(SplitterBuilder, MultBuilder, loc="N", n=5)
T = newNet.s[0, 5:, :5]
T


# Of course this step can be automated for all of the `(ps, vga)` pairs in the in `tuningVals` to yield `tuningMatricesS`.  

# In[ ]:


tuningMatricesS = []
for (psVal, vgaVal) in tuningVals:
    multBank.setAllMults(psVal, vgaVal)
    newNet = BuildNewNetwork(SplitterBuilder, MultBuilder, loc="N", n=5)
    m = newNet.s[0, 5:, :5]
    tuningMatricesS.append(m)
tuningMatricesS = np.array(tuningMatricesS)


# In[ ]:


tuningMatricesS;


# Ideally, this would yield the exact same network scattering matrices as were measured and contained in `tuningMatricesM`.  Of course they won't because each physical device has its own personality and other factors such as varying cable lengths.  We will therefore optimize the PCA weights of each device in simulation in an attempt to create collection of devices which match the real behavior of the experimental devices.
# 
# In order to perform this optimization, we use SciPy's multivariate minimization function `minimize()`.  The format of this 
# `scipy.optimize.minimize(fun, X0)` where `fun` is built such that `fun(X) -> error` where `X` and `X0` are 1D vectors of the real scalars to be optimized.  In order to make this easy, the MultiplierBank comes with two functions `setPersonalityVectors(X)` and `X0 = getPersonalityVectors()`, which grabs the complex PCA weights from all the multipliers as mashes them into a real 1D vector.  The two functions are designed to operate together so that the data

# In[ ]:


X0 = multBank.getPersonalityVectors()


# In[ ]:


def fun(X):
    multBank.setPersonalityVectors(X)
    tuningMatricesS = []
    for (psVal, vgaVal) in tuningVals:
        multBank.setAllMults(psVal, vgaVal)
        newNet = BuildNewNetwork(SplitterBuilder, MultBuilder, loc="N", n=5)
        m = newNet.s[0, 5:, :5]
        tuningMatricesS.append(m)
    tuningMatricesS = np.array(tuningMatricesS)
    error = np.sum(np.abs(tuningMatricesS - tuningMatricesM)**2)
    return error


# In[ ]:


fit = sp.optimize.minimize(fun, X0, method='Nelder-Mead', options={'disp':True})


# In[ ]:


XF = fit.x


# Error when multipliers are the uniform average all devices measured in the PCA:

# In[ ]:


fun(X0)


# Error following fitting the PCA weights:

# In[ ]:


fun(XF)


# In[ ]:


multBank.setPersonalityVectors(XF)


# # Scrap

# In[ ]:


tuningMatricesS = []
for (psVal, vgaVal) in tuningVals:
    multBank.setAllMults(psVal, vgaVal)
    newNet = BuildNewNetwork(SplitterBuilder, MultBuilder, loc="N", n=5)
    m = newNet.s[0, 5:, :5]
    tuningMatricesS.append(m)
tuningMatricesS = np.array(tuningMatricesS)

tuningMatricesS - 


# In[ ]:


physMatrices = []
for (psVal, vgaVal) in tuningVals:
    SetAllSimMults(psVal, vgaVal, multBank)
    time.sleep(1)
    m = MeasurePhysMatrix(5, inSwitchComm, outSwitchComm, vnaComm, delay=0)
    physMatrices.append(m)


# In[ ]:


for loc in multBank.getLocs():
    mult = multBank.getMultByLoc(loc)
    mult.setSettings(psSetting, vgaSetting)


# In[ ]:


mult = multBank.getMultByLoc(loc)
mult.setSettings(psSetting=0, vgaSetting=0)


# In[ ]:


SplitterBuilder(("Sin", 0, 0))


# In[ ]:


MultBuilder(("M", "X", 0, 0))


# In[ ]:


np.allclose(T, Ks)


# In[ ]:


freq = rf.Frequency(start=45, stop=45, npoints=1, unit='mhz', sweep_type='lin')


# In[ ]:


def SplitterBuilder(loc):
    return Build5PortSplitter(freq, loc=loc)


# In[ ]:


SplitterBuilder(("Sin", 0, 0))


# In[ ]:


def MultBuilder(loc):
    (_, locParent, i_in, i_out) = loc
    Tc = Ks[i_out, i_in] * np.sqrt(5)**2
    return BuildMultiplier(Tc, freq, loc)


# In[ ]:


MultBuilder(("M", "X", 0, 0))


# In[ ]:


newNet = BuildNewNetwork(SplitterBuilder, MultBuilder, loc="X", n=5)
T = newNet.s[0, 5:, :5]
T


# In[ ]:


np.allclose(T, Ks)


# In[ ]:


Ks = np.array([[-0.05+0.06j, -0.  -0.13j, -0.07-0.15j,  0.11+0.28j, -0.05-0.18j],
               [-0.1 -0.19j, -0.3 -0.05j, -0.28+0.07j, -0.25+0.28j, -0.11-0.29j],
               [ 0.21-0.18j, -0.08-0.14j,  0.03+0.2j , -0.23+0.24j, -0.06+0.32j],
               [-0.29-0.31j,  0.12+0.09j,  0.08-0.02j,  0.31+0.12j, -0.22-0.18j],
               [-0.18-0.06j,  0.08-0.21j,  0.25-0.18j, -0.26-0.1j ,  0.13+0.1j ]])

