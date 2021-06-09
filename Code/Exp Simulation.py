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


from NetworkBuilding import (BuildMillerNetwork,
                             MillerMultLocsX, MillerCoupLocsX,
                             ConvertThetaPhiToTcX, Build3dBCoupler)


# In[ ]:


from ExpComponents import (Multiplier, MultiplierBank, Build3dBCouplerSim)


# In[ ]:


from Miller import (MillerBuilder)


# In[ ]:


from UtilityMath import (convertArrayToDict, MatrixError, MatrixSqError, makePolarPlot, addMatrixDiff, PolarPlot, ReIm)


# # Library

# # Work

# ## Ideal-ish elements

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


# Finally we'll define a CouplerBuilder based on an ideal device.

# In[ ]:


def CouplerBuilderIdeal(loc):
    return Build3dBCoupler(freq45, loc=loc)


# Finally, we assemble the simulation and determine that its behavior is pretty close to expected.  Deviations are likely due to the integer values used in the multiplier settings.

# In[ ]:


millerNet = BuildMillerNetwork(CouplerBuilderIdeal, MultBuilder, 
                               AttBuilder, n=5, labels=('Uh', 'S', 'V'))
T = millerNet.s[0, 5:, :5]
T


# In[ ]:


plot = makePolarPlot("Goal K vs Realized using Ideal Comps")
addMatrixDiff(plot, Ks, T)
show(plot)


# In[ ]:


MatrixError(T, Ks)


# ## Switch to Sim Couplers

# On this round, let's define a Coupler Builder which makes use of one determined through simulation of the components which includes parasitic losses, and was found to be fairly close to measured devices.

# In[ ]:


def CouplerBuilderSim(loc):
    return Build3dBCouplerSim(freq45, loc=loc)


# And then we assembled a second simulation that utilizes the realistic 3dB couplers.  The deviations are much greater.

# In[ ]:


millerNet2 = BuildMillerNetwork(CouplerBuilderSim, MultBuilder, 
                               AttBuilder, n=5, labels=('Uh', 'S', 'V'))
T = millerNet2.s[0, 5:, :5]
T


# In[ ]:


plot = makePolarPlot("Goal K vs Realized using Realistic Coups and Uncorrected Mults")
addMatrixDiff(plot, Ks, T)
show(plot)


# In[ ]:


MatrixError(T, Ks)


# ## Adjusting the Multipliers to account for the Simulation Couplers

# There are multipliers which go to form MZIs and are trapped between two couplers.  We can generate a list of all such multipliers by messaging the list of all of the couplers.

# In[ ]:


allCouplers = MillerCoupLocsX(5, labels=('Uh', 'V'))
allTrappedMults = []
for loc in allCouplers:
    locList = list(loc)
    locList[0] = 'M'
    allTrappedMults.append(tuple(locList))
allTrappedMults;


# Here is the ideal transmission of a 3dB Coupler.

# In[ ]:


TIdeal = CouplerBuilderIdeal(("M", "Uh", 0, 0, 0)).s[0, 2:, :2]
TIdeal


# Next is the simulated coupler.

# In[ ]:


TSim = CouplerBuilderSim(("M", "Uh", 0, 0, 0)).s[0, 2:, :2]
TSim


# In[ ]:


MatrixError(TIdeal, TSim)


# Next let's see if there is an adjustment factor that minimizes the distance between these two devices.

# In[ ]:


def fError(z):
    zR, zI = z
    error = MatrixSqError(TIdeal, (zR+1j*zI)*TSim)
    return error


# In[ ]:


soln = minimize(fError, [1, 0])
zr, zi = soln.x
zAdjust = zr + 1j*zi
zAdjust


# In[ ]:


plot = makePolarPlot("Ideal vs Sim Coupler with Scalar CF")
addMatrixDiff(plot, TIdeal, TSim)
addMatrixDiff(plot, TIdeal, TSim*zAdjust)
show(plot)


# In[ ]:


MatrixError(TIdeal, TSim*zAdjust)


# We can't apply an arbitary phase/amplitude shift to the couplers, but we can apply it to the local multipliers.  Since each stacked pair of multipliers are between two couplers, we need to apply the adjustment factor "twice", or in other words square it.  Not all multipliers are in MZIs and so we apply this only to the "trapped" mutlipliers.

# In[ ]:


for loc, Tc in TcDict.items():
    mult = multBank.getMultByLoc(loc)
    if loc in allTrappedMults:
        mult.setT(zAdjust**2 * Tc)
    else:
        mult.setT(Tc)


# In[ ]:


millerNet3 = BuildMillerNetwork(CouplerBuilderSim, MultBuilder, 
                               AttBuilder, n=5, labels=('Uh', 'S', 'V'))
T = millerNet3.s[0, 5:, :5]
T


# In[ ]:


plot = makePolarPlot("Goal K vs Realized using Realistic Coups and Corrected Mults")
addMatrixDiff(plot, Ks, T)
show(plot)


# In[ ]:


MatrixError(T, Ks)


# And we see that the error has come down to nearly that of the ideal devices.

# ## Adjusting the Multipliers to account for the Experimental Couplers

# In[ ]:


from ExpComponents import Build3dBCouplerFromData, coupNet1


# In[ ]:


coupNet1.name = 'bar'


# In[ ]:


coupNet1.copy()


# In[ ]:


def CouplerBuilderExp(loc):
    net = coupNet1.copy()
    net.name = str(loc)
    return net


# In[ ]:


CouplerBuilderExp(("C", "Uh", 0, 0, 0)).s[0, 2:, :2]


# There are multipliers which go to form MZIs and are trapped between two couplers.  We can generate a list of all such multipliers by messaging the list of all of the couplers.

# In[ ]:


allCouplers = MillerCoupLocsX(5, labels=('Uh', 'V'))
allTrappedMults = []
for loc in allCouplers:
    locList = list(loc)
    locList[0] = 'M'
    allTrappedMults.append(tuple(locList))
allTrappedMults;


# Here is the ideal transmission of a 3dB Coupler.

# In[ ]:


TIdeal = CouplerBuilderIdeal(("C", "Uh", 0, 0, 0)).s[0, 2:, :2]
TIdeal


# Next is the simulated coupler.

# In[ ]:


TExp = CouplerBuilderExp(("C", "Uh", 0, 0, 0)).s[0, 2:, :2]
TExp


# In[ ]:


MatrixError(TIdeal, TExp)


# Next let's see if there is an adjustment factor that minimizes the distance between these two devices.

# In[ ]:


def fError(z):
    zR, zI = z
    zC = (zR+1j*zI)
    error = MatrixSqError(TIdeal, zC*TExp)
    return error


# In[ ]:


c = 1.2
fError([c*1, c*0.81])


# In[ ]:


soln = minimize(fError, [1, 0])
zr, zi = soln.x
zAdjust = zr + 1j*zi
zAdjust


# In[ ]:


pp = PolarPlot("Ideal vs Exp Coupler with Scalar CF")
pp.addMatrix(TIdeal, "green")
pp.addMatrix(TExp, "red")
pp.addMatrix(TExp*zAdjust, "cyan")
pp.show()


# In[ ]:


MatrixSqError(TIdeal, TExp*zAdjust)


# We can't apply an arbitary phase/amplitude shift to the couplers, but we can apply it to the local multipliers.  Since each stacked pair of multipliers are between two couplers, we need to apply the adjustment factor "twice", or in other words square it.  Not all multipliers are in MZIs and so we apply this only to the "trapped" mutlipliers.

# In[ ]:


for loc, Tc in TcDict.items():
    mult = multBank.getMultByLoc(loc)
    if loc in allTrappedMults:
        mult.setT(zAdjust**2 * Tc)
    else:
        mult.setT(Tc)


# In[ ]:


millerNet3 = BuildMillerNetwork(CouplerBuilderExp, MultBuilder, 
                               AttBuilder, n=5, labels=('Uh', 'S', 'V'))
T = millerNet3.s[0, 5:, :5]
T


# In[ ]:


plot = makePolarPlot("Goal K vs Realized using Realistic Coups and Corrected Mults")
addMatrixDiff(plot, Ks, T)
show(plot)


# In[ ]:


MatrixError(T, Ks)


# In[ ]:


TInv = np.linalg.inv(np.identity(5)-T)
KInv = np.linalg.inv(np.identity(5)-Ks)


# In[ ]:


plot = makePolarPlot("KInv, TInv")
addMatrixDiff(plot, KInv, TInv)
show(plot)


# And we see that the error has come down to nearly that of the ideal devices.

# ## Adjusting the Multipliers to account for the Experimental Couplers with Optimization

# In[ ]:


from ExpComponents import Build3dBCouplerFromData, coupNet1, coupNet2


# In[ ]:


coupNet1.name = 'bar'


# In[ ]:


coupNet1.copy()


# In[ ]:


def CouplerBuilderExp(loc):
    choice = hash((loc))%2
    coupNet = [coupNet1, coupNet2][choice]
    net = coupNet.copy()
    net.name = str(loc)
    return net


# In[ ]:


CouplerBuilderExp(("C", "Uh", 0, 0, 0)).s[0, 2:, :2]


# There are multipliers which go to form MZIs and are trapped between two couplers.  We can generate a list of all such multipliers by messaging the list of all of the couplers.

# In[ ]:


allCouplers = MillerCoupLocsX(5, labels=('Uh', 'V'))
allTrappedMults = []
for loc in allCouplers:
    locList = list(loc)
    locList[0] = 'M'
    allTrappedMults.append(tuple(locList))
allTrappedMults;


# Here is the ideal transmission of a 3dB Coupler.

# In[ ]:


TIdeal = CouplerBuilderIdeal(("C", "Uh", 0, 0, 0)).s[0, 2:, :2]
TIdeal


# Next is the simulated coupler.

# In[ ]:


TExp = (coupNet1.s[0, 2:, :2] + coupNet2.s[0, 2:, :2])/2
TExp


# Next let's see if there is an adjustment factor that minimizes the distance between these two devices.

# In[ ]:


def fError(z):
    zR, zI = z
    zC = (zR+1j*zI)
    error = MatrixSqError(TIdeal, zC*TExp)
    return error


# In[ ]:


soln = minimize(fError, [1, 0])
zr, zi = soln.x
zAdjust = zr + 1j*zi
zAdjust


# In[ ]:


pp = PolarPlot("Ideal vs Exp Coupler with Scalar CF")
pp.addMatrix(TIdeal, "green")
pp.addMatrix(coupNet1.s[0, 2:, :2], "red")
pp.addMatrix(coupNet2.s[0, 2:, :2], "red")
pp.addMatrix(zAdjust*coupNet1.s[0, 2:, :2], "cyan")
pp.addMatrix(zAdjust*coupNet2.s[0, 2:, :2], "cyan")
pp.show()


# In[ ]:


MatrixSqError(TIdeal, TExp*zAdjust)


# We can't apply an arbitary phase/amplitude shift to the couplers, but we can apply it to the local multipliers.  Since each stacked pair of multipliers are between two couplers, we need to apply the adjustment factor "twice", or in other words square it.  Not all multipliers are in MZIs and so we apply this only to the "trapped" mutlipliers.

# In[ ]:


allTrappedMults


# In[ ]:


for loc, Tc in TcDict.items():
    mult = multBank.getMultByLoc(loc)
    if loc in allTrappedMults:
        mult.setT(zAdjust**2 * Tc)
    else:
        mult.setT(Tc)


# In[ ]:


millerNet3 = BuildMillerNetwork(CouplerBuilderExp, MultBuilder, 
                               AttBuilder, n=5, labels=('Uh', 'S', 'V'))
T = millerNet3.s[0, 5:, :5]
T


# In[ ]:


plot = makePolarPlot("Goal K vs Realized using Realistic Coups and Corrected Mults")
addMatrixDiff(plot, Ks, T)
show(plot)


# In[ ]:


mult = multBank.bankByPhysNum[0]
ReIm(mult.TExpected)


# In[ ]:


def getTVecFromMultBank(multBank):
    allTReIm = []
    physNumbers = []
    for idNum, mult in multBank.bankByPhysNum.items():
        physNumbers.append(idNum)
        r,i = ReIm(mult.TExpected)
        allTReIm.extend([r,i])
    return physNumbers, allTReIm


# In[ ]:


physNumbers, allTReIm = getTsFromMultBank(multBank)


# In[ ]:


MatrixError(T, Ks)


# In[ ]:


def setTVecIntoMultBank(multBank, TReIm):
    physNumbers = list(multBank.bankByPhysNum.keys())
    tRe = TReIm[0::2]
    tIm = TReIm[1::2]
    for i, idNum in enumerate(physNumbers):
        t = tRe[i] + 1j*tIm[i]
        mult = multBank.bankByPhysNum[idNum]
        mult.setT(t))


# In[ ]:


def fError(TReIm):
    setTVecIntoMultBank(multBank, TReIm)
    millerNet3 = BuildMillerNetwork(CouplerBuilderExp, MultBuilder, 
                                   AttBuilder, n=5, labels=('Uh', 'S', 'V'))
    T = millerNet3.s[0, 5:, :5]
    return MatrixSqError(T, Ks)


# In[ ]:


soln = minimize(fError, allTReIm)


# In[ ]:


soln.fun


# In[ ]:


TReImFinal = soln.x


# In[ ]:


setTVecIntoMultBank(multBank, TReImFinal)
millerNet3 = BuildMillerNetwork(CouplerBuilderExp, MultBuilder, 
                               AttBuilder, n=5, labels=('Uh', 'S', 'V'))
T = millerNet3.s[0, 5:, :5]
plot = PolarPlot("K vs T using Realistic Coups and Optimized Mults")
plot.addMatrixDiff(Ks, T)
plot.addMatrix()
plot.show()


# In[ ]:


testList = list(range(10))


# In[ ]:


testList[0::2]


# In[ ]:


testList[1::2]


# In[ ]:




