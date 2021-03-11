#!/usr/bin/env python
# coding: utf-8

# #### Stock Imports

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import os, sys, glob


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


# #### Custom Imports

# In[ ]:


from colorize import colorizeComplexArray


# In[ ]:


from UtilityMath import (plotComplexArray, gaussian_filter_complex, complex2DPCA, complex2Dlstsq, PolarPlot)


# In[ ]:


mainQ =(__name__ == '__main__')
mainQ


# # Multiplier

# ## Exp Data Analysis

# Import golden samples and do analysis here.

# In[ ]:


fNames94x94 = glob.glob('..\\GoldenSamples\\MultiplierSamples\\*_94by94.txt')


# In[ ]:


nSamples = 5
pcaComps = 4


# In[ ]:


dataSetList = [np.loadtxt(fNames, dtype=np.complex) for fNames in fNames94x94]
dataSetRough = np.array(dataSetList)
print("dataSetRough.shape", dataSetRough.shape)


# In[ ]:


dataSet = np.empty_like(dataSetRough)
for i in range(len(dataSetRough)):
    dataSet[i] = gaussian_filter_complex(dataSetRough[i], sigma=1.0, mode='nearest')


# In[ ]:


comps, baseWeights = complex2DPCA(dataSet, nComps=pcaComps)
baseWeights


# In[ ]:


vgaSamplePoints = np.loadtxt('..\\GoldenSamples\\MultiplierSamples\\1_94by94Nv.txt')[0]
psSamplePoints  = np.loadtxt('..\\GoldenSamples\\MultiplierSamples\\1_94by94Nv.txt')[0]


# In[ ]:


dataBounds = ((psSamplePoints[0], psSamplePoints[-1]), (vgaSamplePoints[0], vgaSamplePoints[-1]))
dataBounds


# In[ ]:


if mainQ:
    for i in range(pcaComps):
        plotComplexArray(comps[i], maxRad=.05)


# In[ ]:


if mainQ: plotComplexArray(np.sum(comps * baseWeights.reshape(-1,1,1), axis=0), maxRad=1)


# ## Define Element

# In[ ]:


class Multiplier:
    """
    A Multiplier element, derived from physical measurements.  A Multiplier
    takes a 45MHz signal at its input and produces a 45MHz signal at its output,
    however, with a new phase and amplitude such that `A_out = c * A_in`.
    
    Each multiplier takes in two voltages which are specificied digitally as a
    10bit integer [0-1023].  These control a VGA and a pair of phase shifters 
    and loosely correspond to amplitude and phase.
    
    There is enough variability between the fabricated multipliers that special
    techniques are required to characterize their behavior.  Several components
    were measured at an array of vga and ps inputs.  Principal Component
    Analysis was then used to produce several mappings such that a device can be
    adequately described as the weighted sum of those mappings.  Each device is
    supplied with a set of `base weights`, which corresponds to the behavior of
    average component, but can be individually tweaked as more information becomes
    available.
    """
    PCAComps = comps
    baseWeights = baseWeights
    vgaSPs = vgaSamplePoints
    psSPs = psSamplePoints


# In[ ]:


def __init__(self, physNumber=0, loc=(), freq=45e6):
    self.loc = loc                    # The location within the structure.  Assumed to be unique and hashable.
    self.physNumber = physNumber      # The number assigned to the mult used for comms.
    self.weights = np.zeros_like(baseWeights)               # the proportion of the various PCA components
    self.field = None                 # the response of the multiplier as sparsely sampled array
    self.F = None                     # A function such that F(psSetting, vgaSetting) -> Tr + 1j*Ti
    self.vgaSetting = 512               # The VGA setting [0-1023]
    self.psSetting = 512                # The Phase Shifter setting [0-1023]
    self.TExpected = 0+0j             # The expected transmission coefficient.
    self.setWeights(baseWeights)
    self.setT(1+0j)

setattr(Multiplier, "__init__", __init__)


# In[ ]:


def getRFNetwork(self):
    """
    Returns a SciKit-RF network object with a name based on loc
    """
    freq = rf.Frequency(start=45, stop=45, npoints=1, unit='mhz', sweep_type='lin')
    S = np.zeros((len(freq), 2, 2), dtype=np.complex)
    s21 = self.TExpected
    S[:, 1, 0] = s21
    net = rf.Network(name=str(self.loc), frequency=freq, z0=50, s=S)
    return net

setattr(Multiplier, "getRFNetwork", getRFNetwork)


# In[ ]:


def setSettings(self, psSetting, vgaSetting):
    """
    Changes the PS and VGA inputs to new values.  Adjusts T to match.
    """
    self.psSetting = psSetting
    self.vgaSetting = vgaSetting
    self.TExpected = self.F(psSetting, vgaSetting)
    
setattr(Multiplier, "setSettings", setSettings)


# In[ ]:


# def setT(self, T):  # Old version based on scipy.optimize.root.  No obvious way to prevent it from taking large steps.
#     """
#     Changes the expected T to new value.  Adjusts PS and VGA input to match.  
#     Utilizes an inverse function to do so.
#     """
#     self.TExpected = T
#     startPos = [self.psSetting, self.vgaSetting]
#     def FRoot(xvec):
#         x, y = xvec
#         rO = np.real(self.F(x, y) - self.TExpected)
#         iO = np.imag(self.F(x, y) - self.TExpected)
#         return np.array([rO, iO])
#     try:
#         soln = root(FRoot, startPos, method='broyden1', options={'factor':0.1, 'eps':0.001})
#     except ValueError:
#         print("starting over at [512, 512]")
#         soln = root(FRoot, [512, 512], options={'factor':0.1, 'eps':0.001})
#     (ps, vga) = np.round(soln['x']).astype(np.int)
#     self.psSetting = ps
#     self.vgaSetting = vga
    
# setattr(Multiplier, "setT", setT)


# In[ ]:


def dumbRoot2D(F, start, bounds, target):
    iBest = -1
    offsets = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1]])
    pos = np.array(start).copy()
    ((xMin, xMax), (yMin, yMax)) = bounds
    inBounds = lambda xv: xMin <= xv[0] <= xMax and yMin <= xv[1] <= yMax
    onEdge = lambda xv: xv[0] == xMin or xv[0] == xMax or xv[1] == yMax # Does not include yMin
    while iBest != 0:
        evalPts = filter(inBounds, pos + offsets)
        scores = [abs(F(*pt) - target) for pt in evalPts]
        iBest = np.argmin(scores)
        bestOffset = offsets[iBest]
        bestScore = scores[iBest]
        pos = pos + bestOffset
    if onEdge(pos):
        raise ValueError("Optimal soln found on edge.  Not trusting it.")
    if bestScore > 0.02:
        raise ValueError("Optimal soln wasn't very good.  Not trusting it.")
    return (pos, bestScore)


# In[ ]:


def makeEvalDisk(n):
    allPts = np.dstack(np.meshgrid(range(-(n//2), n//2+1), range(-(n//2), n//2+1))).reshape(-1,2) 
    allPts[(n**2)//2] = allPts[0]
    allPts[0] = [0,0]
    return allPts


# In[ ]:


offsets = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1]])
offsets1 = makeEvalDisk(3)
offsets2 = makeEvalDisk(5)


# In[ ]:


def dumbRoot2D(F, start, bounds, target, verbose=False):
    iBest = -1
    offsets0 = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1]])
    offsets1 = makeEvalDisk(3)
    offsets2 = makeEvalDisk(5)
    pos = np.array(start).copy()
    ((xMin, xMax), (yMin, yMax)) = bounds
    inBounds = lambda xv: xMin <= xv[0] <= xMax and yMin <= xv[1] <= yMax
    onEdge = lambda xv: xv[0] == xMin or xv[0] == xMax or xv[1] == yMax # Does not include yMin
    posList = [pos]
    posSet = {tuple(pos)}
    offsets = offsets0
    while iBest != 0:
        evalPts = filter(inBounds, pos + offsets)
        scores = [abs(F(*pt) - target) for pt in evalPts]
        if verbose: print(scores)
        iBest = np.argmin(scores)
        bestOffset = offsets[iBest]
        bestScore = scores[iBest]
        pos = pos + bestOffset
        if verbose: print(pos)
        posList.append(pos)
        if (tuple(pos) in posSet) and (iBest != 0):
            # print(np.array(posList))
            # print("   Loop Back Detected. bestScore:", bestScore)
            offsets = offsets2
        else:
            offsets = offsets0
        posSet.add(tuple(pos))
    if onEdge(pos):
        raise ValueError("   Optimal soln found on edge.  Not trusting it.")
    if bestScore > 0.02:
        raise ValueError("   Optimal soln wasn't very good.  Not trusting it.")
    return (pos, bestScore)


# In[ ]:


def setT(self, T, verbose=False):
    """
    Changes the expected T to new value.  Adjusts PS and VGA input to match.  
    Utilizes an inverse function to do so.
    """
    psSPs, vgaSPs = Multiplier.psSPs, Multiplier.vgaSPs
    bounds = ((psSPs[0], psSPs[-1]), (vgaSPs[0], vgaSPs[-1]))
    (pos, bestScore) = dumbRoot2D(self.F, [512, 512], bounds, T, verbose)
    (ps, vga) = pos
    self.psSetting = ps
    self.vgaSetting = vga
    self.TExpected = self.F(ps, vga)
    
setattr(Multiplier, "setT", setT)


# In[ ]:


def adjustT(self, T, verbose=False):
    """
    Changes the expected T to new value.  Adjusts PS and VGA input to match.  
    Utilizes an inverse function to do so.
    """
    startPos = [self.psSetting, self.vgaSetting]
    psSPs, vgaSPs = Multiplier.psSPs, Multiplier.vgaSPs
    bounds = ((psSPs[0], psSPs[-1]), (vgaSPs[0], vgaSPs[-1]))
    (pos, bestScore) = dumbRoot2D(self.F, startPos, bounds, T, verbose)
    (ps, vga) = pos
    self.psSetting = ps
    self.vgaSetting = vga
    self.TExpected = self.F(ps, vga)
    
setattr(Multiplier, "adjustT", adjustT)


# In[ ]:


def setWeights(self, weights, hold='settings'):
    self.weights[:] = weights[:]
    self.field = np.sum(comps * weights.reshape(-1,1,1), axis=0)
    rF = interp2d(Multiplier.psSPs, Multiplier.vgaSPs, np.real(self.field), kind='linear', bounds_error=True)
    iF = interp2d(Multiplier.psSPs, Multiplier.vgaSPs, np.imag(self.field), kind='linear', bounds_error=True)
    def F(x, y):
        z, = rF(x, y, assume_sorted=True) + 1j*iF(x, y, assume_sorted=True)
        return z
    self.F = F
    if hold == 'settings':
        # Sets settings to orig values, and tweaks T.
        self.setSettings(self.psSetting, self.vgaSetting)
    elif hold == 'T':
        # Sets T to orig values, and tweaks settings.
        self.setT(self.T)

setattr(Multiplier, "setWeights", setWeights)


# Define a Multiplier

# In[ ]:


m1 = Multiplier()


# In[ ]:


if mainQ: plotComplexArray(m1.field, maxRad=1)


# In[ ]:


(psSamplePoints[40], vgaSamplePoints[30])


# In[ ]:


testSample = m1.field[30,40] # y, x
testSampleF = m1.F(psSamplePoints[40], vgaSamplePoints[30])
(testSample, testSampleF)


# In[ ]:


m1.weights


# In[ ]:


m1.setSettings(500, 500)
m1.TExpected, m1.psSetting, m1.vgaSetting


# In[ ]:


T = -0.3-0.3j
m1.setT(T)
m1.TExpected, m1.psSetting, m1.vgaSetting


# In[ ]:


m1.setSettings(psSamplePoints[40], vgaSamplePoints[30])
m1.TExpected, m1.F(psSamplePoints[40], vgaSamplePoints[30]), m1.psSetting, m1.vgaSetting


# In[ ]:


sf = (1.2*np.exp(1j*0.3))
w = baseWeights*sf
m1.setWeights(w, hold='settings')
TNew = m1.TExpected
m1.TExpected, m1.psSetting, m1.vgaSetting


# In[ ]:


TNew/testSampleF , sf


# In[ ]:


testSample2 = m1.field[30,40] # [y, x]
testSampleF2 = m1.F(vgaSamplePoints[40], vgaSamplePoints[30])
testSample2/testSample, sf


# In[ ]:


if mainQ: plotComplexArray(m1.field)


# ## Define Bank

# In[ ]:


class MultiplierBank:
    pass


# In[ ]:


def __init__(self):
    self.bankByLoc = dict()
    self.bankByPhysNum = dict()
    
setattr(MultiplierBank, "__init__", __init__)


# In[ ]:


def addMult(self, mult):
    loc, pNum = mult.loc, mult.physNumber
    if (loc in self.bankByLoc or pNum in self.bankByPhysNum):
        print("not added due to redundancy")
        return
    self.bankByLoc[loc] = mult
    self.bankByPhysNum[pNum] = mult    

setattr(MultiplierBank, "addMult", addMult)


# In[ ]:


def getMults(self):
    return list(self.bankByLoc.values())

setattr(MultiplierBank, "getMults", getMults)


# In[ ]:


def getMultByPhysNum(self, pNum):
    return self.bankByLoc

setattr(MultiplierBank, "getMultByPhysNum", getMultByPhysNum)


# In[ ]:


def getMultByLoc(self, loc):
    return self.bankByLoc[loc]

setattr(MultiplierBank, "getMultByLoc", getMultByLoc)


# In[ ]:


def getLocs(self):
    locs = list(self.bankByLoc.keys())
    locs.sort()
    return locs

setattr(MultiplierBank, "getLocs", getLocs)


# In[ ]:


def getPhysNums(self):
    pNums = list(self.bankByPhysNum.keys())
    pNums.sort()
    return pNums

setattr(MultiplierBank, "getPhysNums", getPhysNums)


# In[ ]:


def getPersonalityVectors(self):
    """
    The personality weights of each multiplier will need to be optimized.  It
    will be necessary to treat all of these as a 1D List.  This provides an
    interface to do so.
    
    Weights are returned in an order based on the multiplier locations.
    """
    locs = self.getLocs()
    listOfWeights = [self.bankByLoc[l].weights for l in locs]
    weights1DComplex = np.array(listOfWeights).flatten()
    weights1DReal = weights1DComplex.view('float')
    return weights1DReal

setattr(MultiplierBank, "getPersonalityVectors", getPersonalityVectors)


# In[ ]:


def setAllMults(self, psVal, vgaVal):
    for mult in self.getMults():
        mult.setSettings(psVal, vgaVal)
        
setattr(MultiplierBank, "setAllMults", setAllMults)        


# In[ ]:


def setPersonalityVectors(self, weights1DReal):
    """
    The personality weights of each multiplier will need to be optimized.  It
    will be necessary to treat all of these as a 1D List.  This provides an
    interface to do so.
    
    Weights are set in an order based on the multiplier locations.
    """
    weights1DComplex = weights1DReal.view('complex')
    locs = self.getLocs()
    vSplit = np.split(weights1DComplex, len(locs))
    for i, loc in enumerate(locs):
        mult = self.bankByLoc[loc]
        mult.setWeights(vSplit[i])

setattr(MultiplierBank, "setPersonalityVectors", setPersonalityVectors)


# In[ ]:


def getRFNetwork(self, loc):
    mult = self.getMultByLoc(loc)
    network = mult.getRFNetwork()
    return network

setattr(MultiplierBank, "getRFNetwork", getRFNetwork)


# In[ ]:


mBank = MultiplierBank()


# In[ ]:


m1 = Multiplier(physNumber=3, loc=('U', 1, 2, 'top'))
m2 = Multiplier(physNumber=4, loc=('U', 1, 2, 'bot'))


# In[ ]:


mBank.addMult(m1)
mBank.addMult(m2)


# In[ ]:


m2.weights


# In[ ]:


mBank.getPersonalityVectors()


# # 3dB Coupler

# ## Couplers from Simulation

# In[ ]:


fname1 = "..\\GoldenSamples\\coupler3dBFromSim.s4p"
tsImport = rf.io.touchstone.Touchstone(fname1)


# In[ ]:


(fCouplerSim, SCouplerSim) = tsImport.get_sparameter_arrays()
i45 = np.argmin(np.abs(fCouplerSim-45e6))
sCouplerSim45 = SCouplerSim[i45]

def Build3dBCouplerSim(freq, loc=()):
    nFreqs = len(freq)
    S = np.zeros((nFreqs, 4, 4), dtype=np.complex)
    Z_0 = 50.
    label = str(loc)
    S[:] = sCouplerSim45    
    net = rf.Network(name=label, frequency=freq, z0=Z_0, s=S)
    return net


# In[ ]:


freq45 = rf.Frequency(start=45, stop=45, npoints=1, unit='mhz', sweep_type='lin')


# In[ ]:


netCouplerSim45 = Build3dBCouplerSim(freq45)
netCouplerSim45.s[0]


# ## Couplers from Measured Data

# In[ ]:


from NetworkBuilding import Build3dBCoupler


# In[ ]:


import glob


# In[ ]:


fnames = glob.glob("..\\GoldenSamples\\CouplerSamples\\??_3dB_[0-9].txt")
fnames


# In[ ]:


# fnames = ["..\\GoldenSamples\\CouplerSamples\\3dB_couplers1.txt",
#           "..\\GoldenSamples\\CouplerSamples\\3dB_couplers2.txt"]


# In[ ]:


coupDataFromFiles000 = [np.loadtxt(f, dtype=np.complex) for f in fnames]


# In[ ]:


def Build3dBCouplerFromData(freq, loc="", force=None):
    if force is None:
        choice = hash(loc)%len(coupDataFromFiles000)
    else:
        choice = force%len(coupDataFromFiles000)
    ((S31, S32), (S41, S42)) = coupDataFromFiles000[choice]
    Z_0 = 50.
    label = str(loc)
    
    nFreqs = len(freq)

    S = np.zeros((nFreqs, 4, 4), dtype=np.complex)
    S[:, 2, 0] = S31
    S[:, 3, 0] = S41
    S[:, 2, 1] = S32
    S[:, 3, 1] = S42
    S[:, 0, 2] = S31
    S[:, 1, 2] = S41
    S[:, 0, 3] = S32
    S[:, 1, 3] = S42
    coupNetwork = rf.Network(name=label, frequency=freq, z0=Z_0, s=S)
    return coupNetwork


# In[ ]:


freq45 = rf.Frequency(start=45, stop=45, npoints=1, unit='mhz', sweep_type='lin')


# In[ ]:


coupNet1 = Build3dBCouplerFromData(freq45, "foo", force=0) # hash("foo")%2 = 1
coupNet2 = Build3dBCouplerFromData(freq45, "bar", force=1) # hash("bar")%2 = 0


# For comparison, here is an ideal coupler.

# In[ ]:


coupNetIdeal = Build3dBCoupler(freq45, couplerConv="LC", loc="")
coupNetIdeal.s[0]


# Next is a coupler based on simulated data.

# In[ ]:


netCouplerSim45 = Build3dBCouplerSim(freq45)
netCouplerSim45.s[0]


# Next are two couplers based on imported data

# In[ ]:


coupNet1.s[0]


# In[ ]:


coupNet2.s[0]


# In[ ]:


10*np.log10(np.abs(-0.214-0.383j))


# In[ ]:


pp = PolarPlot("3dB Couplers")


# In[ ]:


pp.addMatrix(coupNetIdeal.s[0, 2:, :2], 'green')
for i, _ in enumerate(fnames):
    coupNet = Build3dBCouplerFromData(freq45, loc="", force=i)
    pp.addMatrix(coupNet.s[0, 2:, :2], 'red')


# In[ ]:


if mainQ: pp.show()


# In[ ]:




