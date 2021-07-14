#!/usr/bin/env python
# coding: utf-8

# ## Setup

# ### Notebook Customization

# In[ ]:


import IPython


# In[ ]:


css_str = """
<link rel="preconnect" href="https://fonts.gstatic.com">

<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=Playfair+Display+SC&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Lora">
<link href="https://fonts.googleapis.com/css2?family=IM+Fell+Double+Pica:ital@1&display=swap" rel="stylesheet">
    <style>
h1 { color: #7c795d; font-family: 'Playfair Display SC', serif; text-indent: 00px; text-align: center;}
h2 { color: #7c795d; font-family: 'Lora', serif;                text-indent: 00px; text-align: left; }
h3 { color: #7c795d; font-family: 'IM Fell Double Pica', serif; text-indent: 15px; text-align: left; }
h4 { color: #7c795d; font-family: 'Lora', Arial, serif;         text-indent: 30px; text-align: left}
h5 { color: #71a832; font-family: 'IM Fell Double Pica', serif; text-indent: 45px; text-align: left}

"""
IPython.display.HTML(css_str)


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Python Libraries

# Standard Python imports

# In[ ]:


import os, sys, time, glob, random


# It is a good idea to record the starting working directory before it gets changed around.  Note that this can be problematic depending on how you open the notebook, but it works most of the time.

# In[ ]:


baseDir = os.getcwd()
baseDir


# It is also nice to know if what is running is a notebook, or a python script generated from the notebook.

# In[ ]:


mainQ = (__name__ == '__main__')
if mainQ:
    print("This is the main file for this kernel")


# ### Functional Programming

# The `toolz` library has a lot of great functions for performing common operations on iterables, functions, and dictionaries.

# In[ ]:


# from toolz.itertoolz import ()
from toolz.functoolz import (curry, pipe, thread_first)
# from toolz.dicttoolz import ()


# In[ ]:


@curry
def add(x, y): return x + y
@curry
def pow(x, y): return x**y
thread_first(1, add(y=4), pow(y=2))  # pow(add(1, 4), 2)


# In[ ]:


from mini_lambda import InputVar, as_function
_ = as_function
X = InputVar('X')


# In[ ]:


_(X+3)(10)


# In[ ]:


thread_first(1, add(y=4), _(pow(x=2, y=X)))  # pow(2, add(1, 4))


# In[ ]:


thread_first(1, _(X+4), _(2**X))  # pow(add(1, 4), 2)


# ### Scientific Programming

# In[ ]:


from math import *
deg = radians(1)    # so that we can refer to 90*deg
I = 1j              # potentially neater imaginary nomenclature.


# In[ ]:


import numpy as np  # Does high performance dense array operations
np.set_printoptions(edgeitems=30, linewidth=100000,
                    formatter=dict(float=lambda x: "%.3g" % x))
import scipy as sp
import pandas as pd
import PIL 


# In[ ]:


import skimage


# In[ ]:


# Python function compilization.  Makes things very fast.  Function must only include Numpy and basic Python.  No custom classes.
import numba
from numba import njit


# In[ ]:


import sympy
# sp.init_printing(pretty_print=True)
# sp.init_printing(pretty_print=False)


# In[ ]:


import importlib


# In[ ]:


from scipy.optimize import root, minimize
from scipy.interpolate import interp2d


# In[ ]:


from scipy.ndimage import gaussian_filter
from scipy import interpolate


# ### Plotting

# In[ ]:


import bokeh
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
output_notebook()
from bokeh.palettes import Dark2
bokeh.io.curdoc().theme = 'dark_minimal'
palette = Dark2[8]*10


# In[ ]:


import itertools


# In[ ]:


palette = Dark2[8]*10
colors = itertools.cycle(palette)


# In[ ]:


from UtilityMath import plotComplexArray


# In[ ]:


import skrf as rf


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


from UtilityMath import (convertArrayToDict, MatrixError, MatrixSqError, makePolarPlot, addMatrixDiff, PolarPlot, ReIm, 
                         RandomComplexCircularMatrix, PolarPlot)


# ## Definitions (Sim)

# In[ ]:


freq45 = rf.Frequency(start=45, stop=45, npoints=1, unit='mhz', sweep_type='lin')


# First we need to generate labels for the Multipliers.  For the New architecture, this is a simple square grid.  The format is
# 
# `('M', 'N', inputLine, outputLine)` 
# 
# where `'M'` is for "Multiplier", `'N'` is for "New" and `inputLine` and `outputLine` are integers in the range [0,4].

# In[ ]:


allMultLocs = NewMultLocs(5, 'N')
allMultLocs


# Every device has a "Physical Number" that is used for addressing to allow the computer to specify to which device a command is intended.  These are enumarated below.  Similar to SParams, the rows denote output lines while the columns denote input lines.

# In[ ]:


# Be careful here.  A horizontal row in the physical world represents a column in matrix multiplication
multPhysNumberBank = [[31, 32, 33, 34, 35],
                      [11, 12, 13, 14, 15],
                      [16, 17, 18, 19, 20],
                      [21, 22, 23, 24, 25],
                      [26, 27, 28, 29, 30]]
multPhysNumberBank = np.array(multPhysNumberBank).T
multPhysNumberBank


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

# In[ ]:


X0 = multBank.getPersonalityVectors()


# # Large Number Experiments

# In[ ]:


IdM5 = np.identity(5)


# ## Looking for Weird Multiplier Channels

# In[ ]:


XF = np.load("Main_data/personalityVector.npy")


# In[ ]:


weights1DComplex = XF.view('complex')
locs = multBank.getLocs()
pcaWeights = np.array(np.split(weights1DComplex, len(locs)))


# In[ ]:


maxWeight = np.max(np.abs(pcaWeights))


# In[ ]:


def getRandomHexColor():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    hexStr = '#%02X%02X%02X' % (r, g, b)
    return hexStr


# In[ ]:


pp = PolarPlot("PCA Weight Comparison")
pcaWeightsT = pcaWeights.T
for i in range(len(pcaWeightsT)):
    pp.addMatrix(pcaWeightsT[i]/maxWeight, getRandomHexColor())
pp.show()


# In[ ]:


pp = PolarPlot("PCA Weight Comparison")
for i in range(len(pcaWeights)):
    pp.addMatrix(pcaWeights[i]/maxWeight, getRandomHexColor())
pp.show()


# ## Big Kappa

# In[ ]:


# to be uncomment for a new set of values
# inputKernels = [RandomComplexCircularMatrix(0.5, (5,5)) for i in range(1000)]
# np.save("Main_data/largeNInput", np.array(inputKernels))


# In[ ]:


inputKernels = np.load("Main_data/largeNInput.npy")


# In[ ]:


outputMeasurements = np.load("Main_data/largeNOpenLoop.npy")


# In[ ]:


inKK = []
measKK = []
setKK = []
for i in np.arange(1000):
    inKK.append(np.array(outputMeasurements)[i, 0])  # Goal Input Kernel
    setKK.append(np.array(outputMeasurements)[i, 1]) # Experimentally Expected Kernel based on Integer Control Values
    measKK.append(np.array(outputMeasurements)[i, 2]) # Measured Open Loop Kernel


# In[ ]:


pp = PolarPlot("comparisson")
i = 154
pp.addMatrixDiff(np.array(measKK[i]), np.array(inKK[i]))
pp.show()
abs(np.linalg.eig(inKK[i])[0])


# In[ ]:


getEigVals = np.linalg.eigvals


# In[ ]:


def getMaxEigVal(K):
    return thread_first(K, getEigVals, abs, max)


# In[ ]:


eigValOrdering = np.argsort(list(map(getMaxEigVal, inKK)))


# In[ ]:


def MatrixError(m, mTarget):
    """
    Computes how far the average S-Param is from the expected, divided by the magnitude of an SParam in a Unitary Matrix of equivalent size.
    """
    errorSq = (np.abs(m - mTarget)**2).sum()
    nElems = len(inKK[0].flatten())
    aveError = (errorSq**0.5)/nElems
    (nR, nC) = m.shape
    aveValMag = (1/nR)**0.5  # Assuming equal power distribution, this is the average SParam mag
    relAveError = aveError/aveValMag
    return relAveError


# In[ ]:


MatrixError(measKK[0], inKK[0])


# In[ ]:


errors = [MatrixError(measK, goalK) for measK, goalK in zip(measKK, inKK)]


# In[ ]:


fig1 = figure(title="Error of K as Sampled", x_axis_label="sample", y_axis_label="Error")
fig1.line(range(len(errors)), errors, line_width=2)
show(fig1)


# In[ ]:


fig1 = figure(title="Error of K Sorted by Max EigenValue", x_axis_label="Max Eigenvalue", y_axis_label="Error")
fig1.scatter(list(map(getMaxEigVal, inKK)), errors, color='#ff0000', fill_alpha=0.3, line_alpha=0)
show(fig1)


# Below, we compute the percentage error on each element relative to that of a equipower matrix.

# In[ ]:


np.sum(np.array([np.abs(inK - measK) for inK, measK in zip(inKK, measKK)]), axis=0)*(100 / ((1/5)**0.5 * len(inKK)))


# In[ ]:


kGoal = inKK[0]
measK = measKK[0]
def errorFun(X):
    realPert, imagPert = X
    error = (np.abs(kGoal - (1+realPert + I*imagPert)*measK)**2).sum()
    return error


# In[ ]:


errorFun([0,0])


# In[ ]:


pVals = []
for i in range(len(inKK)):
    kGoal = inKK[i]
    measK = measKK[i]
    def errorFun(X):
        realPert, imagPert = X
        error = (np.abs(kGoal - (1+realPert + I*imagPert)*measK)**2).sum()
        return error
    sol = sp.optimize.minimize(errorFun, [0,0])
    pr, pi = sol.x
    p = pr + I*pi
    pVals.append(p)


# In[ ]:


pAve = 1+np.mean(pVals)


# In[ ]:


fig1 = figure(title="Tweak Factor", x_axis_label="sample", y_axis_label="Error")
fig1.line(range(len(errors)), np.real(pVals), line_width=2, color='red')
fig1.line(range(len(errors)), np.imag(pVals), line_width=2, color='blue')
show(fig1)


# In[ ]:


# sParamElemDict = {(r+1, c+1):np.abs(np.array(inKK)[:, r, c]) for r in range(5) for c in range(5)}


# In[ ]:


# for k in sParamElemDict.keys():
#     (r, c) = k
#     fig1 = figure(title="Error of K Sorted by Element", x_axis_label="Element Magnitudes", y_axis_label="Error")
#     fig1.scatter(sParamElemDict[k], errors, color=getRandomHexColor(), fill_alpha=0.3, line_alpha=0, legend_label=str(k))
# show(fig1)


# ## Load Coupler Values

# In[ ]:


coup1 = np.loadtxt("../GoldenSamples/FeedbackCouplerSamples/coupler_1.txt", dtype=complex)
coup2 = np.loadtxt("../GoldenSamples/FeedbackCouplerSamples/coupler_1_v2.txt", dtype=complex)
alpha1 = np.mean([coup1[0,0], coup2[0,0]])
alpha2 = np.mean([coup1[1,1], coup2[1,1]])
beta = np.mean([coup1[0,1], coup1[1,0], coup2[0,1], coup2[1,0]])


# In[ ]:


# Load the coupler parameters
coup1 = np.loadtxt("../GoldenSamples/FeedbackCouplerSamples/coupler_1.txt", dtype=complex)
coup2 = np.loadtxt("../GoldenSamples/FeedbackCouplerSamples/coupler_1_v2.txt", dtype=complex)
alpha1 = np.mean([coup1[0,0], coup2[0,0]])
alpha2 = np.mean([coup1[1,1], coup2[1,1]])
beta = np.mean([coup1[0,1], coup1[1,0], coup2[0,1], coup2[1,0]])


# In[ ]:


coup1, coup2


# In[ ]:


pp = PolarPlot("Couplers")
pp.addMatrix(coup1, color='cyan')
pp.addMatrix(coup2, color='green')
pp.show()


# ## Inverse exp - post processing

# In[ ]:


outputMeasurementsINV = np.load("Main_data/largeNClosedLoop.npy")


# In[ ]:


measCLs = []
for i in np.arange(1000):  
    # inKK_INV.append(np.array(outputMeasurementsINV)[i, 0]) # Goal Input Kernel  (Identical to Open Loop Data Set)
    # setKK_INV.append(np.array(outputMeasurementsINV)[i, 1]) # Experimentally Expected Kernel based on Integer Control Values (Identical to Open Loop Data Set if performed using same Pers Vec)
    measCLs.append(np.array(outputMeasurementsINV)[i, 2]) # Measured Closed Loop Result


# In[ ]:


pAve


# In[ ]:


goal_Invs = [np.linalg.inv(IdM5 - inK) for inK in inKK]
meas_Invs = [((1 - alpha1*alpha2/beta**2)*IdM5 + (alpha1/beta**2)*measCL) for measCL in measCLs]


# In[ ]:


pp = PolarPlot("comparisson")
i = 939
pp.addMatrixDiff(goal_Invs[i], meas_Invs[i])
pp.show()
abs(np.linalg.eig(inKK[i])[0])


# In[ ]:


errors = [MatrixError(meas_Inv, goal_Inv) for meas_Inv, goal_Inv in zip(meas_Invs, goal_Invs)]


# In[ ]:


np.argmin(errors)


# In[ ]:


fig1 = figure(title="Error of Inverse as Sampled", x_axis_label="sample", y_axis_label="Error",  y_range=(0, 1))
fig1.scatter(range(len(errors)), errors, color='#ff0000', fill_alpha=0.3, line_alpha=0)
show(fig1)


# In[ ]:


fig1 = figure(title="Inversion Error Sorted by Max EigenValue", x_axis_label="Max Eigenvalue", y_axis_label="Error", y_range=(0, 1))
fig1.scatter(list(map(getMaxEigVal, inKK)), errors, color='#ff0000', fill_alpha=0.3, line_alpha=0)
show(fig1)


# In[ ]:


pVals = []
for i in range(len(inKK)):
    inK = inKK[i]
    measCL = measCLs[i]
    def errorFun(X):
        realPert, imagPert = X
        goal_Inv = np.linalg.inv(IdM5 - (1 + realPert + I*imagPert)*inK)
        meas_Inv = ((1 - alpha1*alpha2/beta**2)*IdM5 + (alpha1/beta**2)*measCL)                                 
        error = (np.abs(goal_Inv - meas_Inv)**2).sum()
        return error
    sol = sp.optimize.minimize(errorFun, [0,0])
    pr, pi = sol.x
    p = pr + I*pi
    pVals.append(p)


# In[ ]:


fig1 = figure(title="Tweak Factor", x_axis_label="sample", y_axis_label="Error", y_range=(-0.4, 0.4))
fig1.line(range(len(errors)), np.real(pVals), line_width=2, color='red')
fig1.line(range(len(errors)), np.imag(pVals), line_width=2, color='blue')
show(fig1)


# In[ ]:


pAve = 1 + np.mean(pVals)


# In[ ]:


goal_Invs = [np.linalg.inv(IdM5 - (pAve)*inK) for inK in inKK]
meas_Invs = [((1 - alpha1*alpha2/beta**2)*IdM5 + (alpha1/beta**2)*measCL) for measCL in measCLs]


# In[ ]:


pp = PolarPlot("comparisson")
i = 939
pp.addMatrixDiff(goal_Invs[i], meas_Invs[i])
pp.show()
abs(np.linalg.eig(inKK[i])[0])


# In[ ]:


errors = [MatrixError(meas_Inv, goal_Inv) for meas_Inv, goal_Inv in zip(meas_Invs, goal_Invs)]


# In[ ]:


np.argmin(errors)


# In[ ]:


fig1 = figure(title="Error of Inverse as Sampled", x_axis_label="sample", y_axis_label="Error",  y_range=(0, 1))
fig1.scatter(range(len(errors)), errors, color='#ff0000', fill_alpha=0.3, line_alpha=0)
show(fig1)


# In[ ]:


fig1 = figure(title="Inversion Error Sorted by Max EigenValue", x_axis_label="Max Eigenvalue", y_axis_label="Error", y_range=(0, 1))
fig1.scatter(list(map(getMaxEigVal, inKK)), errors, color='#ff0000', fill_alpha=0.3, line_alpha=0)
show(fig1)


# # Scrap

# In[ ]:


inputKernels = np.load("largeNInput_v4.npy")


# In[ ]:


outputMeasurementsINV = []
badMats = []
for inK in inputKernels:
    try:
        setK = calcNewMatrixSettings(inK, multBank, 5,  warn=False)
        setExpMultBank(exp, multBank)
        measK, std = exp.measureSMatrix(delay=2)
        saveData = (inK, setK, measK)
        outputMeasurementsINV.append(saveData)
    except:
        badMats.append(inK)


# In[ ]:


#save or load the measurement results
# np.save("largeNOut_100.npy", np.array(outputMeasurements))
np.save("largeNInput_v4_INV_new.npy", np.array(outputMeasurementsINV))


# In[ ]:


# outputMeasurements=np.load("largeNOpenLoop_v4.npy")
# outputMeasurementsINV=np.load("largeNInput_v4_INV.npy")


# In[ ]:


pp = PolarPlot("comparisson")
i=5
pp.addMatrixDiff(np.array(np.identity(5)-inKK[i]),np.array(np.identity(5)-inKK_INV[i]))
pp.show()
abs(np.linalg.eig(inKK[i])[0])


# In[ ]:


A_goal = np.identity(5)-inKK


# In[ ]:


0*np.identity(5)-inKK


# In[ ]:


A_goal = (np.identity(5)-inKK)
A_goal_inv = np.linalg.inv(A_goal)
A_exp = np.array(measKK_INV)
A_meas_inv = [(alpha1/beta*measKK_INV+(beta-alpha1*alpha2/beta)*np.identity(5))/beta for measK_INV in measK_INV]


# In[ ]:


tr=np.arange(1000, dtype="complex")
trN=np.arange(1000, dtype="complex")
diff=np.arange(1000, dtype="complex")
diff_INV=np.arange(1000, dtype="complex")
for i in np.arange(1000):
    tr[i] = (np.trace(np.dot(A_goal[i],A_meas_inv[i])))/5
    coef=1.25
    trN[i]= abs(np.trace(np.dot(A_goal[i],A_meas_inv[i]/coef)))/5
    diff[i] = MatrixError(inKK[i],measKK[i])/np.linalg.norm(inKK[i])*100
    diff_INV[i] = MatrixError(A_goal_inv[i],A_meas_inv[i])/np.linalg.norm(A_goal_inv[i])*100


# In[ ]:


import matplotlib.pyplot as plt
index = np.arange(1000)
plt.scatter(index, np.real(tr))
plt.xlabel('# random matrix')
plt.ylabel('$|$tr$(A_{target} A^{-1}_{meas}/5|$')
plt.ylim([0, 2.5])
# plt.savefig('BigKappa_rand100_v2.png',dpi=1200)
np.mean(tr)


# In[ ]:


pp = PolarPlot("comparisson")
i=876
coef=np.linalg.norm(A_meas_inv[i]/A_goal_inv[i])/5*0+1
A1=np.array(A_goal_inv[i])
A2=np.array(A_meas_inv[i])/coef
pp.addMatrixDiff(A1,A2)
pp.show()
print(coef,abs(np.linalg.eig(inKK_INV[i])[0]),MatrixError(A1,A2)/np.linalg.norm(A1)*100)
# np.linalg.norm(A_meas_inv[i]/A_goal_inv[i])/5


# In[ ]:


abs(np.linalg.eig(np.identity(5)-inKK_INV[i])[0])


# #### Test Playground

# In[ ]:


from bokeh.models import Range1d

p = figure(plot_width = 600, plot_height = 600,
           x_axis_label = '# random matrix', y_axis_label = '$||\mathbb{K}_{Meas}-\mathbb{K}_{target}||$')
p.y_range=Range1d(0,0.5)
p.circle(1+index, diff, size = 10, color = 'green')
show(p)


# In[ ]:





# In[ ]:


pp = PolarPlot("comparisson")
i=755
pp.addMatrixDiff(np.array(tt[i]),np.array(tt[i]))
pp.show()


# In[ ]:


np.linalg.inv(np.identity(5)-inKK[0])
np.linalg.eig(np.identity(5)-inKK[0])


# In[ ]:


inKK=np.array(outputMeasurements)[55,0]
measKK=np.array(outputMeasurements)[55,2]
tt=np.matmul(np.linalg.inv(inKK),measKK)


# In[ ]:


MatrixSqError(inKK,measKK)


# In[ ]:


test=np.array(outputMeasurements)


# In[ ]:


test[1,1]

