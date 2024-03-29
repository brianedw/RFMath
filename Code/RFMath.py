#!/usr/bin/env python
# coding: utf-8

# ## Imports

# ### Stock Imports

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import os, sys
from time import sleep


# In[ ]:


os.getcwd()


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


from UtilityMath import plotComplexArray


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


from UtilityMath import (convertArrayToDict, MatrixError, MatrixSqError, makePolarPlot, addMatrixDiff, PolarPlot, ReIm, 
                         RandomComplexCircularMatrix, PolarPlot)


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

# ## Definitions (Exp)

# First we define the various devices.

# In[ ]:


inputSwitchComm = SwitchComm(comValue='COM4', portAliases={1:6, 2:5, 3:4, 4:3, 5:2, "test":1})
outputSwitchComm = SwitchComm(comValue='COM3', portAliases={1:3, 2:4, 3:5, 4:6, 5:7, "test":1})
vnaComm = VNAComm()
multBankComm = MultBankComm(comValue='COM5')


# In[ ]:


exp = ExperimentalSetup(inputSwitchComm, outputSwitchComm, multBankComm, vnaComm)


# For convenience, higher level scripts that require coordination between the various devices can be accessed using an `ExperimentalSetup`.

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
allMultLocs;


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


# ## Tuning

# ### Debugging

# In[ ]:


for loc in allMultLocs:
    mult = multBank.getMultByLoc(loc)
    try: 
        multBankComm.blinkMult(mult.physNumber)
    except NameError:
        pass        
    sleep(0.2)


# In[ ]:


inputSwitchComm.setSwitch(5)
outputSwitchComm.setSwitch(5)
vnaComm.getS21AllAt45()


# In[ ]:


coupler1_J21=vnaComm.getS21AllAt45()


# In[ ]:


coupler1_J41=vnaComm.getS21AllAt45()


# In[ ]:


coupler1_J43=vnaComm.getS21AllAt45()


# In[ ]:


coupler1_J23=vnaComm.getS21AllAt45()


# In[ ]:


coupler5= np.array([coupler1_J21,coupler1_J41, coupler1_J23, coupler1_J43])


# In[ ]:


np.save("New_couplers_data/coupler5", coupler5)


# In[ ]:


print(abs(coupler1[:,0]),abs(coupler2[:,0]),abs(coupler3[:,0]),abs(coupler4[:,0]),abs(coupler5[:,0]))


# In[ ]:


outIndex = 1
inIndex = 5
vga, ps = (1000, 1000)
loc = ('M', 'N', inIndex-1, outIndex-1) # ('M', 'N', in, out) :(.
mult = multBank.getMultByLoc(loc)
physNum = mult.physNumber
print(physNum)
multBankComm.setMult(physNum, vga, ps)
inputSwitchComm.setSwitch(inIndex)
outputSwitchComm.setSwitch(outIndex)
sleep(2)
vnaComm.getS21AllAt45()


# In[ ]:


aa=vnaComm.getS21AllAt45()
abs(aa[0])


# In[ ]:


multBankComm.setMult(28, 100, 200)


# In[ ]:


# inputSwitchComm.portAliases=None
# outputSwitchComm.portAliases=None


# In[ ]:


inputSwitchComm.setSwitch(1, verbose=True)
outputSwitchComm.setSwitch(1, verbose=True)
exp.vnaComm.getS21AllAt45()


# In[ ]:


inputSwitchComm.close()
outputSwitchComm.close()


# In[ ]:


outputSwitchComm.setSwitch("test")


# In[ ]:


exp.setMults(0, 100, multBank.getPhysNums())


# In[ ]:


exp.vnaComm.getS21AllAt45()


# In[ ]:


SMat, STD = exp.measureSMatrix(delay=2)


# In[ ]:


np.abs(SMat)


# ### Physical Measurement

# Next we define a series of multiplier set points that we'll use to ascertain the multiplier's PCA weights.

# In[ ]:


tuningPSVals = np.linspace(0, 1023, 25, dtype=np.int)
tuningVGAVals = np.linspace(0, 1023, 25, dtype=np.int)


# In[ ]:


tuningVals = [(ps, vga) for vga in tuningVGAVals for ps in tuningPSVals]


# For each PS, VGA pair, the multipliers are uniformly set and the scattering matrix of the network is measured.

# In[ ]:


tuningMatricesM = []
for (psVal, vgaVal) in tuningVals:
    exp.setMults(int(psVal), int(vgaVal), multBank.getPhysNums())
    time.sleep(1)
    m, std = exp.measureSMatrix(delay=2)
    tuningMatricesM.append(m)
tuningMatricesM = np.array(tuningMatricesM)


# In[ ]:


np.save("Main_data/tuningVals25_15072021", tuningVals)
np.save("Main_data/tuningMatricesM25_15072021", tuningMatricesM)


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


tuningVals = np.load("Main_data/tuningVals25_15072021.npy")
tuningMatricesM = np.load("Main_data/tuningMatricesM25_15072021.npy")


# In[ ]:


def PlotTuningMatrices(tuningMatrices, shape, maxRad):
    """
    tuningMatrices.shape => (N*M, n, n)
    shape = (N, M, n, n)
    """
    N, M, n, n = shape
    tuningMatricesNxN = tuningMatrices.reshape(shape)
    tuningMatricesNxN_List = [[tuningMatricesNxN[r,c] for c in range(M)] for r in range(N)]
    tuningMatrices2D = np.block(tuningMatricesNxN_List)
    plotComplexArray(tuningMatrices2D, maxRad=maxRad)


# In[ ]:


PlotTuningMatrices(tuningMatricesM, (25, 25, 5, 5), maxRad=0.5)


# The simulation builder `BuildNewNetwork` requires that we supply it with two functions, one which creates an RF network object from of a 5-way splitter, and another which creates one of the Multiplier.  We will assume that the splitter is generic and employ a simple theoretical model for that which was imported from our `NetworkBuilding` theoretical simulation notebook.  However, for the Multiplier, we will use the `MultiplierBank` and the `loc` code to extract the model for a multiplier assigned to that specific location in the network. 

# In[ ]:


def MultBuilder(loc):
    return multBank.getRFNetwork(loc)


# In[ ]:


def SplitterBuilder(loc):
    return Build5PortSplitter(freq45, loc=loc)


# As a quick example of a simulation, we set all the multipliers to the same setting, build a network, and examine the transmissive properties of it.

# In[ ]:


# set the values for the models
multBank.setAllMults(psVal=512, vgaVal=512)


# In[ ]:


# building a scikit-RF network object
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


PlotTuningMatrices(tuningMatricesS, (25, 25, 5, 5), maxRad=0.5)


# In[ ]:


PlotTuningMatrices(tuningMatricesM/tuningMatricesS,(25, 25, 5, 5), maxRad=0.5)


# In[ ]:


grossCorrFact = np.mean(tuningMatricesM/tuningMatricesS)
grossCorrFact


# Ideally, this would yield the exact same network scattering matrices as were measured and contained in `tuningMatricesM`.  Of course they won't because each physical device has its own personality and other factors such as varying cable lengths.  We will therefore optimize the PCA weights of each device in simulation in an attempt to create collection of devices which match the real behavior of the experimental devices.
# 
# In order to perform this optimization, we use SciPy's multivariate minimization function `minimize()`.  The format of this 
# `scipy.optimize.minimize(fun, X0)` where `fun` is built such that `fun(X) -> error` where `X` and `X0` are 1D vectors of the real scalars to be optimized.  In order to make this easy, the MultiplierBank comes with two functions `setPersonalityVectors(X)` and `X0 = getPersonalityVectors()`, which grabs the complex PCA weights from all the multipliers as mashes them into a real 1D vector.  The two functions are designed to operate together so that the data

# In[ ]:


X1 = (grossCorrFact*X0.view('complex')).view('float')


# In[ ]:


# X0 = multBank.getPersonalityVectors()  # Defined at definition of multBank


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
    print(error)
    return error


# In[ ]:


fun(X0)


# In[ ]:


fun(X1)


# In[ ]:


fit = sp.optimize.minimize(fun, X1, method='Powell', 
                           options={'disp':True, 'adaptive':True, 'fatol':0.01})


# In[ ]:


XF = multBank.getPersonalityVectors()


# In[ ]:


# XF = fit.x


# Error when multipliers are the uniform average all devices measured in the PCA:

# Error following fitting the PCA weights:

# In[ ]:


fun(XF)


# In[ ]:


multBank.setPersonalityVectors(XF)


# In[ ]:


tuningMatricesS = []
for (psVal, vgaVal) in tuningVals:
    multBank.setAllMults(psVal, vgaVal)
    newNet = BuildNewNetwork(SplitterBuilder, MultBuilder, loc="N", n=5)
    m = newNet.s[0, 5:, :5]
    tuningMatricesS.append(m)
tuningMatricesS = np.array(tuningMatricesS)


# In[ ]:


PlotTuningMatrices(tuningMatricesS, (25, 25, 5, 5), maxRad=0.5)


# In[ ]:


np.save("Main_data/personalityVector_15072021", XF)


# # Set and Measure a Matrix

# In[ ]:


def calcNewMatrixSettings(K, multBank, n, warn=True, verbose=False):
    expK = []
    for i_out in range(n):
        expRow = []
        for i_in in range(n):
            loc = ('M', 'N', i_in, i_out)
            mult = multBank.getMultByLoc(loc)
            T = 5*K[i_out, i_in]
            mult.setT(T, warn=warn, verbose=verbose)
            Texp = mult.TExpected
            expRow.append(Texp)
        expK.append(expRow)
    expK = np.array(expK)
    return (expK/n)


# In[ ]:


def setExpMultBank(exp, multBank):
    physNums = multBank.getPhysNums()
    psSettings = [multBank.getMultByPhysNum(physNum).psSetting for physNum in physNums]
    vgaSettings = [multBank.getMultByPhysNum(physNum).vgaSetting for physNum in physNums]
    exp.setMults(psSettings, vgaSettings, physNums)


# In[ ]:


XF = np.load("Main_data/personalityVector_15072021.npy")


# In[ ]:


multBank.setPersonalityVectors(XF)


# ## Compare Current System to Tuning Matrices

# In[ ]:


tuningVals = np.load("Main_data/tuningVals25_15072021.npy")
tuningMatricesM = np.load("Main_data/tuningMatricesM25_15072021.npy")


# In[ ]:


testCase = 622


# In[ ]:


tuningVals[testCase]


# In[ ]:


(psVal, vgaVal) = tuningVals[testCase]
multBank.setAllMults(psVal, vgaVal)
setExpMultBank(exp, multBank)
m, std = exp.measureSMatrix(delay=2)
print(m)


# In[ ]:


oldM = tuningMatricesM[testCase]


# In[ ]:


oldM


# In[ ]:


MatrixError(m,oldM)


# In[ ]:


tempCOMP=MatrixError(m,oldM)/25
tempCOMP


# ## Inversion Experiment

# In[ ]:


KMidRange = np.full((5,5), fill_value=(-.25+.75j)/5)


# In[ ]:


RandomComplexCircularMatrix(0.5, (5,5));


# In[ ]:


KRand1 = np.array([[-0.395-0.074j,  0.002+0.214j, -0.174+0.235j,  0.456-0.18j , -0.383+0.171j],
                   [-0.461-0.019j, -0.075+0.19j , -0.251+0.301j, -0.132+0.09j , -0.225+0.013j],
                   [-0.409-0.236j, -0.124+0.037j,  0.103+0.197j, -0.436+0.241j, -0.05 +0.148j],
                   [ 0.025-0.064j, -0.183-0.198j, -0.075-0.225j, -0.014+0.166j, -0.053+0.174j],
                   [ 0.373+0.042j, -0.014+0.05j ,  0.034+0.392j,  0.195+0.178j,  0.005-0.079j]])
np.sort(abs(np.linalg.eigvals(KRand1)))[::-1]


# In[ ]:


KRand2 = np.array([[ 0.226+0.27j ,  0.142-0.411j, -0.205-0.386j,  0.146-0.415j,  0.252-0.283j],
                   [-0.383-0.102j,  0.123-0.095j,  0.316-0.183j, -0.422-0.19j ,  0.041+0.079j],
                   [-0.141-0.071j, -0.391+0.113j,  0.065-0.305j,  0.028+0.169j,  0.168+0.385j],
                   [ 0.094-0.225j,  0.21 -0.059j, -0.108+0.427j, -0.139-0.241j,  0.22 +0.272j],
                   [ 0.464-0.107j,  0.446+0.202j, -0.259-0.406j,  0.236-0.308j, -0.005-0.28j ]])
np.sort(abs(np.linalg.eigvals(KRand2)))[::-1]


# In[ ]:


K = KRand2
expName = 'trial3_'


# In[ ]:


KHWExp = calcNewMatrixSettings(K, multBank, 5)
print(KHWExp)


# In[ ]:


np.save(expName+"goalK",K)
np.save(expName+"KHWExp", KHWExp)


# In[ ]:


setExpMultBank(exp, multBank)
m, std = exp.measureSMatrix(delay=2)


# In[ ]:


std


# In[ ]:


plotComplexArray(m, maxRad=np.max(np.abs(m)))


# In[ ]:


# np.save(expName+"measK", m)
np.save(expName+"measKInv", m)


# In[ ]:


spr = np.array([5*[0]]).T


# In[ ]:


plotData = np.hstack((K, spr, KHWExp, spr, m))
plotComplexArray(plotData, maxRad=0.1)


# ## Post Processing

# In[ ]:


expName = 'trial3_'


# In[ ]:


K_goal = np.load(expName+'goalK.npy')
K_HWExp = np.load(expName+'KHWExp.npy')
K_meas = np.load(expName+'measK.npy')
K_measInv = np.load(expName+'measKInv.npy')


# In[ ]:


plotData = np.hstack((K_goal, spr, K_HWExp, spr, K_meas))
maxVal = np.max(np.abs(plotData))
print(maxVal)
plotComplexArray(plotData, maxRad=maxVal)


# In[ ]:


pp = PolarPlot("Big Kappa")
pp.addMatrixDiff(K_goal, K_meas)
# pp.addMatrixDiff(K_goal, K_HWExp)
# pp.addMatrix(K_goal, color='green')
# pp.addMatrix(K_HWExp, color='cyan')
# pp.addMatrix(K_meas, color='red')
pp.show()


# In[ ]:


coup1 = np.loadtxt("../GoldenSamples/FeedbackCouplerSamples/coupler_1.txt", dtype=np.complex)
coup2 = np.loadtxt("../GoldenSamples/FeedbackCouplerSamples/coupler_1_v2.txt", dtype=np.complex)
alpha1 = np.mean([coup1[0,0], coup2[0,0]])
alpha2 = np.mean([coup1[1,1], coup2[1,1]])
beta = np.mean([coup1[0,1], coup1[1,0], coup2[0,1], coup2[1,0]])


# In[ ]:


K_goal_inv = np.linalg.inv(np.identity(5)-K_goal)
K_exp1_inv = np.linalg.inv(np.identity(5)-K_HWExp)
K_exp2_inv = np.linalg.inv(np.identity(5)-K_meas)
K_meas_inv = (alpha1/beta*K_measInv+(beta-alpha1*alpha2/beta)*np.identity(5))/beta;


# In[ ]:


np.mean(abs(K_goal_inv-K_meas_inv))


# In[ ]:


plotData = np.hstack((K_goal_inv, spr, K_exp1_inv, spr, K_exp2_inv, spr, K_meas_inv))
maxV = np.max(np.abs(plotData))
print(maxV)
plotComplexArray(plotData, maxRad=maxV)


# In[ ]:


pp = PolarPlot("Inverse")
pp.addMatrixDiff(K_goal_inv, K_meas_inv)
# pp.addMatrixDiff(K_goal, K_HWExp)
# pp.addMatrix(K_goal, color='green')
# pp.addMatrix(K_HWExp, color='cyan')
# pp.addMatrix(K_meas, color='red')
pp.show()


# ## Large Number Experiments

# ### Big Kappa

# In[ ]:


# to be uncomment for a new set of values
# inputKernels = [RandomComplexCircularMatrix(0.5, (5,5)) for i in range(1000)]
# np.save("Main_data/largeNInput", np.array(inputKernels))


# In[ ]:


inputKernels = np.load("Main_data/largeNInput.npy")


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


# save or load the measurement results
# np.save("Main_data/largeNClosedLoop.npy", np.array(outputMeasurementsINV))
outputMeasurements = np.load("Main_data/largeNOpenLoop.npy")
outputMeasurementsINV = np.load("Main_data/largeNClosedLoop.npy")


# In[ ]:


inKK = []
measKK = []
setKK = []
inKK_INV = []
measKK_INV = []
setKK_INV = []
for i in np.arange(1000):
    inKK.append(np.array(outputMeasurements)[i, 0])  # Goal Input Kernel
    measKK.append(np.array(outputMeasurements)[i, 2]) # Measured Input Kernel
    setKK.append(np.array(outputMeasurements)[i, 1]) # Experimentally Expected Kernel
    inKK_INV.append(np.array(outputMeasurementsINV)[i, 0]) # Goal Input Kernel
    measKK_INV.append(np.array(outputMeasurementsINV)[i, 2]) # Measured Inverted Kernel
    setKK_INV.append(np.array(outputMeasurementsINV)[i, 1]) # Expermintally Expected Kernel


# In[ ]:


# s=np.sign(np.real(RandomComplexCircularMatrix(1, (5,5))))
diff = np.array(range(1000), dtype="float")
comp = np.array(range(1000), dtype="float")
tet = []
for i in np.arange(1000):
    tet.append(measKK[i]-inKK[i])
    diff[i] = MatrixError(measKK[i], inKK[i])/np.linalg.norm(inKK[i])*100
    comp[i] = MatrixError(inKK[i], inKK[i])


# In[ ]:


a = np.linalg.norm(measKK[1]-inKK[1])/np.linalg.norm(inKK[1])*100
b = MatrixError(measKK[1], inKK[1])
print(a, b)


# In[ ]:


# compare the big Kappa (open loop matrices)
import matplotlib.pyplot as plt
index = np.arange(1000)
plt.scatter(index, diff)
plt.xlabel('# random matrix')
plt.ylabel('$||\mathbb{K}_{meas}-\mathbb{K}_{target}||/||\mathbb{K}_{target}||(\%)$')
plt.ylim([0, 15])
plt.savefig('Main_data/fig/BigKappa_rand1000.png',dpi=1200)


# In[ ]:


pp = PolarPlot("comparisson")
i=154
pp.addMatrixDiff(np.array(measKK[i]),np.array(inKK[i]))
pp.show()
abs(np.linalg.eig(inKK[i])[0])


# ### Inverse exp - post processing

# In[ ]:


# Load the coupler parameters
coup1 = np.loadtxt("../GoldenSamples/FeedbackCouplerSamples/coupler_1.txt", dtype=np.complex)
coup2 = np.loadtxt("../GoldenSamples/FeedbackCouplerSamples/coupler_1_v2.txt", dtype=np.complex)
alpha1 = np.mean([coup1[0,0], coup2[0,0]])
alpha2 = np.mean([coup1[1,1], coup2[1,1]])
beta = np.mean([coup1[0,1], coup1[1,0], coup2[0,1], coup2[1,0]])


# In[ ]:


# to be uncomment for a new set of values
# inputKernels = [RandomComplexCircularMatrix(0.5, (5,5)) for i in range(100)]
# np.save("largeNInput_100values", np.array(inputKernels))


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
A_meas_inv = (alpha1/beta*A_exp+(beta-alpha1*alpha2/beta)*np.identity(5))/beta


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

