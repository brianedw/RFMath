#!/usr/bin/env python
# coding: utf-8

# ### stock imports

# In[ ]:


import numpy as np


# In[ ]:


from scipy.optimize import root
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter


# In[ ]:


from colorize import colorizeComplexArray


# In[ ]:


import bokeh
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
output_notebook()
bokeh.io.curdoc().theme = 'dark_minimal'


# In[ ]:


mainQ =(__name__ == '__main__')
mainQ


# # ConvertArrayToDict

# In[ ]:


def convertArrayToDict(array, drop=0, preSpec=()):
    shape = array.shape[:-drop or None]
    nDim = len(shape)
    coords = np.array(np.meshgrid(*map(range, shape), indexing="ij"))
    coords1D = np.moveaxis(coords, 0, len(shape)).reshape(-1, nDim)
    coordsTup = [tuple(c) for c in coords1D]
    d = dict()
    for c in coordsTup:
        data = array[c]
        if np.isfinite(data).all():
            d[preSpec+c] = data
    return d


# In[ ]:


array = np.full(shape=(3,3,4,2), fill_value=np.nan)
array[0,2,3,0] = 2.1
array[0,2,3,1] = 2.3


# In[ ]:


convertArrayToDict(array, drop=1, preSpec=('Alpha', 'Bravo'))


# # Plot Complex Array

# In[ ]:


def plotComplexArray(array, maxRad=10, centerColor='black'):
    pixArray = colorizeComplexArray(array+0.00001j, centerColor=centerColor, maxRad=maxRad)
    (h, w) = array.shape
    img = np.zeros((h, w), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape(h, w, 4)
    view[:, :, 0] = pixArray[:, :, 0]
    view[:, :, 1] = pixArray[:, :, 1]
    view[:, :, 2] = pixArray[:, :, 2]
    view[:, :, 3] = 255
    p = figure(x_range=(0, w), y_range=(0, h), plot_width=800, plot_height=800)
    p = figure()
    p.image_rgba(image=[img], x=0, y=0, dw=w, dh=h)
    show(p)


# In[ ]:


data = np.random.uniform(low=-10, high=10, size=(10, 15)) + 1j*np.random.uniform(low=-10, high=10, size=(10, 15))
data[:3, :3] = 0
if mainQ: plotComplexArray(data, maxRad=10)


# In[ ]:


nx, ny = (1000, 1000)
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
xv, yv = np.meshgrid(x, y)
data = xv + 1j*yv
data = np.where(np.abs(data) > 1, 0, data)
if mainQ: plotComplexArray(data, maxRad=1)


# # Random Complex Matrices

# In[ ]:


def RandomComplexCircularMatrix(r, size):
    """
    Generates a matrix random complex values where each value is
    within a circle of radius `r`.  Values are evenly distributed
    by area.
    """
    rMat = np.random.uniform(0, 1, size=size)**0.5 * r
    pMat = np.random.uniform(0, 2*np.pi, size=size)
    cMat = rMat*(np.cos(pMat) + 1j*np.sin(pMat))
    return cMat


# In[ ]:


RandomComplexCircularMatrix(0.3, (2,2))


# In[ ]:


data = RandomComplexCircularMatrix(10, (100,100))
if mainQ: plotComplexArray(data, maxRad=10)


# In[ ]:


def RandomComplexGaussianMatrix(sigma, size):
    """
    Generates a matrix random complex values where each value is
    within a circle of radius `r`.  Values are evenly distributed
    by area.
    """
    reMat = np.random.normal(0, sigma, size=size)
    imMat = np.random.normal(0, sigma, size=size)
    cMat = reMat + 1j*imMat
    return cMat


# In[ ]:


RandomComplexGaussianMatrix(0.3, (2,2))


# In[ ]:


data = RandomComplexGaussianMatrix(10, (100,100))
if mainQ: plotComplexArray(data, maxRad=3*10)


# In[ ]:


def RandomComplexNaiveMatrix(r, size):
    """
    Generates a matrix random complex values where each value is
    within a circle of radius `r`.  Values are weighted toward the center
    """
    rMat = np.random.uniform(0, r, size=size)
    pMat = np.random.uniform(0, 2*np.pi, size=size)
    cMat = rMat*(np.cos(pMat) + 1j*np.sin(pMat))
    return cMat


# In[ ]:


np.random.exponential()


# In[ ]:


def RandomComplexExponentialMatrix(r, size):
    """
    Generates a matrix random complex values where each value is
    within a circle of radius `r`.  Values are weighted toward the center
    """
    rMat = np.random.exponential(r, size=size)
    pMat = np.random.uniform(0, 2*np.pi, size=size)
    cMat = rMat*(np.cos(pMat) + 1j*np.sin(pMat))
    return cMat


# # Plot Complex Matrix Difference

# In[ ]:


def makePolarPlot(title):
    '''
    This will create a Bokeh plot that depicts the unit circle.

    Requires import bokeh
    '''
    p = bokeh.plotting.figure(plot_width=400, plot_height=400, title=title, 
                              x_range=[-1.1, 1.1], y_range=[-1.1, 1.1])
    p.xaxis[0].ticker=bokeh.models.tickers.FixedTicker(ticks=np.arange(-1, 2, 0.25))
    p.yaxis[0].ticker=bokeh.models.tickers.FixedTicker(ticks=np.arange(-1, 2, 0.25)) 
    p.circle(x = [0,0,0,0], y = [0,0,0,0], radius = [0.25, 0.50, 0.75, 1.0], 
             fill_color = None, line_color='gray')
    p.line(x=[0,0], y=[-1,1], line_color='gray')
    p.line(x=[-1,1], y=[0,0], line_color='gray')
    xs = [0.25, 0.50, 0.75, 1.00]
    ys = [0, 0, 0, 0]
    texts = ['0.25', '0.50', '0.75', '1.00']
    source = bokeh.models.ColumnDataSource(dict(x=xs, y=ys, text=texts))
    textGlyph = bokeh.models.Text(x="x", y="y", text="text", angle=0.3, 
                                  text_color="gray", text_font_size='10px')
    p.add_glyph(source, textGlyph)
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    return p


# In[ ]:


def addMatrixDiff(bokehPlot, m1, m2):
    """
    This will draw lines showing the difference between two 2D matrices.
    """
    p = bokehPlot
    begX = (np.real(m1)).flatten()
    begY = (np.imag(m1)).flatten()
    endX = (np.real(m2)).flatten()
    endY = (np.imag(m2)).flatten()

    xs = np.array([begX, endX]).T.tolist()
    ys = np.array([begY, endY]).T.tolist()
    p.multi_line(xs=xs, ys=ys)

    sourceTarg = bokeh.models.ColumnDataSource(dict(x=begX.tolist(), y=begY.tolist()))
    glyphTarg = bokeh.models.Circle(x="x", y="y", size=10, line_color="green", 
                                    fill_color=None, line_width=3)
    p.add_glyph(sourceTarg, glyphTarg)

    sourceSim = bokeh.models.ColumnDataSource(dict(x=endX.tolist(), y=endY.tolist()))
    glyphSim = bokeh.models.Circle(x="x", y="y", size=5, line_color=None, 
                                   fill_color='red', line_width=3)
    p.add_glyph(sourceSim, glyphSim)
    return p


# In[ ]:


m1 = RandomComplexGaussianMatrix(0.4, (5,5))
m1Error = RandomComplexGaussianMatrix(0.05, (5,5))
m2 = m1 + m1Error
plot = makePolarPlot("Blah Blah Blah")
addMatrixDiff(plot, m1, m2)
if mainQ: show(plot)


# In[ ]:


class PolarPlot:
    
    def __init__(self, title):
        '''
        This will create a Bokeh plot that depicts the unit circle.

        Requires import bokeh
        '''
        p = bokeh.plotting.figure(plot_width=400, plot_height=400, title=title, 
                                  x_range=[-1.1, 1.1], y_range=[-1.1, 1.1])
        p.xaxis[0].ticker=bokeh.models.tickers.FixedTicker(ticks=np.arange(-1, 2, 0.25))
        p.yaxis[0].ticker=bokeh.models.tickers.FixedTicker(ticks=np.arange(-1, 2, 0.25)) 
        phis = np.linspace(0, 2*np.pi, num=181, endpoint=True)
        xs = np.cos(phis)
        ys = np.sin(phis)
        for r in [0.25, 0.50, 0.75, 1.00]:
            p.line(x=(r*xs).tolist(), y=(r*ys).tolist(), line_color='gray')
        p.line(x=[0,0], y=[-1,1], line_color='gray')
        p.line(x=[-1,1], y=[0,0], line_color='gray')
        xs = [0.25, 0.50, 0.75, 1.00]
        ys = [0, 0, 0, 0]
        texts = ['0.25', '0.50', '0.75', '1.00']
        source = bokeh.models.ColumnDataSource(dict(x=xs, y=ys, text=texts))
        textGlyph = bokeh.models.Text(x="x", y="y", text="text", angle=0.3, 
                                      text_color="gray", text_font_size='10px')
        p.add_glyph(source, textGlyph)
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        self.p = p
    
    def addMatrixDiff(self, m1, m2):
        """
        This will draw lines showing the difference between two 2D matrices.
        """
        p = self.p
        begX = (np.real(m1)).flatten()
        begY = (np.imag(m1)).flatten()
        endX = (np.real(m2)).flatten()
        endY = (np.imag(m2)).flatten()

        xs = np.array([begX, endX]).T.tolist()
        ys = np.array([begY, endY]).T.tolist()
        p.multi_line(xs=xs, ys=ys)

        sourceTarg = bokeh.models.ColumnDataSource(dict(x=begX.tolist(), y=begY.tolist()))
        glyphTarg = bokeh.models.Circle(x="x", y="y", size=10, line_color="green", 
                                        fill_color=None, line_width=3)
        p.add_glyph(sourceTarg, glyphTarg)

        sourceSim = bokeh.models.ColumnDataSource(dict(x=endX.tolist(), y=endY.tolist()))
        glyphSim = bokeh.models.Circle(x="x", y="y", size=5, line_color=None, 
                                       fill_color='red', line_width=3)
        p.add_glyph(sourceSim, glyphSim)
    
    
    def addMatrix(self, m1, color='cyan'):
        """
        This will draw lines showing the difference between two 2D matrices.
        """
        p = self.p
        X = (np.real(m1)).flatten()
        Y = (np.imag(m1)).flatten()
        source = bokeh.models.ColumnDataSource(dict(x=X.tolist(), y=Y.tolist()))
        glyph = bokeh.models.Circle(x="x", y="y", size=5, line_color=None, 
                                       fill_color=color, line_width=3)
        p.add_glyph(source, glyph)
        
    def addMatrixSD(self, data, color='cyan'):
        """
        This will draw lines showing the difference between two 2D matrices.
        """
        dataReshaped = data.reshape((-1, 2)).T
        Xs = np.real(dataReshaped[0]).flatten()
        Ys = np.imag(dataReshaped[0]).flatten()
        SDs = np.abs(dataReshaped[1].flatten())
        p = self.p
        source = bokeh.models.ColumnDataSource(dict(x=Xs.tolist(), y=Ys.tolist(), sd=SDs.tolist()))
        glyph = bokeh.models.Circle(x="x", y="y", radius='sd', line_color=None, 
                                       fill_color=color, line_width=1, fill_alpha=0.5)
        p.add_glyph(source, glyph)     

        
    def show(self):
        show(self.p)


# In[ ]:


m1 = RandomComplexGaussianMatrix(0.5, (5,5))
m2 = m1 + RandomComplexGaussianMatrix(0.05, (5,5))

m3 = RandomComplexGaussianMatrix(0.5, (5,5))


# In[ ]:


pp = PolarPlot("Plot Title")
pp.addMatrixDiff(m1, m2)
pp.addMatrix(m3)
if mainQ: pp.show()


# # Passivity

# In[ ]:


def IsPassive(arr, threshold=1):
    u, s, v = np.linalg.svd(arr)
    maxS = max(s)
    passiveQ = (maxS <= threshold)
    return passiveQ


# In[ ]:


def RescaleToUnitary(arr, scaleFactor=1):
    u, s, v = np.linalg.svd(arr)
    maxS = max(s)
    rescaledArr = scaleFactor*arr/maxS
    return rescaledArr


# In[ ]:


arr = RandomComplexCircularMatrix(10, size=(5,5))


# In[ ]:


RescaleToUnitary(arr)


# # ReIm

# The purpose of `ReIm(z)` is to provide similar functionality to Mathematica's `ReIm[z]` function.  It breaks `z` into a real and imaginary component and returns both.  It operates on both scalars and numpy arrays.

# In[ ]:


def ReIm(z):
    return (np.real(z), np.imag(z))


# In[ ]:


ReIm(np.array(0.4+0.2j))


# In[ ]:


ReIm(np.array([0.1+0.2j, 0.3+0.4j]))


# In[ ]:


ReIm(np.array(100))


# # Matrix Errors

# In[ ]:


def MatrixSqError(m, mTarget):
    """
    Computes ||m - mTarget||^2 where ||m|| is the Frobenius norm of M.
    """
    errorSq = (np.abs(m - mTarget)**2).sum()
    return errorSq


# In[ ]:


def MatrixError(m, mTarget):
    """
    Computes ||m - mTarget|| where ||m|| is the Frobenius norm of M.
    """
    errorSq = MatrixSqError(m, mTarget)
    error = errorSq**0.5
    return error


# In[ ]:


def MatrixErrorNormalized(m, mTarget):
    """
    Computes ||m - mTarget||/a where ||m|| is the Frobenius norm of M
    and 'a' is maximum transmissive entropy normalization factor of 
    (1/n)^0.5.

    The purpose of this function is to give a sense of the error within
    a matrix compared to the average values for a matrix of that order.
    """
    errorSq = MatrixSqError(m, mTarget)
    error = errorSq**0.5
    (n,n) = m.shape
    a = (1/n)**0.5
    return error/a


# In[ ]:


a = np.array([[0, 1], [0,   1]])
b = np.array([[0, 1], [0, 0.5]])


# In[ ]:


MatrixSqError(a,b)


# In[ ]:


MatrixError(a,b)


# In[ ]:


MatrixErrorNormalized(a,b)


# # Complex Gaussian Filter

# In[ ]:


def gaussian_filter_complex(b, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0):
    """
    See scipy.ndimage.gaussian_filter for documentation.
    """
    r = gaussian_filter(np.real(b), sigma, order, output, mode, cval, truncate)
    i = gaussian_filter(np.imag(b), sigma, order, output, mode, cval, truncate)
    return r + 1j*i


# # Complex Valued 2D PCA

# Adapted from https://datascience.stackexchange.com/questions/75733/pca-for-complex-valued-data

# In[ ]:


def complex2DPCA(dataSet, nComps):
    nSamples = len(dataSet)
    dataShape = dataSet[0].shape
    compShape = (nComps, ) + dataShape
    matrix = dataSet.reshape(nSamples, -1)
    _, s, vh = np.linalg.svd(matrix, full_matrices=False)
    compsFlat = vh[:nComps]
    comps = compsFlat.reshape(compShape)
    FAve = np.mean(dataSet, axis=0)
    baseWeights = complex2Dlstsq(comps, FAve)
    return comps, baseWeights


# In[ ]:


def complex2Dlstsq(comps, field):
    nComps, nyC, nxC = comps.shape
    fieldFlat = field.flatten()
    compsFlat = comps.reshape(nComps, nyC*nxC)
    weights = np.linalg.lstsq(compsFlat.T, fieldFlat, rcond=None)[0]
    # weights = weights.reshape(-1,1,1)
    return weights


# Below we create a data set consisting of 4 terms, each with their own weighting.

# In[ ]:


ns, nx, ny = (10, 150, 100)
xs = np.linspace(-1, 1, nx)
ys = np.linspace(-1, 1, ny)
xg, yg = np.meshgrid(xs, ys)
kx1, ky1 = [2,-4]
kx2, ky2 = [1, 3]
dataSet = np.zeros(shape=(ns, ny, nx), dtype=np.complex)
for i in range(ns):
    c1 = 1j + RandomComplexGaussianMatrix(0.1, size=1)
    c2 = RandomComplexGaussianMatrix(0.1, size=1)
    F1 = np.exp(1j*(kx1*xg + ky1*yg))
    F2 = np.exp(1j*(kx2*xg + ky2*yg))
    F3 = RandomComplexGaussianMatrix(0.1, size=F1.shape)
    F4 = np.full_like(F1, fill_value=(0.3+0.2j))
    dataSet[i] = c1*F1 + c2*F2 + F3 + 0*F4


# In[ ]:


if mainQ: plotComplexArray(dataSet[0], maxRad=1.5)


# In[ ]:


comps, baseWeights = complex2DPCA(dataSet, nComps=4)


# In[ ]:


prototype = np.sum(comps * baseWeights.reshape(-1,1,1), axis=0)


# In[ ]:


if mainQ: plotComplexArray(prototype, maxRad=1.5)


# In[ ]:


aveF = np.average(dataSet, axis=0)


# In[ ]:


if mainQ: plotComplexArray(prototype-aveF, maxRad=0.1)


# In[ ]:


target = dataSet[0]
weights = complex2Dlstsq(comps, target)
fit = np.sum(comps*weights.reshape(-1,1,1), axis=0)
if mainQ: plotComplexArray(fit, maxRad=1.5)


# In[ ]:


if mainQ: plotComplexArray(fit-target, maxRad=.5)


# # Interpolated Root Finding

# In[ ]:


nx, ny = (100, 93)
xs = np.linspace(0, 1023, nx+1, endpoint=True)
ys = np.linspace(0, 1023, ny+1, endpoint=True)
xg, yg = np.meshgrid(xs, ys)
kx1, ky1 = [2*(2*np.pi)/1023, 0.3*(2*np.pi)/1023]
F1 = np.exp(1j*(kx1*xg + ky1*yg))
F2 = yg/1023
F3 = 0.7*np.exp(-((xg-500)**2 / 300**2))
F4 = RandomComplexGaussianMatrix(0.0001, size=F1.shape)
dataSet = 6*F1*F2*(1-F3)+F4


# In[ ]:


if mainQ: plotComplexArray(dataSet, maxRad=5.5)


# In[ ]:


rF = interp2d(xs, ys, np.real(dataSet), kind='linear', bounds_error=True)
iF = interp2d(xs, ys, np.imag(dataSet), kind='linear', bounds_error=True)

def F(x,y):
    z, = rF(x, y, assume_sorted=True) + 1j*iF(x, y, assume_sorted=True)
    return z


# In[ ]:


F(512, 1023)


# In[ ]:


F(512, 0)


# In[ ]:


target = F(100, 100) + (0.01+0.01j)


# In[ ]:


startPos = [90, 130]

def FRoot(xv, *args):
    x, y = xv
    targ, = args
    rO = np.real(F(x,y) - target)
    iO = np.imag(F(x,y) - target)
    return np.array([rO, iO])

try:
    soln = root(FRoot, startPos, args=target)
except ValueError:
    print("starting over at [512, 512]")
    soln = root(FRoot, [512, 512])
np.round(soln['x']).astype(np.int)


# In[ ]:


def rootRough(F, start):
    evalPts = np.array(start)
    
    F()


# In[ ]:


start = np.array([512,512])
bestOffset = -1
offsets = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1]])
pos = start.copy()
inBounds = lambda xv: 0 <= xv[0] <= 1023 and 0 <= xv[1] <= 1023
while bestOffset != 0:
    evalPts = filter(inBounds, pos + offsets)
    scores = [abs(F(*pt) - (2+1j)) for pt in evalPts]
    bestOffset = np.argmin(scores)
    pos = pos + offsets[bestOffset]


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
    if not onEdge(pos):
        return (pos, bestScore)


# In[ ]:


dumbRoot2D(F, [512, 512], ((0, 1023), (0, 1023)), (1+0j))


# In[ ]:


bounds = ((0, 1023), (0, 1023))


# In[ ]:


inBounds = lambda xv: 0 <= xv[0] <= 1023 and 0 <= xv[1] <= 1023
evalPtsFiltered = filter(inBounds, evalPts)


# In[ ]:


pos = np.array([0, 10])


# In[ ]:


filter(inBounds, pos + offsets)


# In[ ]:


xSamples = np.linspace(0, 1023, 1023+1)
ySamples = np.linspace(0, 1023, 1023+1)
xGrid, yGrid = np.meshgrid(xSamples, ySamples)


# In[ ]:


gridPts = np.dstack((xGrid, yGrid)).reshape(-1, 2)

