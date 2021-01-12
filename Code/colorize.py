#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from matplotlib.colors import hsv_to_rgb


# In[ ]:


def colorizeArray(array2D, min_max=(-1, 1), colors=[(255,0,0), (255,255,255), (0,0,255)], 
                  preFunc=(lambda x: x)):
    colorArray = np.array(colors)
    minVal, maxVal = min_max
    nColors = colorArray.shape[0]
    array_F = preFunc(array2D)
    rescaledX = np.clip((array_F-minVal)/(maxVal-minVal),0,1) # Puts array in range [0,1]
    cPos = rescaledX*(nColors-1) # finds its position in the color array.
    iLow = np.clip(np.floor(cPos).astype(np.uint8), 0, nColors - 2)
    iFrac = cPos - iLow
    lowColors = np.take(colorArray, iLow, axis=0)
    highColors = np.take(colorArray, iLow+1, axis=0)
    iFrac3D = np.expand_dims(iFrac, axis=2)
    outColor = np.round(lowColors*(1 - iFrac3D) + highColors*(iFrac3D)).astype(np.uint8)
    return outColor


# In[ ]:



inArray = np.array([[-2, -0.1, 0.1, 1, 2],[-2, -0.1, 0.1, 1, 2]])
colorizeArray(inArray, min_max=(0,1), 
              colors = [(0,0,0), (255, 0, 0)], 
              preFunc = lambda x: np.abs(x))


# In[ ]:


dims = (4,5)
inArray = np.random.rand(*dims) + 1j*np.random.rand(*dims)


# In[ ]:


def colorizeComplexArray(inArray, maxRad=1, centerColor='white'):
    angleNP = np.angle(inArray+0.001j)/(2*np.pi) # on range [-.5, 0.5]
    offset = np.sign(angleNP)/(-2) + 0.5
    hue = angleNP + offset
    mag = np.clip(np.abs(inArray)/maxRad, 0, 1)
    ones = np.ones_like(hue, np.float)
    if centerColor == 'black':
        HSVChannel = np.array([hue, ones, mag])
    elif centerColor == 'white':
        HSVChannel = np.array([hue, mag, ones])
    else:
        raise Exception(print("centerColor must be in {'white', 'black'}"))
    HSVFloat = np.dstack(HSVChannel)
    rgbOut = np.floor(hsv_to_rgb(HSVFloat)*255.99).astype(np.uint8)
    return rgbOut


# In[ ]:


colorizeComplexArray(inArray, maxRad=0.01, centerColor='white')


# In[ ]:




