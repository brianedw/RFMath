#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
import math


# In[ ]:


#!conda install --yes --prefix {sys.prefix} pyserial


# In[ ]:


import serial
import time


# In[ ]:


import pyvisa


# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO


# # Library

# In[ ]:


get_ipython().system(' python -m serial.tools.list_ports')


# ## Switch

# In[ ]:


class SwitchComm:
    pass


# In[ ]:


def __init__(self, comValue='COM_'):
    """ Creates a communication link to a Metromini controlling a 1:8 RF switch.

    Args: 
        comValue: The serial port controlling the switch.  Example 'COM4'
    """
    self.switchCom = serial.Serial()
    self.comValue = comValue
    self.switchCom.port = comValue
    self.switchCom.close()
    self.switchCom.open()
    time.sleep(0.5)
    response = self.switchCom.read_all()
    print(response.decode("utf-8"))
    
setattr(SwitchComm, '__init__', __init__)


# In[ ]:


def setSwitch(self, i, verbose=False):
    """ Sets the switch value.
    
    Args:
        switchPort (int): The output switch number [0-7].
    """
    self.switchCom.flushInput()
    self.switchCom.write(str.encode(str(i)))
    if verbose:
        time.sleep(2)
        response = self.switchCom.read_all()
        print(response.decode("utf-8"))
        
setattr(SwitchComm, 'setSwitch', setSwitch)


# In[ ]:


def closeSwitch(self):
    """ Closes the communication serial port to the switch.
    """
    self.switchCom.close()
    
setattr(SwitchComm, 'closeSwitch', closeSwitch)


# In[ ]:


# inputSwitch = SwitchComm(comValue='COM1')


# ## MCUs

# In[ ]:


F


# In[ ]:


class MultBankComm:
    writeCode = '2'  # The code which indicates to the Arduino that the instruction is a write operation.  For instance, mult 23 could respond to `23 2 400 500`
    blinkCode = '1'  # The code which indicates to the Arduino that the instruction is a blink request.  For instance, mult 23 could respond to `23 1`
    iMaster = 255    # It is expected that the Multiplier at the head has the number 255.


# In[ ]:


def __init__(self, comValue='COM_'):
    """ Intializes communications with a Multiplier Bank.
    
    Args:
        comValue (str): The COM Port that the head multiplier resides at.  Ex 'COM4'    
    
    """
    self.multBankComm = serial.Serial()
    self.multBankComm = comValue
    self.multBankComm.close()
    self.multBankComm.open()
    time.sleep(0.5)
    response =  self.multBankComm.read_all()
    print(response.decode("utf-8"))
    
setattr(MultBankComm, '__init__', __init__)


# In[ ]:


def setMult(idNum, v1, v2, verbose=False):
    """ Sets the VGA and PS values for a Multiplier
    
    Args:
        idNum (int): target multiplier ID number.
        v1 (int): VGA in range [0-1023]
        v2 (int): PS in range [0-1023]
        verbose (bool): whether or not print return information from the head multiplier.
    """
    self.multBankComm.flushInput()
    # TODO Should this be np.clip(xxx, 0, 1023)
    v1Out = np.clip(v1Out, 1, 1023)
    v2Out = np.clip(v2Out, 1, 1023)
    outString = " ".join((str(idNum), writeCode, str(v1Out), str(v2Out)))    # "10 2 300 1023"
    self.multBankComm.write(str.encode(outString))
    if verbose:
        time.sleep(1)
        response = self.multBankComm.read_all()
        print(response.decode("utf-8"))
        
setattr(MultBankComm, 'setMult', setMult)


# In[ ]:


def setMultBank(self, data, verbose=False):
    """ Sets a bunch of Multipliers.
    
    Args:
        data (dict): expected to be a dictionary of the form `{multID: [v1,v2], multID: [v1,v2], ... }`
    """
    for iMult, vals in data.items():
        v1, v2 = vals
        self.setMCU(iMult, v1, v2, verbose)
        time.sleep(0.2)
        
setattr(MultBankComm, 'setMultBank', setMultBank)


# In[ ]:


def blinkMult(self, multID, verbose=False):
    """ Blinks a multiplier.
    
    Args:
        multID (int): the target multiplier
    """
    self.multBankComm.flushInput()
    outString = " ".join((str(multID), blinkCode))
    self.multBankComm.write(str.encode(outString))
    if verbose:
        time.sleep(2)
        response = self.multBankComm.read_all()
        print(response.decode("utf-8"))
        
setattr(MultBankComm, 'blinkMult', blinkMult)


# In[ ]:


def blinkAll(self, multIDs, verbose=False):
    """ Blinks all multipliers.
    
    Note that the head multiplier is prepended to the list.
    
    Args:
        multIDs (list of ints): the IDs of the target multlipliers.
    """
    allIDs = [iMaster] + multIDs 
    for multID in allIDs:
        self.blinkMCU(multID, verbose)
        time.sleep(5.5) # Needed wait time to allow MCU to return to listenting state.

setattr(MultBankComm, 'blinkAll', blinkAll)


# In[ ]:


def closeMultBank(self):
    """ Closes the Serial Port associated with this multiplier bank.
    """
    self.multBankComm.close()
    
setattr(MultBankComm, 'closeMultBank', closeMultBank)


# ## VNA

# In[ ]:


class VNAComm:
    pass


# In[ ]:


def __init__(self):
    rm = pyvisa.ResourceManager()
    resources = rm.list_resources()
    inst = rm.open_resource(resources[0])
    name = inst.query('*IDN?')
    print(name)
    if True:
        self.vna = inst

setattr(VNAComm, '__init__', __init__)


# In[ ]:


def getS21at45(self):
    """
    VNA state 8
    """
    self.vna.write('MMEM:STOR:FDAT "d:/dummy.csv"')
    result = self.vna.query('MMEM:TRAN? "d:/dummy.csv"')
    impArray = pd.read_csv(StringIO(result), skiprows = 2).to_numpy()
    r = impArray[round((len(impArray)-1)/2), 1]
    i = impArray[round((len(impArray)-1)/2), 2]
    return r + 1j*i

setattr(VNAComm, 'getS21at45', getS21at45)


# In[ ]:


def getS21AllAt45(self):
    """
    For use with VNA State 7
    """
    self.vna.write('MMEM:STOR:FDAT "d:/dummy.csv"')
    result = self.vna.query('MMEM:TRAN? "d:/dummy.csv"')
    impArray = pd.read_csv(StringIO(result), skiprows = 2).to_numpy()
    rvs = impArray[:,1]
    ivs = impArray[:,2]
    zvs = rvs + 1j*ivs
    zAve = np.mean(zvs)
    zSTD = np.std(zvs)
    return (zAve, zSTD)

setattr(VNAComm, 'getS21AllAt45', getS21AllAt45)


# In[ ]:


def getS21freq(self):
    self.vna.write('MMEM:STOR:FDAT "d:/dummy.csv"')
    result = self.vna.query('MMEM:TRAN? "d:/dummy.csv"')
    impArray = pd.read_csv(StringIO(result), skiprows = 2).to_numpy()
    r = impArray[range(len(impArray)), 1]
    i = impArray[range(len(impArray)), 2]
    return r + 1j*i

setattr(VNAComm, 'getS21freq', getS21freq)


# In[ ]:


def closeVNA(self):
    self.vna.close()
    
setattr(VNAComm, 'closeVNA', closeVNA)


# In[ ]:




