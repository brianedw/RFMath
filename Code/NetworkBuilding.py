#!/usr/bin/env python
# coding: utf-8

# #### imports and constants

# In[ ]:


# !pip install scikit-rf


# In[ ]:


# !{sys.executable} -m pip install scikit-rf


# In[ ]:


import os, sys


# In[ ]:


import skrf as rf
import numpy as np
from math import *


# In[ ]:


from Miller import MillerBuilder


# In[ ]:


pi = np.pi
deg = 2*pi/360


# # Scikit-RF Network Building

# Scikit-RF networks are represent collections of microwave components and are mathematically described by their S-Params.  The network object can be created mathematically or by importing experimental data.
# 
# Within this code, we
# - define several elementary network generators: ports, attenuators, multipliers, couplers, and MZIs
# - create functions (Network Combiners) which can glue these networks together in trivial configurations.
# - create functions (Network Builders) which use generators and combiners to create and assmebled complex networks such as the Miller Archicture.
# 
# There is a rather subtle point here.  The Network Builders are designed using a Functional paradigm.  The Network Builders are designed to take "functions" as their inputs, which will either retrieve an existing smaller network model or generate a new one.  This provides a tremendous amount of flexibility as these functions could potentially be referencing an evolving model of an experiment, generating Monte-Carlo style variates, or just generating the nominal devices for checking of correctness.  The argument which is supplied to these functions is a location tag which designates what kind of element it is and where it is located within the network.  The function can then do with that tag what it wishes as long as it returns an appropriately sized network.
# 
# Several examples are given within this notebook for trivial generators that show correctness and the use cases are expanded upon later.
# 

# ## Device Definitions

# Defines a series of functions which generate theoretical `network` objects.

# ### Port

# SciKit-RF expects ports o have the word "port" as part of their name.  The information which follows is irrelavant so long as it is unique within the local network.  The ordering of the ports with regards to SParams comes from their order of introduction within the connections list and has nothing to do with their name.

# In[ ]:


def BuildPort(freq, loc=()):
    # loc expected to be of the form ("P", loc, i) where loc is from the superseding object ('Uh', 'V', etc) and is expected to be internally unique to the generating object.
    network = rf.Circuit.Port(freq, 'port'+str(loc), z0=50)
    return network


# As described above, making a matched load (as opposed to a port) only requires that we not use "port" in the network name.

# In[ ]:


def BuildTrashPort(freq, loc=()):
    # loc expected to be of the form ("P", loc, i) where loc is from the superseding object ('Uh', 'V', etc) and is expected to be internally unique to the generating object.
    network = rf.Circuit.Port(freq, 'trash'+str(loc), z0=50)
    return network


# ### Attenuator

# In[ ]:


def BuildAtten(T, freq, reciprocal=False, loc=()):
    """
    Simple Attenuator model.  Transmitted amplitude is scaled by 'T'.
    """
    Z_0 = 50.
    label = str(loc)
    nFreqs = len(freq)
    S21 = T
    S = np.zeros((nFreqs, 2, 2), dtype=np.complex)
    S[:, 1, 0] = S21
    if reciprocal:
        S[:, 0, 1] = S21
    attenNetwork = rf.Network(name=label, frequency=freq, z0=Z_0, s=S)
    return attenNetwork


# In[ ]:


freq = rf.Frequency(start=40, stop=50, npoints=51, unit='mhz', sweep_type='lin')
att = BuildAtten(0.7, freq, reciprocal=False, loc=("att",1))


# In[ ]:


att


# In[ ]:


att.s[0]


# ### Multiplier

# In[ ]:


def BuildMultiplier(Tc, freq, loc=()):
    """
    Simple Multiplier model.  Transmitted amplitude is scaled by 'T'.
    """
    Z_0 = 50.
    label = str(loc)
    nFreqs = len(freq)
    S21 = Tc
    S = np.zeros((nFreqs, 2, 2), dtype=np.complex)
    S[:, 1, 0] = S21
    multNetwork = rf.Network(name=label, frequency=freq, z0=Z_0, s=S)
    return multNetwork


# In[ ]:


freq = rf.Frequency(start=40, stop=50, npoints=51, unit='mhz', sweep_type='lin')
Tc = 0.7*np.exp(1j*pi*0.3)
mult = BuildMultiplier(Tc, freq, loc=('Mult', 1, 2))


# In[ ]:


mult


# In[ ]:


mult.s[0]


# ### 3dB Coupler

# In[ ]:


def Build3dBCoupler(freq, couplerConv="LC", loc=()):
    """
        This generates the scalar SParameters (not dispersive) 3dB Coupler.

        The MZI consists of two 3dB couplers with two phase shifters.  There are
        many possible designs and conventions.  Here we use the following:
        
        The 3dB couplers have the SParams
        [[0,0,a,b],
         [0,0,b,a],
         [a,b,0,0],
         [b,a,0,0]]
        and the number scheme:
        1 |---| 3
            X    
        2 |---| 4
        In option `couplerConv = 'LC'`, these are `a = -1j/sqrt(2)`, `b = -1/sqrt(2)`.  
        In option  `couplerConv = 'ideal'` these are `a = -1j/sqrt(2)`, `b = 1/sqrt(2)`.
    """
    Z_0 = 50.
    label = str(loc)
    
    nFreqs = len(freq)
    if couplerConv == 'LC':
        a = -1j/sqrt(2)
        b = -1/sqrt(2)
    elif couplerConv == 'ideal':
        a = -1j/sqrt(2)
        b = 1/sqrt(2)
    else:
        raise TypeError("couplerConv should be either 'ideal' or 'LC'")

    S31, S32 = a, b
    S41, S42 = b, a     

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


freq = rf.Frequency(start=40, stop=50, npoints=51, unit='mhz', sweep_type='lin')
coup = Build3dBCoupler(freq, loc=("C", 1, 2))


# In[ ]:


coup


# In[ ]:


coup.s[0]


# ### 5 Port Splitter

# In[ ]:


def Build5PortSplitter(freq, loc=()):
    """
        This generates the scalar SParameters (not dispersive) 5 port splitter.

        The coupler has the SParams
        [[0,a,a,a,a,a],
         [a,0,0,0,0,0],
         [a,0,0,0,0,0],
         [a,0,0,0,0,0],
         [a,0,0,0,0,0],
         [a,0,0,0,0,0]]
        where a = sqrt(1/5).  The numbering scheme is
        
          |---| 2
          |---| 3
        1 |---| 4
          |---| 5
          |---| 6
    """
    Z_0 = 50.
    label = str(loc)
    
    nFreqs = len(freq)
    a = np.sqrt(1/5)
    
    proto = np.array([[0,a,a,a,a,a],
                      [a,0,0,0,0,0],
                      [a,0,0,0,0,0],
                      [a,0,0,0,0,0],
                      [a,0,0,0,0,0],
                      [a,0,0,0,0,0]])

    S = np.zeros((nFreqs, 6, 6), dtype=np.complex)
    S[:] = proto.reshape((1,6,6))
    coupNetwork = rf.Network(name=label, frequency=freq, z0=Z_0, s=S)
    return coupNetwork


# In[ ]:


freq = rf.Frequency(start=40, stop=50, npoints=51, unit='mhz', sweep_type='lin')
coup = Build5PortSplitter(freq, loc=("C", 1, 2))


# In[ ]:


coup


# In[ ]:


coup.s[0]


# In[ ]:


(np.matmul(coup.s[0], [1, 0, 0, 0, 0, 0])**2).sum()


# ### MZI

# #### MZI From Func

# In[ ]:


def BuildMZI(theta, phi, freq, couplerConv="LC", reciprocal=False, loc=()):
    """
        This generates the scalar SParameters (not dispersive) for an MZI.  By
        default it is "forward only," however, if reciprocal=True, then it will work
        in reverse as well.

        The MZI consists of two 3dB couplers with two phase shifters.  There are
        many possible designs and conventions.  Here we use the following:
        
        The 3dB couplers have the SParams
        [[0,0,a,b],
         [0,0,b,a],
         [a,b,0,0],
         [b,a,0,0]]
        and the number scheme:
        1 |---| 3
            X    
        2 |---| 4
        In option `couplerConv = 'LC'`, these are `a = -1j/sqrt(2)`, `b = -1/sqrt(2)`.  
        In option  `couplerConv = 'ideal'` these are `a = -1j/sqrt(2)`, `b = 1/sqrt(2)`.

        The two phase shifters have the transmissions
        T1 = exp(i*(phi + theta))
        T2 = exp(i*(phi - theta))

        Together, they yield a device with the following number scheme.

        1 |---| - [T1] - |---| 3
            X              X    
        2 |---| - [T2] - |---| 4
    """
    Z_0 = 50.
    label = str(loc)
    
    nFreqs = len(freq)
    if couplerConv == 'LC':
        S31 = -1j * np.sin(theta) * np.exp(1j*phi)
        S41 =  1j * np.cos(theta) * np.exp(1j*phi)
        S32 =  1j * np.cos(theta) * np.exp(1j*phi)
        S42 =  1j * np.sin(theta) * np.exp(1j*phi)
    elif couplerConv == 'ideal':
        S31 = -1j * np.sin(theta) * np.exp(1j*phi)
        S41 = -1j * np.cos(theta) * np.exp(1j*phi)
        S32 = -1j * np.cos(theta) * np.exp(1j*phi)
        S42 =  1j * np.sin(theta) * np.exp(1j*phi)
    else:
        raise TypeError("couplerConv should be either 'ideal' or 'LC'")

    S = np.zeros((nFreqs, 4, 4), dtype=np.complex)
    S[:, 2, 0] = S31
    S[:, 3, 0] = S41
    S[:, 2, 1] = S32
    S[:, 3, 1] = S42
    if reciprocal:
        S[:, 0, 2] = S31
        S[:, 1, 2] = S41
        S[:, 0, 3] = S32
        S[:, 1, 3] = S42
    mziNetwork = rf.Network(name=label, frequency=freq, z0=Z_0, s=S)
    return mziNetwork


# In[ ]:


freq = rf.Frequency(start=40, stop=50, npoints=51, unit='mhz', sweep_type='lin')
mzi1 = BuildMZI(0.3*pi, -0.4*pi, freq, couplerConv="LC", loc=("X",2))


# In[ ]:


mzi1


# In[ ]:


mzi1.s[0]


# #### MZI From Components

# In[ ]:


def BuildMZIFromComps(TcData, freq, CouplerBuilder, MultBuilder, couplerConv='LC', loc=()):
    """
    Generates an MZI using a Coupler Builder and a Multiplier Builder.  Local networks
    are generated usng
    
    (Tc1, Tc2) = TcData
    coup1 = CouplerBuilder(freq, couplerConv=couplerConv, loc="coup_L")
    multTop = MultBuilder(Tc1, freq, loc="mult_T")
    multBot = MultBuilder(Tc2, freq, loc="mult_B")
    coup2 = CouplerBuilder(freq, couplerConv=couplerConv, loc="coup_R")
    
    and then assmebled into an MZI network and returned.
    
    """
    
    Z_0 = 50.
    label = str(loc)
    
    (Tc1, Tc2) = TcData
    coup1 = CouplerBuilder(freq, couplerConv=couplerConv, loc="coup_L")
    multTop = MultBuilder(Tc1, freq, loc="mult_T")
    multBot = MultBuilder(Tc2, freq, loc="mult_B")
    coup2 = CouplerBuilder(freq, couplerConv=couplerConv, loc="coup_R")
    
    ports = [rf.Circuit.Port(freq, 'port_'+str(i), z0=Z_0) for i in range(4)]
    conns = [ [(ports[0], 0), (coup1, 0)], [(ports[1], 0), (coup1, 1)],  # Input Ports to coup1
              [(coup1, 2), (multTop, 0)], [(coup1, 3), (multBot, 0)],    # coup1 to multipliers
              [(multTop, 1), (coup2, 0)], [(multBot, 1), (coup2, 1)],    # multipliers to coup2
              [(coup2, 2), (ports[2], 0)], [(coup2, 3), (ports[3], 0)]]  # coup2 to Output Ports
    cir = rf.Circuit(conns)
    net = cir.network
    net.name = label
    return net


# In[ ]:


freq = rf.Frequency(start=40, stop=50, npoints=51, unit='mhz', sweep_type='lin')
theta, phi = 0.3*pi, -0.4*pi
P1 = phi + theta
P2 = phi - theta
(Tc1, Tc2) = np.exp(1j*P1), np.exp(1j*P2)
mzi2 = BuildMZIFromComps((Tc1, Tc2), freq, Build3dBCoupler, BuildMultiplier, couplerConv='LC', loc='mzi1')


# In[ ]:


mzi2


# In[ ]:


mzi2.s[0]


# ## Network Combiners

# These functions are used to take a several `network` objects and assemble them together to form a new `network` object.

# ### Cascading

# In[ ]:


def CascadeNetworks(networks, loc=()):
    """
    Given several nxn networks where the first n/2 ports are "input" and the
    last n/2 are "output", it will stack these circuits in the order presented.
    """
    protoNet = networks[0]
    freq = protoNet.frequency
    (nFreqs, nPorts, nPorts) = protoNet.s.shape
    NN = nPorts//2
    Z_0 = 50.
    label = str(loc)

    "Input Ports"
    inPorts = [BuildPort(freq, ("P", loc, i)) for i in range(0, NN)]
    "Output Ports"
    outPorts = [BuildPort(freq, ("P", loc, i)) for i in range(NN, 2*NN)]

    "Simple Connections"
    portInConnections = [ [(inPorts[i], 0), (networks[0], i)] for i in range(NN)]
    portOutConnections = [ [(networks[-1], NN+i), (outPorts[i], 0)] for i in range(NN)]

    intraConnections = []
    for j in range(len(networks)-1):
        conns = [ [(networks[j], NN+i), (networks[j+1], i)] for i in range(NN)]
        intraConnections.extend(conns)

    "Intra MZI Connections"
    cnx = [*portInConnections, *intraConnections, *portOutConnections]

    "Build the Circuit"
    cir = rf.Circuit(cnx)
    net = cir.network
    net.name = label
    return net


# In[ ]:


freq = rf.Frequency(start=40, stop=50, npoints=51, unit='mhz', sweep_type='lin')
mzi1 = BuildMZI(0.3*pi, -0.4*pi, freq, couplerConv="LC", loc="X1")
mzi2 = BuildMZI(0.3*pi, -0.4*pi, freq, couplerConv="LC", loc="X2")
CascadeNetworks([mzi1, mzi2], loc="casc1")


# ### Stacking

# In[ ]:


def StackNetworks(networks, loc=""):
    """
    Given n two-port networks, it will stack these stack these to form a nxn diagonal network.
    """
    NN = len(networks)
    Z_0 = 50.
    label = str(loc)
    
    freq = networks[0].frequency

    "Input Ports"
    inPorts = [BuildPort(freq, ("P", loc, i)) for i in range(0, NN)]
    "Output Ports"
    outPorts = [BuildPort(freq, ("P", loc, i)) for i in range(NN, 2*NN)]

    "Simple Connections"
    portInConnections =  [ [(inPorts[i], 0), (networks[i], 0)] for i in range(NN)]
    portOutConnections = [ [(networks[i], 1), (outPorts[i], 0)] for i in range(NN)]

    "Intra MZI Connections"
    cnx = [*portInConnections, *portOutConnections]

    "Build the Circuit"
    cir = rf.Circuit(cnx)
    net = cir.network
    net.name = label
    return net


# In[ ]:


freq = rf.Frequency(start=40, stop=50, npoints=51, unit='mhz', sweep_type='lin')
mult1 = BuildMultiplier(.4-0.3j, freq, loc='top')
mult2 = BuildMultiplier(.7+0.2j, freq, loc='bottom')
StackNetworks([mult1, mult2], loc="stack1")


# ## Network Builders

# These functions create network objects which replicate Linear Algebra operations and elements.

# ### Miller Triangle

# In[ ]:


def BuildTriangleNetworkMZI(MZIBuilder, loc=(), n=5):
    """
    Builds a triangular network of MZIs as a SciKit-RF Circuit.

    Inputs:
        PortBuilder:A function that yields an network object
        MZIBuilder: A function such that mziNetwork = MZIBuilder(loc)
                    See notes below.
        label:      Every element in a SciKit-RF circuit must have a unique name.  If the circuit
                    consists of multiple triangles, this can be used to separate them.

    It is assumed an MZI's S-Params can be obtained using
    S = MZIBuilder(theta, phi, freqValues, loc=(i_ch, i_in))
    where theta,phi are in radians, freqValues are [f1, f2, ...] in Hz and the
    returned S is of the shape (nFreqs, NPorts, NPorts).
    """
    Z_0 = 50.
    label = str(loc)
        
    "All of the MZIs"
    MZITri1 = np.empty(shape=(n, n), dtype=object)
    for i_ch in range(n):
        for i_in in range(n - i_ch):
            mziLoc = ("MZI", loc, i_ch, i_in)
            mzi = MZIBuilder(mziLoc)
            # mzi = rf.Network(name="MZI_"+labelX+str(i_ch)+str(i_in), frequency=freq, z0=Z_0, s=s)
            MZITri1[i_ch, i_in] = mzi
            
    freq = MZITri1[0,0].frequency
            
    "Input Ports"
    inPorts = [BuildPort(freq, ("P", loc, i)) for i in range(0, n)]
    "Output Ports"
    outPorts = [BuildPort(freq, ("P", loc, i)) for i in range(n, 2*n)]
    "Absorbers on Input Side of Lower Row"
    trashPortsA = [BuildTrashPort(freq, ("T", loc, i)) for i in range(0, n)]
    "Absorbers on Output Side of Lower Row"
    trashPortsB = [BuildTrashPort(freq, ("T", loc, i)) for i in range(n, 2*n)]

    "Simple Connections"
    portInConnections = [ [(inPorts[i_in], 0), (MZITri1[0, i_in], 0)] for i_in in range(n)]
    portOutConnections = [ [(MZITri1[i_ch, 0], 2), (outPorts[i_ch], 0)] for i_ch in range(n)]
    trashConnectionsA  = [ [(MZITri1[i, 4-i], 1), (trashPortsA[i], 0)] for i in range(n)]
    trashConnectionsB  = [ [(MZITri1[i, 4-i], 3), (trashPortsB[i], 0)] for i in range(n)]

    "Intra MZI Connections"
    mziConnectionsU = []
    for i_ch in range(0, n-1):
        for i_in in range(1, n-i_ch):
            c = [(MZITri1[i_ch, i_in], 2), (MZITri1[i_ch, i_in-1], 1)]
            mziConnectionsU.append(c)
    mziConnectionsL = []
    for i_ch in range(0, n-1):
        for i_in in range(0, n-i_ch-1):
            c = [(MZITri1[i_ch, i_in], 3), (MZITri1[i_ch+1, i_in], 0)]
            mziConnectionsL.append(c)
    cnx = [*portInConnections, *portOutConnections, *trashConnectionsA, *trashConnectionsB, *mziConnectionsU, *mziConnectionsL]

    "Build the Circuit"
    cir = rf.Circuit(cnx)
    net = cir.network
    net.name = label
    return net


# In[ ]:


def BuildTriangleNetworkCoupMult(CouplerBuilder, MultBuilder, loc=(), n=5):
    """
    Builds a triangular network of MZIs as a SciKit-RF Circuit.

    Inputs:
        theta_phi:  Of the form such that (theta, phi) = theta_phi[i_ch, i_in] is the theta, phi value 
                    for the MZI at (i_ch, i_in).
        freq:       SciKit-RF frequency object.
        MZIBuilder: A function such that S = MZIBuilder(theta, phi, freqValues, loc=(i_ch, i_in)).
                    See notes below.
        label:      Every element in a SciKit-RF circuit must have a unique name.  If the circuit
                    consists of multiple triangles, this can be used to separate them.

    It is assumed an MZI's S-Params can be obtained using
    S = MZIBuilder(theta, phi, freqValues, loc=(i_ch, i_in))
    where theta,phi are in radians, freqValues are [f1, f2, ...] in Hz and the
    returned S is of the shape (nFreqs, NPorts, NPorts).
    """
    Z_0 = 50.
    label = str(loc)
        
    "All of the MZIs"
    MultTri = np.empty(shape=(n, n, 2), dtype=object)
    CoupTri = np.empty(shape=(n, n, 2), dtype=object)
    internalConnections = []
    for i_ch in range(n):
        for i_in in range(n - i_ch):
            if i_in < n - i_ch - 1:
                coup1 = CouplerBuilder(("C", loc, i_ch, i_in, 0))
                multTop = MultBuilder(("M", loc, i_ch, i_in, 0))
                multBot = MultBuilder(("M", loc, i_ch, i_in, 1))
                coup2 = CouplerBuilder(("C", loc, i_ch, i_in, 1))
                MultTri[i_ch, i_in] = (multTop, multBot)
                CoupTri[i_ch, i_in] = (coup1, coup2)
                conns = [ [(coup1, 2), (multTop, 0)], [(coup1, 3), (multBot, 0)],    # coup1 to multipliers
                          [(multTop, 1), (coup2, 0)], [(multBot, 1), (coup2, 1)]]    # multipliers to coup2
                internalConnections.extend(conns)
            else: # i_in == NN - i_ch - 1
                multTop = MultBuilder(("M", loc, i_ch, i_in, 0))
                MultTri[i_ch, i_in] = (multTop, None)
                conns = []                                                          # No internal connections for a single component
                internalConnections.extend(conns)
    
    freq = MultTri[0,0,0].frequency
                
    "Input Ports"
    inPorts = [BuildPort(freq, ("P", loc, i)) for i in range(0, n)]
    "Output Ports"
    outPorts = [BuildPort(freq, ("P", loc, i)) for i in range(n, 2*n)]

    "Simple Connections"
    portInConnections = [ [(inPorts[i_in], 0), (CoupTri[0, i_in, 0], 0)] for i_in in range(n-1)]
    portInConnections.append( [(inPorts[n-1], 0), (MultTri[0, n-1, 0], 0)])
    portOutConnections = [ [(CoupTri[i_ch, 0, 1], 2), (outPorts[i_ch], 0)] for i_ch in range(n-1)]
    portOutConnections.append( [(MultTri[n-1, 0, 0], 1), (outPorts[n-1], 0)])

    "Intra MZI Connections"
    mziConnectionsU = []
    for i_ch in range(0, n-1):
        for i_in in range(1, n-i_ch):
            if i_in < n - i_ch - 1:
                # Upper-Right Corner of MZI -> Lower-Left Corner of next MZI
                c = [(CoupTri[i_ch, i_in, 1], 2), (CoupTri[i_ch, i_in-1, 0], 1)]
            else:
                # Lower Mult Output -> Lower-Left Corner of next MZI
                c = [(MultTri[i_ch, i_in, 0], 1), (CoupTri[i_ch, i_in-1, 0], 1)]
            mziConnectionsU.append(c)

    mziConnectionsL = []
    for i_ch in range(0, n-1):
        for i_in in range(0, n-i_ch-1):
            if i_in < n - i_ch - 1 - 1:
                # Lower-Right Corner of MZI -> Upper-Left Corner of next MZI
                c = [(CoupTri[i_ch, i_in, 1], 3), (CoupTri[i_ch+1, i_in, 0], 0)]
            else:
                # Lower-Right Corner of MZI -> Lower-Mult Input
                c = [(CoupTri[i_ch, i_in, 1], 3), (MultTri[i_ch+1, i_in, 0], 0)]
            mziConnectionsL.append(c)
    
    cnx = [*portInConnections, *portOutConnections, *internalConnections, *mziConnectionsU, *mziConnectionsL]

    "Build the Circuit"
    cir = rf.Circuit(cnx)
    net = cir.network
    net.name = label
    return net


# ### Vector

# In[ ]:


def BuildVectorAttenuatorNetwork(AttBuilder, loc=(), n=5):
    """
    Builds a linear network of multipliers as a SciKit-RF Circuit.

    Inputs:
        T:          A vector of transmission coefficients.
        freq:       SciKit-RF frequency object.
        MZIBuilder: A function such that S = MZIBuilder(theta, phi, freqValues, loc=(i_ch, i_in)).
                    See notes below.
        label:      Every element in a SciKit-RF circuit must have a unique name.  If the circuit
                    consists of multiple triangles, this can be used to separate them.

    It is assumed an MZI's S-Params can be obtained using
    netObject = AttBuilder(t, freqValues, loc=i)
    where 't' is in linear scale, freqValues are [f1, f2, ...] in Hz and the
    returned S is of the shape (nFreqs, NPorts, NPorts).
    """
    Z_0 = 50.
    label = str(loc)

    "All of the Attenuators"
    attVec = np.empty(shape=(n,), dtype=object)
    for i in range(n):
        attLoc = ("M", loc, i)
        att = AttBuilder(attLoc)
        attVec[i] = att
        
    freq = attVec[0].frequency
    "Input Ports"
    inPorts = [BuildPort(freq, ("P", loc, i)) for i in range(0, n)]
    "Output Ports"
    outPorts = [BuildPort(freq, ("P", loc, i)) for i in range(n, 2*n)]

    "Simple Connections"
    portInConnections = [ [(inPorts[i], 0), (attVec[i], 0)] for i in range(n)]
    portOutConnections = [ [(attVec[i], 1), (outPorts[i], 0)] for i in range(n)]

    "Intra MZI Connections"
    cnx = [*portInConnections, *portOutConnections]

    "Build the Network"
    cir = rf.Circuit(cnx)
    net = cir.network
    net.name = label
    return net


# ### Miller Complete

# In[ ]:


def BuildMillerNetwork(CouplerBuilder, MultBuilder, AttBuilder, n=5, labels=('Uh', 'S', 'V')):
    """
    Assembles a full Miller Network consisting of two triangles and a vector of attenuators.
    """
    UhNet = BuildTriangleNetworkCoupMult(CouplerBuilder, MultBuilder, loc=labels[0], n=n)
    SNet = BuildVectorAttenuatorNetwork(AttBuilder, loc=labels[1])
    VNet = BuildTriangleNetworkCoupMult(CouplerBuilder, MultBuilder, loc=labels[2], n=n)
    millerNet = CascadeNetworks([UhNet, SNet, VNet])
    return millerNet


# ### Label Generation

# #### Miller

# In[ ]:


def MillerTriangleMultLocs(n, loc):
    """
    Generates all location signatures for all Multipliers in a Miller Triangle.
    """
    multLocs = []
    for i_ch in range(n):
        for i_in in range(n - i_ch):
            for side in range(2):
                multLocs.append(("M", loc, i_ch, i_in, side))
    return multLocs


# In[ ]:


MillerTriangleMultLocs(3, "Uh")


# In[ ]:


def MillerTriangleMultLocsX(n, loc):
    """
    Generates all location signatures for all Multipliers in a Miller Triangle,
    taking into account that the bottom row consists of only one multiplier.
    """
    multLocs = []
    for i_ch in range(n):
        for i_in in range(n - i_ch):
            if i_in < n - i_ch - 1:
                nSides = 2
            else: # i_in == NN - i_ch - 1
                nSides = 1
            for side in range(nSides):
                multLocs.append(("M", loc, i_ch, i_in, side))
    return multLocs


# In[ ]:


MillerTriangleMultLocsX(3, 'Uh')


# In[ ]:


def MillerTriangleCoupLocs(n, loc):
    """
    Generates all location signatures for all 3dB Couplers in a Miller Triangle.
    """
    coupLocs = []
    for i_ch in range(n):
        for i_in in range(n - i_ch):
            for side in range(2):
                coupLocs.append(("C", loc, i_ch, i_in, side))
    return coupLocs


# In[ ]:


MillerTriangleCoupLocs(3, "Uh")


# In[ ]:


def MillerTriangleCoupLocsX(n, loc):
    """
    Generates all location signatures for all 3dB Couplers in a Miller Triangle,
    taking into account that the bottom row consist of only one Multiplier and
    no couplers.
    """
    coupLocs = []
    for i_ch in range(n):
        for i_in in range(n - i_ch - 1):
            for side in range(2):
                coupLocs.append(("C", loc, i_ch, i_in, side))
    return coupLocs


# In[ ]:


MillerTriangleCoupLocsX(3, "Uh")


# In[ ]:


def SVecAllMultLocs(n, loc):
    """
    Generates all location signatures for all Multipliers in a Vector.
    """
    multLocs = []
    for i in range(n):
        multLocs.append(("M", loc, i))
    return multLocs


# In[ ]:


SVecAllMultLocs(3, 'S')


# In[ ]:


def MillerMultLocs(n, labels=('Uh','S','V')):
    """
    Generates all location signatures for all Multipliers in a Miller Triangle.
    """
    locs1 = MillerTriangleMultLocs(n, labels[0]) 
    locs2 = SVecAllMultLocs(n, labels[1])
    locs3 = MillerTriangleMultLocs(n, labels[2])
    return locs1 + locs2 + locs3


# In[ ]:


MillerMultLocs(3, ('Uh','S','V'))


# In[ ]:


def MillerMultLocsX(n, labels=('Uh','S','V')):
    """
    Generates all location signatures for all Multipliers in the Miller Arch,
    taking into account that the bottom row consists of only one multiplier.
    """
    locs1 = MillerTriangleMultLocsX(n, labels[0]) 
    locs2 = SVecAllMultLocs(n, labels[1])
    locs3 = MillerTriangleMultLocsX(n, labels[2])
    return locs1 + locs2 + locs3


# In[ ]:


MillerMultLocsX(3, ('Uh','S','V'))


# In[ ]:


def MillerCoupLocs(n, labels=('Uh', 'V')):
    """
    Generates all location signatures for all 3dB Couplers in the Miller Arch.
    """
    locs1 = MillerTriangleCoupLocs(n, labels[0]) 
    locs3 = MillerTriangleCoupLocs(n, labels[1])
    return locs1 + locs3


# In[ ]:


MillerCoupLocs(3, ('Uh','V'))


# In[ ]:


def MillerCoupLocsX(n, labels=('Uh', 'V')):
    """
    Generates all location signatures for all Multipliers in the Miller Arch,
    taking into account that the bottom row consists of only one multiplier.
    """
    locs1 = MillerTriangleCoupLocsX(n, labels[0]) 
    locs3 = MillerTriangleCoupLocsX(n, labels[1])
    return locs1 + locs3


# In[ ]:


MillerCoupLocsX(3, ('Uh','V'))


# #### New

# In[ ]:


def NewMultLocs(n, loc):
    """
    Generates all location signatures for all Multipliers in a Miller Triangle.
    """
    multLocs = []
    for i_out in range(n):
        for i_in in range(n):
            multLocs.append(("M", loc, i_in, i_out))
    return multLocs


# ### New Architecture

# In[ ]:


def BuildNewNetwork(Splitter5WayBuilder, MultBuilder, loc=(), n=5):
    """
    Builds a network of Multipliers as a SciKit-RF Circuit.

    Inputs:
        MultBuilder: A function that yields an network object
        Splitter5WayBuilder: A function such that mziNetwork = MZIBuilder(loc)
                    See notes below.
        label:      Every element in a SciKit-RF circuit must have a unique name.  If the circuit
                    consists of multiple triangles, this can be used to separate them.

    """
    Z_0 = 50.
    label = str(loc)
        
    "All of the Multipliers"
    multGrid = np.empty(shape=(n, n), dtype=object)
    for i_out in range(n):
        for i_in in range(n):
            multLoc = ("M", loc, i_in, i_out)
            mult = MultBuilder(multLoc)
            multGrid[i_out, i_in] = mult
            
    freq = multGrid[0,0].frequency
            
    "Input Ports"
    inPorts = [BuildPort(freq, ("P", loc, i)) for i in range(0, n)]
    "Output Ports"
    outPorts = [BuildPort(freq, ("P", loc, i)) for i in range(n, 2*n)]
    "ingress splitters"
    inSplitters = [Splitter5WayBuilder(("Si", loc, i)) for i in range(0, n)]
    "egress splitters"
    outSplitters = [Splitter5WayBuilder(("So", loc, i)) for i in range(0, n)]

    "Simple Connections"
    portInConnections = [ [(inPorts[i_in], 0), (inSplitters[i_in], 0)] for i_in in range(n)]
    portOutConnections = [ [(outSplitters[i_out], 0), (outPorts[i_out], 0)] for i_out in range(n)]

    "Splitter to Mult to Splitter Connections"
    internalConnections = []
    for i_in in range(0, n):
        for i_out in range(0, n):
            c1 = [(inSplitters[i_in], i_out+1), (multGrid[i_out, i_in], 0)]
            c2 = [(multGrid[i_out, i_in], 1), (outSplitters[i_out], i_in+1)]
            internalConnections.extend([c1, c2])
    cnx = [*portInConnections, *portOutConnections, *internalConnections]

    "Build the Circuit"
    cir = rf.Circuit(cnx)
    net = cir.network
    net.name = label
    return net


# ## Data Conversions

# In[ ]:


def ConvertThetaPhiToTc(theta_phi):
    """
    Converts an MZI Miller Triangle theta_phi array with shape (NN, NN, 2) to multiplier 
    Tc (ie Complex Transmission) array with shape (NN, NN, 2)  where 
    Tc[2,3,0] = T is the complex transmission of the top multiplier
    at position (i_ch, i_in) = (2, 3).  Tc[2,3,1] would correspond to the bottom
    multiplier.
    """
    NN = theta_phi.shape[0]
    Tc = np.full(shape=(NN, NN, 2), fill_value=np.nan, dtype=np.complex)
    for i_ch in range(NN):
        for i_in in range(NN - i_ch):
            theta, phi = theta_phi[i_ch, i_in]
            Ph1 = phi + theta
            Ph2 = phi - theta
            Tc[i_ch, i_in, 0] = np.exp(1j*Ph1)
            Tc[i_ch, i_in, 1] = np.exp(1j*Ph2)
    return Tc


# In[ ]:


def ConvertThetaPhiToTcX(theta_phi):
    """
    Converts an MZI Miller Triangle theta_phi array with shape (NN, NN, 2) to multiplier 
    Tc (ie Complex Transmission) array with shape (NN, NN, 2)  where 
    Tc[2,3,0] = T is the complex transmission of the top multiplier
    at position (i_ch, i_in) = (2, 3).  Tc[2,3,1] would correspond to the bottom
    multiplier.
    
    Note that the bottom row does not contain MZIs, but rather straight multipliers.
    As the ThetaPhi data was calculated for MZIs, the phase shift from the lack of 
    two 3dB couplers must be taken into account.
    """
    NN = theta_phi.shape[0]
    Tc = np.full(shape=(NN, NN, 2), fill_value=np.nan, dtype=np.complex)
    for i_ch in range(NN):
        for i_in in range(NN - i_ch):
            theta, phi = theta_phi[i_ch, i_in]
            Ph1 = phi + theta
            Ph2 = phi - theta
            if i_in < NN - i_ch - 1:
                Tc[i_ch, i_in, 0] = np.exp(1j*Ph1)
                Tc[i_ch, i_in, 1] = np.exp(1j*Ph2)
            if i_in == NN - i_ch - 1:
                Tc[i_ch, i_in, 0] = np.exp(1j*(Ph1+np.pi))
    return Tc


# In[ ]:


def ConvertSToTc(sTerms):
    """
    Converts a Miller Singular vector with shape (NN) to multiplier 
    Tc (ieComplex Transmission) array with shape (NN)  where 
    Tc[2] = T is the complex transmission of the third multiplier.
    """
    NN = sTerms.shape[0]
    Tc = np.full(shape=(NN, 2), fill_value=np.nan, dtype=np.complex)
    for i in range(NN):
        t = sTerms[i]
        Tc[i] = t
    return Tc


# In[ ]:


# def ConvertSToPhTrDict(sTerms, loc):
#     """
#     Converts an MZI S vector with shape (NN) to multiplier 
#     PT (ie Phase-Transmission) array with shape (NN, 2)  where 
#     PT[2] = (p, t) is the phase and transmission of the third multiplier.
#     """
#     NN = sTerms.shape[0]
#     PhTr = np.zeros(shape=(NN, 2))
#     for i in range(NN):
#         Ph = 0.
#         Tr = sTerms[i]
#         PhTr[i, 0] = Ph
#         PhTr[i, 1] = Tr
#     return PT


# # Tests

# ## Miller Trial - MZI Based

# We begin with a passive kernel, defined below.

# In[ ]:


Ks = np.array([[-0.05+0.06j, -0.  -0.13j, -0.07-0.15j,  0.11+0.28j, -0.05-0.18j],
               [-0.1 -0.19j, -0.3 -0.05j, -0.28+0.07j, -0.25+0.28j, -0.11-0.29j],
               [ 0.21-0.18j, -0.08-0.14j,  0.03+0.2j , -0.23+0.24j, -0.06+0.32j],
               [-0.29-0.31j,  0.12+0.09j,  0.08-0.02j,  0.31+0.12j, -0.22-0.18j],
               [-0.18-0.06j,  0.08-0.21j,  0.25-0.18j, -0.26-0.1j ,  0.13+0.1j ]])


# Next, we perform a simple SVD on it.

# In[ ]:


V, S, Uh = np.linalg.svd(Ks)


# In[ ]:


V


# In[ ]:


S


# In[ ]:


Uh


# We create the MillerBuilder and use it to create theta, phi data for kernel.

# In[ ]:


miller = MillerBuilder(couplerConv='LC', verbose=False)
theta_phi1, sTerms, theta_phi2 = miller.ConvertKToMZI(Ks)


# For instance `theta_phi1[0,0] -> (theta, phi)` for the MZI at `i_ch=1`, `i_in=1`.

# In[ ]:


theta_phi1.shape


# Next we define the three functions required by `BuildTriangleNetoworkMZI` and `BuildVectorAttenuatorNetwork`, namely `MZIBuilder`, and `AttBuilder`.  These two functions are each responsible for creating a scikit-rf network object.  They are supplied with a location tuple, which may be useful in their construction.

# In[ ]:


freq45 = rf.Frequency(start=45, stop=45, npoints=1, unit='mhz', sweep_type='lin')


# In[ ]:


def MZIBuilder1(loc):
    # loc expected to be of the form ("MZI", loc, i_ch, i_in) where loc is from the superseding object ('Uh', 'V', etc) and is expected to be internally unique to the generating object.
    (_, locParent, i_ch, i_in) = loc
    theta, phi = theta_phi1[i_ch, i_in] 
    network = BuildMZI(theta, phi, freq45, couplerConv="LC", reciprocal=False, loc=loc)
    return network


# In[ ]:


def MZIBuilder2(loc):
    # loc expected to be of the form ("MZI", loc, i_ch, i_in) where loc is from the superseding object ('Uh', 'V', etc) and is expected to be internally unique to the generating object.
    (_, locParent, i_ch, i_in) = loc
    theta, phi = theta_phi2[i_ch, i_in] 
    network = BuildMZI(theta, phi, freq45, couplerConv="LC", reciprocal=False, loc=loc)
    return network


# In[ ]:


MZIBuilder1(("MZI", "Uh", 0, 0))


# In[ ]:


def AttBuilder(loc):
    (_, locParent, i) = loc
    T = S[i]
    network = BuildAtten(T, freq45, reciprocal=False, loc=loc)
    return network


# In[ ]:


AttBuilder(("M", 'S', 0))


# We then use these functions to construct the first triagular network and compare it to the desired transmission matrix determined via SVD.

# In[ ]:


netTria1 = BuildTriangleNetworkMZI(MZIBuilder1, loc='Uh')
TTria1 = netTria1.s[0, 5:, :5]


# In[ ]:


np.allclose(TTria1, Uh)


# We do the same for the second triangle.

# In[ ]:


netTria2 = BuildTriangleNetworkMZI(MZIBuilder2, loc='V')
TTria2 = netTria2.s[0, 5:, :5]
TTria2


# In[ ]:


np.allclose(TTria2, V)


# We do the same for the vector of attenuators.

# In[ ]:


netAtt = BuildVectorAttenuatorNetwork(AttBuilder, loc="S")
TAtt = netAtt.s[0, 5:, :5]
TAtt


# In[ ]:


np.allclose(TAtt, np.diag(S))


# Finally, we can test that the three networks, when cascaded, reconstruct the original desired kernel.

# In[ ]:


millerNet = CascadeNetworks([netTria1, netAtt, netTria2])


# In[ ]:


T = millerNet.s[0, 5:, :5]
T


# In[ ]:


np.allclose(Ks, T)


# ## Miller Trial - Multipliers and 3dB Couplers

# We begin with a passive kernel, defined below.

# In[ ]:


Ks = np.array([[-0.05+0.06j, -0.  -0.13j, -0.07-0.15j,  0.11+0.28j, -0.05-0.18j],
               [-0.1 -0.19j, -0.3 -0.05j, -0.28+0.07j, -0.25+0.28j, -0.11-0.29j],
               [ 0.21-0.18j, -0.08-0.14j,  0.03+0.2j , -0.23+0.24j, -0.06+0.32j],
               [-0.29-0.31j,  0.12+0.09j,  0.08-0.02j,  0.31+0.12j, -0.22-0.18j],
               [-0.18-0.06j,  0.08-0.21j,  0.25-0.18j, -0.26-0.1j ,  0.13+0.1j ]])


# Next, we perform a simple SVD on it.

# In[ ]:


V, S, Uh = np.linalg.svd(Ks)


# In[ ]:


V


# In[ ]:


S


# In[ ]:


Uh


# We create the MillerBuilder and use it to create theta, phi data for kernel.

# In[ ]:


miller = MillerBuilder(couplerConv='LC', verbose=False)
theta_phi1, sTerms, theta_phi2 = miller.ConvertKToMZI(Ks)


# For instance `theta_phi1[0,0] -> (theta, phi)` for the MZI at `i_ch=1`, `i_in=1`.

# Next we convert the Theta Phi data to Phase Transmission data.  This is trivial for ideal elements, but once non-ideal elements are involved, the multipliers might be responsible for making up the losses in the couplers.

# In[ ]:


TcData1 = ConvertThetaPhiToTcX(theta_phi1)


# In[ ]:


TcData2 = ConvertThetaPhiToTcX(theta_phi2)


# Finally, we build the network of devices and examine the transmissive portion of the S-Matrix.  Indeed, we find that it is identical to the desired unitary matrix originally provided by the SVD above.

# In[ ]:


freq = rf.Frequency(start=45, stop=45, npoints=1, unit='mhz', sweep_type='lin')


# In[ ]:


def CouplerBuilder(loc):
    return Build3dBCoupler(freq, loc=loc)


# In[ ]:


CouplerBuilder(("C", 'Uh', 0, 0, 0))


# In[ ]:


def MultBuilder1(loc):
    (_, locParent, i_ch, i_in, side) = loc
    Tc = TcData1[i_ch, i_in, side]
    return BuildMultiplier(Tc, freq, loc)


# In[ ]:


def MultBuilder2(loc):
    (_, locParent, i_ch, i_in, side) = loc
    Tc = TcData2[i_ch, i_in, side]
    return BuildMultiplier(Tc, freq, loc)


# In[ ]:


MultBuilder1(("M", 'Uh', 0, 0, 0))


# In[ ]:


def AttBuilder(loc):
    (_, locParent, i) = loc
    T = S[i]
    network = BuildAtten(T, freq45, reciprocal=False, loc=loc)
    return network


# In[ ]:


AttBuilder(("M", 'S', 0))


# In[ ]:


netTria1 = BuildTriangleNetworkCoupMult(CouplerBuilder, MultBuilder1, loc="1", n=5)
TTria1 = netTria1.s[0, 5:, :5]
TTria1


# In[ ]:


np.allclose(TTria1, Uh)


# We do the same for the second triangle.

# In[ ]:


netTria2 = BuildTriangleNetworkCoupMult(CouplerBuilder, MultBuilder2, loc="2")
TTria2 = netTria2.s[0, 5:, :5]
TTria2


# In[ ]:


np.allclose(TTria2, V)


# We do the same for the vector of attenuators.

# In[ ]:


netAtt = BuildVectorAttenuatorNetwork(AttBuilder, loc="S")
TAtt = netAtt.s[0, 5:, :5]
TAtt


# In[ ]:


np.allclose(TAtt, np.diag(S))


# Finally, we can test that the three networks, when cascaded, reconstruct the original desired kernel.

# In[ ]:


millerNet = CascadeNetworks([netTria1, netAtt, netTria2])


# In[ ]:


T = millerNet.s[0, 5:, :5]


# In[ ]:


np.allclose(Ks, T)


# In[ ]:





# ## Miller Trial - Multipliers and 3dB Couplers - Full Architecture

# We begin with a passive kernel, defined below.

# In[ ]:


Ks = np.array([[-0.05+0.06j, -0.  -0.13j, -0.07-0.15j,  0.11+0.28j, -0.05-0.18j],
               [-0.1 -0.19j, -0.3 -0.05j, -0.28+0.07j, -0.25+0.28j, -0.11-0.29j],
               [ 0.21-0.18j, -0.08-0.14j,  0.03+0.2j , -0.23+0.24j, -0.06+0.32j],
               [-0.29-0.31j,  0.12+0.09j,  0.08-0.02j,  0.31+0.12j, -0.22-0.18j],
               [-0.18-0.06j,  0.08-0.21j,  0.25-0.18j, -0.26-0.1j ,  0.13+0.1j ]])


# Next, we perform a simple SVD on it.

# In[ ]:


V, S, Uh = np.linalg.svd(Ks)


# In[ ]:


V


# In[ ]:


S


# In[ ]:


Uh


# We create the MillerBuilder and use it to create theta, phi data for kernel.

# In[ ]:


miller = MillerBuilder(couplerConv='LC', verbose=False)
theta_phi1, sTerms, theta_phi2 = miller.ConvertKToMZI(Ks)


# For instance `theta_phi1[0,0] -> (theta, phi)` for the MZI at `i_ch=1`, `i_in=1`.

# Next we convert the Theta Phi data to Phase Transmission data.  This is trivial for ideal elements, but once non-ideal elements are involved, the multipliers might be responsible for making up the losses in the couplers.

# In[ ]:


TcData1 = ConvertThetaPhiToTcX(theta_phi1)


# In[ ]:


TcData2 = ConvertThetaPhiToTcX(theta_phi2)


# Finally, we build the network of devices and examine the transmissive portion of the S-Matrix.  Indeed, we find that it is identical to the desired unitary matrix originally provided by the SVD above.

# In[ ]:


freq = rf.Frequency(start=45, stop=45, npoints=1, unit='mhz', sweep_type='lin')


# In[ ]:


def CouplerBuilder(loc):
    return Build3dBCoupler(freq, loc=loc)


# In[ ]:


CouplerBuilder(("C", 'Uh', 0, 0, 0))


# In[ ]:


def MultBuilder(loc):
    (_, locParent, i_ch, i_in, side) = loc
    if locParent == 'Uh':
        Tc = TcData1[i_ch, i_in, side]
    elif locParent == 'V':
        Tc = TcData2[i_ch, i_in, side]
    return BuildMultiplier(Tc, freq, loc)


# In[ ]:


MultBuilder(("M", 'Uh', 0, 0, 0))


# In[ ]:


def AttBuilder(loc):
    (_, locParent, i) = loc
    T = S[i]
    network = BuildAtten(T, freq45, reciprocal=False, loc=loc)
    return network


# In[ ]:


AttBuilder(("M", 'S', 0))


# In[ ]:


millerNet = BuildMillerNetwork(CouplerBuilder, MultBuilder, AttBuilder, n=5, labels=('Uh', 'S', 'V'))
T = millerNet.s[0, 5:, :5]
T


# In[ ]:


np.allclose(T, Ks)


# In[ ]:





# ## New Architecture Trial - Multipliers and 5 Port Splitters

# We begin with a passive kernel, defined below.

# In[ ]:


Ks = np.array([[-0.05+0.06j, -0.  -0.13j, -0.07-0.15j,  0.11+0.28j, -0.05-0.18j],
               [-0.1 -0.19j, -0.3 -0.05j, -0.28+0.07j, -0.25+0.28j, -0.11-0.29j],
               [ 0.21-0.18j, -0.08-0.14j,  0.03+0.2j , -0.23+0.24j, -0.06+0.32j],
               [-0.29-0.31j,  0.12+0.09j,  0.08-0.02j,  0.31+0.12j, -0.22-0.18j],
               [-0.18-0.06j,  0.08-0.21j,  0.25-0.18j, -0.26-0.1j ,  0.13+0.1j ]])


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




