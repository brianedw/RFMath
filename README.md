# RFMath

## Purpose:

The goal of this project is to perform linear algebraic computations using digitally controlled analog devices operating at 45MHz.  The heart of these computations is complex matrix multiplication.  Herein, we explore two architectures, one based on MZIs configured according *Miller et al.* and another based on TODO.  The code within this repository is the complete code-base for this project, inluding firmware for the components, drivers for communication, measurement instrument communications, simulation, architecture settings, and device characterization.

## Components:

The experiments utilize a large number of rf components.  This includes evaluation boards, devices which have been designed and built, and measurement aparatus.  These are described below.

### Multipliers
  * 75x
  * Designed and Fabricated in house
  * Given a 45MHz wave with complex amplitude $a$ at its input, it will create render $t a$ at the output, where $t$ is a complex number.  As such, this device functions as a phase shifter, attenuator, and amplifier.  It is based on the XXX VGA and a pair of XXX Phase shifters and is controlled by two analog voltages in the range 0-5VDC.  Each device is coupled to an arduino which recieves serial communication and uses two digitial inputs on the range [0, 1023] to create the two analog voltages, which loosely correspond to phase and amplitude.  However, the precise relationship between the digital instruction and the complex transmission coefficient varies between devices.
  
### 3dB Couplers
  * 75x
  * Designed and Fabricated in house
  * Based on a LC resonant design.  Note that due to parasitic resistances, design sensitivity, and availability of stock component values, the actual rendered transmissions are not ideal.  When two 3dB Couplers are combined with two Multipliers they form an Mach-Zender-Interferometer (MZI).

### 1:5 Splitter
  * 2x
  * Designed and Fabricated in house.  A trivial application of the XXX chip.
  * Based on TODO.

### RF Switch (8 port) 
  * 2x
  * Demo board
  * The switch connects Port-$S$ to one of eight other ports Port-$1-8$.  It operates based on three binary inputs.  We're using the TODO evaluation board.  Unused ports are internally terminated at 50Ohms.  Controlled through an which converts serial port commands to digital outputs.

### ENA 5071C VNA
  * 1x
  * 2 Port Vector Network Analyzer.  Used to measure various individual components, and through the use of the two RF Switches, can be used to measure devices of higher order.  Scripted using TODO.
  
## Process:

*TODO Speculative*

### Device Characterization:
First, we began by characterizing the different component classes.

#### RF Switch
There was minimal difference between both the channels within the devices and between the devices themselves.  It was noted as $t=XX$ TODO.  However, since they were used to give our VNA multiport cabilities and the channels were nearly identical, we simply calibrated the VNA with them in line in cases in which we used these switches.  Raw measurement results and analysis are in `GoldenSamples/SwitchSamples`.

#### 1:5 Splitter:
There was minimal difference between both the channels within the devices and between the devices themselves.  It was noted as $t=TODO$.  Raw measurement results and analysis are in `GoldenSamples/SplitterSamples`.

#### 3dB Couplers:
Five couplers were characteized.  The deviations between them were small, although their behavior was not that of an ideal 3dB Coupler.  Raw measurement results and analysis are in `GoldenSamples/CouplerSamples`.

#### Multipier:
Multipliers and the controlling arduino boards were paired and from hence forth considered as a single unit and collectively referred to as Multipliers.  Ten Multipliers were measured on a grid of digital input values v1 = [0, 11, ..., 1023] and v2 = [0, 11, ..., 1023].  These results were compared using a complex valued Principal Component Analysis (PCA).  Based on training/testing data sets, it was determined that the first three PCA components could adequeately characterize any device and as such any device's personality could be described by three complex numbers.  Note: multiplying all personality coefficients by the same complex number is equivalent to a phase rotation such as would be experienced by connecting a cable to the device.  Given a device's "personality coefficients" and a desired transmission coefficient $t$, one can determine the digital input values.  The actual device characterization was performed with the devices placed in situ within their matrix architecture.  This allows for minor pertubations to be accounted for in the "personality coefficients".  Raw measurement results and analysis are in `GoldenSamples/MultiplierSamples`.

### Matrix Characterization / Personality Determination:
For each architecture in turn (Miller and TODO), the devices were assembled.  A full simulation of the assembled architecture was also created which included connecting cable lengths.  Within this simulation, all components were built from their experimental characterized measurements.  The Multipliers were set to have a "default" personality, which included some information about connecting cable properties.  The transimission values of all the multipliers were set to a strategic value where the personality coefficients were particularly sensitive and the experimental response matrix was measured.  This was done for five other strategic values.  These results were recorded in `MatrixCharacterization`.

With these 6 experimental measurements in hand, the personality coefficients on the multipliers were optimized so as to minimize the difference between the simulated experiments and the measured results.  These personality coefficients were recorded and became part of each multiplier's inverse function which allows us to determine the proper digital inputs to obtain a desired transmission coefficient.

### Matrix Realization:
The multipliers were then set for the test matrix and this was characterized.

### Feedback Realization:
The feedback couplers were then added and the result of the recursion was characterized.

## Philosophy:

This is research code and the goal is rapid development and scientific correctness.  The Jupyter (aka iPython) notebook (.ipynb) is fantastic for this purpose.  However, with a project of this size modularizing across multiple files and classes and connecting them through the use of imports is also necessary.  This is best handled through traditional python files (.py).  To achieve both of these aims, I develop in JupyterLab with the settings that a python file is always created whenever the notebook is saved.  Documentation, small tests, and example codes are placed directly into the Jupyther files.

The jupyter notebooks operate based on cells.  Traditionally, one would put all of the code for a specific function or class into a cell, thus defeating the purpose of jupyter notebook.  As a work-a-round, I define a empty class, and then set the individual methods within it using `setattr()`.

## File Descriptions:

The ordering of the descriptions is from most basic (ie least dependent) to most integrated (ie more dependent).

### Colorize.ipynb
Several functions which are useful converting 2D matrices of complex values into an RGB color space wherein phase is mapped to hue and amplitude is mapped to saturation and brightness.

### Logger.ipynb
A class which allows formatted printing to the console with automatic indenting that increases with the depth of the context.  It is useful for tracing code which is operating several function calls deep.  Good OOP practice would call for abstraction and unit-testing, however when comparing to legacy code, logging can be more efficient.

### UtilityMath.ipynb
Several linear algebra functions which are of general mathematical utility.

### Miller.ipynb
*Dependent on: Logger*

Implements the algorithm described in TODO for converting matrix multiplication into two triangular networks of MZIs separted by a column of attenuators.  The input of the algorithm is a matrix, $K$ and the output is the MZI settings $\theta$ and $\phi$ for each component. 


