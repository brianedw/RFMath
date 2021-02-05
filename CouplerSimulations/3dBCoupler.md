# 3dB Coupler

## Mouser BOM:

81-LQW18ANR13G00D
Inductor 130nH +-2% DCR 1.4Ohm

81-LQW18ANR18J0ZD
Inductor 180nH +-5% 2.2Ohm

581-06031U680FAT2A
Capacitor 68pF 1%

581-06031U101FAT2A
Capacitor 100pF 1%

The goal is to get the most accurate model of the 3dB coupler possible around the design frequency of 45MHz.  Since this is a circuit level description, this boils down to getting the most detailed device models possible.

Both manufacturers have provided a Touchstone files (`*.s2p`) for each device.  In all cases the s2p measurements do not include 45MHz and instead focus on frequecies above this point.  I used CST to compare the measured values up to 1GHz against a parameterized model and in each case optimized these parametric values to get the best fit possible. (`capParamExtraction.cst`, `inductorParamExtraction.cst`).  In all four cases the fit was excellent.  The values were recorded (see `*.png`)  These device models were then used in a simulation of the 3dB coupler. (`FullParametric.cst`) and a polar plot of the results was created (`polar.png`).
