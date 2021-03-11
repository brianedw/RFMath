# Measurements of AD5PS-1+ "5 Ways Core & Wire Power Splitter, 1 - 400 MHz, 50Ω"

These files are measurements across the 5:1 splitters.  A total of ten (TODO: is this right?) were built, named "1" through "10".  These boards are a trivial breakout of the Mini-Circuit AD5PS-1+.  Ideal values for this device would be $S_{XS} = (1/5)^{(1/2)} = 0.447$

The measurements are taken at 45MHz.  Ports are enumerated as Port S and Port1 through Port5.  

The file `sp_X.txt` shows 

$S_{1,S}$<br/>
$S_{2,S}$<br/>
$S_{3,S}$<br/>
$S_{4,S}$<br/>
$S_{5,S}$

for device `X` while `sp_XSTD.txt` shows the standard deviation of the measurement for the same device.

The measurements were performed using the digitally selectable switch<sup>[1](#digiswitch)</sup>.  All ports on the switch were assumed to be identical and a single calibration was used. 

Currently measured devices are 1, 2, 3, 4, 5, 6, 7, and 10.

Example of a measurement file `sp_1.txt`
```
 (4.193069077980985937e-01-1.188417291128507219e-01j)
 (4.181777454677157446e-01-1.210264572851457721e-01j)
 (4.185912810348070279e-01-1.209928710486715686e-01j)
 (4.195498860268132568e-01-1.189496670964912545e-01j)
 (4.177344793820767510e-01-1.224909885990784347e-01j)
```

Example of STD file `sp_1STD.txt`
```
2.061667362856654806e-02
1.987549309514274887e-02
2.465660472284976826e-02
2.011781303839011922e-02
1.992198737651435306e-02

```

All files were created using `NumPy.savetxt()` and can be easily imported using `NumPy.loadtxt()`.

Additionally, as a point of comparison, the files from the manufacturer (`AD5PS-1+_UNIT#1.s6p` and `AD5PS-1+_UNIT#2.s6p`) are also saved herein.

<a name="digiswitch">1</a>: The digital switch was created using the HMC253AQS24 evaluation board.  This board comes with a high-pass capacitors (100pF) on each port.  These were switched for larger values (470pF) which provided little barriar to 45MHz.  Off ports on the HMC253AQS24 chip are nominally matched to 50$\Omega$, but close inspection at 45MHz indicates that they behave closer to an open circuit and are only matched well at much higher frequencies.  Terminators (50Ohm) were placed in parallel on T's to reduce reflections from unused ports.  On the "off" ports, they appear as nearly matched terminations.  The "on" port will show a dramatic reduction in signal, but this can be deembeded out by including the T and terminator in the calibration.  The switch was trivially controlled through a Python script which connected to an Arduino via USB.
