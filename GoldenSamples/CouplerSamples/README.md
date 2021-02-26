In this folder one can find the measurement data files for the 3dB couplers. 

The names of the couplers follow the rule:
MM - Male-Male (used connectors)
MF- Male-Female (used connectors)
and the last number indicates their lab numbering (for proper referencing)
The filed contain a 2x2 complex matrix. The files with the name STD are te standart deviation of the measurements.  

The OLD_Files folder included 12 measurements performed using the switches - However, we observed that the switches introduce some reflections (possibly an impedance missmatch), that distrurbs the overall performance of the couplers. 

The files, MM_3dB_7 and MF_3dB_7 where measured manually, assuning that the unused ports of the 3dB are properly terminated and using a s a refernce port the port #3 of the shitch, for which the VNA was calibrated. 
Similarly the STD files are the standart deviations of the measurements. 



26.2.2021 UPDATE: I repeated the measurements for the same MM and MF 7 3dB coupler, this time using the matching circuitry that reduces reflection loss. It seems that the NEW measurements are relatively closewith the previous values that I extracted manualy. 