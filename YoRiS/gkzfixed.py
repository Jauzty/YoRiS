# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:25:36 2023

@author: aust_
"""
import matplotlib.pyplot as plt
import numpy as np
from Lr_Lkin_convolution import KLF_FRI_FRII, LR, Rx_values, Lmin, Lrr

alpha = 0.54#constants taken from input.pro
beta = 22.1
ar = 0.7 #could consider different values of the exponent to change the slope of FRI/FRII
fcav = 4.0
bb = np.log10(7 * fcav) + 36.0
aa = 0.68
# Convert radio luminosity LR from W/Hz @ 1.4GHz to W/Hz @ 5GHz
L5 = LR - ar * np.log10(5 / 1.4)  # L5 is in W/Hz @ 5GHz
# Convert L5 to erg/s @ 5GHz
L5r = L5 + 7 + 9 + np.log10(5)  # L5r is in erg/s @ 5GHz
# Calculate kinetic luminosity kin
kin = alpha * (L5r) + beta   
# Calculate kinetic luminosity kkin
kkin = aa * (LR - 25) + bb + 7.0  # kkin is in erg/s

def gkFRI(Lbol):
    
    L = [44, 44.5, 45, 45.5, 46, 46.5, 47]
    #gkinI = [0.14, 0.11, 0.07, 0.073, 0.07, 0.08, 0.018]# z = 0.5FRI
    #gkinI = [0.42, 0.42, 0.42, 0.42, 0.42, 0.25, 0.07]# z = 0.5FRII
    #gkinI = [0.15, 0.13, 0.06, 0.04, 0.065, 0.07, 0.025]# z = 0.9 FRI
    gkinI = [0.42, 0.42, 0.42, 0.42, 0.4, 0.23, 0.12]#z = 0.9 FRII
   
    tt = np.where(np.array(L) == Lbol)
    if len(tt[0]) > 0:
        index = tt[0][0]  # Get the first matching index
        gk = gkinI[index]

    nk = len(kin)
    LgLbol = np.zeros(nk)
    
    # gkin for FRI, comparison in the bolometric plane
    for io in range(nk):
        LgLbol[io] = -np.log10(gk) + kin[io]
    #print(gk)
    #print(f' Lboll/Lglbol is {min(Lboll/LgLbol)}, {max(Lboll/LgLbol)}') 
    #lkin/lbol = gk, we dont know gk but if we assume we know it and get it right then Lbol/Lbol should == 1
    return LgLbol

loggkFRI = np.log10([0.15, 0.13, 0.06, 0.04, 0.065, 0.07, 0.025])
loggkFRII = np.log10([0.42, 0.42, 0.42, 0.42, 0.4, 0.23, 0.12])
L = [44, 44.5, 45, 45.5, 46, 46.5, 47]

plt.figure(figsize=(10, 6))
plt.scatter(L, loggkFRI, color='blue', marker='o', label='FRI')
plt.scatter(L, loggkFRII, color='red', marker='d', label='FRII')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Luminosity Lbol')
plt.ylabel('log(gk)')
plt.title('log(gk) vs. Luminosity at z = 0.5')
plt.grid(True)
plt.show()
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300