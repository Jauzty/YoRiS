# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:25:36 2023

@author: aust_
"""
import matplotlib.pyplot as plt
import numpy as np
from Lr_Lkin_convolution import LR

alpha = 0.54#constants taken from input.pro
beta = 22.1
ar = 0.7 #could consider different values of the exponent to change the slope of FRI
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
    #gkinI = [0.85, 0.65, 0.31, 0.2, 0.14, 0.09, 0.025]# z = 0.5FRI
    #gkinI = [0.9, 0.9, 0.9, 0.9, 0.5, 0.35, 0.11]# z = 0.5FRII doesnt really match well full stop
    #gkinI = [0.85, 0.84, 0.35, 0.22, 0.15, 0.08, 0.045]# z = 0.9 FRI
    #gkinI = [0.9, 0.9, 0.9, 0.9, 0.7, 0.37, 0.2]#z = 0.9 FRII
    #gkinI = [0.32, 0.25, 0.19, 0.13, 0.1, 0.07, 0.045]# z = 1.2 FRI
    #gkinI = [2.5, 2.5, 1.4, 0.85, 0.74, 0.45, 0.37]#z = 1.2 FRII
    #gkinI = [0.32, 0.32, 0.32, 0.08, 0.06, 0.06, 0.042]# z = 1.6 FRI
    #gkinI = [2.4, 2.4, 1.5, 0.8, 0.74, 0.45, 0.32]#z = 1.6 FRII
    #gkinI = [0.3, 0.3, 0.3, 0.14, 0.07, 0.049, 0.042]# z = 2 FRI
    #gkinI = [1.4, 1.4, 1.4, 0.8, 0.7, 0.3, 0.2]#z = 2 FRII
    #gkinI = [0.09, 0.09, 0.09, 0.08, 0.055, 0.042, 0.04]# z = 2.4 FRI
    #gkinI = [1.8, 1.8, 1.8, 1.6, 0.7, 0.33, 0.2]#z = 2.4 FRII
    #gkinI = [0.2, 0.2, 0.2, 0.2, 0.045, 0.033, 0.028]# z = 2.8 FRI
    #gkinI = [1.8, 1.8, 1.8, 1.6, 0.8, 0.35, 0.24]#z = 2.8 FRII
    gkinI = [0.17, 0.17, 0.11, 0.1, 0.06, 0.04, 0.026]# z = 3.5 FRI
    #gkinI = [3, 3, 1.8, 1.7, 0.65, 0.3, 0.15]#z = 3.5 FRII
    
    tt = np.where(np.array(L) == Lbol)
    if len(tt[0]) > 0:
        index = tt[0][0]  # Get the first matching index
        gk = gkinI[index]

    nk = len(kkin)
    LgLbol = np.zeros(nk)
    
    # gkin for FRI, comparison in the bolometric plane
    for io in range(nk):
        LgLbol[io] = -np.log10(gk) + kin[io]
    #print(gk)
    #print(f' Lboll/Lglbol is {min(Lboll/LgLbol)}, {max(Lboll/LgLbol)}') 
    #lkin/lbol = gk, we dont know gk but if we assume we know it and get it right then Lbol/Lbol should == 1
    return LgLbol


L = [44, 44.5, 45, 45.5, 46, 46.5, 47]
redshifts = [0.5, 0.9, 1.2, 1.6, 2, 2.4, 2.8, 3.5]

# FRI data
gkinI_FRI = [
    [0.85, 0.65, 0.31, 0.2, 0.14, 0.09, 0.025],  # z = 0.5 FRI
    [0.85, 0.84, 0.35, 0.22, 0.15, 0.08, 0.045],  # z = 0.9 FRI
    [0.32, 0.25, 0.19, 0.13, 0.1, 0.07, 0.045],  # z = 1.2 FRI
    [0.32, 0.32, 0.32, 0.08, 0.06, 0.06, 0.042],  # z = 1.6 FRI
    [0.3, 0.3, 0.3, 0.14, 0.07, 0.049, 0.042],  # z = 2 FRI
    [0.09, 0.09, 0.09, 0.08, 0.055, 0.042, 0.04],  # z = 2.4 FRI
    [0.2, 0.2, 0.2, 0.2, 0.045, 0.033, 0.028],  # z = 2.8 FRI
    [0.17, 0.17, 0.11, 0.1, 0.06, 0.04, 0.026],  # z = 3.5 FRI
]

# FRII data
gkinI_FRII = [
    [0.9, 0.9, 0.9, 0.9, 0.5, 0.35, 0.11],  # z = 0.5 FRII
    [0.9, 0.9, 0.9, 0.9, 0.7, 0.37, 0.2],  # z = 0.9 FRII
    [2.5, 2.5, 1.4, 0.85, 0.74, 0.45, 0.37],  # z = 1.2 FRII
    [2.4, 2.4, 1.5, 0.8, 0.74, 0.45, 0.32],  # z = 1.6 FRII
    [1.4, 1.4, 1.4, 0.8, 0.7, 0.3, 0.2],  # z = 2 FRII
    [1.8, 1.8, 1.8, 1.6, 0.7, 0.33, 0.2],  # z = 2.4 FRII
    [1.8, 1.8, 1.8, 1.6, 0.8, 0.35, 0.24],  # z = 2.8 FRII
    [3, 3, 1.8, 1.7, 0.65, 0.3, 0.15],  # z = 3.5 FRII
]

# Plotting subplots
fig, axs = plt.subplots(2, 4, figsize=(28, 14), sharex=True, sharey=True)

for i in range(2):
    for j in range(4):
        idx = 4 * i + j
        if idx < len(redshifts):
            axs.flatten()[idx].scatter(L, np.log10(gkinI_FRI[idx]), s=100, label=f'FRI at z = {redshifts[i]}')
            axs.flatten()[idx].scatter(L, np.log10(gkinI_FRII[idx]), s =100, label=f'FRII at z = {redshifts[i]}')
            if j >= 0:
                axs.flatten()[idx].set_xlabel('L')
            if j == 0:
                axs.flatten()[idx].set_ylabel('log(gk)')
            axs.flatten()[idx].legend()
            axs.flatten()[idx].grid(True)

if __name__ == "__main__": 
    plt.rcParams['font.size'] = 21
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.dpi'] = 300
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.show()