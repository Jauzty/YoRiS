# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:38:08 2023

@author: aust_
"""
import numpy as np
from QLFs_to_duty_cycles import Lboll, Lboloptdata
from Lr_Lkin_convolution import KLF_FRI_FRII, LR, Rx_values, z, Lmin, Lrr
from scipy.integrate import simps

FRIfracX = [0.0754413, 0.0715513, 0.0629219, 0.0638171, 0.0527830, 0.0948788, 0.1290664, 0.0723229, 0.0323717, 0.1618970, 0.0168045, 0.2352269]
#FRII frac list
FRIIfracX = [0.0308062, 0.0565392, 0.0465047, 0.0291714, 0.0165781, 0.0254845, 0.0162047, 0.0194296, 0.0030720, 0.0420260, 0.0217945, 0.0]

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

def gkFRI(z):
    
    red=[0.5, 0.9, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.8, 4.2, 4.8]
    gkin22I=[0.3, 0.25, 0.13, 0.13, 0.08, 0.07, 0.06, 0.06,0.05,0.08,0.08]
    gkin22I_03=[0.35, 0.3, 0.25, 0.15, 0.1, 0.1, 0.07, 0.09,0.08,0.08,0.05]
    gkin22I_conv=[0.55, 0.45, 0.35, 0.25, 0.25, 0.22, 0.18, 0.2,0.18,0.18,0.18]
    gkinI=[0.2, 0.15, 0.1, 0.1, 0.06, 0.05, 0.05, 0.05,0.04,0.08,0.07]
    gkinI_03=[0.3, 0.2, 0.2, 0.13, 0.08, 0.08, 0.05, 0.08,0.05,0.08,0.05]
    gkinI_conv=[0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.13, 0.15,0.15,0.15,0.15]
    gknew =[0.045820312500000106, 0.015312500000000166, 0.007919921875000182, 0.005698242187500184, 0.004091796875000187, 0.001306152343750192, 0.0010083007812501923, 0.008935546875000177, 0.00036987304687519356, 0.008935546875000177, 0.0005895996093751932]
    
    gk = 0.0  
    gk_03 = 0.
    gk_conv = 0.0  
    gk22 = 0.0  
    gk22_03 = 0.0  
    gk22_conv = 0.0

    
    tt = np.where(np.array(red) == z)
    if len(tt[0]) > 0:
        index = tt[0][0]  # Get the first matching index
        gk = gkinI[index]
        gk_03 = gkinI_03[index]
        gk_conv = gkinI_conv[index]
        gk22 = gkin22I[index]
        gk22_03 = gkin22I_03[index]
        gk22_conv = gkin22I_conv[index]
        newgk = gknew[index]
    
    nk = len(kin)
    LgLbol = np.zeros(nk)
    LgLbol_03 = np.zeros(nk)
    LgLbol_conv = np.zeros(nk)
    LgLbol22 = np.zeros(nk)
    LgLbol22_03 = np.zeros(nk)
    LgLbol22_conv = np.zeros(nk)
    LgLbolnew = np.zeros(nk)
    
    # gkin for FRI, comparison in the bolometric plane
    for io in range(nk):
        LgLbol[io] = -np.log10(gk) + kin[io]
        LgLbol_03[io] = -np.log10(gk_03) + kin[io]
        LgLbol_conv[io] = -np.log10(gk_conv) + kin[io]
        LgLbol22[io] = -np.log10(gk22) + kin[io]
        LgLbol22_03[io] = -np.log10(gk22_03) + kin[io]
        LgLbol22_conv[io] = -np.log10(gk22_conv) + kin[io]
        LgLbolnew[io] = -np.log10(newgk) + kin[io]
    #print(gk)
    #print(f' Lboll/Lglbol is {min(Lboll/LgLbol)}, {max(Lboll/LgLbol)}') 
    #lkin/lbol = gk, we dont know gk but if we assume we know it and get it right then Lbol/Lbol should == 1
    return LgLbolnew
    
    
def gkFRII(z):
    
    redII=[0.5, 0.9, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2]
    gkIIup_conv=[3.5, 1.1, 1., 1., 1.,1., 1.1]
    gkIIup=[3.5, 1.1, 0.8, 1., 1., 1., 1.1]
    gkII21=[0.4, 0.45, 0.5, 0.5, 0.5, 0.55, 0.45,0.45]
    gkII21_03=[0.45, 0.55, 0.55, 0.75, 0.75, 0.7,0.7,0.7]
    gkII21_conv=[0.8, 1., 1.6, 0.75, 0.75, 0.75,1.2]
    gkII22=[0.6, 0.6, 0.65, 0.65, 0.55, 0.55,0.55,0.55]
    gkII22_03=[0.7, 0.7, 0.75, 0.75, 0.75, 0.75,0.75,0.75]
    gkII22_conv=[0.8, 1., 1.6, 0.75, 0.75, 0.75,1.2]
    gtest=[0.85, 0.9, 0.95, 1.0, 1.0, 1.0,1.0, 1.]
    gkIIl=[0.1, 0.1, 0.08, 0.07, 0.07, 0.07, 0.08]
    
    gkup_c = 0
    gkup = 0
    gk21 = 0
    gk21_03 = 0
    gk21_c = 0
    gk22 = 0
    gk22_03 = 0
    gk22_c = 0
    gti = 0
    gkl = 0
    
    tt = np.where(np.array(redII) == z)
    if len(tt[0]) > 0:
        index = tt[0][0]
        gkup_c = gkIIup_conv[index]
        gkup = gkIIup[index]
        gk21 = gkII21[index]
        gk21_03 = gkII21_03[index]
        gk21_c = gkII21_conv[index]
        gk22 = gkII22[index]
        gk22_03 = gkII22_03[index]
        gk22_c = gkII22_conv[index]
        gti = gtest[index]
        gkl = gkIIl[index]
    # gkin FRII - LOWER LIMIT
    # We try to estimate a lower limit on gkin for FRII pop
    # comparing the KLF given by the bright end RLF of Willott2001
    # with the BLF given by the whole X-ray AGN pop, inclusive of Compton thick sources.
    

    nk = len(kkin)
    Lbbolc = np.zeros(nk)
    Lbbol = np.zeros(nk)
    Lbfrii = np.zeros(nk)
    Lbfrii_03 = np.zeros(nk)
    Lbfriiconv = np.zeros(nk)
    Lbfrii22 = np.zeros(nk)
    Lbfrii22_03 = np.zeros(nk)
    Lbfrii22conv = np.zeros(nk)
    LLbtest = np.zeros(nk)
    Llow = np.zeros(nk)

    for io in range(nk):
        Lbbolc[io] = -np.log10(gkup_c) + kkin[io]  # PhikinFRII_conv2
        Lbbol[io] = -np.log10(gkup) + kkin[io]  # PhikFRII
        Lbfriiconv[io] = -np.log10(gk21_c) + kkin[io]
        Lbfrii_03[io] = -np.log10(gk21_03) + kkin[io]
        Lbfrii[io] = -np.log10(gk21) + kkin[io]
        Lbfrii22conv[io] = -np.log10(gk22_c) + kkin[io]
        Lbfrii22_03[io] = -np.log10(gk22_03) + kkin[io]
        Lbfrii22[io] = -np.log10(gk22) + kkin[io]
        LLbtest[io] = -np.log10(gti) + kkin[io]
        Llow[io] = -np.log10(gkl) + kkin[io]  # PhikinFRII_conv2, lower limit on gk

    # gk from comparison between cumulative distributions:
    # at a fixed bolometric luminosity, the ratio between the FRII and the QSO distributions
    # can't exceed ~ 2 (according to van Velzen)
    

if __name__ == "__main__":    
    gkFRI(z)
    gkFRII(z)