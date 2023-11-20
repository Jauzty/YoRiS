# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:16:14 2023

@author: aust_
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt
from Lr_Lkin_convolution import KLF_FRI_FRII, LR, Rx_values, Lrr

zvalues = [0.5, 0.9, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.8, 4.2, 4.8]
efrI=[-3.1, -3.5, -3.3, -3.8, -3.9, -3.5, -3.5]
efrII=[-2.2, -2.4, -2., -2.4, -2.55, -2.4, -2.2]
erII=[-2.8, -2.9, -2.3, -2.8, -3., -2.7, -2.5]

def epsilonrad(zz, zvalues, Lrr, efrI, PhirFRII, PhirFRII21, erII, efrII):
    # FRI and FRII epsilon radio
    # test the amount to go from Lradio to Lbol and how it changes with redshift.
    
    tt = np.where(np.array(zvalues) == zz)[0]
    
    if len(tt[0]) > 0:
        index = tt[0][0]
        lgefr = efrI[index] # epsilon radio for FRI
        eII = erII[index] # epsilon radio for FRII, matching PhibolFRII21 with PhiradioII
        e_II = efrII[index] # FRII, matching PhibolradII with PhiradioII
    
    nr = len(Lrr)
    Lbol_erI = np.zeros(nr)
    Lbol_erII = np.zeros(nr)
    Lbol_ii = np.zeros(nr)
    
    for io in range(nr):
        Lbol_erI[io] = Lrr[io] - lgefr  # both are in erg/s, to be plotted with Phir21, RLF of FRI
        Lbol_erII[io] = Lrr[io] - eII
        Lbol_ii[io] = Lrr[io] - e_II
    
    jac1 = np.abs(np.gradient(Lbol_erII, Lrr))  # jac is 1.0
    jac2 = np.abs(np.gradient(Lbol_ii, Lrr))    # jac is 1.0
    
    PhibolFRII21 = PhirFRII21 * jac1
    PhibolradII = PhirFRII * jac2  # no convolution
    return Lbol_erI


    
