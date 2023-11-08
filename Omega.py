# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:39:51 2023

@author: aust_
"""

"""This script uses an alternate method for calculating the gk ratios using Omega_kin/Omega_rad 
as opposed to the previous method which compared the BLF to the KLF via Lkin = gk*Lbol. We now are
using eq 10 and 11 from La Franca+10."""

import numpy as np
from scipy.integrate import simps
from QLFs_to_duty_cycles import myBolfunc, Lbol
from Lr_Lkin_convolution import kincore, kinlobe
from Ueda_Updated_Py import Ueda_mod, Ueda_14
from UedaXLF_to_RLF import fradioueda21, fradioFRII, fradioueda

red=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
Lr_values = np.linspace(40, 54, 1000)
lrmin = 30 #erg/s
lrmax = 47
Lrr = np.linspace(lrmin, lrmax, 1000)
Lmin = 41
Lx = np.linspace(41, 49, 1000)
Rx = np.linspace(-10, 3, 1000)


def Omega(Lx, Lmin, Lbol, Lrr, Rx, red):
    
    n = 1000
    r = len(red)
    Lxx = myBolfunc(1, Lbol)
    kx = Lbol - Lxx
    kxx=np.interp(Lx,Lxx,kx)
    Lboll = kxx + Lx
    jacbol = np.abs(np.gradient(Lboll, Lx))
    
    ixx = np.where(np.array(Lboll) > 46.5)
    if np.mean(ixx) != -1:
        # Extract values from Lboll using the indices in ixx
        LoggL = Lboll[ixx]
    
    # Get the number of elements in LoggL
    ff = len(LoggL)
    
    # Initialize arrays Phibo21 and Phibolall with zeros
    Phibo21 = np.zeros((r, ff))
    Phibolall = np.zeros((r, ff))
    gkkII = np.zeros(r)
    gkinFRII = np.zeros(r)
    PhiLx21 = np.zeros((r, n))
    Phi_21 = np.zeros((r, n))
    Phi_22 = np.zeros((r, n))
    Phi_23 = np.zeros((r, n))
    PhiLxall = np.zeros((r, n))
    Phixbol21 = np.zeros((r, n))
    Phixbolall = np.zeros((r, n))
    Phir = np.zeros((r, n))
    Phir21 = np.zeros((r, n))
    PhirFRII = np.zeros((r, n))
    PhirFRII21 = np.zeros((r, n))
    Phikin21 = np.zeros((r, n))
    Phikin = np.zeros((r, n))
    PhikinFRII = np.zeros((r, n))
    PhikinFRII21 = np.zeros((r, n))
    fkinI = np.zeros((r, n))
    fkin21 = np.zeros((r, n))
    fradI = np.zeros((r, n))
    frad21 = np.zeros((r, n))
    OkinI = np.zeros(r)
    Okin21 = np.zeros(r)
    OradI = np.zeros(r)
    Orad21 = np.zeros(r)
    gkin = np.zeros(r)
    gkinFRII21 = np.zeros(r)
    gkinFRI = np.zeros(r)
    gkinFRI21 = np.zeros(r)
    fkinII = np.zeros((r, n))
    fkin21II = np.zeros((r, n))
    fradII = np.zeros((r, ff))
    fradII21 = np.zeros((r, ff))
    OkinII = np.zeros(r)
    Okin21II = np.zeros(r)
    OradIIbol = np.zeros(r)
    OradII21 = np.zeros(r)
    OradII = np.array([2.1606037e+37, 2.6143319e+38, 1.7513788e+38, 4.8540429e+40, 3.4929767e+41, 1.7987821e+39, -1.3907784e+39])
    
    # Loop through the calculations
    for ii in range(r):
        PhiLx21[ii, :] = Ueda_14(Lx, red[ii], 0)
        Phi_21[ii, :] = Ueda_14(Lx, red[ii], 1) + PhiLx21[ii, :] 
        Phi_22[ii, :] = Ueda_14(Lx, red[ii], 2) + Phi_21[ii, :] 
        Phi_23[ii, :] = Ueda_14(Lx, red[ii], 3) + Phi_22[ii, :] 
        PhiLxall[ii, :] = Ueda_14(Lx, red[ii], 4) + Phi_23[ii, :]
    
        Phixbol21[ii, :] = PhiLx21[ii, :] * jacbol
        Phixbolall[ii, :] = PhiLxall[ii, :] * jacbol
    
        Phir[ii, :], P = fradioueda(red[ii], Rx, Lrr, Lmin)
        Phir21[ii, :], PhirFRII21[ii, :], P21, PP21 = fradioueda21(red[ii], Rx, Lrr, Lmin)
    
        PhirFRII[ii, :], PP = fradioFRII(red[ii], Rx, Lrr, Lmin)
    
        Phik21, blah = kincore(P21, Phir21[ii, :])
        Phik, kin = kincore(P, Phir[ii, :])
        Phikin21[ii, :] = Phik21
        Phikin[ii, :] = Phik
        
        PhikFRII21, blah = kinlobe(PP21, PhirFRII21[ii, :], Lr_values)
        PhikFRII, kkin = kinlobe(PP, PhirFRII[ii, :], Lr_values)
        PhikinFRII21[ii, :] = PhikFRII21
        PhikinFRII[ii, :] = PhikFRII
    
        fkinI[ii, :] = Phikin[ii, :] * 10 ** kin
        OkinI[ii] = simps(fkinI[ii, :], kin)
        fkin21[ii, :] = Phikin21[ii, :] * 10 ** kin
        Okin21[ii] = simps(fkin21[ii, :], kin)
    
        fradI[ii, :] = Phixbolall[ii, :] * 10 ** Lboll
        OradI[ii] = simps(fradI[ii, :], Lboll)
        frad21[ii, :] = Phixbol21[ii, :] * 10 ** Lboll
        Orad21[ii] = simps(frad21[ii, :], Lboll)
    
        gkinFRI21[ii] = Okin21[ii] / Orad21[ii]
        gkinFRI[ii] = OkinI[ii] / OradI[ii]
    
        fkinII[ii, :] = PhikinFRII[ii, :] * 10 ** kkin
        OkinII[ii] = simps(fkinII[ii, :], kkin)
        fkin21II[ii, :] = PhikinFRII21[ii, :] * 10 ** kkin
        Okin21II[ii] = simps(fkin21II[ii, :], kkin)
    
        Phibolall[ii, :] = np.interp(LoggL, Phixbolall[ii, :], Lboll)
        Phibo21[ii, :] = np.interp(LoggL, PhiLx21[ii, :], Lboll)
        fradII[ii, :] = Phibolall[ii, :] * 10 ** LoggL
        fradII21[ii, :] = Phibo21[ii, :] * 10 ** LoggL
        OradIIbol[ii] = simps(fradII[ii, :], LoggL)
        OradII21[ii] = simps(fradII21[ii, :], LoggL)
    
        gkinFRII[ii] = OkinII[ii] / OradIIbol[ii]
        gkinFRII21[ii] = Okin21II[ii] / OradII21[ii]
        gkkII[ii] = OkinII[ii] / OradII[ii]
    
        gkin[ii] = (OkinII[ii] / OradIIbol[ii]) + (OkinI[ii] / OradI[ii])
    
        print(f'gkinFRI21 is {gkinFRI21}')
        print(f'gkinFRII21 is {gkinFRII21}')
        print(f'gkinFRI is {gkinFRI}')
        print(f'gkinFRII is {gkinFRII}')
        print(min(Lx), min(kin), min(kkin), min(Lboll))
    
    return

Omega(Lx, Lmin, Lbol, Lrr, Rx, red)