# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:26:59 2023

@author: aust_
"""

import numpy as np
from UedaXLF_to_RLF import Pradio, PFRII
from Ueda_Updated_Py import Ueda_14
from scipy.signal import convolve as conv
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp1d

#FRI frac list
FRIfrac = [0.0732533, 0.0524898, 0.0720952, 0.0476212, 0.0436078, 0.0746024, 0.1040458, 0.2458492, 0.0303237, 0.1447227, 0.0501334, 0.1460006]
#FRII frac list
FRIIfrac = [0.0240113, 0.0318131, 0.0232578, 0.0133658, 0.0116493, 0.0183424, 0.0109917, 0.0066876, 0.0009984, 0.0252248, 0.0204266, 0.0]

def kincore(P, Phir):
    ar = 0.7
    alpha = 0.54
    beta = 22.1
    L5 = P - ar*np.log10(5/1.4)
    L5r = L5 + 7 + 9 + np.log10(5)
    kin = alpha*(L5r) + beta
    Phikin = conv(Phir, P, mode = 'same')
    return Phikin, kin

def kinlobe(P, PhirFRII, Lr_values):
    fcav = 4
    bb = np.log10(7*fcav) + 36
    aa = 0.68
    kkin = aa*(P-25) + bb + 7 
    PhikinFRII_conv = convolve(PhirFRII, Lr_values, kkin, kkin, 0.05)
    return PhikinFRII_conv, kkin
    
def convolve(f, y, xmed, x, sigma):
    """written to match the convolution function in IDL as the numpy one is not the same, 
    it previously caused some issues"""
    nx = len(x)
    res = np.zeros(nx, dtype=float)

    for ix in range(nx):
        k = np.exp(-((x[ix] - xmed)**2) / (2.0 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
        res[ix] = simps(f * k, y)

    return res

def KLF_FRI_FRII(LR, Rx, z):
    alpha = 0.54#constants taken from input.pro
    beta = 22.1
    ar = 0.7 #could consider different values of the exponent to change the slope of FRI/FRII
    fcav = 4.0
    bb = np.log10(7 * fcav) + 36.0
    aa = 0.68
    # Convert radio luminosity P from W/Hz @ 1.4GHz to W/Hz @ 5GHz
    L5 = LR - ar * np.log10(5 / 1.4)  # L5 is in W/Hz @ 5GHz
    # Convert L5 to erg/s @ 5GHz
    L5r = L5 + 7 + 9 + np.log10(5)  # L5r is in erg/s @ 5GHz
    # Calculate kinetic luminosity kin
    kin = alpha * (L5r) + beta   
    # Calculate kinetic luminosity kkin
    kkin = aa * (LR - 25) + bb + 7.0  # kkin is in erg/s


    Lr_values = np.linspace(40, 54, 1000)# different LR arrays for calculating FRI and FRII because they dont match otherwise
    Lr_values1 = LR+(7+np.log10(1.4)+9)
    nr = len(Lr_values)
    Phir = np.zeros(nr)
    Phir21 = np.zeros(nr)
    Phirgaus = np.zeros(nr)
    Lx = np.zeros(nr)
    Lx1 = np.zeros(nr)
    temp = np.zeros(nr)
    temp1 = np.zeros(nr)
    Rx_values1 = Lr_values1 - 44 #correct way to calculate Rx works for FRII but not FRI

    for s in range(nr):
        temp = Lr_values[s] - Rx_values
        temp1 = Lr_values1[s] - Rx_values1 #creating separate arrays for Lx values of FRI and FRII
        Lx = temp[::-1]
        Lx1 = temp1[::-1]
        Phi_x = np.zeros(nr)
        Phi_x1 = np.zeros(nr)

        for i in range(0, 5): #This part is very important, we are summing over all the Nh to calculate the total luminosity function in the Radio
            Phi_x = Phi_x + Ueda_14(Lx, z, i)
            Phi_x1 = Phi_x1 + Ueda_14(Lx1, z, i)
            Phi_x[(Lx < Lmin)] = 0
            Phi_x1[(Lx1 < Lmin)] = 0
        Phi_x21 = Ueda_14(Lx, z, 0)
        
        Phi_x[(Lx < Lmin)] = 0
        Phi_x21[(Lx < Lmin)] = 0
        Phi_x1[(Lx1 < Lmin)] = 0
        
        P = Pradio(Lx, Lr_values[s], z)
        PP = PFRII(Lx1, Lr_values1[s], z)
        Phir[s] = simps(P * Phi_x, Lx)
        Phir21[s] = simps(P*Phi_x21, Lx)
        Phirgaus[s] = simps(PP * Phi_x1, Lx1)

    Phikin21conv = conv(Phir21, P, mode = 'same')
    Phikin = conv(Phir, P, mode = 'same') # matches but strange convolution method and not consistent
    PhikinFRII = convolve(Phirgaus, Lr_values, kkin, kkin, 0.05)
        
    kin_interp = np.linspace(min(kin), max(kkin), 1000)
    Phikintot = np.interp(kin_interp, kin, Phikin) + np.interp(kin_interp, kkin, PhikinFRII)
    print(min(kin))
    print(min(kkin))

    if __name__ == "__main__":
        fig, ax = plt.subplots(figsize=(10, 7))
        plt.semilogy(kin, Phikin, label= 'FRI')
        plt.semilogy(kkin, PhikinFRII, label = 'FRII')
        plt.semilogy(kin_interp, Phikintot, label = 'Total Kinetic Output', color = 'black')
        plt.legend(loc='upper right')
        ax.grid()
        ax.set_xlabel('$ \log L_{k} [erg s^{-1}] $')
        ax.set_ylabel('$\log Phi(L_{k}) [Mpc^{-3} dex^{-1}]$')
        plt.title('Kinetic Luminosity Function FRI + FRII')
        plt.rcParams['figure.dpi'] = 300
        plt.ylim(1e-12, 1e-3)
        plt.xlim(40, 48)
        plt.show()
    return PhikinFRII, Phikin, Phikin21conv, kin, kkin

Rx_values = np.linspace(-10, -1, 1000)  # Rx range calculating with LR+(7+np.log10(1.4)+9) -44 does not work idk why
#z_values = [1, 2, 3, 4, 5]
z = 1.25  # Redshift value
Lmin = 41  # Minimum luminosity
LR = np.linspace(15, 32, 1000) #luminosity range in W/Hz based on federicas code

if __name__ == "__main__":
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.dpi'] = 300
    KLF_FRI_FRII(LR, Rx_values, z)
