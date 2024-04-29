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


def kincore(LR, Phir):
    ar = 0.7
    alpha = 0.54
    beta = 22.1
    L5 = LR - ar*np.log10(5/1.4) #W/Hz 5GHz
    L5r = L5 + 7 + 9 + np.log10(5) #erg/s 5GHz
    kin = alpha*(L5r) + beta
    Phikin = conv(Phir, LR, mode = 'same')
    return Phikin, kin

def kinlobe(LR, PhirFRII, Lr_values):
    fcav = 4
    bb = np.log10(7*fcav) + 36
    aa = 0.68
    kkin = aa*(LR-25) + bb + 7 
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

def KLF_FRI_FRII(Lrr, z, LR):
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
    kkin2017 = 0.89 * (LR-25) + 36.05 + 7#erg/s 1.4ghz ineson
    kkin2013 = 0.9 * (LR-25) +35.12 + np.log10(1+z) + 0.58*np.log10(100) + 7 # S and G contains dependance on source size
    kkin2011 = 0.63 * (LR-25) + 37.76 + 7 # o'sullivan erg/s at 1.4ghz
    kkin1999 = 0.86 * (LR-25) + 37.37 + np.log10(15/15) + 7 # o'sullivan erg/s at 1.4ghz

    #Lr_values = np.linspace(39.7, 51, 1000)# different LR arrays for calculating FRI and FRII because they dont match otherwise
    Lr_values = Lrr
    nr = len(Lr_values)
    Phir = np.zeros(nr)
    Phir21 = np.zeros(nr)
    Phirgaus = np.zeros(nr)
    Phirg21 = np.zeros(nr)
    Lx = np.zeros(nr)
    Lx1 = np.zeros(nr)
    temp = np.zeros(nr)
    temp1 = np.zeros(nr)
    Rx_values1 = Lr_values - 44 #correct way to calculate Rx works for FRII but not FRI
    Rx_values = np.linspace(-10, -1, 1000) #FRI

    for s in range(nr):
        temp = Lr_values[s] - Rx_values
        temp1 = Lr_values[s] - Rx_values1 #creating separate arrays for Lx values of FRI and FRII
        Lx = temp[::-1]
        Lx1 = temp1[::-1]
        Phi_x = np.zeros(nr)
        Phi_x1 = np.zeros(nr)
        Phi_xg = np.zeros(nr)

        for i in range(0, 5): #This part is very important, we are summing over all the Nh to calculate the total luminosity function in the Radio
            Phi_x = Phi_x + Ueda_14(Lx, z, i)
            Phi_x1 = Phi_x1 + Ueda_14(Lx1, z, i)
            Phi_x[(Lx < Lmin)] = 0
            Phi_x1[(Lx1 < Lmin)] = 0
        Phi_x21 = Ueda_14(Lx, z, 0)
        Phi_xg = Ueda_14(Lx1, z, 0)
        
        Phi_x[(Lx < Lmin)] = 0
        Phi_xg[(Lx < Lmin)] = 0
        Phi_x21[(Lx < Lmin)] = 0
        Phi_x1[(Lx1 < Lmin)] = 0
        
        P = Pradio(Lx, Lr_values[s], z)
        PP = PFRII(Lx1, Lr_values[s], z)
        Phir[s] = simps(P * Phi_x, Lx)
        Phir21[s] = simps(P*Phi_x21, Lx)
        Phirg21[s] = simps(PP * Phi_xg, Lx1)
        Phirgaus[s] = simps(PP * Phi_x1, Lx1)

    Phir21 = Phir21
    Phirg21 = Phirg21
    Phikin21conv = convolve(Phir21, LR, kin, kin, 0.01)
    Phikin = convolve(Phir, LR, kin, kin, 0.01)
    Phikinscatter = convolve(Phir21, LR, kin, kin, 0.25)
    Phikinscatter2 = convolve(Phir21, LR, kin, kin, 0.47)
    PhikinFRII = convolve(Phirgaus, LR, kkin, kkin, 0.01)
    PhikinFRIIscatter = convolve(Phirg21, LR, kkin, kkin, 0.25)
    PhikinFRIIscatter2 = convolve(Phirg21, LR, kkin, kkin, 0.7)
    PhikinFRII21 = convolve(Phirg21, LR, kkin, kkin, 0.01) #HB
    #PhikinFRII2017 = convolve(Phirg21, Lr_values, kkin2017, kkin2017, 0.25) #ineson
    #PhikinFRII2013 = convolve(Phirg21, Lr_values, kkin2013, kkin2013, 0.25) #shabala
    #PhikinFRII2011 = convolve(Phirg21, Lr_values, kkin2011, kkin2011, 0.25) #o'sullivan 
    #PhikinFRII1999 = convolve(Phirg21, Lr_values, kkin1999, kkin1999, 0.25) #willott
        
    kin_interp = np.linspace(min(kin), max(kkin), 1000)
    Phikintot = np.interp(kin_interp, kin, Phikin21conv) + np.interp(kin_interp, kkin, PhikinFRII21)
    Phikintotscatter = np.interp(kin_interp, kin, Phikinscatter) + np.interp(kin_interp, kkin, PhikinFRIIscatter)
    Phikintotscatter2 = np.interp(kin_interp, kin, Phikinscatter2) + np.interp(kin_interp, kkin, PhikinFRIIscatter2)
    #print(max(kin))
    #print(max(kkin))

    if __name__ == "__main__":
        fig, ax = plt.subplots(figsize=(10, 7))
        #plt.plot(Lr_values, np.log10(Phir21), label= 'FRI')
        plt.plot(kin, np.log10(Phikin21conv), label = 'FRI')
        plt.plot(kkin, np.log10(PhikinFRII21), label = 'FRII', color = 'orange')
        plt.plot(kkin, np.log10(PhikinFRIIscatter), linestyle = '--', color = 'orange')
        plt.plot(kkin, np.log10(PhikinFRIIscatter2), linestyle = ':', color = 'orange')
        #plt.plot(kkin2017, np.log10(PhikinFRII2017), label = 'FRII Ineson+2017')
        #plt.plot(kkin2013, np.log10(PhikinFRII2013), label = 'FRII Shabala+2013')
        #plt.plot(kkin2011, np.log10(PhikinFRII2011), label = 'FRII O''Sullivan+2011')
        #plt.plot(kkin1999, np.log10(PhikinFRII1999), label = 'FRII Willott+1999')
        plt.plot(kin_interp, np.log10(Phikintot), label = 'Total Kinetic Output', color = 'black')
        plt.plot(kin_interp, np.log10(Phikintotscatter), linestyle = '--', color = 'black')
        plt.plot(kin_interp, np.log10(Phikintotscatter2), linestyle = ':', color = 'black')
        plt.legend(loc='upper right')
        ax.grid()
        ax.set_xlabel('$ \log L_{k} [erg s^{-1}] $')
        ax.set_ylabel('$\log Phi(L_{k}) [Mpc^{-3} dex^{-1}]$')
        plt.title('Kinetic Luminosity Function FRI + FRII')
        plt.rcParams['figure.dpi'] = 300
        plt.ylim(-10.5, -3)
        plt.xlim(42, 48)
        plt.show()
    return PhikinFRII, Phikin, Phikin21conv, PhikinFRII21, kin, kkin, Phir21, Phirg21, Phikinscatter, Phikinscatter2, PhikinFRIIscatter, PhikinFRIIscatter2

Rx_values = np.linspace(-10, -1, 1000)  # Rx range taken from federica for FRI
#z_values = [1, 2, 3, 4, 5]
z = 0.1  # Redshift value
Lmin = 41  # Minimum luminosity
Lrr = np.linspace(30, 47, 1000) #erg/s at 5GHz
LR = Lrr -7-np.log10(5)-9 #luminosity range in W/Hz 5 GHz based on federicas code
#LR is the transformation of Lrr into W/Hz at 5 GHz
if __name__ == "__main__":
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.dpi'] = 300
    KLF_FRI_FRII(Lrr, z, LR)
