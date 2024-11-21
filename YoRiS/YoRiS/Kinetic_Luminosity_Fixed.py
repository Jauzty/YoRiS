# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:57:48 2023

@author: aust_
"""

import numpy as np
import matplotlib.pyplot as plt
from Ueda_Updated_Py import Ueda_14
from scipy.integrate import simps
from UedaXLF_to_RLF import Pradio, PFRII
from scipy.signal import convolve as conv

def convolve(f, y, xmed, x, sigma):
    nx = len(x)
    res = np.zeros(nx, dtype=float)

    for ix in range(nx):
        k = np.exp(-((x[ix] - xmed)**2) / (2.0 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
        res[ix] = simps(f * k, y)

    return res

zz=1
L_x=np.linspace(41, 49, 1000)
n=1000
L_rmin = 30 #log units W/Hz
L_rmax = 47 #log units W/Hz
Lrr = np.linspace(L_rmin,L_rmax,n)#define the luminosities in W/Hz     
l_wat=Lrr-7-np.log10(5)-9# calculate the luminosities in erg/s

Rx = Lrr - L_x # calculate Rx rather than just define it

nr=Lrr.size
Phirgaus=np.zeros(nr)
Phir = np.zeros(nr)
temp=np.zeros(nr)

Lmin=41

for s in range(0,nr):
    temp=Lrr[s]-Rx
    Lx=temp[::-1] #reverse, to put in crescent order
#    print(temp[-4:])
#    print(Lx)
#    print(Lx[np.where(Lx < Lmin)])

    #now calculate the XLF given by ueda14.py, considering the whole X-ray AGN population
    f=np.zeros(nr)
    for i in range(0,5): 
        f=f+Ueda_14(Lx,zz,i)
        f[(Lx < Lmin)]=0
        #print (i,f)   
    
    PP = PFRII( Lx, Lrr[s], zz )
    P = Pradio(Lx, Lrr[s], zz)
    Phirgaus[s]=simps(PP*f,Lx)
    Phir[s]=simps(P*f, Lx)

alpha = 0.54
beta = 22.1
ar = 0.7
fcav = 4.0
bb = np.log10(7 * fcav) + 36.0
aa = 0.68
# Convert radio luminosity P from W/Hz @ 1.4GHz to W/Hz @ 5GHz
L5 = l_wat - ar * np.log10(5 / 1.4)  # L5 is in W/Hz @ 5GHz
# Convert L5 to erg/s @ 5GHz
L5r = L5 + 7 + 9 + np.log10(5)  # L5r is in erg/s @ 5GHz
# Calculate kinetic luminosity kin
kin = alpha * (L5r) + beta   
# Calculate kinetic luminosity kkin
kkin = aa * (l_wat - 25) + bb + 7.0  # kkin is in erg/s

Phikin = convolve(Phir, l_wat, kin, kin, 0.01)
PhikinFRII = convolve(Phirgaus, l_wat, kkin, kkin, 0.01)

kin_interp = np.linspace(min(kin), max(kkin), n)
Phikintot = np.interp(kin_interp, kin, Phikin) + np.interp(kin_interp, kkin, PhikinFRII)

if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.plot(kin, np.log10(Phikin), label= 'FRI')
    plt.plot(kkin, np.log10(PhikinFRII), label = 'FRII')
    plt.plot(kin_interp, np.log10(Phikintot), label = 'Total Kinetic Output', color = 'black')    
    plt.legend(loc='upper right')
    ax.grid()
    ax.set_xlabel('$ \log L_{k} [erg s^{-1}] $')
    ax.set_ylabel('$\logPhi(L_{k}) [Mpc^{-3} dex^{-1}]$')
    plt.title('Kinetic Luminosity Function FRI + FRII')
    plt.rcParams['figure.dpi'] = 300
    plt.ylim(-12, -1)
    plt.xlim(42, 48)
    plt.show()  
    
    
    
    
    
    
    
    
    
    