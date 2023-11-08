# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 09:54:36 2023

@author: aust_
"""

import numpy as np
import matplotlib.pyplot as plt
from Ueda_Updated_Py import Ueda_14
from scipy.integrate import simps

def Pradio(Lx, L_r, z):
    #Pradio calculates the probability of having a value Lr given a value Lx
    Norm = 1.0230
    g_r = 0.369
    g_l = 1.69
    Rc = -4.578
    al = 0.109
    az = -0.066
    A = np.sqrt(g_r / g_l)
    # Rx is the discrepancy between radio luminosity and X-ray luminosity for each source
    Rx = L_r - Lx  # Calculate Rx for each L_r

    z0 = max(0.3, min(3.0, z))
    Lx0 = np.clip(Lx, 42.2, 47.0)
    R0 = Rc * (al * (Lx0 - 44) + 1) * (az * (z0 - 0.5) + 1)  # Calculate R0

    DENN1 = 1 + ((R0 - Rx) / g_l) ** 4
    DENN2 = 1 + ((Rx - R0) / g_r) ** 2

    if np.isscalar(Rx):
        if Rx < R0:
            PP = Norm / (A * np.pi * g_l * DENN1)
        else:
            PP = (A * Norm) / (np.pi * g_r * DENN2)
    else:
        PP = np.zeros_like(Rx)
        ix = Rx < R0
        PP[ix] = Norm / (A * np.pi * g_l * DENN1[ix])
        PP[~ix] = (A * Norm) / (np.pi * g_r * DENN2[~ix])

    # where Rx >= 1 PP = 0 because the condition Rx>-1 produces an RLF that over-predicts the number of bright radio sources
    if np.isscalar(Rx) and Rx >= -1:
        PP = 0

    return PP

def PFRII(Lx, L_r, z):
    #Pradio calculates the probability of having a value Lr given a value Lx but for the bright FRII LF from Willot 2001
    # Initialize variables
    norm = 0.0
    LL = 0.0
    
    # Define norm and LL based on zz value
    if z <= 0.25:
        norm = -2.62
        LL = 42.27
    elif 0.25 < z <= 0.5:
        norm = -2.89
        LL = 42.39
    elif 0.5 < z <= 1.0:
        norm = -2.878
        LL = 42.46
    elif 1.0 < z <= 2.0:
        norm = -2.74
        LL = 42.53
    elif 2.0 < z <= 3.0:
        norm = -2.59
        LL = 42.57
    elif 3.0 < z <= 3.5:
        norm = -2.518
        LL = 42.58
    elif 3.5 < z <= 4.0:
        norm = -2.518
        LL = 42.58
    elif 4.0 < z <= 4.5:
        norm = -2.518
        LL = 42.58
    elif 4.5 < z <= 6.5:
        norm = -2.518
        LL = 42.58

    # Define parameters for z-evolution
    rho0h = 0.4
    alphah = 2.27
    zh0 = 1.91
    zh1 = 0.559
    zh2 = 1.378

    # Calculate fzh based on zz and zh0
    fzh = np.exp(-0.5 * ((z - zh0) / zh1) ** 2)
    if z >= zh0:
        fzh = np.exp(-0.5 * ((z - zh0) / zh2) ** 2)

    # Calculate Rx and Rxmed
    Rx = L_r - Lx
    Rxmed = 1.0 - (Lx - LL)

    # Calculate PP
    PP = rho0h * 10**(-alphah * (Rx - Rxmed)) * np.exp(-10 ** (Rxmed - Rx)) * fzh * 10**(norm)
    
    return PP

def fradioueda(z, Rx_values, Lr_values, Lmin):
    nr = len(Lr_values)
    Phir = np.zeros(nr)
    Lx = np.zeros(nr)
    temp = np.zeros(nr)

    for s in range(nr):
        temp = Lr_values[s] - Rx_values
        Lx = temp[::-1]
        Phi_x = np.zeros(nr)

        for i in range(0, 5): #This part is very important, we are summung over all the Nh to calculate the total luminosity function in the Radio
            Phi_x = Phi_x + Ueda_14(Lx, z, i)
            Phi_x[(Lx < Lmin)] = 0

        P = Pradio(Lx, Lr_values[s], z)
        Phir[s] = simps(P * Phi_x, Lx)

    #fig, ax = plt.subplots(figsize=(10, 7))
    
    #plt.semilogy(Lr_values, Phir, label=f'z={z}')
    #plt.legend(loc='upper right')
    #ax.grid()
    #ax.set_xlabel('$ \log L_{r} [erg s^{-1}] $')
    #ax.set_ylabel('$\Phi(L_{r}) [Mpc^{-3} dex^{-1}]$')
    #plt.title('Radio Luminosity Function FRI')
    #plt.rcParams['figure.dpi'] = 300
    #plt.show()
    return Phir, P

def fradioueda21(z, Rx_values, Lr_values, Lmin):
    nr = len(Lr_values)
    Phirgaus21 = np.zeros(nr)
    Phir21 = np.zeros(nr)
    Lx = np.zeros(nr)
    temp = np.zeros(nr)

    for s in range(nr):
        temp = Lr_values[s] - Rx_values
        Lx = temp[::-1]
        Phi_x = np.zeros(nr)

        Phi_x = Ueda_14(Lx, z, 0)
        Phi_x[(Lx < Lmin)] = 0

        P = Pradio(Lx, Lr_values[s], z)
        PP = PFRII(Lx, Lr_values[s], z)
        Phir21[s] = simps(P * Phi_x, Lx)
        Phirgaus21[s] = simps(PP*Phi_x, Lx)
    return Phir21, Phirgaus21, P, PP
        
def fradioFRII( z, Rx_values, Lr_values, Lmin):
    nr = len(Lr_values)
    Phirgaus = np.zeros(nr)
    Phir = np.zeros(nr)
    Lx = np.zeros(nr)
    temp = np.zeros(nr)

    for s in range(nr):
        temp = Lr_values[s] - Rx_values
        Lx = temp[::-1]
        Phi_x = np.zeros(nr)

        for i in range(0, 5): #This part is very important, we are summung over all the Nh to calculate the total luminosity function in the Radio
            Phi_x = Phi_x + Ueda_14(Lx, z, i)
            Phi_x[(Lx < Lmin)] = 0

        PP = PFRII(Lx, Lr_values[s], z)
        P = Pradio(Lx, Lr_values[s], z)
        Phir[s] = simps(P*Phi_x, Lx)
        Phirgaus[s] = simps(PP*Phi_x, x=Lx)

    #fig, ax = plt.subplots(figsize=(10, 7))
    
    #plt.semilogy(Lr_values, Phirgaus, label='FRII')
    #plt.semilogy(Lr_values, Phir, label = 'FRI')
    #plt.legend(loc='upper right')
    #ax.grid()
    #ax.set_xlabel('$ \log L_{r} [erg s^{-1}] $')
    #ax.set_ylabel('$\Phi(L_{r}) [Mpc^{-3} dex^{-1}]$')
    #plt.title('Radio Luminosity Function')
    #plt.rcParams['figure.dpi'] = 300
    #plt.ylim(1e-12, 1e-3)
    #plt.show()
    return Phirgaus, PP
    
def fradioueda_mult_z( z_values, Rx_values, Lr_values, Lmin):
    fig, ax = plt.subplots(figsize=(10, 7))

    for z in z_values:
        nr = len(Lr_values)
        Phir = np.zeros(nr)
        Lx = np.zeros(nr)
        temp = np.zeros(nr)

        for s in range(nr):
            temp = Lr_values[s] - Rx_values
            Lx = temp[::-1]
            Phi_x = np.zeros(nr)

            for i in range(0, 5):
                Phi_x = Phi_x + Ueda_14(Lx, z, i)
                Phi_x[(Lx < Lmin)] = 0

            P = Pradio(Lx, Lr_values[s], z)
            Phir[s] = simps(P * Phi_x, Lx)

        #plt.semilogy(Lr_values, Phir, label=f'z={z}')

    #plt.legend(loc='upper right', fontsize = 14)
    #ax.grid()
    #ax.set_xlabel('$ \log L_{r} [erg s^{-1}] $')
    #ax.set_ylabel('$\Phi(L_{r}) [Mpc^{-3} dex^{-1}]$')
    #plt.ylim(1e-9, 1e-3)
    #plt.title('Radio Luminosity Function')
    #plt.rcParams['figure.dpi'] = 300
    #plt.show()

if __name__ == "__main__":
    # Define the grid of Lr, Rx, and redshift values
    lrmin = 30 #erg/s
    lrmax = 47
    z_values = [0.75, 1, 2, 3.5]
    Lr_values = np.linspace(lrmin, lrmax, 1000)
    L_x = 44  # Luminosity range
    z = 1  # Redshift value
    Lmin = 41  # Minimum luminosity
    Rx_values = Lr_values - L_x
    
    # Call the function to calculate and plot the radio luminosity function
    #fradioueda(z, Rx_values, Lr_values, Lmin)
    #fradioueda_mult_z(z_values, Rx_values, Lr_values, Lmin)
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.weight'] = 'normal'
    


