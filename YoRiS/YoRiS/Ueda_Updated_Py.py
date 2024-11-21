# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:09:09 2023

@author: aust_
"""

import numpy as np
import matplotlib.pyplot as plt

def ueda14(lx, z, Nh):
  
    #Constants
    A = 2.91e-6
    L_s = 43.97
    L_p = 44.0
    g_1 = 0.96
    g_2 = 2.71  #2.71 best-fit, lower lim=2.6
    p_1s = 4.78
    p_2 = -1.5
    p_3 = -6.2
    b_1 = 0.84
    z_sc1 = 1.86
    L_a1 = 44.61
    alpha_1 = 0.29
    z_sc2 = 3.0
    L_a2 = 44.
    alpha_2 = -0.1
    
    nl = len(lx)
    z_c1 = np.zeros_like(lx)
    z_c2 = np.zeros_like(lx)
    p_1 = np.zeros_like(lx)
    e = np.zeros_like(lx)  

    
    for k in range(0,nl):
        if lx[k] <= L_a1: 
            z_c1[k] = z_sc1*(10.**(lx[k]-L_a1))**alpha_1
        if lx[k] > L_a1: 
            z_c1[k] = z_sc1
        if lx[k] <= L_a2:
            z_c2[k] = z_sc2*(10.**(lx[k]-L_a2))**alpha_2
        if lx[k] > L_a2:
            z_c2[k] = z_sc2
    
        p_1[k] = p_1s + b_1*(lx[k] - L_p)
    
        if z <= z_c1[k]:
            e[k] = (1.+z)**p_1[k]
        if z > z_c1[k] and z <= z_c2[k]:
            e[k] = (1.+z_c1[k])**p_1[k] * ((1.+z)/(1.+z_c1[k]))**p_2
        if z > z_c2[k]:
            e[k] = (1.+z_c1[k])**p_1[k] * ((1.+z_c2[k])/(1.+z_c1[k]))**p_2 * ((1.+z)/(1.+z_c2[k]))**p_3
    
    Den1 = (10.**(lx-L_s))**g_1
    Den2 = (10.**(lx-L_s))**g_2
    Den = Den1 + Den2
    
    Phi = (A/Den) * e
    
    Psi=np.zeros_like(lx) 
    bet=0.24
    a1=0.48
    fCTK=1.0 #fraction of Compton tick sources
    Psi0=0.43
    if z < 2.:
        Psi44z=Psi0*(1.+z)**a1 
    else: 
        Psi44z=Psi0*(1.+2.)**a1 
    
    eta=1.7
    Psimax=(1.+eta)/(3.+eta)
    Psimin=0.2
    Psi=np.zeros_like(lx)
    for i in range(0,nl):
        em=Psi44z-bet*(lx[i]-43.75)
        f1=np.array([em,Psimin])
        ff1=f1.max()
        f2=np.array([Psimax,ff1])
        f22=f2.min()
        Psi[i]=f22
    
    lim=(1.+eta)/(3.+eta)

    frac=np.zeros(len(lx))   
    for k in range(0,nl):
        if Psi[k] < lim:
            if Nh == 0:#20.<LogNh<21
                frac[k]=1.- Psi[k] * ((2.+eta)/(1.+eta))
            if Nh == 1:#21.<LogNh<22.
                frac[k]=(1./(1.+eta))*Psi[k]        
            if Nh == 2:#22.<LogNh<23.
                frac[k]=(1./(1.+eta))*Psi[k]
            if Nh == 3:#23.<LogNh<24. 
                frac[k]=(eta/(1.+eta))*Psi[k] 
            if Nh == 4: #24.<LogNh<26
                frac[k]=(fCTK/2.)*Psi[k]
        
        else:
            if Nh == 0:#20.<LogNh<21
                frac[k]=(2./3.)- Psi[k] * ((3.+2.*eta)/(3.+3.*eta)) 
            if Nh == 1:#21.<LogNh<22.
                frac[k]=(1./3.) - Psi[k] * (eta/(3.+3.*eta))
            if Nh == 2:#22.<LogNh<23.
                frac[k]=(1./(1.+eta))*Psi[k]
            if Nh == 3:#23.<LogNh<24.
                frac[k]=(eta/(1.+eta))*Psi[k]
            if Nh == 4:#24.<LogNh<26
                frac[k]=(fCTK/2.)*Psi[k]
    
    
    Phi=frac*Phi
    return Phi

def Ueda_14( L, z, NH):
    A = 2.91e-6
    L_s = 43.97
    L_p = 44.0
    g_1 = 0.96
    g_2 = 2.71
    p_1s = 4.78
    p_2 = -1.5
    p_3 = -6.2
    b_1 = 0.84
    z_sc1 = 1.86
    L_a1 = 44.61
    alpha_1 = 0.29
    z_sc2 = 3.0
    L_a2 = 44.0
    alpha_2 = -0.1

    if isinstance(L, (int, float)):
        L = np.array([L])

    nl = len(L)
    z_c1 = np.zeros(nl)
    z_c2 = np.zeros(nl)
    p_1 = np.zeros(nl)
    e = np.zeros(nl)

    for k in range(nl):
        if L[k] <= L_a1:
            z_c1[k] = z_sc1 * (10**(L[k] - L_a1))**alpha_1
        else:
            z_c1[k] = z_sc1

        if L[k] <= L_a2:
            z_c2[k] = z_sc2 * (10**(L[k] - L_a2))**alpha_2
        else:
            z_c2[k] = z_sc2

        p_1[k] = p_1s + b_1 * (L[k] - L_p)

        if z <= z_c1[k]:
            e[k] = (1 + z)**p_1[k]
        elif z_c1[k] < z <= z_c2[k]:
            e[k] = (1 + z_c1[k])**p_1[k] * ((1 + z) / (1 + z_c1[k]))**p_2
        else:
            e[k] = (1 + z_c1[k])**p_1[k] * ((1 + z_c2[k]) / (1 + z_c1[k]))**p_2 * ((1 + z) / (1 + z_c2[k]))**p_3

    Den1 = (10**(L - L_s))**g_1
    Den2 = (10**(L - L_s))**g_2
    Den = Den1 + Den2

    Phi = (A / Den) * e

    Psi = np.zeros(nl)
    bet = 0.24
    a1 = 0.48
    fCTK = 1.0
    Psi0 = 0.43
    Psi44z = Psi0 * (1 + z)**a1 if z < 2 else Psi0 * (1 + 2)**a1

    eta = 1.7
    Psimax = (1 + eta) / (3 + eta)
    Psimin = 0.2

    for i in range(nl):
        em = Psi44z - bet * (L[i] - 43.75)
        f1 = max(em, Psimin)
        f2 = min(Psimax, f1)
        Psi[i] = f2

    if isinstance(NH, int):
        if NH == 0:
            frac = np.where(Psi < Psimax, 1 - (2 + eta) / (1 + eta) * Psi,
                            2 / 3 - (3 + 2 * eta) / (3 + 3 * eta) * Psi)
        elif NH == 1:
            frac = np.where(Psi < Psimax, (1 / (1 + eta)) * Psi,
                            1 / 3 - eta / (3 + 3 * eta) * Psi)
        elif NH == 2:
            frac = (1 / (1 + eta)) * Psi
        elif NH == 3:
            frac = (eta / (1 + eta)) * Psi
        elif NH == 4:
            frac = (fCTK / 2) * Psi

        Phi = frac * Phi
    return Phi

def Ueda_mod(L, z, NH):
    A = 2.91e-6  # Normalisation factor for Phi so that it matches data and has correct units
    L_s = 43.97  # Luminosity scaling constants
    L_p = 44.0   
    g_1 = 0.96   # Luminosity function parameters
    g_2 = 2.87   
    p_1s = 4.78  
    p_2 = -1.5   
    p_3 = -6.2   
    b_1 = 0.84   
    z_sc1 = 1.86  # Redshift scaling constant 
    L_a1 = 44.61  # Luminosity scaling constant 
    alpha_1 = 0.29  # Luminosity scaling exponent
    z_sc2 = 3.0   # Redshift scaling constant
    L_a2 = 44.0   # Luminosity scaling constant
    alpha_2 = -0.1  # Luminosity scaling exponent
    bet = 0.24    # Scaling factor for Psi, used in determining the value of Psi based on luminosity and redshift.
    a1 = 0.48     # Exponent used for the value of Psi44z.
    fCTK = 1.0    # Used when calculating the value of Psi for specific cases.
    Psi0 = 0.43   # Used in calculating the value of Psi44z, which is a parameter related to Psi.
    eta = 1.7     # Affects the scaling of Psi, min and max values
    Psimax = (1 + eta) / (3 + eta)  # Maximum value of Psi, used to limit Psi in certain conditions.
    Psimin = 0.2  # Minimum value of Psi, used to limit Psi in certain conditions
    
    # Calculate z_c1, z_c2, p_1, e
    # np.where replaces elements where the condition is met, similar to Federicas original code
    z_c1 = np.where(L <= L_a1, z_sc1 * (10 ** (L - L_a1)) ** alpha_1, z_sc1)
    z_c2 = np.where(L <= L_a2, z_sc2 * (10 ** (L - L_a2)) ** alpha_2, z_sc2)
    p_1 = p_1s + b_1 * (L - L_p)
    
    mask_z_c1 = (z <= z_c1)
    mask_z_c2 = (z_c1 < z) & (z <= z_c2)
    #mask_z_c3 = (z > z_c2) not used, not sure why defined
    #e represents the different power law components that differ with z, and Lx
    e = np.where(mask_z_c1, (1 + z) ** p_1, 
                 np.where(mask_z_c2, (1 + z_c1) ** p_1 * ((1 + z) / (1 + z_c1)) ** p_2,
                          (1 + z_c1) ** p_1 * ((1 + z_c2) / (1 + z_c1)) ** p_2 * ((1 + z) / (1 + z_c2)) ** p_3))
    
    #Calculate Den1, Den2, Den ie. relevant luminosity densities
    Den1 = (10 ** (L - L_s)) ** g_1
    Den2 = (10 ** (L - L_s)) ** g_2
    Den = Den1 + Den2

    Phi = (A / Den) * e #Phi is density of AGN as function of NH and z 
    
    #Calculate Psi44z by splitting psi into two redshift regimes, less than or more than z = 2
    Psi44z = Psi0 * ((1 + z) ** a1 if z < 2.0 else (1 + 2) ** a1) 
    
    #Calculate Psi
    em = Psi44z - bet * (L - 43.75) # scaled and shifted psi for a specific redshift
    Psi = np.minimum(np.maximum(em, Psimin), Psimax) #Psi represents the X-ray Luminosity function
    
    #Calculate frac similar but using python logic
    #frac is necessary to differentiate between different regimes of AGN using different modelling methods
    if isinstance(NH, int):
        if NH == 0: #20<LogNH<21
            frac = np.where(Psi < Psimax, 1 - (2 + eta) / (1 + eta) * Psi,
                            2 / 3 - (3 + 2 * eta) / (3 + 3 * eta) * Psi)
        elif NH == 1: #21<LogNH<22
            frac = np.where(Psi < Psimax, (1 / (1 + eta)) * Psi,
                            1 / 3 - eta / (3 + 3 * eta) * Psi)
        elif NH == 2: #22<LogNH<23
            frac = (1 / (1 + eta)) * Psi
        elif NH == 3: #23<LogNH<24
            frac = (eta / (1 + eta)) * Psi 
        elif NH == 4: #24<LogNH<26
            frac = (fCTK / 2) * Psi
        #I have removed the else block as all the conditions are wrapped into the np.where
    
        Phi = frac * Phi
    return Phi

h = 0.7 #70km/s/Mpc
#Define the grid of Lx, NH, and redshift (z) values
Lx_values = np.linspace(42.0, 48.0, 100)  # based on Ueda ranges
NH_values = [0, 1, 2, 3, 4]  #Based on federicas code corresponds to real Nh 20-24
z_values = [0.2, 0.6, 1.0, 1.4, 2.0, 2.4, 3.0, 4.0] # based on Ueda ranges

#Calculate LFs for each combination of Lx, NH, and z
LFs_Lx = np.zeros((len(Lx_values), len(NH_values), len(z_values)))
if __name__ == '__main__':
    for j, NH in enumerate(NH_values):
        for i, Lx in enumerate(Lx_values):
            for k, z in enumerate(z_values):
                Phi_result = Ueda_mod(Lx, z, NH)
                LFs_Lx[i, j, k] = Phi_result
    
    # Create separate figures for each NH range
    for j, NH in enumerate(NH_values):
        fig, ax = plt.subplots(figsize=(10, 8))
    
        for k, z in enumerate(z_values):
            ax.plot(Lx_values, LFs_Lx[:, j, k], label=f'z={z}')
        
        if NH == 0:
            ax.set_title('20<LogNH<21', fontsize = 14)
        elif NH == 1:
            ax.set_title('21<LogNH<22', fontsize = 14)
        elif NH == 2:
            ax.set_title('22<LogNH<23', fontsize = 14)
        elif NH == 3:
            ax.set_title('23<LogNH<24', fontsize = 14)
        elif NH == 4:
            ax.set_title('XLF at 21<LogNH<26')
            
        ax.set_yscale('log')
        ax.grid()
        ax.set_xlabel('log Lx [erg s^-1]')
        ax.set_ylabel('dphi/dlogLx (XLF) [Mpc^-3]')
        ax.tick_params(axis='y')
        ax.legend(title='Redshift (z)', fontsize = 14)
    
    plt.tight_layout()
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.dpi'] = 300
    plt.show()