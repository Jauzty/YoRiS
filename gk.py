# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:38:08 2023

@author: aust_
"""
import numpy as np
from UedaXLF_to_RLF import fradioueda21, fradioFRII, fradioueda
from dutyforspecz import dutyforspecz, bin_edges
from QLFs_to_duty_cycles import Phixbol24, Phixbol20, Lboll, Lunif, l_2500, PhiBbol, Lboloptdata, sigmaBbold, sigmaBbolu, LBdata
from Lr_Lkin_convolution import KLF_FRI_FRII, LR, Rx_values, z, Lmin
from scipy.integrate import simps

lrmin = 30 #erg/s
lrmax = 47
Lr_values = np.linspace(lrmin, lrmax, 1000)
Phir = fradioueda(z, Rx_values, Lr_values, Lmin)
PhiradioII = dutyforspecz(Lunif, l_2500, PhiBbol, Lboloptdata, sigmaBbold, sigmaBbolu, LBdata, bin_edges, z)
Phir21, Phirgaus21, P = fradioueda21(z, Rx_values, Lr_values, Lmin)
Phikin, PhikinFRII, Phikin21conv, kin, kkin = KLF_FRI_FRII(LR, Rx_values, z)
Phirgaus = fradioFRII(z, Rx_values, Lr_values, Lmin)
jac = np.abs(np.gradient(P, kin))
Jac = np.abs(np.gradient(P, kkin))
Phikin21 = Phir21*jac #obtain Phikin for two different absorbtions without convolution, instead use jacobian 
PhikinI = Phir*jac
PhikinFRII21 = Phirgaus21*Jac#similar for PhikinFRII for two absorbtions with different jacobian
PhikFRII = Phirgaus*Jac


def gkFRI(z):
    
    red=[0.5, 0.9, 1.25, 1.6, 2.0, 2.4, 2.8, 3.25, 3.75, 4.25, 4.75]
    gkin22I=[0.3, 0.25, 0.13, 0.13, 0.08, 0.07, 0.06, 0.06,0.05,0.08,0.08]
    gkin22I_03=[0.35, 0.3, 0.25, 0.15, 0.1, 0.1, 0.07, 0.09,0.08,0.08,0.05]
    gkin22I_conv=[0.55, 0.45, 0.35, 0.25, 0.25, 0.22, 0.18, 0.2,0.18,0.18,0.18]
    gkinI=[0.2, 0.15, 0.1, 0.1, 0.06, 0.05, 0.05, 0.05,0.04,0.08,0.07]
    gkinI_03=[0.3, 0.2, 0.2, 0.13, 0.08, 0.08, 0.05, 0.08,0.05,0.08,0.05]
    gkinI_conv=[0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.13, 0.15,0.15,0.15,0.15]
    
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
    
    nk = len(kin)
    LgLbol = np.zeros(nk)
    LgLbol_03 = np.zeros(nk)
    LgLbol_conv = np.zeros(nk)
    LgLbol22 = np.zeros(nk)
    LgLbol22_03 = np.zeros(nk)
    LgLbol22_conv = np.zeros(nk)
    
    # gkin for FRI, comparison in the bolometric plane
    for io in range(nk):
        LgLbol[io] = -np.log10(gk) + kin[io]
        LgLbol_03[io] = -np.log10(gk_03) + kin[io]
        LgLbol_conv[io] = -np.log10(gk_conv) + kin[io]
        LgLbol22[io] = -np.log10(gk22) + kin[io]
        LgLbol22_03[io] = -np.log10(gk22_03) + kin[io]
        LgLbol22_conv[io] = -np.log10(gk22_conv) + kin[io]
    
    
    # Omega_rad
    fradI = Phixbol24 * 10**Lboll
    OradI = simps(fradI, Lboll)
    
    frad21 = Phixbol20 * 10**Lboll
    Orad21 = simps(frad21, Lboll)
    
    # Omega_kin
    fkinI = Phikin * 10**kin #PhikinI is without convolution and result is very different
    OkinI = simps(fkinI, kin)
    
    fkin21 = Phikin21conv * 10**kin #Phikin21 is without convolution and result very different
    Okin21 = simps(fkin21, kin)
    
    gk1 = OkinI / OradI

    gk21_FRI = Okin21 / Orad21
    print(f'OradI is {OradI}')
    print(f'OkinI is {OkinI}')
    print(f"gk1 is {gk1}")
    print(f'gk21_FRI is {gk21_FRI}')
    
def gkFRII(z):
    
    redII=[0.5, 0.9, 1.25, 1.6, 2.0, 2.4, 2.8, 3.25]
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
    pasL = 0.05
    cumW = np.cumsum(np.flip(pasL * PhikinFRII21))
    cumSDSS = np.cumsum(np.flip(pasL * PhiradioII))
    
    frad = Phixbol24 * 10**Lboll
    Orad = simps(frad, Lboll)
    
    fkin_2 = PhikinFRII21 * 10**kkin
    Okin_2 = simps(fkin_2, kkin)
    
    fkin = PhikFRII * 10**kkin  # Jacobian of rad to kin is 1.0
    Okin = simps(fkin, kkin)
    # Optionally filter data based on Lboloptdata
    yr = np.where(np.array(Lboloptdata) > 45.5)
    Ldata = Lboloptdata[yr]
    Phiopt = PhiradioII[yr]
    
    # Calculate radiative power energy density (Omega_rad) for filtered data
    fradII = Phiopt * 10**Ldata
    OradII = simps(fradII, Ldata)
    
    gkFRII = Okin/Orad #close to right but not
    gk21_FRII = Okin_2/OradII #couldnt be more wrong not sure why
    # Print the results
    print("Okin:", Okin)
    print("Orad:", Orad)
    print(f'gkFRII is {gkFRII}')
    print(f'gk21_FRII is {gk21_FRII}')
    

if __name__ == "__main__":    
    gkFRI(z)
    gkFRII(z)