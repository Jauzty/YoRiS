# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:00:06 2023

@author: aust_
"""
from Ueda_Updated_Py import Ueda_14, Ueda_mod
from UedaXLF_to_RLF import fradioueda, fradioFRII
import numpy as np
from UedaXLF_to_RLF import Pradio, PFRII
from scipy.io import readsav
import matplotlib.pyplot as plt
from scipy.integrate import simps

Lrmin = 21
Lrmax = 28
Lr = np.linspace(Lrmax, Lrmin, 1000)#in W/Hz @1.4GHz, defined between 21 and 28 for calculating the duty cycle
Lx = np.linspace(42, 46.5, 1000) # Lx is the X-ray luminosity in erg/s
Lrr=Lr+(7+np.log10(1.4)+9)# Lrr is the transformation of Lr into erg/s at 1.4 Ghz
Lrrmin = Lrmin+(7+np.log10(1.4)+9)
Lrrmax = Lrmax+(7+np.log10(1.4)+9)# Lrrmin and Lrrmax are in erg/s, transformation of Lrmin and Lrmax
Rx = Lrr - Lx
z=0.5
L_r = 44

def duty(z, L_r, Lr, Lx, Lrr, Rx):
    # z is the redshift given by the user
    # K = 1.0, fraction of Compton-thick sources in the XLF by Ueda
    Lmin = 41.0 #erg/s, minimum Lx
    # Phitot is the TOTAL RLF, with all the radio luminosities Lrr in erg/s
    # L_r is the radio luminosity at which to evaluate the inverse probability P(Lx|L_r,zz)
    # PQ is the inverse probability P(Lx|L_r,zz)
    # U is the radio duty cycle
    # PhiLxall is the XLF by Ueda with K = 1.0
    # XLF by Ueda, inclusive of Compton thick

    PhiLxall = Ueda_mod(Lx, z, 0) + Ueda_mod(Lx, z, 1) + Ueda_mod(Lx, z, 2) + Ueda_mod(Lx, z, 3) + Ueda_mod(Lx, z, 4)

    # RLF TOT, starts from Lrad 21. in W/Hz @1.4GHz
    Phir_tot = fradioueda(z, Rx, Lrr, Lmin) + fradioFRII(z, Rx, Lrr, Lmin)

    n = len(Lx)

    U = np.zeros(n)
    for s in range(n):
        Lrv = Rx + Lx[s]
        ix = np.where((Lrv >= Lrrmin) & (Lrv <= Lrrmax))
        if np.isscalar(ix) and ix >= -1:
            PP = 0
        else:
            PP = Pradio(Lx[s], Lrv, z) + PFRII(Lx[s], Lrv, z)
        U[s] = simps(PP / np.interp(Lrv, Lr, np.log10(np.abs(Phir_tot)))) * PhiLxall[s]

    # for checking the P(Lx|L_r,zz) we use the TOTAL RLF,
    # with all the radio luminosities in erg/s, so we use the
    # variable Lrr and the RLF Phitot
    Rmax = np.max(Rx)
    Rmin = np.min(Rx)

    nx = len(Lx)
    QP = np.zeros(nx)

    for k in range(nx):
        Rp = Lrr - Lx[k]
        ix = np.where((Rp < Rmin) | (Rp > Rmax))
        P = Pradio(Lx[k], Lrr, z) + PFRII(Lx[k], Lrr, z)
        if np.isscalar(ix) and ix >= -1:
            PP = 0
        else:
            PP = np.interp(L_r, Lrr, P / Phir_tot)
        QP[k] = PP * PhiLxall[k]

    return QP, U

