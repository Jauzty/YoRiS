# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 17:27:15 2023

@author: aust_
"""
import numpy as np

def myBolfunc(TAG, L):
    # Bolometric Correction Function
    # input L is in ergs/s, this is the bolometric Luminosity
    
    Ls = L - np.log10(4.0) - 33.0  # to convert to solar Luminosities

    if TAG == 1:
        # 1: Marconi+04 per X 2-10 kev
        L_qq = Ls - 1.54 - (0.24 * (Ls - 12.0)) - (0.012 * (Ls - 12.0)**2) + (0.0015 * (Ls - 12.0)**3)
        L_qq = L_qq + 33.0 + np.log10(4.0)  # convert back to erg/s
        L_x = L_qq
        L_qq = 0.0
        return L_x

    elif TAG == 2:
        # 2: Marconi+04 per OPT B band
        L_qq = Ls - 0.80 + (0.067 * (Ls - 12.0)) - (0.017 * (Ls - 12.0)**2) + (0.0023 * (Ls - 12.0)**3)
        L_qq = L_qq + 33.0 + np.log10(4.0)  # convert back to erg/s
        L_B = L_qq
        L_qq = 0.0
        return L_B

def reverse_myBolfunc(TAG, L_bol):
    
    Ls = L_bol - np.log10(4.0) - 33.0  # to convert to solar Luminosities

    if TAG == 1:
        # 1: Marconi+04 per X 2-10 kev
        L_qq = Ls + 1.54 + (0.24 * (Ls - 12.0)) + (0.012 * (Ls - 12.0)**2) - (0.0015 * (Ls - 12.0)**3)
        L_qq = L_qq + 33.0 + np.log10(4.0)  # convert back to erg/s
        L_x = L_qq
        L_qq = 0.0
        return L_x

    elif TAG == 2:
        # 2: Marconi+04 per OPT B band
        L_qq = Ls + 0.80 - (0.067 * (Ls - 12.0)) + (0.017 * (Ls - 12.0)**2) - (0.0023 * (Ls - 12.0)**3)
        L_qq = L_qq + 33.0 + np.log10(4.0)  # convert back to erg/s
        L_B = L_qq
        L_qq = 0.0
        return L_B

arr = np.linspace(42.8, 46.2, 100)
rev = reverse_myBolfunc(1, arr)
new = myBolfunc(1, arr)
print(rev)
    