# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:39:05 2023

@author: aust_
"""

import numpy as np
from scipy.optimize import minimize
from scipy.io import readsav
from Lr_Lkin_convolution import KLF_FRI_FRII, LR, Rx_values, z, Lmin, Lrr
from QLFs_to_duty_cycles import Lboll
PhikinFRII, Phikin, Phikin21conv, kin, kkin = KLF_FRI_FRII(Lrr, Rx_values, z, LR)

#data = readsav(r'C:\Users\aust_\YoRiS\shen_catalogue.sav')
#Lbol = data['lbol'] 
#Lbol = Lbol[Lbol != 0]


'''
Lbol in bin 1 is: 44.11554511800629, 47.26809213074085 #z=0.5
Lbol in bin 2 is: 44.04744705747671, 47.48280156362307 #z=0.9 
Lbol in bin 3 is: 44.118014041941954, 47.762020570418144 #z=1.2
Lbol in bin 4 is: 44.13896030322051, 47.92086746192046 #z=1.6
Lbol in bin 5 is: 44.370556221287025, 48.17485160464779 #z=2
Lbol in bin 6 is: 44.72351073262212, 48.190012140986475 #z=2.4
Lbol in bin 7 is: 44.769447734708756, 48.25953514420532 #z=2.8
Lbol in bin 8 is: 45.3457755301005, 48.29106814755614 #z=3.2
Lbol in bin 9 is: 46.06562853262776, 48.05299973758891 #z=3.8
Lbol in bin 10 is: 46.10988961170089, 47.98624680120893 #z=4.2
Lbol in bin 11 is: 46.28894259941676, 48.00268647786993 #z=4.8'''


def gkFRI(z, kin):
    red = [0.5, 0.9, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.8, 4.2, 4.8]
    gkin22I = [0.3, 0.25, 0.13, 0.13, 0.08, 0.07, 0.06, 0.06, 0.05, 0.08, 0.08]
    gkin22I_03 = [0.35, 0.3, 0.25, 0.15, 0.1, 0.1, 0.07, 0.09, 0.08, 0.08, 0.05]
    gkin22I_conv = [0.55, 0.45, 0.35, 0.25, 0.25, 0.22, 0.18, 0.2, 0.18, 0.18, 0.18]
    gkinI = [0.2, 0.15, 0.1, 0.1, 0.06, 0.05, 0.05, 0.05, 0.04, 0.08, 0.07]
    gkinI_03 = [0.3, 0.2, 0.2, 0.13, 0.08, 0.08, 0.05, 0.08, 0.05, 0.08, 0.05]
    gkinI_conv = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.13, 0.15, 0.15, 0.15, 0.15]
    nk = len(kin)
    LgLbol = np.zeros(nk)
    LgLbol_03 = np.zeros(nk)
    LgLbol_conv = np.zeros(nk)
    LgLbol22 = np.zeros(nk)
    LgLbol22_03 = np.zeros(nk)
    LgLbol22_conv = np.zeros(nk)
    
    if z == 0.5:
        Lbol = np.linspace(44.11554511800629, 47.26809213074085, 1000)
    elif z == 0.9:
        Lbol = np.linspace(44.04744705747671, 47.48280156362307, 1000)
    elif z == 1.25:
        Lbol = np.linspace(44.118014041941954, 47.762020570418144, 1000)
    elif z == 1.6:
        Lbol = np.linspace(44.13896030322051, 47.92086746192046, 1000)
    elif z == 2.0:
        Lbol = np.linspace(44.370556221287025, 48.17485160464779, 1000)
    elif z == 2.4:
        Lbol = np.linspace(44.72351073262212, 48.190012140986475, 1000)
    elif z == 2.8:
        Lbol = np.linspace(44.769447734708756, 48.25953514420532, 1000)
    elif z == 3.25:
        Lbol = np.linspace(45.3457755301005, 48.29106814755614, 1000)
    elif z == 3.75:
        Lbol = np.linspace(46.06562853262776, 48.05299973758891, 1000)
    elif z == 4.25:
        Lbol = np.linspace(46.10988961170089, 47.98624680120893, 1000)
    elif z == 4.75:
        Lbol = np.linspace(46.28894259941676, 48.00268647786993, 1000)
            
    def chi_squared(params, kin, LgLbol_target, LgLbol_03_target, LgLbol_conv_target, LgLbol22_target, LgLbol22_03_target, LgLbol22_conv_target):
        gk, gk_03, gk_conv, gk22, gk22_03, gk22_conv = params

        LgLbol = -np.log10(gk) + kin
        LgLbol_03 = -np.log10(gk_03) + kin
        LgLbol_conv = -np.log10(gk_conv) + kin
        LgLbol22 = -np.log10(gk22) + kin
        LgLbol22_03 = -np.log10(gk22_03) + kin
        LgLbol22_conv = -np.log10(gk22_conv) + kin

        residuals = np.concatenate([LgLbol - LgLbol_target,
                                    LgLbol_03 - LgLbol_03_target,
                                    LgLbol_conv - LgLbol_conv_target,
                                    LgLbol22 - LgLbol22_target,
                                    LgLbol22_03 - LgLbol22_03_target,
                                    LgLbol22_conv - LgLbol22_conv_target])

        chi_squared_value = np.sum(residuals**2)
        return chi_squared_value

    # Normalize the data
    kin_normalized = kin / 1e40  # Adjust the denominator as needed

    tt = np.where(np.array(red) == z)
    if len(tt[0]) > 0:
        index = tt[0][0]  # Get the first matching index
        initial_guess = [gkinI[index], gkinI_03[index], gkinI_conv[index],
                         gkin22I[index], gkin22I_03[index], gkin22I_conv[index]]


        result = minimize(chi_squared, initial_guess, args=(
            kin_normalized, Lbol, LgLbol_03, LgLbol_conv, LgLbol22, LgLbol22_03, LgLbol22_conv), method='Nelder-Mead', bounds=None)

        best_params = result.x
        print("Best Parameters:", best_params)

    for io in range(nk):
        LgLbol[io] = -np.log10(best_params[0]) + kin[io]
        LgLbol_03[io] = -np.log10(best_params[1]) + kin[io]
        LgLbol_conv[io] = -np.log10(best_params[2]) + kin[io]
        LgLbol22[io] = -np.log10(best_params[3]) + kin[io]
        LgLbol22_03[io] = -np.log10(best_params[4]) + kin[io]
        LgLbol22_conv[io] = -np.log10(best_params[5]) + kin[io]
    print(Lbol/LgLbol) 
    #lkin/lbol = gk, we dont know gk but if we assume we know it and get it right then lkin/lbol should == 1
gkFRI(z, kin)
