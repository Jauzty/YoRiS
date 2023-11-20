# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:57:16 2023

@author: aust_
"""
from time import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from Lr_Lkin_convolution import KLF_FRI_FRII, LR, Rx_values, z, Lrr
from QLFs_to_duty_cycles import Lboll, Lboloptdata, PhiBbol
from gk import gkFRI, gkFRII
t1 = time()

red=[0.5, 0.9, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.8, 4.2, 4.8]

best_gk = []
for z in red:
    print(z)
    PhikinFRII, Phikin, Phikin21conv, PhikinFRII21, kin, kkin = KLF_FRI_FRII(Lrr, Rx_values, z, LR, None, None)
    
    # Simplified objective function for optimizing gk alone
    def objective_function(gk, kin, LgLbol_target):
        LgLbol = -np.log10(gk) + kin #both kin and Lglbol are logged so we have to have loggk in the eq
        residuals = LgLbol - LgLbol_target
        chi_squared_value = np.sum(residuals**2)
        return chi_squared_value
    gk_init = 0.1
    LgLbol = gkFRI(z)
    LgLbol_target = np.interp(LgLbol, np.linspace(0, 1, len(Lboloptdata)), Lboloptdata)
    # Run optimization for gk alone
    result = minimize(objective_function, gk_init, args=(kin, LgLbol_target), method='Nelder-Mead')
    best_gk.append(result.x[0])
    # Calculate the reduced chi-squared
    num_data_points = len(LgLbol_target)
    num_parameters = 1  # Since we are fitting only 'gk'
    degrees_of_freedom = num_data_points - num_parameters
    reduced_chi_squared = result.fun / degrees_of_freedom
    
    # Print the results for each redshift
    #print(f"Results for redshift {z}:")
    #print("Best value for gk:", result.x[0])
    #print("Chi-squared value:", result.fun)
    #print("Reduced chi-squared:", reduced_chi_squared)
    #print("\n")
    
print(best_gk)
#print(np.log10(best_gk))
# Redshifts

# Results for FRI
'''
gk_fri = [0.0009757499999999965, 0.0008242499999999971, 0.0005504999999999983, 0.0004477499999999986, 
          0.00025612499999999925, 0.0001676249999999996, 0.00014681249999999967, 7.284374999999998e-05, 
          4.185937500000003e-05, 4.296093750000003e-05, 3.431250000000004e-05]
log_gk_fri = [-3.01066144, -3.08394104, -3.25924268, -3.34896441, -3.59154803, -3.77566121, -3.83323697, 
              -4.1376077, -4.37820726, -4.36692625, -4.46454764]

# Results for FRII
gk_frii = [0.0004886718749999987, 0.0004132812499999985, 0.0002759765624999981, 0.00022460937499999798, 
           0.00012851562499999772, 8.398437499999762e-05, 7.363281249999758e-05, 3.6523437499997495e-05, 
           2.0971679687497453e-05, 2.1533203124997452e-05, 1.7187499999997443e-05]
log_gk_frii = [-3.31098266, -3.3837543, -3.5591278, -3.64857212, -3.89104407, -4.07580151, -4.13292861, 
               -4.43742835, -4.67836678, -4.66689136, -4.76478729]

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(red, log_gk_fri, color='blue', marker='o', label='FRI')
plt.scatter(red, log_gk_frii, color='red', marker='^', label='FRII')

plt.xlabel('Redshift (z)')
plt.ylabel('log(gk)')
plt.title('log(gk) vs. Redshift for FRI and FRII')
plt.legend()
plt.grid(True)
plt.show()'''

t2 = time()
print(f'time in minutes = {t2-t1}')

