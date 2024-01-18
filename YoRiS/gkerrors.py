# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:56:25 2024

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

red = [0.5, 0.9, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2]

# Data points for FRI
gknewM = [0.1, 0.07, 0.04, 0.02, 0.025, 0.016, 0.012, 0.02]  # matched for mid luminosity

# Error bars for FRI
gknewH = [0.1, 0.085, 0.065, 0.05, 0.053, 0.037, 0.025, 0.0265]  # lower bounds
gknewL = [0.21, 0.16, 0.07, 0.06, 0.05, 0.03, 0.015, 0.034]  # upper bounds

# Data points for FRII
gk_friiL = [0.9, 0.9, 0.9, 1, 0.7, 0.6, 0.6, 0.6]  # match at low luminosity
gk_friiM = [0.8, 0.7, 0.7, 0.65, 0.55, 0.5, 0.5, 0.5]  # match at mid luminosity
gk_friiH = [0.7, 0.55, 0.51, 0.42, 0.33, 0.35, 0.33, 0.3]  # high L

log_gk_fri_L = np.log10(gknewL)
log_gk_fri_M = np.log10(gknewM)
log_gk_fri_H = np.log10(gknewH)

log_gk_friiL = np.log10(gk_friiL)
log_gk_friiM = np.log10(gk_friiM)
log_gk_friiH = np.log10(gk_friiH)

# Plotting FRI
plt.figure(figsize=(10, 6))
# For the 5th, 6th, and 7th points, FRI low is the middle point
fri_low_error = np.array([log_gk_fri_L[4:7] - log_gk_fri_H[4:7], log_gk_fri_M[4:7] - log_gk_fri_L[4:7]])
plt.scatter(red[4:7], log_gk_fri_L[4:7], color='red', marker='o')
plt.errorbar(red[4:7], log_gk_fri_L[4:7], yerr=fri_low_error, fmt='none', color='red', capsize=5)

# For the rest of the points, FRI high is the middle point
fri_high_error = np.array([log_gk_fri_H[:4] - log_gk_fri_L[:4], log_gk_fri_M[:4] - log_gk_fri_H[:4]])
plt.scatter(red[:4], log_gk_fri_H[:4], color='red', marker='o')
plt.errorbar(red[:4], log_gk_fri_H[:4], yerr=fri_high_error, fmt='none', color='red', capsize=5)
fri_high_error1 = np.array([log_gk_fri_H[7:] - log_gk_fri_L[7:], log_gk_fri_M[7:] - log_gk_fri_H[7:]])
plt.scatter(red[7:], log_gk_fri_H[7:], color='red', marker='o', label='FRI')
plt.errorbar(red[7:], log_gk_fri_H[7:], yerr=fri_high_error1, fmt='none', color='red', capsize=5)


# Plotting FRII
plt.scatter(red, log_gk_friiM, color='black', marker='^', label='FRII')
plt.errorbar(red, log_gk_friiM, yerr=[log_gk_friiM - log_gk_friiL, log_gk_friiH - log_gk_friiM], fmt='none', color='black', capsize=5)

plt.legend(loc="center right")
plt.xlabel('Redshift (z)')
plt.ylabel('log(gk)')
plt.title('log(gk) vs. Redshift')
plt.grid(True)
plt.show()

# Other plotting
newe = [-0.2, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1]
efriinew = [-2.6, -2.6, -2.4, -2.6, -2.6, -2.6, -2.5, -2.6]

plt.figure(figsize=(10, 6))
plt.scatter(red, newe, color='blue', marker='o', label='FRI')
plt.scatter(red, efriinew, color='red', marker='^', label='FRII')

plt.xlabel('Redshift (z)')
plt.ylabel('Epsilon')
plt.title('Epsilon vs. Redshift for FRI and FRII')
plt.legend()
plt.grid(True)
plt.show()

t2 = time()
print(f'time in minutes = {t2-t1}')