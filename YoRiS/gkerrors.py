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

red = [0.5, 0.9, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.2, 4.8]

# Data points for FRI manual
gknew =[0.15, 0.13, 0.1, 0.09, 0.07, 0.05, 0.04, 0.07, 0.07, 0.07, 0.12]

# Data points for FRII
gk_friiL = [0.9, 0.9, 0.9, 1, 0.7, 0.6, 0.6, 0.6]  # match at low luminosity
gk_friiM = [0.8, 0.7, 0.7, 0.65, 0.55, 0.5, 0.5, 0.5]  # match at mid luminosity
gk_friiH = [0.7, 0.55, 0.51, 0.42, 0.33, 0.35, 0.33, 0.3]  # high L
gk_mcmc = np.array([0.1504, 0.1459, 0.0926, 0.058, 0.0577, 0.0521, 0.0387, 0.0492, 0.0439, 0.0431, 0.0582])
mcmcsigmapos = np.array([0.003, 0.0016, 0.0023, 0.0014, 0.0014, 0.0003, 0.0003, 0.0004, 0.0018, 0.0026, 0.0028])
mcmcsigmaneg = np.array([0.0045, 0.0021, 0.0012, 0.0009, 0.0009, 0.0003, 0.0002, 0.0004, 0.0015, 0.0022, 0.0024])
gk_mcmcfrii = np.array([0.64, 0.68, 0.42, 0.45, 0.40, 0.45, 0.44, 0.39, 0.29, 0.41, 0.28])
mcmcsigmaposfrii = np.array([0.21, 0.29, 0.12, 0.17, 0.11, 0.12, 0.1, 0.09, 0.1, 0.12, 0.09])
mcmcsigmanegfrii = np.array([0.11, 0.13, 0.08, 0.1, 0.09, 0.07, 0.07, 0.07, 0.08, 0.09, 0.08])
# Calculate the logarithm and its errors
log_gk_mcmc = np.log10(gk_mcmc)
log_gk_mcmc_pos_err = mcmcsigmapos / (gk_mcmc * np.log(10))
log_gk_mcmc_neg_err = mcmcsigmaneg / (gk_mcmc * np.log(10))
mcmc_error = np.vstack((log_gk_mcmc_neg_err, log_gk_mcmc_pos_err))


log_gk_fri = np.log10(gknew)

log_gk_friiL = np.log10(gk_friiL)
log_gk_friiM = np.log10(gk_friiM)
log_gk_friiH = np.log10(gk_friiH)

plt.figure(figsize=(10, 6))
plt.scatter(red, log_gk_mcmc, color = 'red', marker = 's', s = 8, label = 'MCMC match')
plt.scatter(red, log_gk_fri, color = 'grey', marker = 'd', s = 20, label = 'Manual match at L = 46')
plt.errorbar(red, log_gk_mcmc, yerr=mcmc_error, fmt='none', color='red', capsize=4)

"""
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
plt.errorbar(red[7:], log_gk_fri_H[7:], yerr=fri_high_error1, fmt='none', color='red', capsize=5)"""


# Plotting FRII
#plt.scatter(red, log_gk_friiM, color='black', marker='^', label='FRII')
#plt.errorbar(red, log_gk_friiM, yerr=[log_gk_friiM - log_gk_friiL, log_gk_friiH - log_gk_friiM], fmt='none', color='black', capsize=5)

plt.legend(loc="upper center")
plt.xlabel('Redshift (z)')
plt.ylabel('log(gk)')
plt.title('log(gk) vs. Redshift')
plt.grid(True)
plt.show()

t2 = time()
print(f'time in seconds = {t2-t1}')