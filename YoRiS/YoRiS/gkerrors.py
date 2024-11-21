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
redII = [0.5, 0.9, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2]

# Data points for FRI manual
gknew =[0.15, 0.13, 0.1, 0.09, 0.07, 0.05, 0.04, 0.07, 0.07, 0.07, 0.12]

#shen
mean_gk = [0.107, 0.194, 0.247, 0.290, 0.319, 0.381, 0.466, 0.530]  # Mean of gk for each bin frii
mean_error_gk = [0.036, 0.022, 0.022, 0.020, 0.023, 0.038, 0.059, 0.061]  # Mean error on gk for each bin frii
mean_gk_FRI = [0.014, 0.045, 0.071, 0.096, 0.125, 0.180, 0.200, 0.238]  # Mean of gk for each bin fri
mean_error_gk_FRI = [0.019, 0.019, 0.020, 0.017, 0.018, 0.023, 0.026, 0.030]  # Mean error on gk for each bin fri
BH_fixed_gk_FRI = [0.012, 0.06, 0.133, 0.129, 0.268, 0.352, 0.377, 0.493]


# Data points for FRII
gk_friiL = [0.9, 0.9, 0.9, 1, 0.7, 0.6, 0.6, 0.6]  # match at low luminosity
gk_friiM = [0.8, 0.7, 0.7, 0.65, 0.55, 0.5, 0.5, 0.5]  # match at mid luminosity
gk_friiH = [0.7, 0.55, 0.51, 0.42, 0.33, 0.35, 0.33, 0.3]  # high L
gk_mcmc = np.array([0.1504, 0.1459, 0.0926, 0.058, 0.0577, 0.0521, 0.0387, 0.0492, 0.0439, 0.0431, 0.0582])
mcmcsigmapos = np.array([0.003, 0.0016, 0.0023, 0.0014, 0.0014, 0.0003, 0.0003, 0.0004, 0.0018, 0.0026, 0.0028])
mcmcsigmaneg = np.array([0.0045, 0.0021, 0.0012, 0.0009, 0.0009, 0.0003, 0.0002, 0.0004, 0.0015, 0.0022, 0.0024])

gk_mcmcwithscatter = np.array([0.6053, 0.4875, 0.4089, 0.2429, 0.1828, 0.3731, 0.4268, 0.3325, 0.1404, 0.2235, 0.1695])
mcmcsigmaposscatter = np.array([0.0354, 0.0306, 0.0295, 0.0337, 0.0285, 0.0383, 0.0575, 0.0392, 0.0354, 0.0375, 0.0358])
mcmcsigmanegscatter = np.array([0.0475, 0.0328, 0.0365, 0.0402, 0.0348, 0.2636, 0.3691, 0.1546, 0.0436, 0.0352, 0.0431])

gk_mcmcfullscatter = np.array([0.7396, 0.5573, 0.2527, 0.1575, 0.0709, 0.1666, 0.2203, 0.1683, 0.1959, 0.4647, 0.2056])
mcmcsigmaposfullscatter = np.array([0.0558, 0.0502, 0.0312, 0.0324, 0.0390, 0.0400, 0.0179, 0.1407, 0.0530, 0.0907, 0.1314])
mcmcsigmanegfullscatter = np.array([0.0601, 0.0567, 0.0262, 0.0174, 0.0121, 0.0132, 0.0241, 0.1070, 0.0585, 0.0977, 0.0541])

gk_mcmcwithscatterFRII = np.array([1.4740, 1.3333, 0.9637, 0.6391, 0.5308, 0.4268, 0.2962, 0.6937])
mcmcsigmaposscatterFRII = np.array([0.0520, 0.0316, 0.0319, 0.0182, 0.0253, 0.0072, 0.0039, 0.0187])
mcmcsigmanegscatterFRII = np.array([0.0498, 0.0300, 0.0276, 0.0198, 0.0214, 0.0052, 0.0037, 0.0201])

gk_mcmcfrii = np.array([0.64, 0.68, 0.42, 0.45, 0.40, 0.45, 0.44, 0.39])
mcmcsigmaposfrii = np.array([0.21, 0.29, 0.12, 0.17, 0.11, 0.12, 0.1, 0.09])
mcmcsigmanegfrii = np.array([0.11, 0.13, 0.08, 0.1, 0.09, 0.07, 0.07, 0.07])

gk_FRII = np.array([0.7, 0.55, 0.51, 0.42, 0.33, 0.35, 0.33, 0.3])
log_gk_FRII = np.log10(gk_FRII)
# Calculate the logarithm and its errors
log_gk_mcmcnoscatter = np.log10(gk_mcmc)
log_gk_mcmcnoscatterFRII = np.log10(gk_mcmcfrii)
log_gk_mcmc = np.log10(gk_mcmcwithscatter)
log_gk_mcmcFRII = np.log10(gk_mcmcwithscatterFRII)
log_gk_mcmc_pos_errnoscatter = mcmcsigmapos / (gk_mcmc * np.log(10))
log_gk_mcmc_neg_errnoscatter = mcmcsigmaneg / (gk_mcmc * np.log(10))
log_gk_mcmc_pos_errnoscatterFRII = mcmcsigmaposfrii / (gk_mcmcfrii * np.log(10))
log_gk_mcmc_neg_errnoscatterFRII = mcmcsigmanegfrii / (gk_mcmcfrii * np.log(10))
log_gk_mcmc_pos_err = mcmcsigmaposscatter / (gk_mcmcwithscatter * np.log(10))
log_gk_mcmc_neg_err = mcmcsigmanegscatter / (gk_mcmcwithscatter * np.log(10))
log_gk_mcmc_pos_errFRII = mcmcsigmaposscatterFRII / (gk_mcmcwithscatterFRII * np.log(10))
log_gk_mcmc_neg_errFRII = mcmcsigmanegscatterFRII / (gk_mcmcwithscatterFRII * np.log(10))

log_gk_mcmcfullscatter = np.log10(gk_mcmcfullscatter)
log_gk_mcmc_pos_errfullscatter = mcmcsigmaposfullscatter / (gk_mcmcfullscatter * np.log(10))
log_gk_mcmc_neg_errfullscatter = mcmcsigmanegfullscatter / (gk_mcmcfullscatter * np.log(10))

mcmc_errorfullscatter = np.vstack((log_gk_mcmc_neg_errfullscatter, log_gk_mcmc_pos_errfullscatter))
mcmc_errornoscatter = np.vstack((log_gk_mcmc_neg_errnoscatter, log_gk_mcmc_pos_errnoscatter))
mcmc_errornoscatterFRII = np.vstack((log_gk_mcmc_neg_errnoscatterFRII, log_gk_mcmc_pos_errnoscatterFRII))
mcmc_errorFRII = np.vstack((log_gk_mcmc_neg_errFRII, log_gk_mcmc_pos_errFRII))
mcmc_error = np.vstack((log_gk_mcmc_neg_err, log_gk_mcmc_pos_err))


log_gk_fri = np.log10(gknew)

log_gk_friiL = np.log10(gk_friiL)
log_gk_friiM = np.log10(gk_friiM)
log_gk_friiH = np.log10(gk_friiH)

plt.figure(figsize=(10, 6))
plt.scatter(red, log_gk_mcmc, color = 'red', marker = 's', s = 20, label = 'FRI')
plt.scatter(redII, log_gk_mcmcFRII, color = 'blue', marker = '*', s = 20, label = 'FRII')
plt.errorbar(redII, np.log10(mean_gk), yerr=mean_error_gk, fmt='^', color='green', ecolor='green', capsize=5, label='Shen FRII')
plt.errorbar(redII, np.log10(mean_gk_FRI), yerr=mean_error_gk_FRI, fmt='+', color='magenta', ecolor='magenta', capsize=5, label='Shen FRI')

#plt.scatter(red, log_gk_mcmcnoscatter, color = 'green', marker = 'v', s = 20, label = 'MCMC match, no scatter, FRI')
#plt.scatter(redII, log_gk_mcmcnoscatterFRII, color = 'purple', marker = '8', s = 20, label = 'MCMC match, no scatter, FRII')
#plt.scatter(red, log_gk_fri, color = 'grey', marker = 'd', s = 20, label = 'Manual match, no scatter, FRI')
#plt.scatter(redII, log_gk_FRII, color = 'orange', marker = '<', s = 20, label = 'Manual match, no scatter, FRII')
plt.scatter(0.025, -0.31, color = 'black', marker = 'o', s = 30, label = 'LeMMINGs result, all sources')
plt.errorbar(0.025, -0.31, yerr=0.42/np.sqrt(280), fmt='none', color = 'black', capsize = 5)
#plt.errorbar(red, log_gk_mcmcnoscatter, yerr=mcmc_errornoscatter, fmt='none', color='green', capsize=4)
#plt.errorbar(redII, log_gk_mcmcnoscatterFRII, yerr=mcmc_errornoscatterFRII, fmt='none', color='purple', capsize=4)
plt.errorbar(redII, log_gk_mcmcFRII, yerr=mcmc_errorFRII, fmt='none', color='blue', capsize=4)
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
plt.errorbar(red[7:], log_gk_fri_H[7:], yerr=fri_high_error1, fmt='none', color='red', capsize=5)

red = [0.5, 0.9, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.2, 4.8]
redII = [0.5, 0.9, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2]

scatterFRI = np.array([0.495, 0.4805, 0.5574, 0.5990, 0.5036, 0.7755, 0.8725, 0.7059, 0.4676, 0.5761, 0.4055])
scatterFRIpos = np.array([0.0127, 0.0156, 0.0178, 0.0319, 0.0378, 0.0221, 0.0277, 0.0236, 0.0525, 0.0342, 0.0422])
scatterFRIneg = np.array([0.0165, 0.0172, 0.0221, 0.0491, 0.0564, 0.3238, 0.5730, 0.1376, 0.0908, 0.0411, 0.0672])

scatterFRII = np.array([0.4378, 0.3314, 0.3201, 0.3155, 0.2989, 0.2398, 0.2289, 0.4186])
scatterFRIIpos = np.array([0.0114, 0.0068, 0.0098, 0.0075, 0.01, 0.0042, 0.0043, 0.0084])
scatterFRIIneg = np.array([0.0103, 0.0062, 0.0073, 0.0074, 0.0087, 0.0031, 0.0038, 0.0076])

print(np.mean(scatterFRI))
print(np.mean(scatterFRII))

mcmc_errorFRI = np.vstack((scatterFRIneg, scatterFRIpos))
mcmc_errorFRII = np.vstack((scatterFRIIneg, scatterFRIIpos))

plt.figure(figsize=(10, 6))
plt.scatter(red, scatterFRI, color = 'red', marker = 's', s = 20, label = 'Scatter, FRI')
plt.scatter(redII, scatterFRII, color = 'blue', marker = '^', s = 20, label = 'Scatter, FRII')
plt.errorbar(redII, scatterFRII, yerr=mcmc_errorFRII, fmt='none', color='blue', capsize=4)
plt.errorbar(red, scatterFRI, yerr=mcmc_errorFRI, fmt='none', color='red', capsize=4)

plt.legend(fontsize=16)
plt.xlabel('Redshift (z)')
plt.ylabel('Scatter (dex)')
plt.title('Scatter vs. Redshift')
plt.grid(True)
plt.rcParams['font.size'] = 18
plt.show()
"""



# Plotting FRII
#plt.scatter(red, log_gk_friiM, color='black', marker='^', label='FRII')
#plt.errorbar(red, log_gk_friiM, yerr=[log_gk_friiM - log_gk_friiL, log_gk_friiH - log_gk_friiM], fmt='none', color='black', capsize=5)

plt.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), fontsize=18, framealpha=0)
plt.ylim(-2.0, 0.25)
plt.xlabel('Redshift (z)')
plt.ylabel('$log(g_{k})$')
plt.title('$log(g_{k})$ vs. Redshift')
plt.grid(True)
plt.rcParams['font.size'] = 22
plt.show()

t2 = time()
print(f'time in seconds = {t2-t1}')