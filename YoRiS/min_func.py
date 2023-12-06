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
red = [0.5, 0.9, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2]

gknewL =[0.21, 0.16, 0.07, 0.06, 0.05, 0.03, 0.015, 0.034] #matched for low luminosity
gknewL2 =[0.082, 0.075, 0.045, 0.03, 0.035, 0.023, 0.012, 0.02]# case 2
gknewL4 =[0.1, 0.08, 0.047, 0.03, 0.035, 0.023, 0.012, 0.02]# case 4
gknewM =[0.1, 0.07, 0.04, 0.02, 0.025, 0.016, 0.012, 0.02] #matched for mid luminosity
gknewH =[0.1, 0.085, 0.065, 0.05, 0.053, 0.037, 0.025, 0.0265]#high L
gknewH2= [0.065, 0.055, 0.035, 0.025, 0.02, 0.015, 0.015, 0.015] # high L new convolution
log_gk_fri_L = np.log10(gknewL)
log_gk_fri_L2 = np.log10(gknewL2)
log_gk_fri_L4 = np.log10(gknewL4)
log_gk_fri_M = np.log10(gknewM)
log_gk_fri_H = np.log10(gknewH)
loggkfrinew = np.log10(gknewH2)


# Results for FRII
gk_friiL = [0.9, 0.9, 0.9, 1, 0.7, 0.6, 0.6, 0.6]# match at low luminosity 
gk_friiM = [0.8, 0.7, 0.7, 0.65, 0.55, 0.5, 0.5, 0.5]# match at mid luminosity
gk_friiH = [0.7, 0.55, 0.51, 0.42, 0.33, 0.35, 0.33, 0.3]#high L
log_gk_friiL = np.log10(gk_friiL)
log_gk_friiM = np.log10(gk_friiM)
log_gk_friiH = np.log10(gk_friiH)

# Plotting
plt.figure(figsize=(10, 6))
#plt.scatter(red, log_gk_fri_L, color='blue', marker='o', label='FRI Low L case 5')
#plt.scatter(red, log_gk_fri_L4, color='red', marker='d', label='FRI Low L case 4')
#plt.scatter(red, log_gk_fri_L2, color='green', marker='*', label='FRI Low L case 2', alpha = 0.9)
#plt.scatter(red, log_gk_fri_M, color='red', marker='o', label='FRI Mid L')
#plt.scatter(red, log_gk_fri_H, color='black', marker='o', label='FRI High L')
#plt.scatter(red, log_gk_friiL, color='blue', marker='^', label='FRII Low L')
#plt.scatter(red, log_gk_friiM, color='red', marker='^', label='FRII Mid L')
plt.scatter(red, log_gk_friiH, color='black', marker='^', label='FRII High L')
plt.scatter(red, loggkfrinew, color ='red', marker = 'd', label = 'FRI High L')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Redshift (z)')
plt.ylabel('log(gk)')
plt.title('log(gk) vs. Redshift; low luminsoity La-Franca cases 2, 4, 5')
plt.grid(True)
plt.show()
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

t2 = time()

newe = [-0.2, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1]# new edited for fri
efriinew = [-2.6, -2.6, -2.4, -2.6, -2.6, -2.6, -2.5, -2.6] # new edited for 

plt.figure(figsize=(10, 6))
plt.scatter(red, newe, color='blue', marker='o', label='FRI')
plt.scatter(red, efriinew, color='red', marker='^', label='FRII')

plt.xlabel('Redshift (z)')
plt.ylabel('Epsilon')
plt.title('Epsilon vs. Redshift for FRI and FRII')
plt.legend()
plt.grid(True)
plt.show()

print(f'time in minutes = {t2-t1}')

