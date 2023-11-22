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

gknew = [0.15867187500000018, 0.07500000000000005, 0.03271484375000014, 
            0.02199218750000015, 0.02156250000000015, 0.009384765625000176,
            0.008310546875000177, 0.033906250000000124, 0.005273437500000184,
            0.03628906250000013, 0.008359375000000177]
log_gk_fri = np.log10(gknew)

# Results for FRII
gk_frii = [0.7609375000000024, 0.7, 0.8, 0.8, 0.45, 0.5, 0.5, 0.5]
log_gk_frii = np.log10(gk_frii)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(red, log_gk_fri, color='blue', marker='o', label='FRI')
plt.scatter(red, log_gk_frii, color='red', marker='^', label='FRII')

plt.xlabel('Redshift (z)')
plt.ylabel('log(gk)')
plt.title('log(gk) vs. Redshift for FRI and FRII')
plt.legend()
plt.grid(True)
plt.show()

t2 = time()
print(f'time in minutes = {t2-t1}')

