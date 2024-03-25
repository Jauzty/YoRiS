# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:14:59 2023

@author: aust_
"""
import glob
from time import time
import numpy as np
from scipy.optimize import minimize
from Ueda_Updated_Py import Ueda_14, Ueda_mod
import matplotlib.pyplot as plt
from QLFs_to_duty_cycles import mi_to_L2500, L2500_to_kev, myBolfunc
from Lr_Lkin_convolution import KLF_FRI_FRII, LR, Lrr
from gkzfixed import gkFRI
t1 = time()

Lx = np.linspace(41, 49, 1000)
Lbol = np.linspace(40.0, 50, 1000)
Lboldiscrete = [44, 44.5, 45, 45.5, 46, 46.5, 47]

zz = 3.5
if __name__ == "__main__":
    fig, axes = plt.subplots(2, 4, figsize=(22, 14), sharex = True, sharey = True)
    axes = axes.flatten()

#FRI frac list optical
FRIfracopt = [0.0747402, 0.0684142, 0.0622467, 0.0882428, 0.0516555, 0.0832443, 0.1088309, 0.0600215, 0.0308876, 0.0509483, 0.0133806, 0.2078514]
#FRII frac list
FRIIfracopt = [0.0237775, 0.0314957, 0.0234358, 0.0174848, 0.0134677, 0.0158409, 0.0088633, 0.0125791, 0.0021857, 0.0257798, 0.0179971, 0.0]
#FRI frac list xray
FRIfracX = [0.0754413, 0.0715513, 0.0629219, 0.0638171, 0.0527830, 0.0948788, 0.1290664, 0.0723229, 0.0323717, 0.1618970, 0.0168045, 0.2352269]
#FRII frac list
FRIIfracX = [0.0308062, 0.0565392, 0.0465047, 0.0291714, 0.0165781, 0.0254845, 0.0162047, 0.0194296, 0.0030720, 0.0420260, 0.0217945, 0.0]

file = r'C:\Users\aust_\YoRiS\QLFS\QLF8.txt'
dataQLF = np.genfromtxt(file, dtype=float)
PhikinFRII, Phikin, Phikin21conv, PhikinFRII21, kin, kkin, Phir21, Phirg21, Phikinscatter, Phikinscatter2, PhikinFRIIscatter, PhikinFRIIscatter2 = KLF_FRI_FRII(Lrr, zz, LR)
# Extract necessary data columns
mi2_column = dataQLF[:, 0]
PhiMi = dataQLF[:, 1]
PhiMiu = dataQLF[:, 2] #upper bound
PhiMil = dataQLF[:, 3] #lower bound

# Call the mi_to_L2500 function to convert magnitudes to L2500 Ångström luminosities
l_2500, Phi_l_2500 = mi_to_L2500(mi2_column, PhiMi)
l_2500, Phi_l_2500u = mi_to_L2500(mi2_column, PhiMiu)
l_2500, Phi_l_2500l = mi_to_L2500(mi2_column, PhiMil)   
l_kev, Phi_kev = L2500_to_kev(l_2500, Phi_l_2500)

sigmaL2500d=(Phi_l_2500) - (Phi_l_2500l)   #linear uncertainties
sigmaL2500u=(Phi_l_2500u) - (Phi_l_2500)

Phi_20 = Ueda_mod(Lx, zz, 0)# only unabsorbed
#converting the XLF into bolometric using K-correction
Lxx=myBolfunc(1,Lbol)
kx=Lbol-Lxx
kxx=np.interp(Lx,Lxx,kx)
Lboll=kxx+Lx
Phixbol20=Phi_20*np.abs(np.gradient(Lxx,Lx)) #just absorbtion of NH <21
Phixbol20frac = Phixbol20*FRIfracX[7] #log for comparison and account for fraction of FRI/FRII radio sources in the sample

#print(np.abs(np.gradient(Lboll,Lx)))
#converting the QLF into the B-band bolometric luminosity
alpha_opt = 0.5
LB=myBolfunc(2,Lbol)
kb=Lbol-LB
LBdata=l_2500+alpha_opt*np.log10(6.7369/11.992)
LBdata=LBdata + 14 + np.log10(6.7369)
kbb=np.interp(LBdata,LB,kb)
Lboloptdata=kbb+LBdata
PhiB=(10**Phi_l_2500)*np.abs(np.gradient(LBdata,l_2500))
PhiBd=(10**Phi_l_2500l)*np.abs(np.gradient(LBdata,l_2500))
PhiBu=(10**Phi_l_2500u)*np.abs(np.gradient(LBdata,l_2500))
PhiBbol=PhiB*np.abs(np.gradient(Lboloptdata,LBdata))*((FRIfracopt[7])) #account for the fraction of sources in the optical sample
PhiBbold=PhiBd*np.abs(np.gradient(Lboloptdata,LBdata))*FRIfracopt[7]
PhiBbolu=PhiBu*np.abs(np.gradient(Lboloptdata,LBdata))*FRIfracopt[7]
sigmaBbold= (np.log10(PhiBbol) - np.log10(PhiBbold))
sigmaBbolu= (np.log10(PhiBbolu) - np.log10(PhiBbol))

for i, L in enumerate(Lboldiscrete):  
    Lbolgk = gkFRI(L)
    ax = axes[i]
    if __name__ == "__main__":
        ax.plot(Lbolgk, np.log10(Phikin21conv), color='black', linestyle='-', label='NH<21')
        ax.scatter(Lboloptdata, np.log10(PhiBbol), marker='o', color='navy', s=30, edgecolors='black', label=f'QLF sample at matched at L = {L}')
        for xi, yi, y_err_lower_i, y_err_upper_i in zip(Lboloptdata, np.log10(PhiBbol), sigmaBbold, sigmaBbolu):
            ax.errorbar(xi, yi, yerr=[[y_err_lower_i], [y_err_upper_i]], fmt='none', capsize=3, color='navy')
        if i >= 3:
            ax.set_xlabel('$\log L_{bol} [erg s^{-1}]$')
        if i % 4 == 0:
            ax.set_ylabel('$\log \Phi(L_{k}) [Mpc^{-3} dex^{-1}]$')
            
        x_values = [43.5, 44, 44.5, 45, 45.5, 46, 46.5, 47, 47.5, 48]           
        ax.set_xticks([L + 0.5 for L in x_values[:-1]])
        tick_labels = [str(int(i)) if i % 1 == 0 else '' for i in ax.get_xticks()]
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(-10.5, -3.5)
        ax.set_xlim(43, 48.5)
        ax.legend(loc='upper left', fontsize=14, fancybox = True, framealpha = 0.0)
        ax.grid(True)
        
if __name__ == "__main__": 
    plt.rcParams['font.size'] = 21
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.dpi'] = 300
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.show()
t2 = time()
print(f'time in minutes is {(t2-t1)/60}')    
