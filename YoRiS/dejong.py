# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:43:43 2023

@author: aust_
"""
import glob
from time import time
import numpy as np
import matplotlib.pyplot as plt
from UedaXLF_to_RLF import fradioFRII

t1 = time()

file_list = glob.glob(r'C:\Users\aust_\YoRiS\*de jong.txt')
file_list += glob.glob(r'C:\Users\aust_\YoRiS\*WangSmolcic.txt')
file_list.sort()

if __name__ == "__main__":
    fig, axes = plt.subplots(3, 3, figsize=(32, 22), sharex=True, sharey=True)

    for i, file_name in enumerate(file_list):
        if "0.15de jong" in file_name:
            z = 0.15
        elif "0.3WangSmolcic" in file_name:
            z = 0.3
        elif "0.4de jong" in file_name:
            z = 0.4
        elif "0.65de jong" in file_name:
            z = 0.65
        elif "0.75WangSmolcic" in file_name:
            z = 0.75
        elif "1.05de jong" in file_name:
            z = 1.05
        elif "1.75WangSmolcic" in file_name:
            z = 1.75
        elif "2.0de jong" in file_name:
            z = 2.0
        elif "3.5WangSmolcic" in file_name:
            z = 3.5
        
        lrmin = 30 #erg/s
        lrmax = 47
        Lr_values = np.linspace(lrmin, lrmax, 1000)
        Rx_values = np.linspace(-10, -1, 1000) #FRI
        Rx_values1 = Lr_values - 44 #FRII
        P = Lr_values-7-np.log10(1.4)-9
        Phir, Phirgausfake = fradioFRII(z, Rx_values, Lr_values, 41)
        Phirfake, Phirgaus = fradioFRII(z, Rx_values1, Lr_values, 41)
        Phirtot = Phir + Phirgaus
        
        if "WangSmolcic" in file_name:
            data = np.genfromtxt(file_name, dtype=float, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
            densFRI = np.log10(data[:, 1])
            Lum = data[:, 0]
            FRIerrpos = np.log10(data[:, 3])-densFRI
            FRIerrneg = np.abs(densFRI-np.log10(data[:, 2]))
            densFRIS = np.log10(data[:, 5])
            LumS = data[:, 4]
            FRIerrposS = np.log10(data[:, 7])-densFRIS
            FRIerrnegS = densFRIS-np.log10(data[:, 6])
            densFRIC = np.log10(data[:, 9])
            LumC = data[:, 8]
            FRIerrposC = np.log10(data[:, 11])-densFRIC
            FRIerrnegC = densFRIC-np.log10(data[:, 10])
            densFRISlaus = np.log10(data[:, 13])
            LumSlaus = data[:, 12]
            FRIerrposSlaus = np.log10(data[:, 15])-densFRISlaus
            FRIerrnegSlaus = densFRISlaus-np.log10(data[:, 14])
        else:
            data = np.genfromtxt(file_name, dtype=float, usecols=(0, 1, 2, 3, 4, 5, 6))
            densFRI = data[:, 1]
            densFRII = data[:, 2]
            Lum = data[:, 0]
            FRIerrpos = data[:, 3]
            FRIerrneg = data[:, 4]
            FRIIerrpos = data[:, 5]
            FRIIerrneg = data[:, 6]


        ax = axes.flatten()[i]
        if "WangSmolcic" in file_name:
            ax.plot(P, np.log10(Phir), label = f'FRI RLF z = {z}', linestyle = "--", color = "black")
            ax.plot(P, np.log10(Phirtot), label = f'FRII RLF z = {z}', color = "blue")
            ax.errorbar(Lum, densFRI, yerr=[FRIerrpos, FRIerrneg], fmt='d', label=f'Wang+2024, z={z}', alpha = 0.5, color = 'red', capsize = 4, markersize = 13)
            ax.errorbar(LumS, densFRIS,  yerr=[FRIerrposS, FRIerrnegS], fmt='X', label=f'Smolcic+2017, z={z}', alpha = 0.5, color = 'green', capsize = 4, markersize = 13)
            ax.errorbar(LumC, densFRIC,  yerr=[FRIerrposC, FRIerrnegC], fmt='s', label=f'Ceraj+2018, z={z}', alpha = 0.5, color = 'purple', capsize = 4, markersize = 13)
            ax.errorbar(LumSlaus, densFRISlaus,  yerr=[FRIerrposSlaus, FRIerrnegSlaus], fmt='o', label=f'Slaus+2023, z={z}', alpha = 0.5, color = 'navy', capsize = 4, markersize = 13)
        else:

            ax.errorbar(Lum, densFRI, yerr=[FRIerrpos, -FRIerrneg], fmt='d', label=f'DeJong+2023 FRI, z={z}', alpha = 0.5, color = 'red', capsize = 4, markersize = 13)
            ax.errorbar(Lum, densFRII, yerr=[FRIIerrpos, -FRIIerrneg], fmt='o', label=f'DeJong+2023 FRII, z={z}', alpha = 0.5, color = 'green', capsize = 4, markersize = 13)
            ax.plot(P, np.log10(Phir), label = f'FRI RLF z = {z}', linestyle = "--", color = "black")
            ax.plot(P, np.log10(Phirtot), label = f'FRII RLF z = {z}', color = "blue")
        
        if i == 6  or i == 7 or i == 8:
            ax.set_xlabel('$\log (L_{R}) (W/Hz^{-1})$')
        if i == 0 or i == 3 or i == 6:
            ax.set_ylabel('$\log \Phi(L_{R}) [Mpc^{-3} dex^{-1}]$')
        ax.set_ylim(-9.5, -2.5)
        ax.set_xlim(16, 29)
        ax.set_xticks(range(18, 29, 2))
        ax.set_yticks(range(-9, -2, 1))
        ax.legend(prop = {"size": 25}, framealpha = 0.0)
        ax.grid(True)

    plt.tight_layout()
    plt.rcParams['font.size'] = 40
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.dpi'] = 500
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.show()

t2 = time()
print(f"Time taken: {t2 - t1:.2f} seconds")
