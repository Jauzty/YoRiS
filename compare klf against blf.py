# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:41:09 2023

@author: aust_
"""
import glob
from time import time
import numpy as np
from scipy.optimize import minimize
from Ueda_Updated_Py import Ueda_14, Ueda_mod
import matplotlib.pyplot as plt
from QLFs_to_duty_cycles import mi_to_L2500, L2500_to_kev, myBolfunc
from Lr_Lkin_convolution import KLF_FRI_FRII, LR, Rx_values, Lrr
from gk import gkFRI, gkFRII
t1 = time()

#FRI frac list optical
FRIfracopt = [0.0747402, 0.0684142, 0.0622467, 0.0882428, 0.0516555, 0.0832443, 0.1088309, 0.0600215, 0.0308876, 0.0509483, 0.0133806, 0.2078514]
#FRII frac list
FRIIfracopt = [0.0237775, 0.0314957, 0.0234358, 0.0174848, 0.0134677, 0.0158409, 0.0088633, 0.0125791, 0.0021857, 0.0257798, 0.0179971, 0.0]
#FRI frac list xray
FRIfracX = [0.0754413, 0.0715513, 0.0629219, 0.0638171, 0.0527830, 0.0948788, 0.1290664, 0.0723229, 0.0323717, 0.1618970, 0.0168045, 0.2352269]
#FRII frac list
FRIIfracX = [0.0308062, 0.0565392, 0.0465047, 0.0291714, 0.0165781, 0.0254845, 0.0162047, 0.0194296, 0.0030720, 0.0420260, 0.0217945, 0.0]

zvalues = [0.5, 0.9, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.8, 4.2, 4.8]

file_list = glob.glob(r'C:\Users\aust_\YoRiS\QLFS\QLF*.txt')
file_list.sort()
#anyone else using this will have to change to their correct directory
Lx = np.linspace(41, 49, 1000)
Lbol = np.linspace(40.0, 50, 1000)

optimized_LgLbol_list = []

"""if __name__ == "__main__":
    fig, axes = plt.subplots(3, 4, figsize=(22, 14), sharex = True, sharey = True)
    axes = axes.flatten()"""
    
best_gk =[]
for i, (zz, file_name) in enumerate(zip(zvalues, file_list)): #take the qlf files and separate them based on their z
    if file_name == r'C:\Users\aust_\YoRiS\QLFS\QLF1.txt':
        z = 0.5
    elif file_name == r'C:\Users\aust_\YoRiS\QLFS\QLF2.txt':
        z = 0.9
    elif file_name == r'C:\Users\aust_\YoRiS\QLFS\QLF3.txt':
        z = 1.2
    elif file_name == r'C:\Users\aust_\YoRiS\QLFS\QLF4.txt':
        z = 1.6    
    elif file_name == r'C:\Users\aust_\YoRiS\QLFS\QLF5.txt':
        z = 2.0    
    elif file_name == r'C:\Users\aust_\YoRiS\QLFS\QLF6.txt':
        z = 2.4    
    elif file_name == r'C:\Users\aust_\YoRiS\QLFS\QLF7.txt':
        z = 2.8
    elif file_name == r'C:\Users\aust_\YoRiS\QLFS\QLF8.txt':
        z = 3.2        
    elif file_name == r'C:\Users\aust_\YoRiS\QLFS\QLF9.txt':
        z = 3.8        
    elif file_name == r'C:\Users\aust_\YoRiS\QLFS\QLFA10.txt':
        z = 4.2        
    elif file_name == r'C:\Users\aust_\YoRiS\QLFS\QLFA11.txt':
        z = 4.8        
    elif file_name == r'C:\Users\aust_\YoRiS\QLFS\QLFA12.txt':
        z = 5.8  

    # Use np.genfromtxt to load data with varying number of columns
    dataQLF = np.genfromtxt(file_name, dtype=float)
    PhikinFRII, Phikin, Phikin21conv, PhikinFRII21, kin, kkin = KLF_FRI_FRII(Lrr, Rx_values, zz, LR, FRIfracX[i], FRIIfracX[i])
    print(zz)
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
    
    Phi_20 = Ueda_mod(Lx, z, 0)# only unabsorbed
    #converting the XLF into bolometric using K-correction
    Lxx=myBolfunc(1,Lbol)
    kx=Lbol-Lxx
    kxx=np.interp(Lx,Lxx,kx)
    Lboll=kxx+Lx
    Phixbol20=Phi_20*np.abs(np.gradient(Lxx,Lx)) #just absorbtion of NH <21
    Phixbol20frac = Phixbol20*FRIfracX[i] #log for comparison and account for fraction of FRI/FRII radio sources in the sample

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
    PhiBbol=PhiB*np.abs(np.gradient(Lboloptdata,LBdata))*((FRIfracopt[i])) #account for the fraction of sources in the optical sample
    PhiBbold=PhiBd*np.abs(np.gradient(Lboloptdata,LBdata))*FRIfracopt[i]
    PhiBbolu=PhiBu*np.abs(np.gradient(Lboloptdata,LBdata))*FRIfracopt[i]
    sigmaBbold= (np.log10(PhiBbol) - np.log10(PhiBbold))
    sigmaBbolu= (np.log10(PhiBbolu) - np.log10(PhiBbol))
    LgLbol = gkFRI(zz)
    # Define the objective function to minimize    
    # Simplified objective function for optimizing gk alone
    def objective_function(gk, kin, LgLbol_target):
        LgLbol = -np.log10(gk) + kin #both kin and Lglbol are logged so we have to have loggk in the eq
        residuals = LgLbol - LgLbol_target
        chi_squared_value = np.sum(residuals**2)
        return chi_squared_value
    gk_init = 0.1
    LgLbol_target = np.interp(LgLbol, PhiBbol, Lboloptdata)
    # Run optimization for gk alone
    result = minimize(objective_function, gk_init, args=(kin, LgLbol_target), method='Nelder-Mead')
    best_gk.append(result.x[0])
    # Calculate the reduced chi-squared
    num_data_points = len(LgLbol_target)
    num_parameters = 1  # Since we are fitting only 'gk'
    degrees_of_freedom = num_data_points - num_parameters
    reduced_chi_squared = result.fun / degrees_of_freedom
    """
    # Print the results for each redshift
    #print(f"Results for redshift {z}:")
    #print("Best value for gk:", result.x[0])
    #print("Chi-squared value:", result.fun)
    #print("Reduced chi-squared:", reduced_chi_squared)
    #print("\n")
    def objective_function(LgLbol, *args):
        Phikin21conv, Lboloptdata, PhiBbol = args
        # Calculate the difference between the model (Phikin21conv) and the data (PhiBbol)
        Phibol_interp = np.interp(LgLbol, Lboloptdata, PhiBbol)
        diff = np.log10(Phikin21conv) - np.log10(Phibol_interp)
        return np.sum(diff**2)  # You can customize this objective function based on your specific needs
    
    # Perform the minimization
    result = minimize(objective_function, LgLbol, args=(Phikin21conv, Lboloptdata, PhiBbol), method = 'L-BFGS-B')
    # Get the optimized value for LgLbol
    optimized_LgLbol = result.x
    
    # Now you can use optimized_LgLbol in your plot or further analysis
    print(max(LgLbol)-max(optimized_LgLbol))
    
    if __name__ == "__main__":
        ax = axes[i]
        ax.plot(LgLbol, np.log10(Phikin21conv), color='black', linestyle='-', label='NH<21')
        ax.scatter(Lboloptdata, np.log10(PhiBbol), marker='o', color='navy', s=30, edgecolors='black', label=f'QLF sample at z = {z}')
        for xi, yi, y_err_lower_i, y_err_upper_i in zip(Lboloptdata, np.log10(PhiBbol), sigmaBbold, sigmaBbolu):
            ax.errorbar(xi, yi, yerr=[[y_err_lower_i], [y_err_upper_i]], fmt='none', capsize=3, color='navy')
        # sets the axes labels correctly, 3 on the y and 4 on the x axis
        if i // 4 == 2:
            ax.set_xlabel('$\log L_{bol} [erg s^{-1}]$')
        if i % 4 == 0:
            ax.set_ylabel('$\log \Phi(L_{k}) [Mpc^{-3} dex^{-1}]$')
        ax.set_ylim(-10.5, -3.5)
        ax.set_xlim(43, 48.5)
        ax.legend(loc='lower left', fontsize=14, fancybox = True, framealpha = 0.0)
        ax.grid(True)"""

if __name__ == "__main__": 
    print(best_gk)
    plt.rcParams['font.size'] = 21
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.dpi'] = 300
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.show()
t2 = time()
print(f'time in minutes is {(t2-t1)/60}')    
