# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:13:10 2023

@author: aust_
"""
import numpy as np
from Lr_Lkin_convolution import z as zz
from scipy.io import readsav
import matplotlib.pyplot as plt
from QLFs_to_duty_cycles import Lunif, l_2500, PhiBbol, Lboloptdata, sigmaBbold, sigmaBbolu, LBdata, poisson_gehrels, dutyjiang

def find_closest_edges(z, zregions):
    """finds the closest two z values based on the z used in gk.py"""
    # Initialize closest_edges with the first and last values in zregions as a default.
    closest_edges = [zregions[0], zregions[-1]]
    #Iterate through zregions to find the two bin edges closest to the given z value.
    for i in range(len(zregions) - 1):
        # If z falls within the current bin, update closest_edges accordingly.
        if zregions[i] <= z <= zregions[i+1]:
            closest_edges = [zregions[i], zregions[i+1]]
            break
    
    return closest_edges

def redefine_zregions(z, zregions): 
    """will take the closest two edges and then redfine the list to
    be used in the code, where before it returned many, it will now only return for one chunk of z"""
    closest_edges = find_closest_edges(z, zregions)
     # Construct a new list, new_zregions, containing only the values within the closest edges.
    new_zregions = [edge for edge in zregions if closest_edges[0] <= edge <= closest_edges[1]]
    
    return new_zregions

zregions = [0.3, 0.68, 1.06, 1.44, 1.82, 2.2, 2.6, 3.0, 3.5, 4.0, 4.5, 5.0, 6.5]
bin_edges = redefine_zregions(zz, zregions)

def dutyforspecz(Lunif, l_2500, PhiBbol, Lboloptdata, sigmaBbold, sigmaBbolu, LBdata, bin_edges, zz):
    alpha_opt = 0.5
    #extract data from the two relevant files
    data = readsav(r'C:\Users\aust_\Downloads\shen_catalogue.sav')
    z = data['z']
    first_FR_type = data['first_FR_type']
    Lbol = data['lbol']
    LogL3000 = data['logl3000']
    LogL1350 = data['logl1350']
    LogL5100 = data['logl5100']
    file_path = r'C:\Users\aust_\Downloads\vanvelzen.txt'
    data2 = np.genfromtxt(file_path, names=['bol', 'fracII'], dtype=None, delimiter=None)
    bol = data2['bol']
    fracII = data2['fracII'] / 100.0

    X = Lbol#assuming tagbol == 0 this is the simplest way to create the X array

    #if zz <= 2.2:
     #   X = LogL3000 #take 3000A straight from shen catalogue
    #elif zz > 2.2: #at high redshift the correction must be applied slightly differently
     #   xx = LogL1350 #take 1350A from shen catalogue
      #  xx = xx - np.log10(2.2207)-15
       # X = xx + alpha_opt*np.log10(9.9931/2.2207)
        #X = X + np.log10(9.9931)+14 #makes it back to 3000A
        
    #LogL2500 = X - alpha_opt*np.log10(9.9931/11.992)# convert from 3000 to 2500
    #LogL2500 = LogL2500 - 14 - np.log10(9.9931) #from erg/s to erg/s/hz required for input into function
    #LogL2kev, b = L2500_to_kev(LogL2500, Phi_l_2500) #convert to 2-10kev x ray
    #X = LogL2kev

    nl = len(Lunif)
    #define the bin edges where z inbetween two consecutive values is all kept together
    
    # Create an array of bin assignments for each element in z
    zbin = np.digitize(z, bin_edges)
    
    # Initialize an array to store the indices of data points in each bin
    zbin_indices = []
    bin_indices = np.where(zbin == 1)[0]
    # Create separate arrays for each bin
    zbin_indices.append(bin_indices)
    zvalues = z[bin_indices]
    first_FR_type_values = first_FR_type[bin_indices]
    z1 = zvalues - 0.35
    z2 = zvalues + 0.35
    
    #initialise arrays , ratios are 1 dimensional errors are two dimensional
    ratioall = np.zeros(nl)
    sigALL = np.zeros((nl, 2))
    ratioI = np.zeros(nl)
    sigI = np.zeros((nl, 2))
    ratioII = np.zeros(nl)
    sigII = np.zeros((nl, 2))
    ratiojiang = np.zeros(nl)
    sigJiang = np.zeros(nl)
    ratiovanvelzen = np.zeros(nl)
    #return datapoints where the FR type and z values are within certain range
    ixopt = np.where((first_FR_type_values == 0) & (zvalues < z2) & (zvalues > z1))
    ixrad = np.where((first_FR_type_values > 0) & (zvalues < z2) & (zvalues > z1))
    ixI = np.where((first_FR_type_values == 1) & (zvalues < z2) & (zvalues > z1))
    ixII = np.where((first_FR_type_values == 2) & (zvalues < z2) & (zvalues > z1))
    
    if np.mean(ixopt) != -1:
        Lopt = X[ixopt]
    if np.mean(ixrad) != -1:
        Lrad = X[ixrad]
    if np.mean(ixI) != -1:
        LradI = X[ixI]
    if np.mean(ixII) != -1:
        LradII = X[ixII]
    
    maxl = max(LradI) + 0.5 if np.all(zvalues > 3.75) else max(LradII) + 0.2
    Lunif = np.linspace(44, maxl, nl)
    
    # Initialize arrays to store ratio calculations and errors
    A = np.zeros(1)
    B = np.zeros(1)
    C = np.zeros(1)
    D = np.zeros(1)
    
    # Loop through the range of kx values in the length of the Lunif array
    for kx in range(nl - 1):
        # Calculate the indices for elements within specific ranges based on conditions
        ixo = np.where((Lopt > Lunif[kx]) & (Lopt < Lunif[kx + 1]))
        
        # Check if there are any elements within the calculated indices
        if len(ixo[0]) > 0:
            # Sum the values in Lopt within the specified range
            A = np.sum(Lopt[ixo])
    
        # Repeat the above process for Lrad, LradI, and LradII
        ixr = np.where((Lrad > Lunif[kx]) & (Lrad < Lunif[kx + 1]))
        if len(ixr[0]) > 0:
            B = np.sum(Lrad[ixr])
    
        ixrI = np.where((LradI > Lunif[kx]) & (LradI < Lunif[kx + 1]))
        if len(ixrI[0]) > 0:
            C = np.sum(LradI[ixrI])
    
        ixrII = np.where((LradII > Lunif[kx]) & (LradII < Lunif[kx + 1]))
        if len(ixrII[0]) > 0:
            D = np.sum(LradII[ixrII])
    
        # Calculate ratios based on the accumulated values of B and combinations of A and B
        ratioall[kx] = B / float(A + B)
        ratioI[kx] = C / float(A + B)
        ratioII[kx] = D / (A + B)
    
        # Calculate positive and negative errors for the ratios using poisson_gehrels
        res = poisson_gehrels(B)
        sigALL_up = res[:, 0] / float(A + B)
        sigALL_lw = res[:, 1] / float(A + B)
        sigALL[kx, 0] = sigALL_up
        sigALL[kx, 1] = sigALL_lw
    
        resI = poisson_gehrels(C)
        sigI_up = resI[:, 0] / float(A + B)
        sigI_lw = resI[:, 1] / float(A + B)
        sigI[kx, 0] = sigI_up
        sigI[kx, 1] = sigI_lw
    
        resII = poisson_gehrels(D)
        sigII_up = resII[:, 0] / (A + B)
        sigII_lw = resII[:, 1] / (A + B)
        sigII[kx, 0] = sigII_up
        sigII[kx, 1] = sigII_lw
    
        # Calculate ratiojiang and sigJiang based on a different formula
        ratiojiang[kx] = B / float(A)
        sigJiang[kx] = np.sqrt(B) / float(A)
    
        # Calculate ratiovanvelzen based on D and a combination of A and C
        ratiovanvelzen[kx] = D / float(A + C)
    
    nn = len(zvalues)
    M2500 = np.linspace(-29, -17, nn)
    fjiang = 10 ** dutyjiang(zvalues, M2500)
    lgfnu = -20 + np.log10(3.63)
    lgf2500 = lgfnu - 0.4 * M2500
    jac1 = np.abs(np.gradient(lgf2500, M2500))
    LogL2500 = lgf2500 + np.log10(4 * np.pi) + (2.0 * 19) + 2 * np.log10(3.0857)
    lgl3000 = LogL2500 + alpha_opt * np.log10(9.9931 / 11.992)
    lgl3000 = lgl3000 + np.log10(9.9931) + 14
    fjiang3000 = fjiang * jac1
    
    Lunifmono = Lunif - np.log10(9.9931) - 14.0  # Lunif is in log unit erg/s, convert to erg/s/Hz
    LBshen = Lunifmono + alpha_opt * np.log10(6.7369 / 9.9931)
    LLB = LBshen + np.log10(6.7369) + 14.0  # LLB in erg/s
    ratiovanvelzenbol = ratiovanvelzen * np.abs(np.gradient(LBshen, Lunifmono))
    
    J=np.abs(np.gradient(LBdata,l_2500))*np.abs(np.gradient(Lboloptdata,LBdata))
    ratioVV = np.interp(Lboloptdata, Lunif, ratiovanvelzen)
    PhiradioVV = PhiBbol * ratioVV * J
    fracVV = np.interp(Lboloptdata, bol, fracII)
    #PhiradiodataVV =  PhiBbol * fracII * J causes an error as the array lengths are not compatible
    
    ratiodataII = np.interp(Lboloptdata, Lunif, ratioII)
    ratiodata = np.interp(Lboloptdata, Lunif, ratioI)
    ratiodatatot = np.interp(Lboloptdata, Lunif, ratioall)
    Phiradio = PhiBbol * ratiodata
    PhiradioII = PhiBbol * ratiodataII
    Phiradiotot = PhiBbol * ratiodatatot
    sigintIId = np.interp(Lboloptdata, Lunif, sigII[:, 1])
    sigintIIu = np.interp(Lboloptdata, Lunif, sigII[:, 0])
    sigintd = np.interp(Lboloptdata, Lunif, sigII[:, 1])
    sigintu = np.interp(Lboloptdata, Lunif, sigII[:, 0])
    sigintTotd = np.interp(Lboloptdata, Lunif, sigALL[:, 1])
    sigintTotu = np.interp(Lboloptdata, Lunif, sigALL[:, 0])
    sigRadioIId = PhiradioII * np.sqrt((sigmaBbold / PhiBbol) ** 2 + (sigintIId / ratiodataII) ** 2)
    sigRadioIIu = PhiradioII * np.sqrt((sigmaBbolu / PhiBbol) ** 2 + (sigintIIu / ratiodataII) ** 2)
    sigRadiod = Phiradio * np.sqrt((sigmaBbold / PhiBbol) ** 2 + (sigintIId / ratiodata) ** 2)
    sigRadiou = Phiradio * np.sqrt((sigmaBbolu / PhiBbol) ** 2 + (sigintIIu / ratiodata) ** 2)
    sigRadioTotd = Phiradiotot * np.sqrt((sigmaBbold / PhiBbol) ** 2 + (sigintTotd / ratiodatatot) ** 2)
    sigRadioTotu = Phiradiotot * np.sqrt((sigmaBbolu / PhiBbol) ** 2 + (sigintTotu / ratiodatatot) ** 2)
    #calculating the average fractions for rescaling the luminosity functions
    if __name__ == "__main__":
        #plots the datapoints
        plt.scatter(Lunif, ratioall, marker='o', label='ALL', color='blue')
        plt.scatter(Lunif, ratioI, marker='s', label='FRI', color='green')
        plt.scatter(Lunif, ratioII, marker='^', label='FRII', color='red')
        plt.scatter(Lunif, ratiojiang, marker='*', label='Radio Loud Jiang+07', color='purple')
                
        #plots their error bars
        for xi, yi, y_err_lower_i, y_err_upper_i in zip(Lunif, ratioall, sigALL[:, 0], sigALL[:, 1]):
            plt.errorbar(xi, yi, yerr=[[y_err_lower_i], [y_err_upper_i]], fmt='none', capsize=3, color='blue')
        for xi, yi, y_err_lower_i, y_err_upper_i in zip(Lunif, ratioI, sigI[:, 0], sigI[:, 1]):
            plt.errorbar(xi, yi, yerr=[[y_err_lower_i], [y_err_upper_i]], fmt='none', capsize=3, color='green')
        for xi, yi, y_err_lower_i, y_err_upper_i in zip(Lunif, ratioII, sigII[:, 0], sigII[:, 1]):
            plt.errorbar(xi, yi, yerr=[[y_err_lower_i], [y_err_upper_i]], fmt='none', capsize=3, color='red')
        for xi, yi, y_err_lower_i, y_err_upper_i in zip(Lunif, ratiojiang, sigJiang[:], sigJiang[:]):
            plt.errorbar(xi, yi, yerr=[[y_err_lower_i], [y_err_upper_i]], fmt='none', capsize=3, color='purple')
                
    if __name__ == "__main__":
        plt.rcParams['font.size'] = 24
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['figure.dpi'] = 300
        plt.tight_layout()
        plt.legend(loc = 'upper center', fontsize = 10, ncol =4)
        plt.yscale('log')
        plt.ylim(1e-3, 5e-1)
        plt.xlim(43.9, 47.6)
        plt.show()
    return PhiradioII

print()
if __name__ == "__main__":
    dutyforspecz(Lunif, l_2500, PhiBbol, Lboloptdata, sigmaBbold, sigmaBbolu, LBdata, bin_edges, zz)