# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 13:40:48 2023

@author: aust_
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
from Ueda_Updated_Py import Ueda_mod, Ueda_14
from scipy.io import readsav

#FRI frac list optical
FRIfracopt = [0.0747402, 0.0684142, 0.0622467, 0.0882428, 0.0516555, 0.0832443, 0.1088309, 0.0600215, 0.0308876, 0.0509483, 0.0133806, 0.2078514]
#FRII frac list
FRIIfracopt = [0.0237775, 0.0314957, 0.0234358, 0.0174848, 0.0134677, 0.0158409, 0.0088633, 0.0125791, 0.0021857, 0.0257798, 0.0179971, 0.0]
#FracALL list
Fracallopt = [0.0985177, 0.0999099, 0.0856825, 0.0657518, 0.0651232, 0.0990852, 0.1176942, 0.0699071, 0.0321780, 0.0570827, 0.0313777, 0.2078514]

#FRI frac list xray
FRIfracX = [0.0754413, 0.0715513, 0.0629219, 0.0638171, 0.0527830, 0.0948788, 0.1290664, 0.0723229, 0.0323717, 0.1618970, 0.0168045, 0.2352269]
#FRII frac list
FRIIfracX = [0.0308062, 0.0565392, 0.0465047, 0.0291714, 0.0165781, 0.0254845, 0.0162047, 0.0194296, 0.0030720, 0.0420260, 0.0217945, 0.0]
#FracALL list
FracALLX = [0.1062475, 0.0996258, 0.0877799, 0.0929886, 0.0631226, 0.1203633, 0.1376054, 0.0879205, 0.0336231, 0.1653578, 0.0385990, 0.2352269]

def dutyjiang(z, M2500):
    b_0 = -0.132
    b_z = -2.052
    b_M = -0.183

    frac_qq = b_0 + b_z * np.log10(1 + z) + b_M * (M2500 + 26)
    dutyfrac = frac_qq
    frac_qq = 0

    return dutyfrac

def poisson_gehrels(k):
    # Input k is a vector containing nonnegative integer elements.
    # This function calculates the sigma resulting from the one-sided Poisson CL
    # (equivalent to Gaussian statistics 1 sigma) up to k = 10 following Gehrels 1986.
    if isinstance(k, (int, float)):
        k = np.array([k])
    n = len(k)
    p_up = [1.841, 3.3, 4.638, 5.918, 7.163, 8.382, 9.584, 10.77, 11.95, 13.11, 14.27]
    p_lw = [0., 0.173, 0.708, 1.367, 2.086, 2.84, 3.62, 4.419, 5.232, 6.057, 6.891]
    res = np.zeros((n, 2))

    for i in range(n):
        if k[i] > 10:
            res_u = np.sqrt(k[i])
            res_l = res_u
        else:
            pp = int(k[i])
            res_u = p_up[pp] - k[i]
            res_l = k[i] - p_lw[pp]
        res[i, 0] = res_u
        res[i, 1] = res_l

    return res

def mi_to_L2500(mi2, PhiMi):
    # mi2 in input are Mi(z=2) magnitudes
    # PhiMi is the optical LF defined as a function of Mi(z=2), already in log
    # the output lgL2500 is in log, erg/s/Hz

    # Richards+06:
    a = np.log10(4 * np.pi)
    b = (19.0 * 2.0) + 2.0 * np.log10(3.08)  # this term is d^2, where d=10pc=3.08e19 cm
    mag = mi2 + 48.60 + 2.5 * np.log10(1 + 2)
    lgL2500 = a + b - 0.4 * mag  # erg/s/Hz

    Phi_L2500 = PhiMi - np.log10(0.4)  # the jacobian of the transformation is np.log10(0.4)

    return lgL2500, Phi_L2500

def L2500_to_kev(lgl2500, Phi_L2500):
    # Go from lgl2500 to lgL2keV using equations from Lusso+10 and Comastri+95.
    # Both luminosities in input are in erg/s/Hz, and the final luminosities are in erg/s.
    # The input quasars LF are already in log units.

    lgL2kevmed = 0.760 * lgl2500 + 3.508
    lgL2kevmed = lgL2kevmed + 18.0 - np.log10(4.1357)  # conversion to erg/s/keV, L in log
    alphaX = -0.8
    den = 1.0 - 0.8  # alphaX + 1
    lgL2_10kev = lgL2kevmed - alphaX * np.log10(2.0) + np.log10(10**den - 2**den) - np.log10(den)

    Phi_Lx = Phi_L2500 - np.log10(0.760)  # jacobian of the transformation

    return lgL2_10kev, Phi_Lx

def myBolfunc(TAG, L):
    # Bolometric Correction Function
    # input L is in ergs/s, this is the bolometric Luminosity
    
    Ls = L - np.log10(4.0) - 33.0  # to convert to solar Luminosities

    if TAG == 1:
        # 1: Marconi+04 per X 2-10 kev
        L_qq = Ls - 1.54 - (0.24 * (Ls - 12.0)) - (0.012 * (Ls - 12.0)**2) + (0.0015 * (Ls - 12.0)**3)
        L_qq = L_qq + 33.0 + np.log10(4.0)  # convert back to erg/s
        L_x = L_qq
        L_qq = 0.0
        return L_x

    elif TAG == 2:
        # 2: Marconi+04 per OPT B band
        L_qq = Ls - 0.80 + (0.067 * (Ls - 12.0)) - (0.017 * (Ls - 12.0)**2) + (0.0023 * (Ls - 12.0)**3)
        L_qq = L_qq + 33.0 + np.log10(4.0)  # convert back to erg/s
        L_B = L_qq
        L_qq = 0.0
        return L_B
    
    elif TAG == 3: #should really be in reverse func because input is x ray not lbol
        #Duras+2020 hard x ray bolometric correction as a function of the bolometric luminosity
        a = 10.96 # +- 0.06
        b = 11.93 # +- 0.01
        c = 17.79 # +- 0.1
        L_1 = a * (1 + (Ls/b)**c)
        L_C= L_1 + 33 + np.log10(4)
        return L_C # scatter is 0.27
    
    elif TAG == 4: #same here
        #Duras+2020 hard x ray bolometric correction as a function of the hard x ray luminosity
        a = 15.33 # +- 0.06
        b = 11.48 # +- 0.01
        c = 16.20 # +- 0.16
        L_2 = a * (1 + ((Ls)/b)**c)
        L_D= L_2 + 33 + np.log10(4)
        return L_D # scatter is 0.37
    
def reverse_myBolfunc(TAG, L_bol): #opposite of myBolfunc
    
    Ls = L_bol - np.log10(4.0) - 33.0  # to convert to solar Luminosities

    if TAG == 1:
        # 1: Marconi+04 per X 2-10 kev
        L_qq = Ls + 1.54 + (0.24 * (Ls - 12.0)) + (0.012 * (Ls - 12.0)**2) - (0.0015 * (Ls - 12.0)**3)
        L_qq = L_qq + 33.0 + np.log10(4.0)  # convert back to erg/s
        L_x = L_qq
        L_qq = 0.0
        return L_x

    elif TAG == 2:
        # 2: Marconi+04 per OPT B band
        L_qq = Ls + 0.80 - (0.067 * (Ls - 12.0)) + (0.017 * (Ls - 12.0)**2) - (0.0023 * (Ls - 12.0)**3)
        L_qq = L_qq + 33.0 + np.log10(4.0)  # convert back to erg/s
        L_B = L_qq
        L_qq = 0.0
        return L_B

Lunif = np.linspace(43, 47, 10)

def dutySDSSshen(Lunif, l_2500, PhiBbol, Lboloptdata, sigmaBbold, sigmaBbolu, LBdata):
     #we want to see the contribution of radio sources in the Phibol - Lbol plane
     #coming from the optical QSO, also in the Phibol-optical and xray planes

    alpha_opt = 0.5
    #extract data from the two relevant files
    data = readsav(r'C:\Users\aust_\YoRiS\shen_catalogue.sav')
    z = data['z']
    first_FR_type = data['first_FR_type']
    Lbol = data['lbol'] #change to lopt and Lx for qso lf and xray lf
    LogL3000 = data['logl3000']
    LogL1350 = data['logl1350']
    LogL5100 = data['logl5100']
    file_path = r'C:\Users\aust_\\YoRiS\vanvelzen.txt'
    data2 = np.genfromtxt(file_path, names=['bol', 'fracII'], dtype=None, delimiter=None)
    bol = data2['bol']
    fracII = data2['fracII'] / 100.0
    print(first_FR_type)
    

    X = Lbol #assuming tagbol == 0 this is the simplest way to create the X array

    z1 = z - 0.35
    z2 = z + 0.35
    nl = len(Lunif)
    #define the bin edges where z inbetween two consecutive values is all kept together
    bin_edges = [0.3, 0.68, 1.06, 1.44, 1.82, 2.2, 2.6, 3.0, 3.5, 4.0, 4.5, 5.0, 6.5]
    
    # Create an array of bin assignments for each element in z
    zbin = np.digitize(z, bin_edges)
    
    # Initialize an array to store the indices of data points in each bin
    zbin_indices = []
    if __name__ == "__main__":
        fig, axes = plt.subplots(3, 4, figsize=(22, 14), sharex = True, sharey = True)
        axes = axes.flatten()
        
    # Create separate arrays for each bin
    for bin_num in range(1, 13):
        if __name__ == "__main__":
            ax = axes[bin_num-1]
        bin_indices = np.where(zbin == bin_num)[0]  # Get the indices for the current bin
        zbin_indices.append(bin_indices)
        zvalues = z[bin_indices]
        first_FR_type_values = first_FR_type[bin_indices]
        z1 = zvalues - 0.35
        z2 = zvalues + 0.35
        Lbol_vals = Lbol[bin_indices]
        Lbol_vals = Lbol_vals[Lbol_vals != 0]
        #print(f'Lbol in bin {bin_num} is: {min(Lbol_vals)}, {max(Lbol_vals)}')
        
        #if bin_num <= 6:
            #X = LogL3000 #take 3000A straight from shen catalogue
        #elif bin_num > 6: #at high redshift the correction must be applied slightly differently
            #xx = LogL1350 #take 1350A from shen catalogue
            #xx = xx - np.log10(2.2207)-15
            #X = xx + alpha_opt*np.log10(9.9931/2.2207)
            #X = X + np.log10(9.9931)+14 #makes it back to 3000A
        
        #LogL2500 = X - alpha_opt*np.log10(9.9931/11.992)# convert from 3000 to 2500
        #LogL2500 = LogL2500 - 14 - np.log10(9.9931) #from erg/s to erg/s/hz required for input into function
        #LogL2kev, b = L2500_to_kev(LogL2500, Phi_l_2500) #convert to 2-10kev x ray
        #X = LogL2kev
        
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
            ratioall[kx] = (B / float(A + B))
            ratioI[kx] = (C / float(A + B))
            ratioII[kx] = (D / (A + B))
        
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
        print(ratiodata)
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
        xmin = 43.9
        xmax = 46.9
        avg1 = np.nanmean(ratioI)
        avg2 = np.nanmean(ratioII)
        avgall = np.nanmean(ratioall)
        #print(f'in bin {bin_num} avg1 is {avg1}, avg2 is {avg2}, avgall is {avgall}')
        
        if __name__ == "__main__":
            #plots the datapoints
            ax.scatter(Lunif, ratioall, marker='o', label='ALL', color='blue')
            ax.scatter(Lunif, ratioI, marker='s', label='FRI', color='green')
            ax.scatter(Lunif, ratioII, marker='^', label='FRII', color='red')
            ax.scatter(Lunif, ratiojiang, marker='*', label='Radio Loud Jiang+07', color='purple')
            
            #plots their error bars
            for xi, yi, y_err_lower_i, y_err_upper_i in zip(Lunif, ratioall, sigALL[:, 0], sigALL[:, 1]):
                ax.errorbar(xi, yi, yerr=[[y_err_lower_i], [y_err_upper_i]], fmt='none', capsize=3, color='blue')
            for xi, yi, y_err_lower_i, y_err_upper_i in zip(Lunif, ratioI, sigI[:, 0], sigI[:, 1]):
                ax.errorbar(xi, yi, yerr=[[y_err_lower_i], [y_err_upper_i]], fmt='none', capsize=3, color='green')
            for xi, yi, y_err_lower_i, y_err_upper_i in zip(Lunif, ratioII, sigII[:, 0], sigII[:, 1]):
                ax.errorbar(xi, yi, yerr=[[y_err_lower_i], [y_err_upper_i]], fmt='none', capsize=3, color='red')
            for xi, yi, y_err_lower_i, y_err_upper_i in zip(Lunif, ratiojiang, sigJiang[:], sigJiang[:]):
                ax.errorbar(xi, yi, yerr=[[y_err_lower_i], [y_err_upper_i]], fmt='none', capsize=3, color='purple')
        
            if bin_num >= 9:# sets the axes labels correctly, 3 on the y and 4 on the x axis
                ax.set_xlabel('log L_opt(3000A) [erg/s]')
            if bin_num % 4 == 1:
                ax.set_ylabel('f_rad')
                
            ax.set_yscale('log')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(0.001, 0.9)
            if bin_num ==1:# take only the legend from one figure to put at the top of the plot    
                handles, labels = ax.get_legend_handles_labels()
                zleg = 0.5#define a z for each legend so that they are kept separate to make less clunky
            elif bin_num ==2:
                zleg = 0.9
            elif bin_num ==3:
                zleg = 1.2
            elif bin_num ==4:
                zleg = 1.6
            elif bin_num ==5:
                zleg = 2.0
            elif bin_num ==6:
                zleg = 2.4
            elif bin_num ==7:
                zleg = 2.8
            elif bin_num ==8:
                zleg = 3.2
            elif bin_num ==9:
                zleg = 3.8
            elif bin_num ==10:
                zleg = 4.2
            elif bin_num ==11:
                zleg = 4.8
            elif bin_num ==12:
                zleg = 5.8
                
            ax.legend([f'z = {zleg}'], loc='lower left', handlelength=0)
    
            ax.grid()
    if __name__ == "__main__":
        plt.rcParams['font.size'] = 24
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['figure.dpi'] = 300
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        lines_labels = [ax.get_legend_handles_labels()]#fixes clunkiness
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]#^
        fig.legend(lines, labels, loc='upper center', ncol=4)#^
        plt.show()
        
    return PhiradioII


#equivalent to compxopt 
file_list = glob.glob(r'C:\Users\aust_\YoRiS\QLFS\QLF*.txt')
file_list.sort()
#anyone else using this will have to change to their correct directory
Lx = np.linspace(41, 49, 1000)
Lbol = np.linspace(40.0, 50, 1000)

if __name__ == "__main__":
    fig, axes = plt.subplots(3, 4, figsize=(22, 14), sharex = True, sharey = True)
    axes = axes.flatten()

for i, file_name in enumerate(file_list): #take the qlf files and separate them based on their z
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
    Phi_21 = Ueda_14(Lx, z, 1) + Phi_20 #upto nh<22
    Phi_22 = Ueda_14(Lx, z, 2) + Phi_21 #upto nh<23
    Phi_23 = Ueda_14(Lx, z, 3) + Phi_22 #upto nh<24
    Phi_24 = Ueda_14(Lx, z, 4) + Phi_23 #upto nh<26 (CTK)
    
    #converting the XLF into bolometric using K-correction
    Lxx=myBolfunc(1,Lx)
    kx=Lx-Lxx
    kxx=np.interp(Lx,Lxx,kx)
    Lboll=kxx+Lx
    Phixbol20=Phi_20*np.abs(np.gradient(Lxx,Lx)) #just absorbtion of NH <21
    Phixbol21=Phi_21*np.abs(np.gradient(Lxx,Lx))
    Phixbol22=Phi_22*np.abs(np.gradient(Lxx,Lx))
    Phixbol23=Phi_23*np.abs(np.gradient(Lxx,Lx))
    Phixbol24=Phi_24*np.abs(np.gradient(Lxx,Lx)) #sum of all absorbtions
    
    Phixbol20frac = Phixbol20*FracALLX[i] #log for comparison and account for fraction of FRI/FRII radio sources in the sample
    Phixbol21frac = Phixbol21*FracALLX[i]
    Phixbol22frac = Phixbol22*FracALLX[i]
    Phixbol23frac = Phixbol23*FracALLX[i]
    Phixbol24frac = Phixbol24*FracALLX[i]
    
    #print(np.abs(np.gradient(Lboll,Lx)))
    #converting the QLF into the B-band bolometric luminosity
    alpha_opt = 0.5
    LB=myBolfunc(2,l_kev)
    kb=l_kev-LB
    LBdata=l_2500+alpha_opt*np.log10(6.7369/11.992) #use spectral index to convert to bolometric
    LBdata=LBdata + 14 + np.log10(6.7369)
    kbb=np.interp(LBdata,LB,kb)
    Lboloptdata=kbb+LBdata
    PhiB=(10**Phi_l_2500)*np.abs(np.gradient(LBdata,l_2500))
    PhiBd=(10**Phi_l_2500l)*np.abs(np.gradient(LBdata,l_2500))
    PhiBu=(10**Phi_l_2500u)*np.abs(np.gradient(LBdata,l_2500))
    PhiBbol=PhiB*np.abs(np.gradient(Lboloptdata,LBdata))*((Fracallopt[i])) #account for the fraction of sources in the optical sample
    PhiBbold=PhiBd*np.abs(np.gradient(Lboloptdata,LBdata))*Fracallopt[i]
    PhiBbolu=PhiBu*np.abs(np.gradient(Lboloptdata,LBdata))*Fracallopt[i]
    sigmaBbold= PhiBbol - PhiBbold
    sigmaBbolu= PhiBbolu - PhiBbol
    logPhi = np.log10(PhiBbol)
    log_phi_pos_err = np.abs(sigmaBbolu / (PhiBbol * np.log(10)))
    log_phi_neg_err = np.abs(sigmaBbold / (PhiBbol * np.log(10)))
    log_phi_pos_err = np.where(np.isnan(log_phi_pos_err), np.nanmean(log_phi_pos_err), log_phi_pos_err)
    log_phi_pos_err = np.where(log_phi_pos_err == 0, np.nanmean(log_phi_pos_err), log_phi_pos_err)
    log_phi_neg_err = np.where(np.isnan(log_phi_neg_err), np.nanmean(log_phi_neg_err), log_phi_neg_err)
    log_phi_neg_err = np.where(log_phi_neg_err == 0, np.nanmean(log_phi_neg_err), log_phi_neg_err)

    if __name__ == "__main__":
        ax = axes[i]
        ax.plot(Lboll, np.log10(Phixbol20frac), color='black', linestyle='-', label='NH<21')
        ax.plot(Lboll, np.log10(Phixbol21frac), color='red', linestyle='--', label='NH<22')
        ax.plot(Lboll, np.log10(Phixbol22frac), color='purple', linestyle='-.', label='NH<23')
        ax.plot(Lboll, np.log10(Phixbol23frac), color='green', linestyle=':', label='NH<24')
        ax.plot(Lboll, np.log10(Phixbol24frac), color='orange', linestyle=(0, (2, 1)), label='NH<26')
        ax.scatter(Lboloptdata, np.log10(PhiBbol), marker='o', color='navy', s=30, edgecolors='black', label=f'QSO LFs at z = {z}')
        for xi, yi, y_err_lower_i, y_err_upper_i in zip(Lboloptdata, np.log10(PhiBbol), log_phi_neg_err, log_phi_pos_err):
            ax.errorbar(xi, yi, yerr=[[y_err_lower_i], [y_err_upper_i]], fmt='none', capsize=5, color='navy')
        # sets the axes labels correctly, 3 on the y and 4 on the x axis
        if i // 4 == 2:
            ax.set_xlabel('$\log L_{bol} [erg s^{-1}]$')
        if i % 4 == 0:
            ax.set_ylabel('$\log \Phi(L_{bol}) [Mpc^{-3} dex^{-1}]$')
        ax.set_ylim(-11.5, -3.5)
        ax.set_xlim(44, 48.5)
        ax.legend(loc='lower left', fontsize=16, fancybox = True, framealpha = 0.0)
        ax.grid(True)

#if __name__ == "__main__":    
 #   plt.rcParams['font.size'] = 21
  #  plt.rcParams['font.family'] = 'serif'
   # plt.rcParams['figure.dpi'] = 300
    #plt.tight_layout()
    #plt.subplots_adjust(wspace=0.0, hspace=0.0)
    #plt.show()
if __name__  == "__main__":
    dutySDSSshen(Lunif, l_2500, PhiBbol, Lboloptdata, sigmaBbold, sigmaBbolu, LBdata)
