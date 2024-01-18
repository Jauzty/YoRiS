# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:34:07 2024

@author: aust_
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.integrate
from Ueda_Updated_Py import Ueda_14, Ueda_mod

def Pradio(Lx, L_r, z):
    #Pradio calculates the probability of having a value Lr given a value Lx
    #this is case 5 we can aslo try case 2 and 4 to see if this effects our gk
    #Norm = 1.0230 #case 5
    #Norm = 1.0620 #case 2
    Norm = 1.0652 #case 4
    #g_r = 0.369 #case 5
    #g_r = 0.476 #case 2
    g_r = 0.429 #case 4
    #g_l = 1.69 #case 5
    #g_l = 1.93 #case 2
    g_l = 1.7 #case 4
    #Rc = -4.578 #case 5
    #Rc = -4.313 #case 2
    Rc = -4.386 #case 4
    #al = 0.109 #case 5
    #al= 0 #case 2
    al = 0.056 #case 4
    #az = -0.066 #case 5
    az = 0 #case 2/4
    A = np.sqrt(g_r / g_l)
    # Rx is the discrepancy between radio luminosity and X-ray luminosity for each source
    Rx = L_r - Lx  # Calculate Rx for each L_r

    z0 = max(0.3, min(3.0, z))
    Lx0 = np.clip(Lx, 42.2, 47.0)
    R0 = Rc * (al * (Lx0 - 44) + 1) * (az * (z0 - 0.5) + 1)  # Calculate R0

    DENN1 = 1 + ((R0 - Rx) / g_l) ** 4
    DENN2 = 1 + ((Rx - R0) / g_r) ** 2

    if np.isscalar(Rx):
        if Rx < R0:
            PP = Norm / (A * np.pi * g_l * DENN1)
        else:
            PP = (A * Norm) / (np.pi * g_r * DENN2)
    else:
        PP = np.zeros_like(Rx)
        ix = Rx < R0
        PP[ix] = Norm / (A * np.pi * g_l * DENN1[ix])
        PP[~ix] = (A * Norm) / (np.pi * g_r * DENN2[~ix])

    # where Rx >= 1 PP = 0 because the condition Rx>-1 produces an RLF that over-predicts the number of bright radio sources
    if np.isscalar(Rx) and Rx >= -1:
        PP = 0

    return PP


Lmin=41.0
nr=500
Rmax=-1.
Rmin=-10.
Rx=np.linspace(Rmin, Rmax, nr)
l_wat=np.linspace(15.,30.,nr) #W Hz-1, monochromatic, log units
Lr=l_wat+7.+np.log10(1.4)+9.  #erg/s, 1.4L_1.4 log unit
z=1.0


nr=Lr.size
Phir=np.zeros(nr)
temp=np.zeros(nr)

#P=np.zeros(nr)

for s in range(0,nr):
    temp=Lr[s]-Rx
    Lx=temp[::-1] #reverse, to put in crescent order
#    print(temp[-4:])
#    print(Lx)
#    print(Lx[np.where(Lx < Lmin)])

    #now calculate the XLF given by ueda14.py, considering the whole X-ray AGN population
    f=np.zeros(nr)
    for i in range(0,5): 
        f=f+Ueda_14(Lx,z,i)
        f[(Lx < Lmin)]=0
        #print (i,f)   
    
    P = Pradio( Lx, Lr[s],z )

    Phir[s]=scipy.integrate.simps(P*f,Lx)


fig, ax = plt.subplots(figsize=(10, 7))

plt.plot(l_wat,np.log10(Phir), label='z=%f' %z)
leg = plt.legend(loc='upper right')
ax.grid()
plt.xlim(16.,28.)
plt.ylim(-10, -3)
ax.set_xlabel('$ \log L_{1.4 GHz} [W Hz^{-1}]  $')
ax.set_ylabel('$\Phi(L_{1.4 GHz}) [Mpc^{-3} dex^{-1}]$')
