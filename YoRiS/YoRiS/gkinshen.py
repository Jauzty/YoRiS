# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:46:50 2024

@author: aust_
"""
import numpy as np
from astropy.cosmology import Planck18 as cosmo
import matplotlib.pyplot as plt
from scipy.stats import norm, pearsonr

alpha = 0.54  # constants taken from input.pro
beta = 22.1
ar = 0.7  # could consider different values of the exponent to change the slope of FRI/FRII
fcav = 4.0
bb = np.log10(7 * fcav) + 36.0
aa = 0.68
alpha_opt = 0.5
filename = r'C:\Users\aust_\YoRiS\shennew.txt'

# Load the data, handling missing values
data = np.genfromtxt(filename, usecols=(0, 1, 2, 3, 4, 5), delimiter='|', skip_header=3, filling_values=0)

# Extract the columns
redshift = data[:, 0]
rfflux6 = data[:, 3]  # 5GHz in mJy
loglbol = data[:, 1]
flag = data[:, 2]
logl5100 = data[:, 4]
ledd = data[:, 5]
rffluxJy = rfflux6 / 1000  # now in Jy
LB5100 = np.log10(3500) + logl5100

# Convert redshift to distance using astropy
distances = cosmo.luminosity_distance(redshift).value * 10**6  # distances in parsecs (pc)
# Calculate the luminosities in erg/s/Hz
Lum = 4 * np.pi * distances**2 * (1 + redshift)**2 * 9.5 * 10**13 * rffluxJy  # not frequency dependent
Lum = Lum * 5e9  # now Lum is in erg/s

radio_detected_mask = (Lum != 0) & (loglbol != 0) & (flag == 1) # & (logl5100 != 0)
Lum_non_zero = np.log10(Lum[radio_detected_mask])
LBdata = loglbol[radio_detected_mask]
LB5100 = LB5100[radio_detected_mask]
Edd = ledd[radio_detected_mask]
L5 = Lum_non_zero
redshift_detected = redshift[radio_detected_mask]

Ledd = 10**LBdata/10**Edd
BHmass = Ledd/(1.3*10**38)
print(BHmass)

# Define redshift bins
redshift_bins = [0, 0.5, 0.9, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2]

for i in range(len(redshift_bins) - 1):
    z_min = redshift_bins[i]
    z_max = redshift_bins[i + 1]
    
    # Mask for the current redshift bin
    bin_mask = (redshift_detected >= z_min) & (redshift_detected < z_max)
    print(f"Number of sources in redshift bin {z_min}-{z_max}: {np.sum(bin_mask)}")
    
    L5_bin = L5[bin_mask]
    LBdata_bin = LBdata[bin_mask]
    Edd_bin = Edd[bin_mask]
    
    # Calculate kin
    kin_all = []
    for _ in range(100):
        kin = alpha * (L5_bin) + beta + np.random.normal(0, 0.47, len(L5_bin))
        kin_all.append(kin)

    kin_all = np.array(kin_all)
    average_kin = np.mean(kin_all, axis=0)

    kkin_all = []
    L5r_bin = L5_bin - 7 - 9 - np.log10(5)
    L14_bin = L5r_bin + ar * np.log10(1.4 / 5)
    for _ in range(100):
        kkin = aa * (L14_bin - 25) + bb + 7 + np.random.normal(0, 0.7, len(L5_bin))
        kkin_all.append(kkin)

    kkin_all = np.array(kkin_all)
    average_kkin = np.mean(kkin_all, axis=0)
    
    gk_bin = np.log10(10**average_kkin / 10**LBdata_bin)
    
    # Filter out non-finite values
    gk_valid_bin = gk_bin[np.isfinite(gk_bin)]
    Edd_valid_bin = Edd_bin[np.isfinite(gk_bin)]

    
    # Fit a Gaussian curve
    mu, std = norm.fit(gk_valid_bin)
    
    # Calculate the correlation coefficient if there's enough data
    if len(gk_valid_bin) >= 2 and len(Edd_valid_bin) >= 2:
        correlation_coefficient, _ = pearsonr(Edd_valid_bin, gk_valid_bin)
    else:
        correlation_coefficient = np.nan
        print(f"Not enough data for Pearson correlation in redshift bin {z_min}-{z_max}")
        
    mu = 10**mu
    std = std/np.sqrt(np.sum(bin_mask))
    
    # Print results for the current redshift bin
    print(f"Redshift bin: {z_min}-{z_max}")
    print(f"  Mean of gk: {mu:.3f}")
    print(f"  Mean error on gk: {std:.3f}")
    print(f"  Correlation coefficient (gk vs Eddington Ratio): {correlation_coefficient:.3f}\n")