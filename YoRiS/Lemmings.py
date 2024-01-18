# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:33:12 2023

@author: aust_
"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
from QLFs_to_duty_cycles import myBolfunc
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

t1 = time()

filename = r'C:\Users\aust_\YoRiS\LeMMINGs.txt'

data = np.genfromtxt(filename, skip_footer=(20))
l_opt = data[:, 5]
l_rad_core = data[:, 6]

non_zero_mask = (l_opt != 0)

# Apply boolean indexing to both l_opt and l_rad_core
l_opt_non_zero = l_opt[non_zero_mask]
l_rad_core_non_zero = l_rad_core[non_zero_mask]

#converting optical L_OIII luminosity into bolometric luminosity. OIII is 5007 A
#using bolometric correction from Spinoglio+2023
#LBdata = 3.31 + 0.88*l_opt_non_zero
#using bolometric correction from Runnoe+2012
#LBdata = 4.89 + 0.91*l_opt_non_zero
#using Nemmen and Brotherton +2010
#LBdata = 11.7 + 0.76*l_opt_non_zero
#using heckman et al 2004
LBdata = np.log10(3500) + l_opt_non_zero

#Using Merloni Heinz 2007 and Heckman and Best 2014 to take core lrad to lkin 
#LeMMINGs is 1.5 GHz not 1.4 GHz for rad and starts out as erg/s
alpha = 0.54#constants taken from input.pro
beta = 22.1
ar = 0.7 #could consider different values of the exponent to change the slope of FRI/FRII
fcav = 4.0
bb = np.log10(7 * fcav) + 36.0
aa = 0.68
#convert lemmings radio luminosity to w/hz @ 1.5ghz
LW = l_rad_core_non_zero - 7 - 9 - np.log10(1.5)
# Convert radio luminosity LW from W/Hz @ 1.5GHz to W/Hz @ 5GHz
L5 = LW - ar * np.log10(5 / 1.5)  # L5 is in W/Hz @ 5GHz / 1.5 not 1.4
#convert back to erg/s
L5 = L5 + 7 + 9 + np.log10(5)
# Calculate kinetic luminosity kin, Merloni and Heinz 2007
kin = alpha * (L5) + beta   
#calculate kkin using Heckman and Best +2014
kkin = aa * (LW - 25) + bb + 7  # kkin is in erg/s
print(kkin)
#calculate gk on source by source basis
gk = (10**kkin / 10**LBdata)
print(min(gk), max(gk))

# Define bins for histogram
#bins = np.arange(42, 58, 0.2)
#bins = np.arange(36, 46, 0.2)

# Create histogram without normalization
#hist, bin_edges = np.histogram(LBdata, bins=bins)
#hist, bin_edges = np.histogram(kin, bins=bins)

# Apply normalization to the histogram
#plt.hist(LBdata, bins=bins, edgecolor='black')
#plt.hist(kin, bins=bins, edgecolor='black')

# Define bins for histogram
bins = np.arange(0, 5, 0.1 )

# Create histogram without normalization
hist, bin_edges = np.histogram(gk, bins=bins)

# Check if gk is valid
gk_valid = gk[np.isfinite(gk)]

# Fit a Gaussian (normal) distribution to the valid data
mu, std = norm.fit(gk_valid)

# Plot the histogram
plt.hist(gk, bins=bins, density=True, alpha=0.75, color='g', edgecolor='black')

# Plot the fitted Gaussian curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

# Customize the plot
plt.title('LeMMINGs $g_{k}$ Histogram')
plt.xlabel('Kinetic to Bolometric ratio')
plt.ylabel('Number of Galaxies')
plt.legend(['Fit ($\mu={:.2f}$, $\sigma={:.2f}$)'.format(mu, std), 'Histogram'], fontsize = 10)

# Calculate FWHM 
fwhm = 2 * np.sqrt(2 * np.log(2)) * std

# Display the FWHM on the plot
#plt.text(0.835, 0.9, f'FWHM = {fwhm:.3f}', fontsize=12, color='red')

# Reshape the data for compatibility with sklearn's KDE
'''gk_valid = gk_valid.reshape(-1, 1)

# Fit the KDE
kde = KernelDensity(bandwidth=0.012)  # You can adjust the bandwidth
kde.fit(gk_valid)

# Generate data points for the KDE curve
x_vals = np.linspace(min(gk_valid), max(gk_valid), 1000).reshape(-1, 1)
log_dens = kde.score_samples(x_vals)

# Plot the histogram
plt.hist(gk, bins=bins, density=True, alpha=0.6, color='g', edgecolor='black', label='Histogram')

# Plot the KDE curve
plt.plot(x_vals, np.exp(log_dens), 'r', linewidth=2, label='KDE')

# Customize the plot
plt.title('LeMMINGs $g_{k}$ Histogram with KDE')
plt.xlabel('Kinetic to Bolometric ratio')
plt.ylabel('Galaxy Number Density')

# Add a legend
plt.legend()'''

# Add horizontal grid lines with custom spacing
plt.grid(axis='y', which='major', linestyle='-', linewidth='0.5')  # Major grid lines
plt.grid(axis='y', which='minor', linestyle=':', linewidth='0.5')

plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 1000

plt.show()

# Calculate the midpoint of each bin
bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2

# Calculate the average value of gk using the histogram counts as weights
average_gk = np.average(bin_midpoints, weights=hist)

# Print or use the average_gk value as needed
print(f'Average gk: {average_gk}')
print(f'FWHM: {fwhm}')

t2 = time()
print(f'Time in minutes: {(t2 - t1) / 60}')
