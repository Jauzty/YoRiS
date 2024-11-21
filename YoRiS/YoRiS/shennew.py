# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:35:45 2024

@author: aust_
"""
import numpy as np
from astropy.cosmology import Planck18 as cosmo
import matplotlib.pyplot as plt
from scipy.stats import norm
from Ueda_Updated_Py import Ueda_mod, Ueda_14
from scipy.io import readsav
from scipy.stats import pearsonr

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

#using the edd ratio and lbol you can find eddington luminsoity, then divide by 1.3*10**38 ergs/s will give you solar masses
#in l functions we are driven mostly by the characteristic luminsosity of the function, driven by average active bh mass
#edd = lbol/ledd so ledd = lbol/edd


# Convert redshift to distance using astropy
distances = cosmo.luminosity_distance(redshift).value * 10**6  # distances in parsecs (pc)
# Calculate the luminosities in erg/s/Hz
Lum = 4 * np.pi * distances**2 * (1 + redshift)**2 * 9.5 * 10**13 * rffluxJy  # not frequency dependent
Lum = Lum * 5e9  # now Lum is in erg/s

radio_detected_mask = (Lum != 0) & (loglbol != 0) & (flag == 2) # & (logl5100 != 0)
FRImask = (flag == 1)
FRIImask = (flag == 2)
Lum_non_zero = np.log10(Lum[radio_detected_mask])
LBdata = loglbol[radio_detected_mask]
LB5100 = LB5100[radio_detected_mask]
Edd = ledd[radio_detected_mask]
L5 = Lum_non_zero
Ledd = LBdata/Edd
BHmass = Ledd/(1.3*10**38)
print(BHmass)

kin_all = []
for _ in range(100):
    kin = alpha * (L5) + beta + np.random.normal(0, 0.47, len(L5))  # discuss scatter make gk dist more stable use shen for source by source gk
    kin_all.append(kin)

kin_all = np.array(kin_all)
average_kin = np.mean(kin_all, axis=0)

kkin_all = []
L5r = L5 - 7 - 9 - np.log10(5)
L14 = L5r + ar * np.log10(1.4 / 5)
for _ in range(100):
    # Calculate kkin with random noise expects W/Hz
    kkin = aa * (L14 - 25) + bb + 7 + np.random.normal(0, 0.7, len(L5))
    # Append the kkin array to the list
    kkin_all.append(kkin)

kkin_all = np.array(kkin_all)
average_kkin = np.mean(kkin_all, axis=0)
print(min(LBdata), max(LBdata))

gk = np.log10(10**average_kkin / 10**LBdata)

# Filter out non-finite values
gk_valid = gk[np.isfinite(gk)]
Edd_valid = Edd[np.isfinite(gk)]

# Fit a Gaussian curve
mu, std = norm.fit(gk_valid)

# Plot histogram
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(gk_valid, bins=20, density=True, color='skyblue', edgecolor='black', alpha=0.7)

# Plot the Gaussian curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

# Add vertical line for typical value
plt.axvline(x=-1.301, color='red', linestyle='--', linewidth=2, label='Typical Value at $g_{k}$ = 0.05')

# Add labels, title, legend, and grid
plt.xlabel('log(gk)')
plt.ylabel('Density')
plt.title('Histogram of gk Values with Gaussian Fit')
plt.legend(['Fit ($\mu={:.2f}$, $\sigma={:.2f}$)'.format(mu, std), 'Typical Value $g_{k}$ = 5%'], fontsize=12, loc='upper right')
plt.xlim(-3, 2)
plt.grid(True)

# Plot gk vs Edd
plt.subplot(1, 2, 2)
correlation_coefficient, _ = pearsonr(Edd, gk)

# Calculate the slope and intercept of the regression line
slope, intercept = np.polyfit(Edd, gk, 1)

# Create an array of x values for the regression line
x_values = np.linspace(min(Edd), max(Edd), 100)

# Calculate the corresponding y values using the regression equation
y_values = slope * x_values + intercept
plt.scatter(Edd_valid, gk_valid, color='skyblue', alpha=0.7, s=1)
plt.plot(x_values, y_values, color='black', label='Regression Line')
plt.text(0.1, 0.1, f'Correlation: {correlation_coefficient:.2f}', transform=plt.gca().transAxes, color='red')
plt.xlabel('log(Eddington Ratio)')
plt.ylabel('log(gk)')
plt.title('gk vs Eddington Ratio')
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot average_kin vs LBdata with color gradient for Edd
plt.figure(figsize=(8, 6))

# Scatter plot with color gradient for Edd
scatter = plt.scatter(LBdata, average_kkin, c=Edd, cmap='viridis', alpha=0.7, edgecolors='none', s=10)

# Calculate the correlation coefficient
correlation_coefficient_avg_kin, _ = pearsonr(LBdata, average_kkin)

# Calculate the slope and intercept of the regression line
slope_avg_kin, intercept_avg_kin = np.polyfit(LBdata, average_kkin, 1)

# Create an array of x values for the regression line
x_values_avg_kin = np.linspace(min(LBdata), max(LBdata), 100)

# Calculate the corresponding y values using the regression equation
y_values_avg_kin = slope_avg_kin * x_values_avg_kin + intercept_avg_kin

# Plot the regression line
plt.plot(x_values_avg_kin, y_values_avg_kin, color='black', label='Regression Line')

# Add a color bar
cbar = plt.colorbar(scatter)
cbar.set_label('Eddington Ratio')

# Add text annotation for correlation coefficient
plt.text(0.05, 0.95, f'Correlation: {correlation_coefficient_avg_kin:.2f}', transform=plt.gca().transAxes, color='red', fontsize=12, verticalalignment='top')

# Add labels, title, legend, and grid
plt.xlabel('log(Lbol)')
plt.ylabel('log(lkin)')
plt.title('Lkin vs Lbol with Edd ratio (FRII)')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()







