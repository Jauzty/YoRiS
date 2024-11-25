# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:33:12 2023

@author: aust_
"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy.stats import pearsonr
from QLFs_to_duty_cycles import myBolfunc, reverse_myBolfunc

t1 = time()

filename = r'C:\Users\aust_\YoRiS\LeMMINGs.txt'
filename2 = r'C:\Users\aust_\YoRiS\LeMMINGs_xray.txt'

data = np.genfromtxt(filename, skip_footer=(21))
l_x = data[:, 10]
l_opt = data[:, 5]
l_rad_core = data[:, 6]
morph = data[:, 9]
sigma = data[:, 2]
mbulge = data[:, 3]
mgal = data[:, 4]
radiuskpc = data[:, 8]

#creating masks
x_ray_detected_mask = (l_x != 0)
xopt_mask = (l_x != 0) & (l_opt != 0)
radio_detected_mask = (morph != 0) & (l_opt != 0) # all galaxies that had the core detected in radio
undetected_mask = (morph == 0) & (l_opt != 0) # all galaxies that were undetected core radio
core_identified_mask = (morph != 0) & (morph != 6) & (l_opt != 0) #makes up nearly all of the radio_detected so not worth plotting
core_unidentified_mask = (morph == 6) & (l_opt != 0) #only 4 of them so not worth plotting
class_A_mask = (morph == 1) & (l_opt != 0) #A has enough galaxies for meaningful results the rest do not so they are also grouped together
class_B_mask = (morph == 2) & (l_opt != 0)
class_C_mask = (morph == 3) & (l_opt != 0)
class_D_mask = (morph == 4) & (l_opt != 0)
class_E_mask = (morph == 5) & (l_opt != 0)
not_A_mask = (morph != 0) & (morph != 1) & (morph != 6) & (l_opt != 0) #BCDE merged together as they are all small
non_zero_mask = (l_opt != 0)
#masks for the plots against mbulge, sigma, radius, they all have 0 values individually
bulgemask = (mbulge != 0) & (l_opt != 0)
sigmamask = (sigma != 0) & (l_opt != 0) & (l_x != 0 )
radiusmask = (radiuskpc != 0) & (l_opt != 0)
mgalmask = (mgal != 0) & (l_opt != 0) & (l_x != 0)

# Apply boolean indexing to both l_opt and l_rad_core
xray_non_zero = l_x[x_ray_detected_mask]
xopt_nonzero = l_x[xopt_mask]
xoptsigma_nonzero = l_x[sigmamask]
xoptmgal_nonzero = l_x[mgalmask]
optx_nonzero = l_opt[xopt_mask]
optxmgal_nonzero = l_opt[mgalmask]
optxsigma_nonzero = l_opt[sigmamask]

l_opt_non_zero = l_opt[non_zero_mask]
l_rad_core_non_zero = l_rad_core[non_zero_mask]
#apply also the morphology masks
l_opt_radio_detected = l_opt[radio_detected_mask]
l_rad_radio_detected = l_rad_core[radio_detected_mask]

l_rad_undetected = l_rad_core[undetected_mask]
l_opt_undetected = l_opt[undetected_mask]

l_rad_core_ident = l_rad_core[core_identified_mask]
l_opt_core_ident = l_opt[core_identified_mask]

l_rad_core_unident = l_rad_core[core_unidentified_mask]
l_opt_core_unident = l_opt[core_unidentified_mask]

l_rad_A = l_rad_core[class_A_mask]
l_opt_A = l_opt[class_A_mask]

l_rad_B = l_rad_core[class_B_mask] #not enough sources
l_opt_B = l_opt[class_B_mask]

l_rad_C = l_rad_core[class_C_mask] #not enough sources
l_opt_C = l_opt[class_C_mask]

l_rad_D = l_rad_core[class_D_mask]
l_opt_D = l_opt[class_D_mask]

l_rad_E = l_rad_core[class_E_mask]
l_opt_E = l_opt[class_E_mask]

l_rad_BCDE = l_rad_core[not_A_mask]
l_opt_BCDE = l_opt[not_A_mask]

l_rad_sigma = l_rad_core[sigmamask]
l_opt_sigma = l_opt[sigmamask]

l_rad_bulge = l_rad_core[bulgemask]
l_opt_bulge = l_opt[bulgemask]

l_rad_radius = l_rad_core[radiusmask]
l_opt_radius = l_opt[radiusmask]

l_rad_mgal = l_rad_core[mgalmask]
l_opt_mgal = l_opt[mgalmask]

sigmanonzero = sigma[sigmamask]
mbulgenonzero = mbulge[bulgemask]
radiusnonzero = radiuskpc[radiusmask]
mgalnonzero = mgal[mgalmask]

logBHmass = 8.32 + 5.64*np.log10(sigmanonzero/200) #Mconnell +2013
#logBHmass = np.log10(sigmanonzero**5.4) #Van Den Bosch +2016

logBHmass2 = 7.45 + 1.05*np.log10(10**mgalnonzero/1e11)



#converting optical L_OIII luminosity into bolometric luminosity. OIII is 5007 A
#using bolometric correction from Spinoglio+2023
#LBdata = 3.31 + 0.88*l_opt_non_zero
#using bolometric correction from Runnoe+2012
#LBdata = 4.89 + 0.91*l_opt_non_zero
#using Nemmen and Brotherton +2010
#LBdata = 11.7 + 0.76*l_opt_non_zero
#using heckman et al 2004
LBdata = np.log10(3500) + l_opt_non_zero
LBdataopt = np.log10(3500) + optx_nonzero
LBdataoptmgal = np.log10(3500) + optxmgal_nonzero
LBdataoptsigma = np.log10(3500) + optxsigma_nonzero
LBdataxray = reverse_myBolfunc(1, xopt_nonzero)
LBdataxraysigma = reverse_myBolfunc(1, xoptsigma_nonzero)
LBdataxraymgal = reverse_myBolfunc(1, xoptmgal_nonzero)

LBdataradiodetected = np.log10(3500) + l_opt_radio_detected
LBdataundetected = np.log10(3500) + l_opt_undetected
LBdataA = np.log10(3500) + l_opt_A
LBdataB = np.log10(3500) + l_opt_B
LBdataC = np.log10(3500) + l_opt_C
LBdataD = np.log10(3500) + l_opt_D
LBdataE = np.log10(3500) + l_opt_E
LBdataBCDE = np.log10(3500) + l_opt_BCDE
LBdatasigma = np.log10(3500) + xoptsigma_nonzero
LBdatabulge = np.log10(3500) + l_opt_bulge
LBdataradius = np.log10(3500) + l_opt_radius
LBdatamgal = np.log10(3500) + xoptmgal_nonzero
Eddratio = np.log10(10**LBdataoptsigma/10**logBHmass)
Eddratio2 = np.log10(10**LBdataoptmgal/10**logBHmass2)
print(len(Eddratio))
print(len(Eddratio2))

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
LWradiodetected = l_rad_radio_detected - 7 - 9 - np.log10(1.5)
LWundetected = l_rad_undetected - 7 - 9 - np.log10(1.5)
LWA = l_rad_A - 7 - 9 - np.log10(1.5)
LWB = l_rad_B - 7 - 9 - np.log10(1.5)
LWC = l_rad_C - 7 - 9 - np.log10(1.5)
LWD = l_rad_D - 7 - 9 - np.log10(1.5)
LWE = l_rad_E - 7 - 9 - np.log10(1.5)
LWBCDE = l_rad_BCDE - 7 - 9 - np.log10(1.5)
LWradius = l_rad_radius - 7 - 9 - np.log10(1.5)
LWbulge = l_rad_bulge - 7 - 9 - np.log10(1.5)
LWsigma = l_rad_sigma - 7 - 9 - np.log10(1.5)
LWmgal = l_rad_mgal - 7 - 9 - np.log10(1.5)
# Convert radio luminosity LW from W/Hz @ 1.5GHz to W/Hz @ 5GHz
L5 = LW - ar * np.log10(5 / 1.5)  # L5 is in W/Hz @ 5GHz / 1.5 not 1.4
L5radiodetected = LWradiodetected - ar * np.log10(5 / 1.5)
L5undetected = LWundetected - ar * np.log10(5 / 1.5)
L5A = LWA - ar * np.log10(5 / 1.5)
L5B = LWB - ar * np.log10(5 / 1.5)
L5C = LWC - ar * np.log10(5 / 1.5)
L5D = LWD - ar * np.log10(5 / 1.5)
L5E = LWE - ar * np.log10(5 / 1.5)
L5BCDE = LWBCDE - ar * np.log10(5 / 1.5)
L5radius = LWradius - ar * np.log10(5 / 1.5)
L5sigma = LWsigma - ar * np.log10(5 / 1.5)
L5bulge = LWbulge - ar * np.log10(5 / 1.5)
L5mgal = LWmgal - ar * np.log10(5 / 1.5)
#convert back to erg/s
L5 = L5 + 7 + 9 + np.log10(5)
L5radiodetected = L5radiodetected + 7 + 9 + np.log10(5)
L5undetected = L5undetected + 7 + 9 + np.log10(5)
L5A = L5A + 7 + 9 + np.log10(5)
L5B = L5B + 7 + 9 + np.log10(5)
L5C = L5C + 7 + 9 + np.log10(5)
L5D = L5D + 7 + 9 + np.log10(5)
L5E = L5E + 7 + 9 + np.log10(5)
L5BCDE = L5BCDE + 7 + 9 + np.log10(5)
L5radius = L5radius + 7 + 9 + np.log10(5)
L5sigma = L5sigma + 7 + 9 + np.log10(5)
L5bulge = L5bulge + 7 + 9 + np.log10(5)
L5mgal = L5mgal + 7 + 9 + np.log10(5)
# Calculate kinetic luminosity kin, Merloni and Heinz 2007
#include a gaussian scatter of 0.47 in the scaling from LR to LK
kin_all = []
for _ in range(100):    
    kin = alpha * (L5sigma) + beta + np.random.normal(0, 0.47, len(L5sigma)) #discuss scatter make gk dist more stable use shen for source by source gk
    kin_all.append(kin)
    
kin_all = np.array(kin_all)
average_kin = np.mean(kin_all, axis=0)
    
kinradiodetected = alpha * (L5radiodetected) + beta  
kinundetected = alpha * (L5undetected) + beta  
kinA = alpha * (L5A) + beta  
kinB = alpha * (L5B) + beta 
kinC = alpha * (L5C) + beta 
kinD = alpha * (L5D) + beta 
kinE = alpha * (L5E) + beta 
kinBCDE = alpha * (L5BCDE) + beta 
kinradius = alpha * (L5radius) + beta
kinsigma = alpha * (L5sigma) + beta
kinbulge = alpha * (L5bulge) + beta   
kinmgal = alpha * (L5mgal) + beta
#calculate kkin using Heckman and Best +2014
#include a gaussian scatter of 0.7 when scaling LR to LK and average over stochastic method

kkin_all = []
for _ in range(100):
    # Calculate kkin with random noise
    kkin = aa * (LW - 25) + bb + 7 + np.random.normal(0, 0.7, len(LW))
    # Append the kkin array to the list
    kkin_all.append(kkin)
    
kkin_all = np.array(kkin_all)
average_kkin = np.mean(kkin_all, axis=0)

kkinradiodetected = aa * (LWradiodetected - 25) + bb + 7
kkinundetected = aa * (LWundetected - 25) + bb + 7
kkinA = aa * (LWA - 25) + bb + 7
kkinB = aa * (LWB - 25) + bb + 7
kkinC = aa * (LWC - 25) + bb + 7
kkinD = aa * (LWD - 25) + bb + 7
kkinE = aa * (LWE - 25) + bb + 7
kkinBCDE = aa * (LWBCDE - 25) + bb + 7
kkinbulge = aa * (LWbulge - 25) + bb + 7
kkinradius = aa * (LWradius - 25) + bb + 7
kkinsigma = aa * (LWsigma - 25) + bb + 7
kkinmgal = aa * (LWmgal - 25) + bb + 7
#calculate gk on source by source basis
gk = np.log10(10**average_kin / 10**LBdataoptsigma)
gk2 = np.log10(10**average_kkin / 10**LBdata)
gksigma = np.log10(10**kinsigma / 10**LBdatasigma)
gkradius = np.log10(10**kkinradius / 10**LBdataradius)
gkbulge = np.log10(10**kkinbulge / 10**LBdatabulge)
gkmgal = np.log10(10**kinmgal / 10**LBdatamgal)

# Define bins for histogram
#bins = np.arange(42, 58, 0.2)
#bins = np.arange(36, 46, 0.3)

# Create histogram without normalization
#hist, bin_edges = np.histogram(LBdata, bins=bins)
#hist, bin_edges = np.histogram(kkin, bins=bins)

# Apply normalization to the histogram
#plt.hist(LBdata, bins=bins, edgecolor='black')
#plt.hist(kkin, bins=bins, edgecolor='black')

# Define bins for histogram
bins = np.arange(-4, 2, 0.2)

# Create histogram without normalization
hist, bin_edges = np.histogram(gk, bins=bins)

# Check if gk is valid
gk_valid = gk[np.isfinite(gk)]

# Fit a Gaussian (normal) distribution to the valid data
mu, std = norm.fit(gk_valid)

# Plotting the histogram
plt.figure(figsize=(10, 8))
plt.hist(gk, bins=bins, density=True, alpha=0.75, color='b', edgecolor='black')
# Plot the fitted Gaussian curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.axvline(x=-1.301, color='red', linestyle='--', linewidth=2, label='Typical Value at $g_{k}$ = 0.05')
# Customize the plot
plt.legend(['Fit ($\mu={:.2f}$, $\sigma={:.2f}$)'.format(mu, std), 'Typical Value $g_{k}$ = 5%'], fontsize=12)
plt.title(' $g_{k}$ for class A sources')

# Customize the plot
plt.xlabel('Log Kinetic to Bolometric Efficiency $g_{k}$')
plt.ylabel('Number density of Galaxies')
plt.grid(axis='y', which='major', linestyle='-', linewidth='0.5')  # Major grid lines
plt.grid(axis='y', which='minor', linestyle=':', linewidth='0.5')

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 1000

plt.show()
"""
correlation_coefficient, _ = pearsonr(Lbdataxray, LBdataopt)
plt.figure(figsize=(8, 6))
plt.scatter(Lbdataxray, LBdataopt, alpha=0.7)
plt.title('Lbol from OIII  vs Lbol from X-Ray')
plt.xlabel('Log Lbol from X-Ray')
plt.ylabel('Log Lbol from OIII ')
plt.text(0.1, 0.1, f'Correlation: {correlation_coefficient:.2f}', transform=plt.gca().transAxes, color='red')

# Calculate the slope and intercept of the regression line
slope, intercept = np.polyfit(Lbdataxray, LBdataopt, 1)
# Create an array of x values for the regression line
x_values = np.linspace(min(Lbdataxray), max(Lbdataxray), 100)
# Calculate the corresponding y values using the regression equation
y_values = slope * x_values + intercept
plt.plot(x_values, y_values, color='black', label='Regression Line')

plt.show()
"""
"""
# Scatter plot for gk vs Bulge Mass
correlation_coefficient, _ = pearsonr(mbulgenonzero, gkbulge)
plt.figure(figsize=(8, 6))
plt.scatter(mbulgenonzero, gkbulge, alpha=0.7)
plt.title('$g_k$ vs Bulge Mass')
plt.xlabel('Log Bulge Mass')
plt.ylabel('$g_k$')
plt.text(0.1, 0.1, f'Correlation: {correlation_coefficient:.2f}', transform=plt.gca().transAxes, color='red')
plt.show()

# Scatter plot for gk vs Radius
plt.figure(figsize=(8, 6))
plt.scatter(np.log10(radiusnonzero), gkradius, alpha=0.7)
plt.title('$g_k$ vs Radius')
plt.xlabel('Log Radius (kpc)')
plt.ylabel('$g_k$')
plt.show()

# Scatter plot for gk vs Sigma
plt.figure(figsize=(8, 6))
plt.scatter(sigmanonzero, gksigma, alpha=0.7)
plt.title('$g_k$ vs Sigma')
plt.xlabel('Sigma')
plt.ylabel('$g_k$')
plt.show()
"""
correlation_coefficient2, _ = pearsonr(Eddratio, gksigma)

# Calculate the slope and intercept of the regression line
slope, intercept = np.polyfit(Eddratio, gksigma, 1)

# Create an array of x values for the regression line
x_values = np.linspace(min(Eddratio), max(Eddratio), 100)

# Calculate the corresponding y values using the regression equation
y_values = slope * x_values + intercept

# Plot the scatter plot for gk vs Eddratio from sigma
plt.figure(figsize=(10, 14))

plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
plt.scatter(Eddratio, gksigma, alpha=0.7, label='Data Points')
plt.plot(x_values, y_values, color='black', label='Regression Line')
plt.title('$g_k$ vs Eddington Ratio from $\sigma$')
plt.xlabel('$\log(L_{bol}/M_{BH})$ using OIII')
plt.ylabel('$Log g_k$ using X-Ray')
plt.text(0.1, 0.1, f'Correlation: {correlation_coefficient2:.2f}', transform=plt.gca().transAxes, color='red')
plt.legend(loc="upper right")

correlation_coefficient3, _ = pearsonr(Eddratio2, gkmgal)

# Calculate the slope and intercept of the regression line
slope, intercept = np.polyfit(Eddratio2, gkmgal, 1)

# Create an array of x values for the regression line
x_values = np.linspace(min(Eddratio2), max(Eddratio2), 100)

# Calculate the corresponding y values using the regression equation
y_values = slope * x_values + intercept

# Plot the scatter plot for gk vs Eddratio from Mgal
plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
plt.scatter(Eddratio2, gkmgal, alpha=0.7, label='Data Points')
plt.plot(x_values, y_values, color='black', label='Regression Line')
plt.title('$g_k$ vs Eddington Ratio from $M_{gal}$')
plt.xlabel('$\log(L_{bol}/M_{BH})$ using OIII')
plt.ylabel('$Log g_k$ using X-ray')
plt.text(0.1, 0.1, f'Correlation: {correlation_coefficient3:.2f}', transform=plt.gca().transAxes, color='red')
plt.legend(loc="upper right")

plt.tight_layout()  # Adjust layout for better spacing
plt.show()

# Calculate the midpoint of each bin
bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2

# Calculate the average value of gk using the histogram counts as weights
average_gk = np.average(bin_midpoints, weights=hist)

# Plot average_kin vs LBdata with color gradient for Edd
plt.figure(figsize=(8, 6))

# Scatter plot with color gradient for Edd
scatter = plt.scatter(LBdataoptsigma, average_kin, c=Eddratio, cmap='viridis', alpha=0.7, edgecolors='none', s=40)

# Calculate the correlation coefficient
correlation_coefficient_avg_kin, _ = pearsonr(LBdataoptsigma, average_kin)

# Calculate the slope and intercept of the regression line
slope_avg_kin, intercept_avg_kin = np.polyfit(LBdataoptsigma, average_kin, 1)

# Create an array of x values for the regression line
x_values_avg_kin = np.linspace(min(LBdataoptsigma), max(LBdataoptsigma), 100)

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
plt.title('Lkin vs Lbol with Edd ratio (LeMMINGs)')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()

t2 = time()
print(f'Time in minutes: {(t2 - t1) / 60}')
