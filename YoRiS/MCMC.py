# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:50:36 2024

@author: aust_
"""

import numpy as np
from matplotlib import pyplot as plt
import emcee
import corner
from scipy.stats import norm

Lrr = np.linspace(30, 47, 1000) #erg/s at 5GHz
LR = Lrr -7-np.log10(5)-9 #luminosity range in W/Hz 5 GHz based on federicas code

alpha = 0.54#constants taken from input.pro
beta = 22.1
ar = 0.7 #could consider different values of the exponent to change the slope of FRI/FRII
fcav = 4.0
bb = np.log10(7 * fcav) + 36.0
aa = 0.68
# Convert radio luminosity LR from W/Hz @ 1.4GHz to W/Hz @ 5GHz
L5 = LR - ar * np.log10(5 / 1.4)  # L5 is in W/Hz @ 5GHz
# Convert L5 to erg/s @ 5GHz
L5r = L5 + 7 + 9 + np.log10(5)  # L5r is in erg/s @ 5GHz
# Calculate kinetic luminosity kin
kin = alpha * (L5r) + beta   
# Calculate kinetic luminosity kkin
kkin = aa * (LR - 25) + bb + 7.0  # kkin is in erg/s

def find_crossing_points(y_values, threshold):
    # Find indices where y values cross the threshold
    crossing_indices = np.where(np.diff((y_values < threshold).astype(int)) != 0)[0]

    return crossing_indices

def loglike(theta, Lboloptdata, err, PhiBbol, Phikin21conv):
# Log liklihood function
# vtot data should be the bol lum
# model will be the lf
        gk, scatter = theta[0], theta[1]
        nk = len(kin)
        LgLbol = np.zeros(nk)
        
        for io in range(nk):
            LgLbol[io] = -np.log10(gk) -np.log10(scatter) + kin[io]
            
        #cut_index = find_crossing_points(Phikin21conv, 1e-10)
        #midpoint = (LgLbol[cut_index[:-1]] + LgLbol[cut_index[1:]]) / 2.01
        #cut_point = np.argmax(LgLbol >= midpoint)
        
        #Phikin21conv = Phikin21conv[cut_point:]
        #LgLbol = LgLbol[cut_point:]
        
        #max_Phikin21conv = np.max(Phikin21conv)
        #indices_to_keep = PhiBbol <= max_Phikin21conv  
        # Filters indices where PhiBbol is not higher than max Phikin21conv
    
        #Lboloptdata = Lboloptdata[indices_to_keep]
        #PhiBbol = PhiBbol[indices_to_keep]
        
        y_interp = np.interp(Lboloptdata, LgLbol, Phikin21conv)
        y = np.log10(y_interp)
        PhiBbol = np.log10(PhiBbol)
        # Find indices where y values are 25
        #deleted_indices_y = np.where(y == 25)[0]
        
        # Delete values in y array that are 25
        #y = y[y != 25]
        
        # Delete corresponding points in PhiBbol array
        #PhiBbol = np.delete(PhiBbol, deleted_indices_y)
        #err = err[indices_to_keep]
        
        return -0.5*np.sum(np.power(y - PhiBbol, 2) / np.power(err, 2)) 
   
def logprior(theta):
    gk, scatter = theta  # Access the first element of the 2D array
    #prior_mean_gk = 0.15
    #prior_stddev_gk = 0.05
    #log_prior_gk = norm.logpdf(gk, loc=prior_mean_gk, scale=prior_stddev_gk)
    
    # No need to compute log_prior_scatter since it's not used

    if 0.01 < gk < 3 and 0.01 < scatter < 10:
        return 0
    else:
        return -np.inf
       
def logprob(theta, Lboloptdata, err, PhiBbol, Phikin21conv):
    p = theta[0], theta[1]
    p_input = np.array([])  # Pass theta as a 2D array
    p_input = np.append(p_input, p)
    lp = logprior(p_input)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(theta, Lboloptdata, err, PhiBbol, Phikin21conv)
   
def MCMC_Fit(Lboloptdata, err, PhiBbol, Phikin21conv):
    ndim, nwalkers = 2, 200 
    popt = 0.15 # starting positions, can set to whatever you want
    pos = [popt + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)] # add random noise to each position
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, args=(Lboloptdata, err, PhiBbol, Phikin21conv), a=2, threads=20) # MCMC sampler, I use an older version of emcee
    samples = sampler.run_mcmc(pos, 1000, progress=True) # set number of steps (10,000 here)
    samples = sampler.chain[:, 100:, :].reshape((-1, ndim)) # remove burn-in (1,000 points here)
    sampler_chain = sampler.get_chain(discard=0) # extract the chain
    labels = ["$g_{k}$", "Scatter"]  # set parameter labels

    # plot chains
    #fig, axes = plt.subplots(ndim, sharex=True, figsize=(8, 9))
    #for i in range(ndim):
        #ax = axes[i]
        #ax.plot(sampler_chain[:, :, i], "k", alpha=0.4)
        #ax.set_xlim(0, len(sampler_chain) + 100)  # Adjusted x-axis limits for burn-in
        #ax.set_ylabel(labels[i])
        #ax.set_xlabel("step number")
        #ax.yaxis.set_label_coords(-0.1, 0.5)
    #fig.tight_layout()

    #plot contours with 1 sigma percentiles
    corner.corner(samples, labels= labels, show_titles=True, quantiles=[0.16, 0.5, 0.84], title_quantiles=[0.16, 0.5, 0.84])
    plt.show()

    # Extract parameter statistics
    median_gk = np.round(np.median(sampler.flatchain[:, 0]), 5)
    lower_gk, upper_gk = np.round(np.percentile(sampler.flatchain[:, 0], [16, 84]), 5)

    gk_stats = (median_gk, upper_gk - median_gk, median_gk - lower_gk)
    return gk_stats, sampler_chain, labels, samples, sampler