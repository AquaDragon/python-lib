'''
NAME:           rfea.py
AUTHOR:         swjtang
DATE:           13 Feb 2021
DESCRIPTION:    A toolbox of functions related to energy analyzer analysis.
------------------------------------------------------------------------------
to reload module:
import importlib
importlib.reload(<module>)
------------------------------------------------------------------------------
'''
import h5py
import importlib
import numpy as np
import re
import scipy
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

import lib.toolbox as tbx


def find_Ti(xx, yy, plot=0):
    # Find Ti from curve fitting of the decaying part of -dI/dV
    xoff = 100    # offset used for curve fitting gaussian
    
    # Assume the peak of the gaussian is the maximum point
    iimax = np.where((yy == np.max(yy)))[0][0]

    # Try searching for local minima (end point of curve fit)
    try_arr = np.arange(iimax, len(yy), 40)
    prev = try_arr[0]
    for iitry in try_arr[1:]:
        iimin = iimax + np.argmin(yy[iimax:iitry])
        if (iimin < iitry) & (iimin == prev):
            break
        prev = iimin

    def gauss_func(x, a, b, c, x0):
        return a * np.exp(-b * (x-x0)**2) + c

    if plot != 0:
        plt.plot(xx[max(iimax-xoff,0)], yy[max(iimax-xoff,0)], 'o')
        plt.plot(xx[iimin], yy[iimin], 'o')
    
    guess = [yy[iimax]-yy[iimin], 0.2, yy[iimin], xx[iimax]]

    popt, pcov = curve_fit(gauss_func, xx[max(iimax-xoff,0):iimin],
                           yy[max(iimax-xoff,0):iimin], p0=guess)

    # Plot the points if desired    
    if plot != 0:
        plt.plot(xx[max(iimax-xoff,0):iimin], 
                 gauss_func(xx[max(iimax-xoff,0):iimin], *popt))

    # Returns width of gaussian; b = 1/(2*sigma^2)
    return(popt)#1/np.sqrt(2*popt[1]))