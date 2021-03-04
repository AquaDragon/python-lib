'''
NAME:           rfea.py
AUTHOR:         swjtang
DATE:           03 Mar 2021
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

''' --------------------------------------------------------------------------
    DATA ANALYSIS FUNCTIONS
------------------------------------------------------------------------------
'''


def find_Ti(xx, yy, plot=0, width=40, xmax=0, xoff=10):
    # Find Ti from curve fitting of the decaying part of -dI/dV
    # xoff = offset used for curve fitting gaussian

    # Assume the peak of the gaussian is the maximum point
    iimax = np.argmax(yy)

    # Try searching for local minima (end point of curve fit)
    if xmax == 0:
        try_arr = np.arange(iimax, len(yy), width)
        prev = try_arr[0]
        for iitry in try_arr[1:]:
            iimin = iimax + np.argmin(yy[iimax:iitry])
            if (iimin < iitry) & (iimin == prev):
                break
            prev = iimin
    else:
        iimin = len(yy)-1

    def gauss_func(x, a, b, c, x0):
        return a * np.exp(-b * (x-x0)**2) + c

    if plot != 0:
        plt.plot(xx[max(iimax-xoff, 0)], yy[max(iimax-xoff, 0)], 'o')
        plt.plot(xx[iimin], yy[iimin], 'o')

    guess = [yy[iimax]-yy[iimin], 0.2, yy[iimin], xx[iimax]]

    popt, pcov = curve_fit(gauss_func, xx[max(iimax-xoff, 0):iimin],
                           yy[max(iimax-xoff, 0):iimin], p0=guess)

    # Plot the points if desired
    if plot != 0:
        plt.plot(xx[max(iimax-xoff, 0):iimin],
                 gauss_func(xx[max(iimax-xoff, 0):iimin], *popt), '--')

    # Returns full width of gaussian; b = 1/kT = 1/(2*sigma^2)
    return popt, pcov  # (1/np.sqrt(*popt[1]))


# Calculates the current derivative -dI/dV
def IVderiv(volt, curr, scale=1, nwindow=31, polyn=2, **kwargs):
    return -tbx.smooth(np.gradient(tbx.smooth(curr, nwindow=nwindow,
                       polyn=polyn, **kwargs)), nwindow=nwindow, polyn=polyn,
                       **kwargs)*scale


''' --------------------------------------------------------------------------
    PLOTTING FUNCTIONS
------------------------------------------------------------------------------
'''


def plot_IVderiv(volt, curr=None, deriv=None, xoff=0, yoff=0, nwindow=31,
                 polyn=2, **kwargs):
    # Plots the current derivative -dI/dV
    # curr/deriv: specify if the input is a current trace or -dI/dV
    # curr would be 2nd input if specified without keyword
    if deriv is None:
        plt.plot(volt+xoff, IVderiv(volt, curr, nwindow=nwindow, polyn=polyn) +
                 yoff, **kwargs)
    elif curr is None:
        plt.plot(volt+xoff, deriv+yoff, **kwargs)

''' --------------------------------------------------------------------------
    NAMING FUNCTIONS
------------------------------------------------------------------------------
'''


def fname_gen(series, date='2021-01-28', folder='/data/swjtang/RFEA/', ch=2):
    # Plots the current derivative -dI/dV
    return '{0}{1}/C{2}{3}.txt'.format(folder, date, ch, series)
