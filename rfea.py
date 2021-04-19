'''
NAME:           rfea.py
AUTHOR:         swjtang
DATE:           19 Apr 2021
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

import lib.find_multiref_phase as fmp
import lib.toolbox as tbx
import lib.spikes as spike


''' --------------------------------------------------------------------------
    DATA ANALYSIS FUNCTIONS
------------------------------------------------------------------------------
'''
# Find Ti from curve fitting of the decaying part of -dI/dV
def find_Ti(xx, yy, plot=0, width=40, xmax=0, xoff=10):
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

    if plot != 0:
        plt.plot(xx[max(iimax-xoff, 0)], yy[max(iimax-xoff, 0)], 'o')
        plt.plot(xx[iimin], yy[iimin], 'o')

    guess = [yy[iimax]-yy[iimin], 1, yy[iimin], xx[iimax]]

    popt, pcov = curve_fit(gauss_func, xx[max(iimax-xoff, 0):iimin],
                           yy[max(iimax-xoff, 0):iimin], p0=guess)

    # Plot the points if desired
    if plot != 0:
        plt.plot(xx[max(iimax-xoff, 0):iimin],
                 gauss_func(xx[max(iimax-xoff, 0):iimin], *popt), '--')

    # Returns full width of gaussian; b = 1/kT = 1/(2*sigma^2)
    return popt, pcov  # (1/np.sqrt(*popt[1]))


# Finds Ti from an exponential plot; The curve starts from some max value then
# decays exponentially.
def find_Ti_exp(volt, curr, plot=0, startpx=100, endpx=100, mstime=0, fid=0,
                save=1):
    '''
    startpx = number of pixels to count at the start to determine max value
    endpx   = number of pixels to count at the end to determine min value
    '''
    # Smooth the curve
    temp = tbx.smooth(curr, nwindow=11)

    # Take gradient to find peak of dI/dV, thats where to cut the exp fit off
    gradient = np.gradient(tbx.smooth(temp, nwindow=51))
    argmin = np.argmin(gradient)

    # Determine start and end values of the curve
    start = np.mean(temp[:startpx])
    end = np.mean(temp[-endpx:])

    guess = [start-end, 2, end, volt[argmin]]
    bound_down = [0.1*(start-end), 0, end-start, volt[0]]
    bound_up = [+np.inf, 50, start, volt[-1]]
    try:
        popt, pcov = curve_fit(exp_func, volt[argmin:], temp[argmin:], p0=guess,
                               bounds=(bound_down, bound_up))
    except:
        return None, None, None

    Vp = volt[np.argmin(abs([start for ii in volt] - exp_func(volt, *popt)))]
    Ti = popt[1]
    Ti_err = np.sqrt(np.diag(pcov))[1]

    if plot != 0:
        tbx.prefig(xlabel='peak pulse potential [V]', 
                   ylabel='current [$\mu$A]')
        plt.plot(volt, temp, label='{0} ms'.format(mstime))
        plt.title('{0} exponential fit, t = {1:.2f} ms'.format(fid, mstime),
                  fontsize = 20)
        plt.plot(volt[argmin:], temp[argmin:])
        plt.plot(volt, [start for ii in volt], '--')
        plt.plot(volt, [end for ii in volt], '--')
        plt.plot(volt[argmin-20:], exp_func(volt[argmin-20:], *popt), '--')
        plt.ylim(np.min(temp)*0.96, np.max(temp)*1.04)
        if save == 1:
            tbx.savefig('./img/{0}-currpeak-exp-fit-{1:.2f}ms.png'.format(
                        fid, mstime))

    return Vp, Ti, Ti_err


# Calculates the current derivative -dI/dV
def IVderiv(curr, scale=1, nwindow=51, polyn=3, **kwargs):
    smoo1 = tbx.smooth(curr, nwindow=nwindow, polyn=polyn, **kwargs)
    grad1 = -np.gradient(smoo1)
    smoo2 = tbx.smooth(grad1, nwindow=nwindow, polyn=polyn, **kwargs)
    return smoo2*scale
    # return -tbx.smooth(np.gradient(tbx.smooth(curr, nwindow=nwindow,
    #                    polyn=polyn, **kwargs)), nwindow=nwindow, polyn=polyn,
    #                    **kwargs)*scale


# Given arrays of (1) step number, (2) argtime and (3) peak current
# corresponding to each peak, returns peaks that fulfil input conditions
def select_peaks(step_arr, argtime_arr, peakcurr_arr, step=0, trange=[0,1]):
    ind = np.where((argtime_arr > trange[0]) & (argtime_arr <= trange[1]) &
                   (step_arr == step))
    return peakcurr_arr[ind]


# Determines the actual time (in ms) from the start of the discharge
def mstime(time, ind, start=5, off=0):
    # start: [ms] the start time of the trigger
    # off:   [px] the offset of the analyzed slice of data
    return time[int(ind)+off]*1e3 + start


''' --------------------------------------------------------------------------
    ANALYSIS ROUTINES
------------------------------------------------------------------------------
'''
def condavg_rfea(data, bdata, nsteps, nshots, trange=None, btrange=None,
                 ref=[0, 0], bref=None):
    ''' ----------------------------------------------------------------------
    Conditionally avarage shift of RFEA current data.
    INPUTS:   data    = np.array with the data to be conditionally averaged.
              bdata   = np.array with the phase information (usually bdot)
              nsteps  = Number of steps in the voltage sweep
              nshots  = Number of shots for each step in the voltage sweep
              trange  = Time range to store conditionally averaged data.
              btrange = Time range of the conditional averaging (bdot)
    OPTIONAL: ref = [step, shot] number of the reference shot
              bref = Inputs a reference shot for conditional averaging
    '''
    if trange is not None:
        t1, t2 = trange
    else:
        t1, t2 = 0, data.shape[0]

    if btrange is not None:
        bt1, bt2 = btrange
    else:
        bt1, bt2 = t1, t2

    # Current array, shifted in phase
    curr_arr = np.zeros((t2-t1, nsteps, nshots))
    # Array shows number of shots skipped because cross-correlation fails
    skip_arr = np.zeros(nsteps)

    # Determine the reference shot in bdata
    if bref is None:
        bref = bdata[bt1:bt2, ref[0], ref[1]]

    for step in range(nsteps):
        skips = 0
        for shot in range(nshots):
            tbx.progress_bar([step, shot], [nsteps, nshots], ['nsteps',
                             'nshots'])
            bsig = bdata[bt1:bt2, step, shot]
            xlag = fmp.lagtime(bref, bsig, quiet=1, threshold=0.7)['xlag']

            if xlag is not None:
                curr_arr[:, step, shot] = np.roll(data[t1:t2, step, shot, 0],
                                                  -xlag)
            else:
                skips += 1
        skip_arr[step] = skips

    # Calculates factor so that mean_curr takes mean of shots not skipped
    factor = [nshots/(nshots - ii) for ii in skip_arr]
    mean_curr = np.mean(curr_arr, axis=2) * factor

    return mean_curr, curr_arr, bref


def calc_Ti_arr(volt, curr, res=1, trange=None, dt=1, **kwargs):
    ''' ----------------------------------------------------------------------
    Calculate Ti and Vp from an array of RFEA IV curves
    INPUTS:   volt = np.array of voltage data
              curr = np.array of current data
    OPTIONAL: res    = [Ohm] Resistance used to measure current
              trange = [start, stop] time indices
              dt     = Number of indices to skip 
    '''
    if trange is not None:
        t1, t2 = trange
    else:
        t1, t2 = 0, curr.shape[0]

    tarr = np.arange(t1, t2, dt)
    nt = tarr.shape[0]

    # Define arrays
    Ti = np.empty(nt)
    Vp = np.empty(nt)
    errTi = np.empty(nt)

    for ii in range(nt):
        tt = tarr[ii]
        tbx.progress_bar([ii], [nt])
        Vp[ii], Ti[ii], errTi[ii] = find_Ti_exp(volt, curr[tt,:]/res, plot=0,
                                                save=0, **kwargs)
    return Ti, Vp, errTi

''' --------------------------------------------------------------------------
    PLOTTING FUNCTIONS
------------------------------------------------------------------------------
'''
# Plots the current derivative -dI/dV (if input is deriv do a regular plot)
def plot_IVderiv(volt, curr, xoff=0, yoff=0, nwindow=51, polyn=3, **kwargs):
    plt.plot(volt + xoff, IVderiv(curr, nwindow=nwindow, polyn=polyn) +
             yoff, **kwargs)


# Browses the data and returns the selected trange
def browse_data(data, x=None, y=None, step=0, shot=0, chan=0, trange=[0,-1]):
    t1 = trange[0]
    t2 = np.min([data.shape[0], trange[1]])
    tbx.prefig(xlabel='time [px]', ylabel='magnitude')

    if (x == None) and (y == None):
        temp = data[:, step, shot, chan]
        plt.title('step = {0}, shot = {1}, chan = {2}, trange = [{3}, {4}]'.\
            format(step, shot, chan, trange[0], trange[1]), fontsize=20)
    else:
        temp = data[:, x, y, step, shot, chan]
        plt.title('(x, y) = ({0:.2f}, {1:.2f}), step = {2}, shot = {3}, '
            ' chan = {4}, trange = [{5}, {6}]'.format(
            x, y, step, shot, chan, trange[0], trange[1]), fontsize=20)
    
    plt.plot(temp)
    plt.plot([t1,t1], [np.min(temp), np.max(temp)], 'orange')
    plt.plot([t2,t2], [np.min(temp), np.max(temp)], 'orange')

    return t1, t2

# Plot function to show the bounded region of integrated B used for condavg
def show_bint_range(data, trange=None, fid=0):
    tbx.prefig(xlabel='time [px]', ylabel='B-int')
    plt.plot(data)
    if trange is not None:
        t1, t2 = trange
        plt.plot([t1,t1], [np.min(data), np.max(data)], 'orange')
        plt.plot([t2,t2], [np.min(data), np.max(data)], 'orange')
        tbx.savefig('./img/{0}-condavg-range.png'.format(fid))
    else:
        t1, t2 = 0, len(data)
    return t1, t2


''' --------------------------------------------------------------------------
    NAMING FUNCTIONS
------------------------------------------------------------------------------
'''
def fname_gen(series, date='2021-01-28', folder='/data/swjtang/RFEA/', ch=2):
    # Plots the current derivative -dI/dV
    return '{0}{1}/C{2}{3}.txt'.format(folder, date, ch, series)


''' --------------------------------------------------------------------------
    BASIC MATH FUNCTIONS
------------------------------------------------------------------------------
'''
def gauss_func(x, a, b, c, x0):
    return a * np.exp(-b * (np.sqrt(x)-np.sqrt(x0))**2) + c


def exp_func(x, a, b, c, x0):
    return a * np.exp(-(x-x0)/b) + c
