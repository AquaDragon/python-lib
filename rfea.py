'''
NAME:           rfea.py
AUTHOR:         swjtang
DATE:           05 Feb 2023
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
import numpy.polynomial.polynomial as poly
import re
import scipy
from scipy.optimize import curve_fit
from matplotlib import animation, pyplot as plt

import lib.find_multiref_phase as fmp
import lib.fname_tds as fn
import lib.read_lapd_data as rd
import lib.toolbox as tbx
import lib.spikes as spike


class params(object):
    def __init__(self, fid, nsteps, nshots, res, ch_volt=0, ch_curr=3,
                 ch_bdot=None, ch_bint=None):
        self.fid = fid
        self.fname = fn.fname_tds(fid, old=0)
        self.nsteps = nsteps
        self.nshots = nshots
        self.res = res

        # Channel info
        self.ch_volt = ch_volt
        self.ch_curr = ch_curr
        self.ch_bdot = ch_bdot or ch_bint

        # Flag to determine if input channel is B-integrated
        if ch_bint is not None:
            self.f_bint = 1
        else:
            self.f_bint = 0

        # Store parameter data arrays
        self.volt = None
        self.time = None
        self.xpos = None
        self.ypos = None
        self.tarr = None         # time array for plotting Ti/Vp

        # Store dataset parameters
        self.nt = None
        self.dt = None
        self.t1, self.t2 = None, None    # [px] Area of interest
        self.bt1, self.bt2 = None, None  # [px] B-int bounds for Xcorrelation

    # Set analysis times
    def set_time(self, t1=None, t2=None, bt1=None, bt2=None):
        if t1 is not None:
            self.t1 = t1
        if t2 is not None:
            self.t2 = t2
        if bt1 is not None:
            self.bt1 = bt1
        if bt2 is not None:
            self.bt2 = bt2


''' ----------------------------------------------------------------------
    GET DATA METHODS
--------------------------------------------------------------------------
'''


class data():
    def __init__(self, obj):
        self.obj = obj

    # Get voltage step data
    def get_volt(self, quiet=0, rshot=None, **kwargs):
        if rshot is None:
            rshot = [1]
        dataset = rd.read_lapd_data(
            self.obj.fname, nsteps=self.obj.nsteps, nshots=self.obj.nshots,
            rchan=[self.obj.ch_volt], rshot=rshot, quiet=quiet, **kwargs)

        # check dimension of dataset output
        if len(np.shape(np.array(dataset['data']))) == 6:
            # Arr[nt, nx, ny, shot, chan, step]
            temp = np.transpose(dataset['data'], (0, 1, 2, 5, 3, 4))
            data = temp[:, 0, 0, ...]
        else:
            # Arr[nt, shot, chan, step]
            data = np.transpose(dataset['data'], (0, 3, 1, 2))

        # x100 = (50x from voltmeter, x2 from 1M/50 Ohm digitizer mismatch)
        self.obj.volt = np.mean(data[10000:35000, :, 0, 0]*100, axis=0)
        return self.obj.volt

    # Get current and bdot data
    def get_dataset(self, quiet=0, **kwargs):
        dataset = rd.read_lapd_data(
            self.obj.fname, nsteps=self.obj.nsteps, nshots=self.obj.nshots,
            rchan=[self.obj.ch_curr, self.obj.ch_bdot], quiet=quiet, **kwargs)

        datatemp = dataset['data']
        self.obj.nt = datatemp.shape[0]

        # check dimension of dataset output
        if len(np.shape(np.array(dataset['data']))) == 6:
            # Arr[nt, nx, ny, shot, chan, step]
            temp = np.transpose(dataset['data'], (0, 1, 2, 5, 3, 4))
            data = temp[:, 0, 0, ...]
        else:
            # Arr[nt, shot, chan, step]
            data = np.transpose(dataset['data'], (0, 3, 1, 2))

        if self.obj.ch_curr < self.obj.ch_bdot:
            curr = data[..., 0]
            bdot = data[..., 1]
        else:
            bdot = data[..., 0]
            curr = data[..., 1]

        self.obj.time = dataset['time']
        self.obj.dt = dataset['dt'][0]
        self.obj.xpos = dataset['x']
        self.obj.ypos = dataset['y']
        return curr, bdot

    # Get description of datarun
    def get_desc(self, **kwargs):
        dataset = rd.read_lapd_data(
            self.obj.fname, nsteps=1, nshots=1,
            rchan=[0], rshot=[0], quiet=1, **kwargs)
        print(dataset['desc'])

    ''' ----------------------------------------------------------------------
        ION TEMPERATURE (Ti) MANIPULATION
    --------------------------------------------------------------------------
    '''
    def calc_Ti_arr(self, volt, curr, dt=1, ca=0, **kwargs):
        ''' ---------------------------------------------------------
        Calculate Ti and Vp from an array of RFEA IV curves
        INPUTS:   volt = np.array of voltage data
                  curr = np.array of current data
        OPTIONAL: dt   = Number of indices to skip
        '''
        if ca != 0:
            t1, t2 = 0, curr.shape[0]
            tarr = np.arange(t1, t2, dt)
            self.obj.tarr = np.arange(self.obj.t1+t1, self.obj.t1+t2, dt)
        else:
            t1 = self.obj.t1 or 0
            t2 = self.obj.t2 or curr.shape[0]
            tarr = np.arange(t1, t2, dt)
            self.obj.tarr = tarr

        ntarr = tarr.shape[0]

        # Define arrays
        Ti = np.empty(ntarr)
        Vp = np.empty(ntarr)
        errTi = np.empty(ntarr)

        for ii in range(ntarr):
            tt = tarr[ii]
            tbx.progress_bar([ii], [ntarr])
            Vp[ii], Ti[ii], errTi[ii] = find_Ti_exp(
                volt, curr[tt, :]/self.obj.res, plot=0, save=0, **kwargs)
        return Ti, Vp, errTi

    def plot_TiVp(self, Ti, Vp, ca=0, limTi=20):
        tt = np.array([self.mstime(tt) for tt in self.obj.tarr])
        temp = np.where(Ti < limTi)

        text = 'conditional average, exponential fit'
        if ca == 1:
            text = 'YES ' + text
            svname = 'yes-condavg'
        else:
            text = 'NO ' + text
            svname = 'no-condavg'

        tbx.prefig(xlabel='time [ms]', ylabel='$T_i$ [eV]')
        plt.title('{0} $T_i$ vs time, {1} (all times)'.format(self.obj.fid,
                  text), fontsize=25)
        plt.plot(tt[temp], tbx.smooth(Ti, nwindow=51)[temp])
        tbx.savefig('./img/{0}-{1}-Ti-vs-time.png'.format(self.obj.fid,
                                                          svname))

        tbx.prefig(xlabel='time [ms]', ylabel='$V_p$ [V]')
        plt.title('{0} $V_p$ vs time, {1} (all times)'.format(self.obj.fid,
                  text), fontsize=25)
        plt.plot(tt, Vp)
        tbx.savefig('./img/{0}-{1}-Vp-vs-time.png'.format(self.obj.fid,
                                                          svname))

    ''' ----------------------------------------------------------------------
        B-DOT MANIPULATION
    --------------------------------------------------------------------------
    '''
    def integrate_bdot(self, bdot, axis=0):
        # Method to integrate the B-dot signal. Also checks if input is B-int.
        # if bdot is None:
        #     print("** Running method: get_dataset...")
        #     self.obj.get_dataset(quiet=1)
        if self.obj.f_bint == 1:
            print("** Input B-data is already integrated. Saving bint...")
            bint = bdot
        else:
            bint = tbx.bdot.bint(bdot, axis=axis)
        return bint

    def plot_bint_range(self, bint, step=0, shot=0):
        # Plot function to show the bounded region of integrated B used for
        # conditional averaging
        # INPUTS: bdata = 1D data array
        if bint is None:
            print("** Running method: integrate_bdot...")
            self.obj.integrate_bdot()

        bdata = bint[:, step, shot]

        tbx.prefig(xlabel='time [px]', ylabel='B-int')
        plt.plot(bdata)
        bt1 = self.obj.bt1 or 0
        bt2 = self.obj.bt2 or len(bdata)

        plt.plot([bt1, bt1], [np.min(bdata), np.max(bdata)], 'orange')
        plt.plot([bt2, bt2], [np.min(bdata), np.max(bdata)], 'orange')
        plt.title('integrated B, step={0}, shot={1}'.format(step, shot),
                  fontsize=20)
        tbx.savefig('./img/{0}-condavg-range.png'.format(self.obj.fid))

    def plot_bint_shift(self, bint, curr=None, step=0, shot=0, ref=None):
        if ref is None:
            ref = [0, 0]
        # Plots the reference bint/current with a test shot
        bref = bint[self.obj.bt1:self.obj.bt2, ref[0], ref[1]]
        bdata = bint[self.obj.bt1:self.obj.bt2, step, shot]
        xlag = fmp.lagtime(bref, bdata)['xlag']
        if xlag is not None:
            tbx.prefig()
            plt.title('integrated B signals', fontsize=25)
            plt.plot(bref, label='reference')
            plt.plot(bdata, label='original')
            plt.plot(np.roll(bdata, -xlag), label='shift')
            plt.legend(fontsize=20)

            if curr is not None:
                curr0 = self.obj.curr[self.obj.bt1:self.obj.bt2, ref[0],
                                      ref[1]]
                curr1 = self.obj.curr[self.obj.bt1:self.obj.bt2, step, shot]
                tbx.prefig()
                plt.title('current signals', fontsize=25)
                plt.plot(curr0, label='reference')
                plt.plot(np.roll(curr1, -xlag), label='shift')
                plt.legend(fontsize=20)
            else:
                print("** curr = None, current not plotted")

    ''' ----------------------------------------------------------------------
        CONDITIONAL AVERAGING ROUTINE
    --------------------------------------------------------------------------
    '''
    def condavg(self, bint, curr, bref=None, ref=None):
        ''' ------------------------------------------------------------------
        Conditionally avarage shift of RFEA current data.
        INPUTS:   data    = np.array with the data to be conditionally
                             averaged.
                  bdata   = np.array with the phase information (usually bdot)
                  nsteps  = Number of steps in the voltage sweep
                  nshots  = Number of shots for each step in the voltage sweep
                  trange  = Time range to store conditionally averaged data.
                  btrange = Time range of the conditional averaging (bdot)
        OPTIONAL: ref = [step, shot] number of the reference shot
                  bref = Inputs a reference shot for conditional averaging
        '''
        # Set default values
        if (self.obj.t1 is None) and (self.obj.t2 is None):
            self.obj.t1, self.obj.t2 = 0, curr.shape[0]
            print("** condavg t1, t2 undefined, setting defaults t1, t2 = {0},"
                  " {1}". format(self.obj.t1, self.obj.t2))
        if (self.obj.bt1 is None) and (self.obj.bt2 is None):
            self.obj.bt1, self.obj.bt2 = self.obj.t1, self.obj.t2
            print("** condavg bt1, bt2 undefined, setting defaults bt1, bt2 ="
                  " {0}, {1}". format(self.obj.bt1, self.obj.bt2))
        if ref is None:
            ref = [0, 0]

        # Current array, shifted in phase
        curr_arr = np.zeros((self.obj.t2-self.obj.t1, self.obj.nsteps,
                             self.obj.nshots))
        # Array shows number of shots skipped because cross-correlation fails
        skip_arr = np.zeros(self.obj.nsteps)

        # Determine the reference shot in bdata
        if bref is None:
            bref = bint[self.obj.bt1:self.obj.bt2, ref[0], ref[1]]

        for step in range(self.obj.nsteps):
            skips = 0
            for shot in range(self.obj.nshots):
                tbx.progress_bar([step, shot], [self.obj.nsteps,
                                 self.obj.nshots], ['nsteps', 'nshots'])
                bsig = bint[self.obj.bt1:self.obj.bt2, step, shot]
                xlag = fmp.lagtime(bref, bsig, quiet=1, threshold=0.7)['xlag']

                if xlag is not None:
                    curr_arr[:, step, shot] = np.roll(
                        curr[self.obj.t1:self.obj.t2, step, shot], -xlag)
                else:
                    skips += 1
            skip_arr[step] = skips

        factor = np.zeros(len(skip_arr))
        # Calculates factor so that mean_curr takes mean of shots not skipped
        for ii in range(len(skip_arr)):
            if (self.obj.nshots - skip_arr[ii] > 0):
                factor[ii] = self.obj.nshots/(self.obj.nshots - skip_arr[ii])
            else:
                print(self.obj.nshots, skip_arr[ii])
                print('factor = 0 for step {0}, all shots skipped!'.format(ii))
        mean_condavg_curr = np.mean(curr_arr, axis=2) * factor

        # Calculate rejection rate
        _ = fmp.reject_rate(skip_arr)

        return mean_condavg_curr, bref

    ''' ----------------------------------------------------------------------
        GENERAL DATA ANALYSIS FUNCTIONS
    --------------------------------------------------------------------------
    '''
    def mstime(self, *args, **kwargs):
        return trigtime(self.obj.time, *args, **kwargs)

    def mean_current(self, curr):
        return np.mean(curr, axis=2)/self.obj.res * 1e6    # [uA]

    def plot_IV(self, volt, curr, times=None):
        if times is None:
            times = [15000, 17500, 20000, 25000, 30000]

        # IV response
        tbx.prefig(xlabel='Peak pulse voltage [V]', ylabel='Current [$\mu$A]')
        for tt in times:
            plt.plot(volt, curr[tt, :], label='{0:.2f} ms'.format(
                     self.mstime(tt, start=5)))
        plt.legend(fontsize=20)
        plt.title('Average IV response, NO conditional averaging, {0} shots'.
                  format(self.obj.nshots), fontsize=20)
        tbx.savefig('./img/{0}-average-IV-response.png'.format(self.obj.fid))

        # IV derivative
        tbx.prefig(xlabel='Peak pulse voltage [V]', ylabel='-dI/dV')
        for tt in times:
            deriv = IVderiv(curr[tt, :], nwindow=51)
            plt.plot(volt, deriv, label='{0:.2f} ms'.format(
                self.mstime(tt, start=5)))
        plt.legend(fontsize=20)
        plt.title('Average IV-deriv, NO conditional averaging, {0} shots'.
                  format(self.obj.nshots), fontsize=20)
        tbx.savefig('./img/{0}-average-IV-deriv.png'.format(self.obj.fid))


''' --------------------------------------------------------------------------
    SINGLE DISTRIBUTION FUNCTION ANALYSIS
------------------------------------------------------------------------------
'''


class dfunc():
    def __init__(self, x, y, xrange=None):
        clip = int(len(x)*0.07)
        self.x = x[clip:-clip]    # Voltage array
        self.y = y[clip:-clip]    # -dI/dV array

        # Stored values
        if xrange is not None:
            self.xrange = xrange
        else:
            self.xrange = [30, 95]
            self.set_xrange()

        self.rms = 0
        self.update_rms()

        self.guess = None
        self.bounds = None

    # Define fit functions ---------------------------------------------------
    @staticmethod
    def onegauss_func(x, x1, a1, b1, x2, a2, b2, c):
        return a1 * np.exp(-(x-x1)**2/(2 * b1**2)) + c

    @staticmethod
    def twogauss_func(x, x1, a1, b1, x2, a2, b2, c):
        return a1 * np.exp(-(x-x1)**2/(2 * b1**2)) +\
               a2 * np.exp(-(x-x2)**2/(2 * b2**2)) + c

    @staticmethod
    def gauss(x, x1, a1, b1, c):
        return a1 * np.exp(-(x-x1)**2/(2 * b1**2)) + c

    # Calculate the max noise value
    def update_rms(self):
        ind = np.where((self.x < self.xrange[0]) | (self.x > self.xrange[1]))
        self.rms = np.amax(abs(self.y[ind]))
        return self.rms

    # Recalculate the xrange based on the given data
    def set_xrange(self):
        # Indices for peak value, guess L and R values
        argx1 = np.argmin(abs(self.x - self.xrange[0]))
        argx2 = np.argmin(abs(self.x - self.xrange[1]))
        argmax = np.argmax(self.y[argx1:argx2]) + argx1

        # Search for first minimum point
        def argL_func(peakL):
            if len(peakL) > 0:
                test = peakL[-1] + argx1
                # Avoid cutting off in the middle of bimodal distribution
                if self.y[test]/self.y[argmax] > 0.2:
                    peakL = peakL[:-1]
                    return argL_func(peakL)
                return test
            else:
                return argx1

        def argR_func(peakR):
            if len(peakR) > 0:
                test = peakR[0] + argmax
                if self.y[test]/self.y[argmax] > 0.2:
                    peakR = peakR[1:]
                    return argR_func(peakR)
                return test
            else:
                return argx2

        peakL, _ = scipy.signal.find_peaks(-self.y[argx1:argmax])
        argL = argL_func(peakL)

        peakR, _ = scipy.signal.find_peaks(-self.y[argmax:argx2])
        argR = argR_func(peakR)

        self.xrange = [self.x[argL], self.x[argR]]
        return self.xrange

    # Set default values if no input is specified ----------------------------
    # Array is for gaussfit (x1, a1, b1, x2, a2, b2, c)
    def set_guess(self, onegauss=None, guess_range=None):
        bm = None   # flag
        if guess_range is None:
            self.guess = self.xrange
        else:
            self.guess = guess_range

        a1 = np.amax(self.y)
        argb1 = np.argwhere(self.y > np.amax(self.y)/2)
        b1 = (self.x[argb1[-1]][0]-self.x[argb1[0]][0])/(2*np.sqrt(2*np.log(2)))
        a2 = self.update_rms()    # this is self.rms

        def xrpct(pct):
            return pct * (self.xrange[1]-self.xrange[0]) + self.xrange[0]

        # Find peaks with amplitudes greater than noise level and at least
        #  1 eV apart
        dx = self.x[1]-self.x[0]
        peaks, prop = scipy.signal.find_peaks(self.y, height=1.5*a2,
                                              width=1/dx, distance=1/dx)
        parg = np.where((self.x[peaks] > self.guess[0]) &
                        (self.x[peaks] < self.guess[1]))
        peaks = peaks[parg]
        # If there are exactly two peaks then absolutely use bimodal
        if len(peaks) == 2:
            w1, w2 = prop['width_heights'][parg]
            guess = [self.x[peaks[0]], self.y[peaks[0]], np.amin([15, dx*w1]),
                     self.x[peaks[1]], self.y[peaks[1]], np.amin([15, dx*w2]),
                     0]
            if onegauss is None:
                bm = 1    # Bimodal
            else:
                mu1, a1, b1, mu2, a2, b2, c = guess
                if a2 > a1:
                    guess = [mu2, a2, b2, mu1, a1, b1, c]
        elif len(peaks) == 1:
            w1 = prop['width_heights'][parg][0]
            if self.x[peaks]-xrpct(0.71) > self.x[peaks]-xrpct(0.32):
                guess_x2 = xrpct(0.32)
            else:
                guess_x2 = xrpct(0.71)
            guess = [self.x[peaks][0], self.y[peaks][0], np.amin([15, dx*w1]),
                     guess_x2, a2, 12, 0]
        else:
            # Guess from plot
            guess = [xrpct(0.32), a1, b1, xrpct(0.71), a2, 12, 0]

        # Guesses should not have zero because it will trigger the boundary
        # condition.
        return guess, bm

    def set_bounds(self):
        return [(self.xrange[0], self.rms, 0,
                 self.xrange[0], self.rms, 0, -self.rms/2),
                (self.xrange[1], 1.1*np.amax(self.y), 15,
                 self.xrange[1], 1.1*np.amax(self.y), 15, self.rms/2)]

    # Fitting function for distribution function -----------------------------
    def gaussfit(self, guess=None, bounds=None, onegauss=None, bm=None,
                 **kwargs):
        # Set default guess and boundaries
        if guess is None:
            guess, bm = self.set_guess(onegauss=onegauss, **kwargs)
        if bounds is None:
            bounds = self.set_bounds()

        if onegauss is None:
            fitfunc = self.twogauss_func
        else:
            fitfunc = self.onegauss_func

        try:
            popt, _ = scipy.optimize.curve_fit(fitfunc, self.x, self.y,
                                               p0=guess, bounds=bounds)
            # Add case: Have to use bimodal if double peaked
            if bm or (self.bimodal_test(popt) is not None) or \
                    (popt[4] != 0):
                # Rejection cases for bimodal:
                #   1. The choice of fit is unimodal/Maxwellian
                if onegauss is not None:
                    if popt[4] > popt[1]:
                        mu1, a1, b1, mu2, a2, b2, c = popt
                        popt = [mu2, a2, b2, mu1, 0, b1, c]
                    else:
                        popt[4] = 0
                #   2. One of the peaks is less than the noise level
                elif popt[1] < 1.5*self.rms:
                    mu1, a1, b1, mu2, a2, b2, c = popt
                    popt = [mu2, a2, b2, mu1, 0, b1, c]
                elif popt[4] < 1.5*self.rms:
                    popt[4] = 0
                #   3. One of the peaks is 4x smaller than the other
                elif popt[1]/popt[4] < 0.25:
                    mu1, a1, b1, mu2, a2, b2, c = popt
                    popt = [mu2, a2, b2, mu1, 0, b1, c]
                elif popt[4]/popt[1] < 0.25:
                    popt[4] = 0
                #   4. The two peaks are on top of each other
                elif abs(popt[0]-popt[3]) < (popt[2] + popt[5])/2:
                    popt[4] = 0

            # Calculate least squares for error
            arg = np.argwhere(self.y > 1.5*self.rms)
            lsq = np.sum((self.y[arg] - fitfunc(self.x[arg], *popt))**2)
            return popt, lsq
        except (RuntimeError, ValueError):
            return None, None

    # Plot components of the distribution function ---------------------------
    def dfplot(self, x, y, popt, lsq, fitfunc, color='red', window=None,
               label=None, **kwargs):
        color = 'red' #0e10e6' # for PRL figure consistency
        # Given popt, figure out if unimodal or bimodal
        if self.bimodal_test(popt) is not None:
            bu = 'Bimodal'    # Bimodal
        else:
            bu = 'Unimodal'   # Unimodal

        if np.sign(popt[6]) >= 0:
            csign = '+'
        else:
            csign = ' '
        if fitfunc is self.twogauss_func:
            # wlabel = '{0}: '.format(bu) + \
            #          '$x_1$ = {1:.2f}, $A_1$ = {2:.2f}, $b_1$ = {3:.2f}, '\
            #          '$x_2$ = {4:.2f}, $A_2$ = {5:.2f}, $b_2$ = {6:.2f}, '\
            #          '$c$ = {7:.2f} ({8})'.format(lsq, *popt, label)
            wlabel = r'${1:.1f} * exp\left(-\dfrac{{(V-{0:.1f})^2}}{{2*({2:.1f})^2}}\right) + $'\
                     r'${4:.1f} * exp\left(-\dfrac{{(V-{3:.1f})^2}}{{2*({5:.1f})^2}}\right) {7} $'\
                     r'${6:.2f}$'.format(*popt, csign)
            # A2
            window.plot(x, self.gauss(x, popt[3], popt[4], popt[5], popt[6]),
                        '--', linewidth=3, color=color, alpha=0.7)
        else:
            bu = 'Unimodal'   # Unimodal
            # wlabel = '{0}: '.format(bu) + \
            #          '$x_1$ = {1:.2f}, $A_1$ = {2:.2f}, $b_1$ = {3:.2f}, '\
            #          '$c$ = {7:.2f} ({8})'.format(lsq, *popt, label)
            wlabel = r'${1:.1f} * exp\left(-\dfrac{{(V-{0:.1f})^2}}{{2*({2:.1f})^2}}\right) + $'\
                     '{6:.2f}'.format(*popt)

        # swap wlabel
        window.plot(x, fitfunc(x, *popt), label='Bi-Maxwellian (best fit)', color=color,
                    linewidth=5)
        # A1
        window.plot(x, self.gauss(x, popt[0], popt[1], popt[2], popt[6]), 
                    '--', linewidth=3, color=color, alpha=0.7)

    # Multiple function analysis. Plot best curve from least squares. --------
    def bestfit(self, rec_guess=None, window=None, lsq=1e6, **kwargs):
        # rec_guess = A guess value to be passed to check for better guesses

        # Guess #1: Maxwellian / unimodal
        popt, lsq1 = self.gaussfit(onegauss=1, **kwargs)
        if lsq1 is not None:
            if (lsq1 < lsq):
                popt[4] = 0
                lsq = lsq1
                color = 'green'
                fitfunc = self.onegauss_func

        # Guess #2: Maxwellian + beam / bimodal
        popt2, lsq2 = self.gaussfit(**kwargs)
        # print('popt2', popt2, lsq2)
        # More conditions for rejecting a bimodal distribution
        if (lsq2 is not None) and (popt2 is not None) and (popt is not None):
            # Primary peak has comparable amplitude and width to unimodal
            if (popt2[1]/popt[1] > 0.80) and (popt2[2]/popt[2] > 0.80):
                pass
            # The fit has to improve least squares by at least a factor of 10
            elif (lsq2 < 0.1*lsq) and (popt2[4] != 0):
                popt, lsq = popt2, lsq2
                color = 'red'
                fitfunc = self.twogauss_func

        # # If an improved guess is submitted, use that
        popt3, lsq3 = self.gaussfit(guess=rec_guess, **kwargs)
        # print('popt2', popt2, lsq2)
        # More conditions for rejecting a bimodal distribution
        if (lsq3 is not None) and (popt3 is not None) and (popt is not None):
            # Primary peak has comparable amplitude and width to unimodal
            if (popt3[1]/popt[1] > 0.80) and (popt3[2]/popt[2] > 0.80):
                pass
            # The fit has to improve least squares by at least a factor of 10
            elif (lsq3 < 0.1*lsq) and (popt3[4] != 0):
                popt, lsq = popt3, lsq3
                color = 'purple'
                fitfunc = self.twogauss_func

        if (window is not None) and (popt is not None):
            self.dfplot(self.x, self.y, popt, lsq, fitfunc, window=window,
                        color=color, label='{0:.2f}'.format(lsq))
        return popt

    # Test of bimodality by Robertson & Fryer (1969), Scandainavian
    # Actuarial Journal
    @staticmethod
    def bimodal_test(popt):
        # Check if input is None?
        if popt is None:
            return None    # Unimodal

        # Require mu1 <= mu2
        if popt[0] <= popt[3]:
            mu1, a1, b1, mu2, a2, b2, c = popt
        else:
            mu2, a2, b2, mu1, a1, b1, c = popt

        # Define constants
        p1 = (a1-c) * b1 * np.sqrt(2*np.pi)
        p2 = (a2-c) * b2 * np.sqrt(2*np.pi)
        p = p1 / (p1+p2)
        sigma1 = b1
        sigma2 = b2
        mu = (mu2-mu1)/(sigma1)
        sigma = sigma2/sigma1

        mu0 = np.sqrt((2*(sigma**4 - sigma**2 + 1)**1.5 - (2*sigma**6 -
                      3*sigma**4 - 3*sigma**2 + 2)) / sigma**2)

        if mu <= mu0:
            return None    # Unimodal
        else:
            # Solve cubic equation
            coeff = [mu*sigma**2, -mu**2, -mu*(sigma**2-2), sigma**2-1]
            roots = np.array([])
            for r in poly.polyroots(coeff):
                # Check for (1) real roots, (2) less than mu, (3) greater
                # than zero.
                if (np.imag(r) == 0) and (mu > r) and (r > 0):
                    roots = np.append(roots, r)
            roots = np.sort(roots)

            def p_root(value):
                invp = 1 + (sigma**3 * value / (mu-value)) * np.exp(
                    -value**2/2 + ((value-mu)/sigma)**2/2)
                return 1/invp

            # Should only have two distinct real roots
            p1 = p_root(roots[0])
            if len(roots) > 1:
                p2 = p_root(roots[1])
                if (p1 < p) and (p < p2):
                    return 1       # Bimodal
                else:
                    return None    # Unimodal
            else:
                return None

    # Plot a single frame of the movie
    def movie_frame(self, tt, volt, curr, xx, yy, ygrad, amp=1, window=plt,
                    xlabel=None, labels=None, ynoise=None):
        if labels is None:
            labeltext = ''
            # window.plot(self.x, yy, color='#0e10e6')
            # window.plot(self.x, curr[tt, :], 'grey', alpha=0.7,
            #             color='#f78f2e')
        else:
            labeltext = '$-dI/dV$ * {0}'.format(amp)
            # window.plot(self.x, yy, label='current (Savitzky-Golay)',
            #             color='#0e10e6')
            # window.plot(self.x, curr[self.tt, :], alpha=0.7,
            #             label='current (original)', color='#f78f2e')

        window.plot(self.x, self.y, label=labeltext, color='blue', alpha=0.75)

        # Plot the noise level
        if ynoise is not None:
            window.fill_between([self.x[0], self.x[-1]], [ynoise, ynoise], [0,0],
                                color='green', alpha=0.1)

        # Find peaks of yy and mark them
        peaks, _ = scipy.signal.find_peaks(self.y, height=ynoise, distance=5)
        parg = np.where((self.x[peaks] > self.guess[0]) &
                        (self.x[peaks] < self.guess[1]))
        peaks = peaks[parg]
        #window.plot(self.x[peaks], self.y[peaks], 'x')  # disable for PRL

        if window is not plt:
            if xlabel is not None:
                window.set_xlabel('Discriminator Grid Voltage [V]', fontsize=40)
            window.set_ylabel('arb. units', fontsize=40)
            window.set_ylim(-25, 220) #PRL
            #window.set_ylim([self.y.min()*1.1, self.y.max()*1.5])
        else:
            if xlabel is not None:
                window.xlabel('Potential [V]', fontsize=30)
            window.ylabel('magnitude', fontsize=30)
            #window.ylim([self.y.min()*1.1, self.y.max()*1.5])
            window.ylim(-25, 220) #PRL
        window.tick_params(labelsize=30)
        window.legend(fontsize=25, loc='upper left')


# Function to smooth the data then calculate popt
def calc_popt(volt, curr, factor=1e6/9.08e3, snw=41, passes=3, gamp=60,
              popt0=None, guess_range=None, **kwargs):
    # Input only 1D array (i.e. cacurrA[tt,:])
    # Smooth the I-V curve
    sgx, sgy, sgyg = sgsmooth(curr*factor, nwindow=snw, repeat=passes)

    # Calculate popt
    df = dfunc(volt[sgx], sgyg*gamp, **kwargs)
    popt = df.bestfit(rec_guess=popt0, guess_range=guess_range)

    noise = 1.5*df.rms
    return popt, sgx, sgy, sgyg, factor, noise

''' --------------------------------------------------------------------------
    JOINT DISTRIBUTION FUNCTION ANALYSIS
------------------------------------------------------------------------------
'''


class join_dfunc():
    def __init__(self, time, voltL, voltR, currL, currR, trange=None,
                 dV=1, xrange=None, yrange=None, fid='fid'):
        # Store inputs
        self.time = time
        self.voltL = voltL
        self.voltR = voltR
        self.volt = np.concatenate([np.flip(-voltL), voltR])
        self.currL = currL
        self.currR = currR
        self.dV = dV
        self.fid = fid

        # Define the shape of currL/currR
        tL, nstepL = currL.shape
        tR, nstepR = currR.shape

        # Expecting trange to be an array [t1, t2] for range of movie
        if trange is None:
            self.nt = len(time)
            self.t1 = 0
            self.t2 = self.nt
        else:
            self.nt = trange[1] - trange[0]
            self.t1 = trange[0]
            self.t2 = trange[1]

        # Plotting parameters:
        if yrange is None:
            self.yrange = [-0.0035, 0.035]
        else:
            self.yrange = yrange

        if xrange is None:
            self.xrange = [-40, 40]
        else:
            self.xrange = xrange

        # Define parameters to be used later
        self.arrTT = None
        self.arrTi = None
        self.enflag = None

    # Function to join the two distribution functions
    @staticmethod
    def set_dfunc(voltL, voltR, dataL, dataR, dV=1, nwindow=41, nwindowR=None,
                  order=3, nosmooth=None, **kwargs):
        # Create distribution function using two data arrays.
        # Find max, cut the curve, do it for the other side, then join them
        # at the top. Normalize to the mag of one side. Inputs are IV traces.

        if nwindowR is None:
            nwindowR = nwindow
        else:
            nwindowR = nwindowR

        # Calculate gradL/gradR. Note that the length of grad is reduced by
        # the window size and is even: int(nwindow/2)*2
        xL, yL, gradL = sgsmooth(dataL, nwindow=nwindow, repeat=order)
        xR, yR, gradR = sgsmooth(dataR, nwindow=nwindowR, repeat=order)

        vL = voltL[xL]
        vR = voltR[xR]

        dfuncL = dfunc(vL, gradL, **kwargs)
        dfuncR = dfunc(vR, gradR, **kwargs)

        # tbx.prefig()
        poptL = dfuncL.bestfit(window=None)
        poptR = dfuncR.bestfit(window=None)
        # plt.legend(fontsize=20, loc='upper left')

        # Choose leftmost peak if it is bimodal
        def check_popt(popt, grad, vLR):
            arg = np.argmax(grad)    # Default is peak value

            # Change this value if it is bimodal is used
            bflag = dfunc.bimodal_test(popt)
            if bflag is not None:
                if popt is not None:
                    if popt[4] in [0, None]:
                        pp = popt[0]
                    else:
                        if popt[1] > popt[4]:
                            pp = popt[0]
                        else:
                            pp = popt[3]  # pp = np.min([popt[0], popt[3]])
                        arg = np.argmin(abs(vLR-pp))
            return arg

        argL = check_popt(poptL, gradL, vL)
        argR = check_popt(poptR, gradR, vR)

        # Slice curves and only keep the right side
        sliceL = np.array(gradL[argL:])
        sliceR = np.array(gradR[argR:])

        # Normalize wrt right side of the curve
        factor = gradR[argR] / gradL[argL]
        index = np.arange(-len(sliceL), len(sliceR))
        dfLR = np.concatenate([np.flip(sliceL)*factor, sliceR])

        # Slice voltL/voltR as well, but also shift the starting value to zero
        if (voltL is not None) and (voltR is not None):
            vL = np.array(vL[argL:]) - vL[argL]
            vR = np.array(vR[argR:]) - vR[argR]
            vLvR = np.concatenate([np.flip(-vL), vR])
        else:
            vLvR = index*dV

        return index, dfLR, vLvR

    # For data that is already processed, input is volt and dfunc
    @staticmethod
    def join_processed(voltL, voltR, dfuncL, dfuncR):

        # Function to get all relevant parameters
        def get_params(volt, dfunc):
            peaks, _ = scipy.signal.find_peaks(dfunc)
            try:
                if len(peaks) == 1:
                    arg = int(peaks[0])
                else:
                    arg = int(np.amin(peaks[0]))
            except:
                print(dfunc)

            p_mag = np.amax(dfunc)
            p_volt = np.array(volt[arg:]) - volt[arg]
            p_slice = dfunc[arg:]

            return p_mag, p_volt, p_slice

        magL, vL, sliceL = get_params(voltL, dfuncL)
        magR, vR, sliceR = get_params(voltR, dfuncR)

        vLvR = np.concatenate([np.flip(-vL), vR])
        factor = sliceR[0] / sliceL[0]
        dfLR = np.concatenate([np.flip(sliceL)*factor, sliceR])
        index = np.arange(-len(sliceL), len(sliceR))

        return index, dfLR/np.amax(dfLR), vLvR

    def calc_enint(self, dt=1, **kwargs):
        nsteps = int(len(self.currL[:, 0])/dt)
        arrTi = np.zeros(nsteps)
        for step in range(nsteps):
            tbx.progress_bar(step, nsteps)
            tt = dt * step
            # function can handle None
            _, dfunc, vLvR = self.set_dfunc(self.voltL, self.voltR,
                                            self.currL[tt, :],
                                            self.currR[tt, :], **kwargs)
            arrTi[step] = enint(vLvR, dfunc)

        self.arrTT = self.time[[ii*dt+self.t1 for ii in range(nsteps)]]*1e3+5
        self.arrTi = arrTi
        self.enflag = 1

    # Plot Ti calculated from the energy integral
    def plot_enint(self, limTi=100):
        tbx.prefig(xlabel='time [ms]', ylabel='average $E$ [eV]')
        plt.title('{0} average energy (combined distribution '
                  'function)'.format(self.fid), fontsize=20)

        test = np.where(self.arrTi < limTi)
        plt.plot(self.arrTT[test], self.arrTi[test])
        tbx.savefig('./img/{0}-Ti-distfunc.png'.format(self.fid))

    def movie(self, nstep=500, limTi=100):
        nframes = self.nt // nstep
        test = np.where(self.arrTi < limTi)

        # Plot movie to look at distribution function evolution
        if self.enflag is not None:
            fig = plt.figure(figsize=(16, 9))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
        else:
            fig = plt.figure(figsize=(16, 4.5))
            ax2 = fig.add_subplot(111)

        def generate_frame(i):
            tt = i*nstep
            if self.enflag is not None:
                ax1.clear()
                ax1.set_title('{0} energy integral'.format(self.fid),
                              fontsize=25)
                ax1.plot(self.arrTT[test], self.arrTi[test])
                ax1.plot(np.repeat(trigtime(self.time, tt, off=self.t1), 2),
                         [np.amin(self.arrTi[test])*1.1,
                         np.amax(self.arrTi[test])*1.1], color='orange')
                ax1.set_xlabel('time [ms]', fontsize=30)
                ax1.set_ylabel('average $E$ [eV]', fontsize=30)

            ax2.clear()
            ax2.set_title('Distribution function (positive towards old '
                          'LaB$_6$), t ={0:.3f} ms [{1}]'.format(trigtime(
                              self.time, tt, off=self.t1), tt), fontsize=20)
            _, dfunc, vLvR = self.set_dfunc(self.voltL, self.voltR,
                                            self.currL[tt, :],
                                            self.currR[tt, :])
            ax2.plot(vLvR, dfunc)
            ax2.set_xlabel('Potential [V]', fontsize=30)
            ax2.set_ylabel('f(V)', fontsize=30)
            ax2.tick_params(labelsize=30)
            ax2.set_ylim(self.yrange)
            ax2.set_xlim(self.xrange)

            plt.tight_layout()

            print('\r', 'Generating frame {0}/{1} ({2:.2f}%)...'
                  .format(i+1, nframes, (i+1)/nframes*100), end='')

        anim = animation.FuncAnimation(fig, generate_frame,
                                       frames=nframes, interval=25)
        anim.save('./videos/{0}-dfunc-combine.mp4'.format(self.fid))


''' --------------------------------------------------------------------------
    ENERGY INTEGRAL CALCULATION
------------------------------------------------------------------------------
'''


def enint(volt, dfunc):
    den = np.sum([jj/np.sqrt(abs(ii)) for ii, jj in zip(volt, dfunc)
                 if ii != 0])
    vavg = np.sum([jj*np.sqrt(abs(ii)) for ii, jj in zip(volt, dfunc)
                  if ii != 0])
    return 2*vavg/den


''' ----------------------------------------------------------------------
    REGULAR DISTRIBUTION FUNCTION ROUTINES
--------------------------------------------------------------------------
'''


def get_dfunc(cacurr, snw=41, passes=3):
    # Gets the distribution function from a single I-V plot
    nt, nvolt = cacurr.shape
    nvolt -= 2*(snw//2)

    cacurr_sm = np.empty((nt, nvolt))
    grad_sm = np.empty((nt, nvolt))

    for tt in range(nt):
        tbx.progress_bar(tt, nt, label='tt')
        x, y, grad = sgsmooth(cacurr[tt, :], nwindow=snw, repeat=passes)
        cacurr_sm[tt, :] = y
        grad_sm[tt, :] = grad
    return x, cacurr_sm, grad_sm


def get_dfunc2(cacurrL, cacurrR, voltL, voltR, nwindow=41, order=3, dt=1):
    # Joins two distribution functions from two different I-V plots
    nt, nvolt = cacurrL.shape
    nt = int(nt/dt)
    dV = (voltL[-1]-voltL[0])/(nvolt-1)
    nv = nvolt//2
    vrange = np.arange(-nv, nv+1) * dV

    dfunc_arr = np.empty((nt, nvolt))

    for ii in range(0, nt):
        tt = dt * ii
        tbx.progress_bar(ii, nt, label='tt')
        index, func, revolt = join_dfunc.set_dfunc(
                                voltL, voltR, cacurrL[tt, :], cacurrR[tt, :],
                                nwindow=nwindow, order=order)
        index += nv
        # aaa = np.where((index >= 0) & (index < vrange.shape))
        dfunc_arr[ii, index] = func

    return vrange, dfunc_arr


''' ----------------------------------------------------------------------
    REGULAR NON-CLASS FUNCTIONS
--------------------------------------------------------------------------
'''


def trigtime(time, ind, start=5, off=0):
    # Determines the actual time (in ms) from the start of the discharge
    # start: [ms] the start time of the trigger (recorded)
    # off:   [px] the offset of the analyzed slice of data
    return time[int(ind)+off]*1e3 + start


def IVderiv(curr, scale=1, nwindow=51, nwindow2=None, polyn=3, **kwargs):
    # Calculates the current derivative -dI/dV
    smoo1 = tbx.smooth(curr, nwindow=nwindow, polyn=polyn, **kwargs)
    grad1 = -np.gradient(smoo1)
    if nwindow2 is not None:
        smoo2 = tbx.smooth(grad1, nwindow=nwindow2, polyn=polyn, **kwargs)
        return smoo2*scale
    else:
        return grad1*scale


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

    if plot != 0:
        plt.plot(xx[max(iimax-xoff, 0)], yy[max(iimax-xoff, 0)], 'o')
        plt.plot(xx[iimin], yy[iimin], 'o')

    def gauss_func(x, a, b, c, x0):
        return a * np.exp(-b * (np.sqrt(x)-np.sqrt(x0))**2) + c

    guess = [yy[iimax]-yy[iimin], 1, yy[iimin], xx[iimax]]

    popt, pcov = curve_fit(gauss_func, xx[max(iimax-xoff, 0):iimin],
                           yy[max(iimax-xoff, 0):iimin], p0=guess)

    # Plot the points if desired
    if plot != 0:
        plt.plot(xx[max(iimax-xoff, 0):iimin],
                 gauss_func(xx[max(iimax-xoff, 0):iimin], *popt), '--')

    # Returns full width of gaussian; b = 1/kT = 1/(2*sigma^2)
    return popt, pcov  # (1/np.sqrt(*popt[1]))


def find_Ti_exp(volt, curr, startpx=100, endpx=100, plot=0, mstime=0,
                fid=None, save=0):
    '''
    # Finds Ti from an exponential plot; The curve starts from some max
    # value then decays exponentially.
    startpx = number of pixels to count at the start to determine max value
    endpx   = number of pixels to count at the end to determine min value
    '''
    # Smooth the curve
    temp = tbx.smooth(curr, nwindow=11)

    # Take gradient to find peak of dI/dV, thats where to cut the exp fit off
    gradient = np.gradient(tbx.smooth(temp, nwindow=51))
    argmin = np.argmin(gradient)

    # Determine start and end values of the curve
    vstart = np.mean(temp[:startpx])
    vend = np.mean(temp[-endpx:])

    def exp_func(x, a, b, c, x0):
        return a * np.exp(-(x-x0)/b) + c

    guess = [vstart-vend, 2, vend, volt[argmin]]
    bound_down = [0.1*(vstart-vend), 0, vend-vstart, volt[0]]
    bound_up = [+np.inf, 50, vstart, volt[-1]]
    try:
        popt, pcov = curve_fit(exp_func, volt[argmin:], temp[argmin:],
                               p0=guess, bounds=(bound_down, bound_up))
    except:
        return None, None, None

    Vp = volt[np.argmin(abs([vstart for ii in volt] - exp_func(
        volt, *popt)))]
    Ti = popt[1]
    Ti_err = np.sqrt(np.diag(pcov))[1]

    if plot != 0:
        tbx.prefig(xlabel='Discriminator Grid Voltage (V)',
                   ylabel='Current ($\mu$A)')
        plt.plot(volt, temp, color='#0e10e6')  # ,label='{0} ms'.format(mstime)
        plt.title('exponential fit, t = {0:.2f} ms'.format(mstime),
                  fontsize=20)
        # plt.plot(volt[argmin:], temp[argmin:], color='#9208e7')
        plt.plot(volt, [vstart for ii in volt], '--', color='#5cd05b')
        plt.plot(volt, [vend for ii in volt], '--', color='#5cd05b')
        plt.plot(volt[argmin-20:], exp_func(volt[argmin-20:], *popt), '--',
                 label='$T_i$ = {0:.2f} eV'.format(Ti), color='#ff4900',
                 linewidth=2)
        plt.ylim(np.min(temp)*0.96, np.max(temp)*1.04)
        plt.legend(fontsize=20, loc='upper right')
        if save == 1:
            tbx.savefig('./img/{0}-IV-expfit-{1:.2f}ms.png'.format(
                        fid, mstime))

    return Vp, Ti, Ti_err


def select_peaks(step_arr, argtime_arr, peakcurr_arr, step=0, trange=None):
    # Given arrays of (1) step number, (2) argtime and (3) peak current
    # corresponding to each peak, returns peaks that fulfil input
    # conditions.
    # Set default trange
    if trange is None:
        trange = [0, 1]

    ind = np.where((argtime_arr > trange[0]) & (argtime_arr <= trange[1]) &
                   (step_arr == step))
    return peakcurr_arr[ind]


''' ----------------------------------------------------------------------
    PLOTTING FUNCTIONS
--------------------------------------------------------------------------
'''


def plot_volt(volt):
    tbx.prefig(xlabel='step', ylabel='voltage [V]')
    plt.plot(volt)


def plot_IVderiv(volt, curr, xoff=0, yoff=0, nwindow=51, polyn=3,
                 **kwargs):
    # Plots the current derivative -dI/dV (if input is deriv do a regular
    # plot)
    plt.plot(volt + xoff, IVderiv(curr, nwindow=nwindow, polyn=polyn) +
             yoff, **kwargs)


def browse_data(data, x=None, y=None, step=0, shot=0, chan=0, trange=None):
    # Browses the data and returns the selected trange
    # Set default trange
    if trange is None:
        trange = [0, -1]

    t1 = trange[0]
    t2 = np.min([data.shape[0], trange[1]])
    tbx.prefig(xlabel='time [px]', ylabel='magnitude')

    if (x is None) and (y is None):
        temp = data[:, step, shot, chan]
        plt.title("step = {0}, shot = {1}, chan = {2}, trange = [{3},"
                  " {4}]". format(step, shot, chan, trange[0], trange[1]),
                  fontsize=20)
    else:
        temp = data[:, x, y, step, shot, chan]
        plt.title('(x, y) = ({0:.2f}, {1:.2f}), step = {2}, shot = {3}, '
                  ' chan = {4}, trange = [{5}, {6}]'.format(x, y, step, shot,
                                                            chan, trange[0],
                                                            trange[1]),
                  fontsize=20)

    plt.plot(temp)
    plt.plot([t1, t1], [np.min(temp), np.max(temp)], 'orange')
    plt.plot([t2, t2], [np.min(temp), np.max(temp)], 'orange')

    return t1, t2


''' ----------------------------------------------------------------------
    NAMING FUNCTIONS
--------------------------------------------------------------------------
'''


def fname_gen(series, date='2021-01-28', folder='/data/swjtang/RFEA/',
              ch=2):
    # Plots the current derivative -dI/dV
    return '{0}{1}/C{2}{3}.txt'.format(folder, date, ch, series)


''' ----------------------------------------------------------------------
    DISTRIBUTION FUNCTIONS & SMOOTHING
--------------------------------------------------------------------------
'''


def rsmooth(data, repeat=2, nwindow=31, **kwargs):
    temp = data
    while repeat > 0:
        temp = tbx.smooth(temp, nwindow=nwindow, **kwargs)
        repeat -= 1
    return temp


def sgsmooth(data, nwindow=31, repeat=5):
    # Savitzky-Golay smoothing
    # returns (1) appropriately formatted x-values, (2) smoothed data,
    # (3) gradient
    xval = np.arange(len(data))
    x = xval[int(nwindow/2):-int(nwindow/2)]

    y = rsmooth(data, repeat=repeat, nwindow=nwindow)[int(
            nwindow/2):-int(nwindow/2)]

    grad_y = -np.gradient(y)

    return x, y, grad_y


def exv2(volt, data, **kwargs):
    # Expectation value of v^2 divide by density expectation value
    try:
        nwindow = kwargs.nwindow
    except:
        nwindow = 31    # default value of nwindow in rsmooth

    grad = -np.gradient(rsmooth(data, **kwargs))

    # Cut -dI/dV at the peak then symmetrize
    arg = np.argmax(grad[nwindow:-nwindow]) + nwindow
    cut = np.array(grad[arg:-nwindow])

    arrE = volt[arg:]-volt[arg]

    den = np.sum([jj/np.sqrt(ii) for ii, jj in zip(arrE, cut) if ii != 0])
    vavg = np.sum([jj*np.sqrt(ii) for ii, jj in zip(arrE, cut) if ii != 0])

    return vavg/den
