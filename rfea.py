'''
NAME:           rfea.py
AUTHOR:         swjtang
DATE:           02 Jun 2021
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

        # dataset output is Arr[nt, shot, chan, step]
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
        data = np.transpose(datatemp, (0, 3, 1, 2))
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

    def plot_TiVp(self, Ti, Vp, ca=0):
        tt = np.array([self.mstime(tt) for tt in self.obj.tarr])

        text = 'conditional average, exponential fit'
        if ca == 1:
            text = 'YES '+text
            svname = 'yes-condavg'
        else:
            text = 'NO '+text
            svname = 'no-condavg'

        tbx.prefig(xlabel='time [ms]', ylabel='$T_i$ [eV]')
        plt.title('{0} $T_i$ vs time, {1} (all times)'.format(self.obj.fid,
                  text), fontsize=25)
        # plt.fill_between(tt, tbx.smooth(self.obj.Ti-self.obj.errTi,
        #                  nwindow=51),
        # tbx.smooth(self.obj.Ti+self.obj.errTi, nwindow=51), alpha=0.2)
        plt.plot(tt, tbx.smooth(Ti, nwindow=51))
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

    def plot_bint_shift(self, bint, curr=None, step=0, shot=0):
        # Plots the reference bint/current with a test shot
        bref = bint[self.obj.bt1:self.obj.bt2, 0, 0]
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
                curr0 = self.obj.curr[self.obj.bt1:self.obj.bt2, 0, 0]
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
    CREATE JOINT DISTRIBUTION FUNCTION MOVIE
------------------------------------------------------------------------------
'''
class dfunc_movie():
    def __init__(self, time, currL, currR, trange=None, voltL=None, voltR=None,
                 dV=1, nstep=500, xrange=None, yrange=None, fid='fid'):
        # Store inputs
        self.time = time
        self.currL = currL
        self.currR = currR
        self.dV = dV
        self.fid = fid

        # Define the shape of currL/currR
        tL, nstepL = currL.shape
        tR, nstepR = currR.shape

        # Expecting trange to be an array [t1, t2] for range of movie
        if trange is None:
            nt = len(time)
            self.t1 = 0
            self.t2 = nt
        else:
            nt = trange[1] - trange[0]
            self.t1 = trange[0]
            self.t2 = trange[1]

        self.nstep = nstep
        self.nframes = nt // nstep

        if (voltL is None) & (voltR is None):
            self.voltL = None
            self.voltR = None
            self.volt = np.arange(-nstepL, nstepR, dV)
        else:
            self.voltL = voltL
            self.voltR = voltR
            self.volt = np.concatenate([np.flip(-voltL), voltR])

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

    def movie(self):
        # Plot movie to look at distribution function evolution
        if self.enflag is not None:
            fig = plt.figure(figsize=(16, 9))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
        else:
            fig = plt.figure(figsize=(16, 4.5))
            ax2 = fig.add_subplot(111)

        def generate_frame(i):
            tt = i*self.nstep

            if self.enflag is not None:
                ax1.clear()
                ax1.set_title('{0} energy integral'.format(self.fid),
                              fontsize=25)
                ax1.plot(self.arrTT, self.arrTi)
                ax1.plot(np.repeat(trigtime(self.time, tt, off=self.t1), 2),
                         [np.amin(self.arrTi)*1.1, np.amax(self.arrTi)*1.1],
                         color='orange')
                ax1.set_xlabel('time [ms]', fontsize=30)
                ax1.set_ylabel('$T_i$ [eV]', fontsize=30)

            ax2.clear()
            ax2.set_title('Distribution function (positive towards old '
                          'LaB$_6$), t ={0:.3f} ms [{1}]'.format(trigtime(
                              self.time, tt, off=self.t1), tt), fontsize=20)
            if (self.voltL is None) or (self.voltR is None):
                index, func, revolt = dfunc(self.currL[tt, :],
                                            self.currR[tt, :])
                revolt = index*self.dV
            else:
                index, func, revolt = dfunc(self.currL[tt, :],
                                            self.currR[tt, :],
                                            voltL=self.voltL, voltR=self.voltR)
            ax2.plot(revolt, func)

            bbb = np.where(revolt <= 0)
            ax2.plot(np.flip(-revolt[bbb]), np.flip(func[bbb]))

            ax2.set_xlabel('Potential [V]', fontsize=30)
            ax2.set_ylabel('f(V)', fontsize=30)
            ax2.tick_params(labelsize=20)
            ax2.set_ylim(self.yrange)
            ax2.set_xlim(self.xrange)

            plt.tight_layout()

            print('\r', 'Generating frame {0}/{1} ({2:.2f}%)...'
                  .format(i+1, self.nframes, (i+1)/self.nframes*100), end='')

        anim = animation.FuncAnimation(fig, generate_frame,
                                       frames=self.nframes, interval=25)
        anim.save('./videos/{0}-dfunc-combine.mp4'.format(self.fid))

    # Calculate Ti using the energy integral
    def calc_enit(self, dt=1):
        self.en_int_dt = dt
        nsteps = int(len(self.currL[:, 0])/dt)
        arrTi = np.zeros(nsteps)
        for step in range(nsteps):
            tbx.progress_bar(step, nsteps)
            tt = dt * step
            if (self.voltL is None) or (self.voltR is None):
                index, func, revolt = dfunc(self.currL[tt, :],
                                            self.currR[tt, :])
                revolt = index*self.dV
            else:
                index, func, revolt = dfunc(self.currL[tt, :],
                                            self.currR[tt, :],
                                            voltL=self.voltL, voltR=self.voltR)
            den = np.sum([jj/np.sqrt(abs(ii)) for ii, jj in zip(revolt, func)
                          if ii != 0])
            vavg = np.sum([jj*np.sqrt(abs(ii)) for ii, jj in zip(revolt, func)
                           if ii != 0])
            arrTi[step] = vavg/den

        self.arrTT = self.time[[ii*dt+self.t1 for ii in range(nsteps)]]*1e3+5
        self.arrTi = arrTi
        self.enflag = 1

    # Plot Ti calculated from the energy integral
    def plot_enit(self):
        if self.arrTi is not None:
            tbx.prefig(xlabel='time [ms]', ylabel='$T_i$ [eV]')
            plt.title('{0} $T_i$ from energy integral (combined distribution '
                      'function)'.format(self.fid), fontsize=20)
            plt.plot(self.arrTT, self.arrTi)
            tbx.savefig('./img/{0}-Ti-distfunc.png'.format(self.fid))
        else:
            print('arrTi not found, run "<var>.calc_enit()"')


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
        tbx.prefig(xlabel='peak pulse potential [V]',
                   ylabel='current [$\mu$A]')
        plt.plot(volt, temp, label='{0} ms'.format(mstime))
        plt.title('{0} exponential fit, t = {1:.2f} ms'.format(fid, mstime),
                  fontsize=20)
        plt.plot(volt[argmin:], temp[argmin:])
        plt.plot(volt, [vstart for ii in volt], '--')
        plt.plot(volt, [vend for ii in volt], '--')
        plt.plot(volt[argmin-20:], exp_func(volt[argmin-20:], *popt), '--')
        plt.ylim(np.min(temp)*0.96, np.max(temp)*1.04)
        if save == 1:
            tbx.savefig('./img/{0}-currpeak-exp-fit-{1:.2f}ms.png'.format(
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


def dfunc(dataL, dataR, voltL=None, voltR=None, nwindow=31, nwindowR=None,
          order=20):
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
    # voltL/voltR to match length of gradL/gradR
    if (voltL is not None) and (voltR is not None):
        vL = voltL[int(nwindow/2):-int(nwindow/2)]
        vR = voltR[int(nwindowR/2):-int(nwindowR/2)]

    # Find the max
    argL = np.argmax(gradL)
    argR = np.argmax(gradR)

    # Slice curves and only keep the right side
    sliceL = np.array(gradL[argL:])
    sliceR = np.array(gradR[argR:])
    # Slice voltL/voltR as well, but also shift the starting value to zero
    if (voltL is not None) and (voltR is not None):
        vL = np.array(vL[argL:]) - vL[argL]
        vR = np.array(vR[argR:]) - vR[argR]
        revolt = np.concatenate([np.flip(-vL), vR])
    else:
        revolt = None

    # Normalize wrt right side of the curve
    factor = gradR[argR] / gradL[argL]
    index = np.arange(-len(sliceL), len(sliceR))

    return index, np.concatenate([np.flip(sliceL)*factor, sliceR]), revolt


def get_dfunc(cacurr, snw=41, passes=3):
    nt, nvolt = cacurr.shape
    nvolt -= 2*(snw//2)

    cacurr_sm = np.empty((nt,nvolt))
    grad_sm = np.empty((nt,nvolt))
    
    for tt in range(nt):
        tbx.progress_bar(tt, nt, label='tt')
        x, y, grad = sgsmooth(cacurr[tt,:], nwindow=snw, repeat=passes)
        cacurr_sm[tt,:] = y
        grad_sm[tt,:] = grad
    return x, cacurr_sm, grad_sm
