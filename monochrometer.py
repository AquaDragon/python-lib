'''
NAME:           monochrometer.py
AUTHOR:         swjtang
DATE:           01 Jun 2021
DESCRIPTION:    A toolbox of functions related to the monochrometer / 
                spectrometer.
----------------------------------------------------------------------------
to reload module:
import importlib
importlib.reload(<module>)
-------------------------------------------------------------------------------
'''
from matplotlib import pyplot as plt
import numpy as np
import pickle
import scipy.signal
import scipy.optimize
from sklearn.linear_model import LinearRegression

import lib.toolbox as tbx


class pat():
    # Routines that are used to unpack the data written by Pat's monochrometer
    # program.
    def __init__():
        pass

    def write_csv(wavelengths, bintimes, histogram, nshots=1, fname=None):
        ''' ------------------------------------------------------------------
        Pat's routine to bin monochrometer data and write to csv file.
        INPUTS:
            wavelengths = 1D array of wavelengths
            bintimes    = 1D array of each bin's start time
            histogram   = 2D array of histogram data (#wavelengths x #times)
        OPTIONAL:
            nshots = Total number of shots per wavelength step
            fname  = Name of the savefile
        '''
        nlambda = wavelengths.shape[0]
        nt = bintimes.shape[0]

        if fname is None:
            fname = '2021-05-12 6405.2(.05)6407.7 400shots portB 20u slits'\
                    'coll above midplane'
        save_arr = np.zeros((nlambda+1, nt))
        save_arr[0, 0] = np.nan               # top left corner
        save_arr[0, 1:] = bintimes[:-1]       # first row is times
        save_arr[1:, 0] = wavelengths         # first column is wavelengths
        save_arr[1:, 1:] = histogram/nshots   # rest of array is histogram data
        csv_fn = fname+".csv"
        np.savetxt(csv_fn, save_arr, delimiter=',')
        print('wrote histograms to file "', csv_fn, '"', sep='')
        print('  first row = times')
        print('  first col = wavelengths')


    def unpickleh(fname=None, ttotal=20, tbin=100, nshots=200, nlambda=1,
                  hist=None):
        ''' ------------------------------------------------------------------
        Routine to unpickle the monochometer data then bin using histograms.
        INPUTS:
            fname   = Full filename + directory of the file
            ttotal  = [ms] Total duration of a single shot
            tbin    = Width of bin in pixels
            nshots  = Number of shots per wavelength step
            nlambda = Number of wavelength steps of the monochrometer
        '''
        pf = open(fname, "rb")    # pickle file
        ii = 0
        print('Unpickling the data...')

        while True:
            try:
                header = pickle.load(pf)
                ref = pickle.load(pf)

                # Initialization step, have to read one line of data first
                if ii == 0:
                    if hist is None:
                        nbins = int(ref.shape[0]/tbin)  # Total number of bins
                    elif hist =='8pi':
                        nbins = 9
                    elif hist == 'onoff':
                        nbins = 4
                    hist_arr = np.zeros((nlambda, nbins-1))  # Histogram
                    hlist = np.linspace(0, ttotal, nbins)
                    dt = ttotal/nbins

                pt = pickle.load(pf)
                py = pickle.load(pf)

                ## Changes way data is manipulated here
                if hist is None:
                    arr = histogram.regular(pt, bins=hlist)  # regular
                elif hist == '8pi':
                    arr = histogram.h8pi(ref, pt, ttotal=ttotal)  # ref peaks
                elif hist == 'onoff':
                    arr = histogram.onoff(ref, pt)

                hist_arr[ii//nshots] += arr

                ii += 1
            except EOFError:
                break
        print('Unpickling complete! Hist_arr shape = {0}'.format(
              hist_arr.shape))
        pf.close()
        return hlist, hist_arr, dt


class histogram():
    def __init__(self):
        pass

    def regular(data, **kwargs):
        hist, _ = np.histogram(data, **kwargs)
        return hist

    def h8pi(ref, data, ttotal=None):
        nt = len(ref)
        xval = np.array(range(nt))

        if ttotal is not None:
            time = np.linspace(0, ttotal, len(ref), endpoint=False)
        else:
            time = np.arange(len(ref))

        # Find peaks, use them to separate each cycle
        peaks, _ = scipy.signal.find_peaks(ref, distance=100, prominence=0.1)

        # Screen peaks after 10ms
        aaa = np.where((time[peaks] >= 10) & (time[peaks] < 15))
        peaks = peaks[aaa]

        arr = np.zeros((8))
        for ii in range(len(peaks)-1):
            bins = np.linspace(time[peaks[ii]], time[peaks[ii+1]], 9)
            hist, _ = np.histogram(data, bins=bins)
            arr += hist

        return arr

    def onoff(ref, data):
        # Time of : background on (start), rope on, rope off, signal end time
        bins = [0, 9, 15, 20]
        hist, _ = np.histogram(data, bins=bins)
        return hist


class cfit():
    def __init__(self):
        pass

    def desc():
        print('g - Gaussian, L - Lorentzian')
        print('L1/L2 = [nm] wavelength of 1st/2nd peak')
        print('a1/a2 = [count] amplitude of 1st/2nd peak')
        print('b1/b2 = [nm] width of 1st/2nd peak')
        print('c     = [count] baseline of plot')
        print('--')

    def gauss(wavelength, hist_arr, ind=1, title=None):
        # Input wavelength in nm
        xx = wavelength
        yy = hist_arr[:,ind]

        iimin = np.argmin(yy)
        iimax = np.argmax(yy)

        def twogauss_func(x, x1, a1, b1, x2, a2, b2, c):
            return a1 * np.exp(-((x-x1)/b1)**2) + a2 * np.exp(-((x-x2)/b2)**2) + c

        def twolorenz_func(x, x1, a1, b1, x2, a2, b2, c):
            return a1 / (1 + ((x-x1)/b1)**2) + a2 / (1 + ((x-x2)/b2)**2) + c

        tbx.prefig(xlabel='wavelength [nm]', ylabel='count')
        plt.step(xx, yy)
        if title is not None:
            plt.title(title, fontsize=30)

        # Initial guess
        guess = [320.3325, yy[iimax]-yy[iimin], 0.02, 320.32,
                 yy[iimax]-yy[iimin], 0.01, yy[iimin]]

        # Set boundaries
        bounds = [(320.325, 0, 0, 320.30, 0, 0, -np.inf),
                  (320.35, +np.inf, 0.2, 320.32, +np.inf, 0.2, +np.inf)]
        try:
            poptg, pcovg = scipy.optimize.curve_fit(twogauss_func, xx, yy,
                                                    p0=guess, bounds=bounds)
            poptl, pcovl = scipy.optimize.curve_fit(twolorenz_func, xx, yy,
                                                    p0=guess, bounds=bounds)

            print(ind, 'g L1={0:.3f}, a1={1:.1f}, b1={2:.3f}, L2={3:.3f}, '
                  'a2={4:.1f}, b2={5:.3f}, c={6:.1f}'.format(*poptg))
            print(ind, 'L L1={0:.3f}, a1={1:.1f}, b1={2:.3f}, L2={3:.3f}, '
                  'a2={4:.1f}, b2={5:.3f}, c={6:.1f}'.format(*poptl))
            print('--')
            plt.plot(xx, twogauss_func(xx, *poptg), label='gaussian')
            plt.plot(xx, twolorenz_func(xx, *poptl), label='lorentzian')
            plt.legend(fontsize=20)
        except RuntimeError:
            pass



class plot():
    def __init__(self):
        pass

    def hist8pi(wavelength, hist_arr, date='2021-05-xx', port='41B'):
        # Expecting wavelength in units of angstrom
        tbx.prefig(xlabel='wavelength [nm]', ylabel='counts')
        plt.title('{0} monochrometer port {1}'.format(date, port), fontsize=20)
        for ii in range(8):
            plt.plot(wavelength/10, hist_arr[:,ii],
                     label='{0:.2f}$\pi$'.format(ii/4))
        plt.legend(fontsize=20, loc='upper left')
        tbx.savefig('./img/{0}-mc-plot-8pi.png'.format(date))

    def onoff(wavelength, hist_arr, date='2021-05-xx', port='41B'):
        # Expecting wavelength in units of angstrom
        tbx.prefig(figsize=(16,7), xlabel='wavelength [nm]', ylabel='counts')
        plt.title('{0} monochrometer port {1}'.format(date, port), fontsize=20)
        labels = ['background (0-9ms)', 'rope on (9-15ms)',
                  'afterglow (15-20ms)']
        for ii in range(3):
            plt.step(wavelength/10, hist_arr[:,ii], label=labels[ii])
        plt.legend(fontsize=20, loc='upper left')
        tbx.savefig('./img/{0}_mc-plot-onoff.png')

        # Use the peaks to bin the histogram
        # bins = np.append(np.append(0, time[peaks]), time[-1])
        # hist, _ = np.histogram(data, bins=bins)
        # print('{0:.2f}'.format(time[peaks[0]]), hist,
        #       '{0:.2f}'.format(time[peaks[-1]]), len(data))

        # bin arrays by oscillation number
        # create blank array to track unbinned
        # arr = np.zeros(21)
        # blank = np.zeros(21)
        # nb = len(hist)
        # arr[:nb-1] = hist[:-1]
        # arr[-1] = hist[-1]
        # blank[nb-1:-1] = 1
        # return arr, blank

        # dt = time[peaks[1]] - time[peaks[0]]
        # print(dt, peaks[1]-peaks[0])

        # tbx.prefig()
        # plt.plot(xval, ref)
        # plt.plot(xval[peaks], ref[peaks], 'o', markersize=10)


class wavelength():
    def __init__(self):
        pass

    def calc(start, end, dL=1, factor=1):
        nlambda = int((end-start)/dL)
        wavelength = np.arange(start, end, dL) * factor
        return nlambda, wavelength


class angle():
    # Calculate volume observed by the optics
    def __init__(self):
        pass

    def linreg(X, y):
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        #print('R^2 = {0}'.format(r2))

        b0 = reg.intercept_
        b1 = reg.coef_
        #print(b0, b1)
        return b0, b1, r2