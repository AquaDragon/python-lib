'''
NAME:           monochrometer.py
AUTHOR:         swjtang
DATE:           21 May 2021
DESCRIPTION:    A toolbox of functions related to the monochrometer / 
                spectrometer.
----------------------------------------------------------------------------
to reload module:
import importlib
importlib.reload(<module>)
-------------------------------------------------------------------------------
'''
import copy
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
from scipy.io.idl import readsav    # for read_IDL_sav
import scipy.signal


class mc:
    ''' ----------------------------------------------------------------------
        MONOCHROMETER ROUTINES
    --------------------------------------------------------------------------
    '''
    def __init__():
        pass

    def pat_data_csv(wavelengths, bintimes, histogram, nshots=1, fname=None):
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


    def pat_unpickle_hist(fname=None, ttotal=20, tbin=100, nshots=200,
                          nlambda=1):
        ''' ------------------------------------------------------------------
        Routine to unpickle the monochometer data written by Pat.
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

                tbx.prefig()
                plt.plot(ref)

                # Initialization step, have to read one line of data first
                if ii == 0:
                    nbins = int(ref.shape[0]/tbin)  # Total number of bins
                    hist_arr = np.zeros((nlambda, nbins-1))  # Histogram
                    hlist = np.linspace(0, ttotal, nbins)
                    dt = ttotal/nbins

                pt = pickle.load(pf)
                py = pickle.load(pf)
                
                ## Changes way data is manipulated here
                hist, _ = np.histogram(pt, bins=hlist)
                hist_arr[ii//nshots] += hist

                ii += 1
                break
            except EOFError:
                break
        print('Unpickling complete!')
        pf.close()
        return hlist, hist_arr, dt