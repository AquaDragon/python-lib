'''
NAME:               average_fft_data.py
AUTHOR:             swjtang
DATE:               20 Apr 2021
DESCRIPTION:        Function to average FFT
REQUIRED INPUTS:    data = A formatted numpy array in order of
                           (nt,nx,ny,nshots,nchan)
                    time = A formatted numpy array with nt elements
OPTIONAL INPUTS:    bfield    = If set to 1, perform B-field integration
                    meanrange = The range over which the mean value is taken
                                for b_int
                    frange    = Frequency range of the displayed plot
                    fid       = filename, file ID
                    fname     = filename, name of saved plot
                    ch1, ch2  = filename, channel numbers
------------------------------------------------------------------------------
to reload module:
import importlib
importlib.reload(<module>)
------------------------------------------------------------------------------
'''
import lib.toolbox as tbx
import numpy as np
import os
import scipy.constants as const
from matplotlib import pyplot as plt


def average_fft_data(data, time, bfield=1, meanrange=None, frange=None, fid=0,
                     fname=None, ch1='(ch1)', ch2='(ch2)'):
    # Set default values
    if meanrange is None:
        meanrange = [0, 1000]
    if frange is None:
        frange = [2, 30]

    # Check inputs
    dt = time[1]-time[0]
    name = '{0}_FFT_ch{1}_{2}'.format(fid, ch1, ch2)
    if fname is not None:
        fname = name

    # Integrate the B-field signal
    if bfield == 1:
        temp1 = tbx.filter_bint(data, meanrange=meanrange)
    else:
        temp1 = data

    # Average the FFT over ALL dimensions
    avgfft, freqarr = tbx.average_fft(temp1, time)
    freqarr = freqarr/1e3    # [kHz]

    # Plot the FFT and save it
    temp2 = tbx.plot_fft(avgfft, freqarr, units='kHz', frange=frange,
                         title='Average FFT plot', fname=fname,
                         save=name+'.png')

    # Find the peak frequency of the flux rope oscillation
    x1 = tbx.value_locate_arg(freqarr, frange[0])
    x2 = tbx.value_locate_arg(freqarr, frange[1])

    peakfreq = freqarr[tbx.value_locate_arg(avgfft[x1:x2],
                       np.amax(avgfft[x1:x2]))+x1]
    x_period = int(np.ceil(1/(peakfreq*1e3*dt)))

    print('Peak frequency = {0:.2f} kHz ({1} pixels in data)'.
          format(peakfreq, x_period))

    return peakfreq, x_period
