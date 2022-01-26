'''
NAME:           get_bdot_calib.py
AUTHOR:         swjtang
DATE:           25 Jan 2022
DESCRIPTION:    Generates NA, the effective area of the probe (# turns * area)
                from the B-dot calibration files (different for Bx, By, Bz)
'''
import numpy as np
import struct         # for binary structure files
import glob           # file path manipulation
import os             # file path manipulation
import re             # regular expressions
import scipy.constants as const
from scipy.optimize import curve_fit
from matplotlib import animation, cm, pyplot as plt

# custom programs
from lib.toolbox import *
from lib.fname_tds import fname_tds
from lib.binary_to_ascii import b2a

# fname_ext = os.path.basename(fff)
# fname     = os.path.splitext(fname_ext)[0]
# print([os.path.basename(item) for item in flist])


# ----------------------------------------------------------------------------
# checks if the calibration files exist for a given probe
def probe_check(probe, fdir):
    flist = [os.path.basename(item) for item in glob.glob(fdir+'*.DAT')]
    calib = [probe+item for item in ['_BX.DAT', '_BY.DAT', '_BZ.DAT']]
    exist_check = [item in flist for item in calib]

    chreq = ', '.join([a for (a, b) in zip(['Bx', 'By', 'Bz'], exist_check)
                       if not b])
    if len(chreq) > 0:
        print('!!! Missing calibration for Bdot #{0}: ({1})'.
              format(probe, chreq))
        return None
    else:
        return calib


# ----------------------------------------------------------------------------
# List all the probes in the directory. Assumes filenames in the format
#  ""<probe>_BX.DAT".
def get_probe_list(fdir):
    names = [os.path.basename(item) for item in glob.glob(fdir+'*.DAT')]
    print('List of probes with calibrations: ',
          np.unique(np.array([re.split('_', item)[0] for item in names])))


# ----------------------------------------------------------------------------
# plots all 3 calibration curves
def calib_3plots(probeid, data1, data2, data3):
    fig, ax1 = plt.subplots(figsize=(10, 11.25/2))
    fig.subplots_adjust(hspace=0.05)

    lm1 = ax1.plot(data1['freq']/1e3, data1['logmag'], label='Bx LOGMAG')
    lm2 = ax1.plot(data2['freq']/1e3, data2['logmag'], label='By LOGMAG')
    lm3 = ax1.plot(data3['freq']/1e3, data3['logmag'], label='Bz LOGMAG')
    ax1.set_title('Calibration plots for Bdot probe #{0}'. format(probeid),
                  fontsize=18)
    ax1.set_ylabel('LOGMAG [dB]', fontsize=16)

    ax3 = ax1.twinx()
    ph1 = ax3.plot(data1['freq']/1e3, data1['phase'], linestyle=':',
                   label='Bx PHASE')
    ph2 = ax3.plot(data2['freq']/1e3, data2['phase'], linestyle=':',
                   label='By PHASE')
    ph3 = ax3.plot(data3['freq']/1e3, data3['phase'], linestyle=':',
                   label='Bz PHASE')
    ax3.set_ylabel('PHASE (degrees)', fontsize=16)
    lns = lm1+lm2+lm3+ph1+ph2+ph3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0, fontsize=16)

    ax1.tick_params(axis='both', labelsize=20)
    ax3.tick_params(axis='both', labelsize=20)


# ----------------------------------------------------------------------------
# PROBE AREA CALIBRATION
def area_calib(data, g=10, r=5.4e-2, label='', quiet=0, debug=0):
    # the data needs to contain frequency and logmag data in dict
    freq = data['freq']       # [Hz]
    logmag = data['logmag']   # [dB]

    ratioAR = [10**(ii/20) for ii in logmag]

    def fline(x, AA, BB):  # define straight line function y=f(x)
        return AA*x + BB

    # the gradient contains the area value
    popt, _ = curve_fit(fline, 2*np.pi*freq, ratioAR)  # fit to data x, y
    
    mu = const.mu_0
    area = popt[0] / (32 * (4/5)**1.5 * g * mu / r)    # here popt[0] is omega
    qprint(quiet, label+' Effective area = (Probe area * turns), NA = {0:.4f} '
           ' [cm^2]'.format(area*1e4))

    # Here we want to plot in frequency so multiply by 2pi so that we get omega
    popt[0] *= 2*np.pi

    if (debug != 0) and (quiet == 0):
        plt.figure(figsize=(8, 4.5))
        plt.title(label, fontsize=20)
        plt.plot(freq/1e3, ratioAR)
        plt.plot(freq/1e3, fline(freq, *popt))
        plt.xlabel('Frequency [kHz]', fontsize=16)
        plt.ylabel('magnitude A/R', fontsize=16)
        plt.tick_params(axis='both', labelsize=20)
        plt.legend(['original', 'best fit line'], fontsize=16)

    return area   # output is [m^2]


# ----------------------------------------------------------------------------
# This is the main routine, call this function
def get_bdot_calib(probeid='1', fdir='/home/swjtang/bdotcalib/', quiet=0,
                   debug=None, output=0, ch=2, **kwargs):
    # exception list of probe calibrations in channel 1 of VNA
    # July 2021: The new calibrations are almost exclusively channel 1
    if probeid in ['11', '20', '21', 'C12', 'C14']:
        ch = 1
    if quiet == 0:
        get_probe_list(fdir)
    bnames = probe_check(probeid, fdir)

    data1 = b2a(fdir+bnames[0], ch=ch, output=output, quiet=quiet, **kwargs)
    data2 = b2a(fdir+bnames[1], ch=ch, output=output, quiet=quiet, **kwargs)
    data3 = b2a(fdir+bnames[2], ch=ch, output=output, quiet=quiet, **kwargs)

    if debug is not None:
        calib_3plots(probeid, data1, data2, data3)

    areas = np.empty(3)
    for ii in range(3):
        data = [data1, data2, data3]
        label = ['BX', 'BY', 'BZ']
        areas[ii] = area_calib(data[ii], label=label[ii], quiet=quiet,
                               debug=debug)

    temp = {
        'probeid': probeid,
        'areas':   areas
    }
    return temp
