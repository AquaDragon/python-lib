'''
NAME:           spikes.py
AUTHOR:         swjtang
DATE:           03 Mar 2021
DESCRIPTION:    A toolbox of functions relating to spike finding & statistics.
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

import lib.toolbox as tbx


def spike_finder(data, flip=1, **kwargs):
    # prom - required prominence of peaks
    if flip == 1:
        sampledata = -(data - np.max(data))
    else:
        sampledata = data - np.max(data)

    if 'distance' not in kwargs.keys():
        # [px] minimum horizontal dist between peaks
        kwargs['distance'] = 1000

    if 'width' not in kwargs.keys():
        # [px] required width of peaks
        kwargs['width'] = 100

    peaks, prop = scipy.signal.find_peaks(sampledata, **kwargs)
    # fwhm = scipy.signal.peak_widths(-sampledata, peaks, rel_height=0.5)
    # (fwhm output is the same as 'widths', 'width_heights', 'left_ips',
    #  'right_ips' in prop)
    return peaks, prop


def get_width_height(data, window=None, prom=0):
    # note: peaks[] gives the position of peaks in time, use that for the
    #  filter by time
    nt, nx, ny, nshots, nchan = data.shape

    if (window is None) | (window >= nt):
        bins = 1
    else:
        bins = int(np.ceil(nt/window))

    local_stats = np.zeros((3, bins, nx, ny))

    global_width, global_height, global_peaktime = [], [], []
    for xx in range(nx):
        for yy in range(ny):
            local_width, local_height, local_peaktime = [], [], []
            for shot in range(nshots):
                for chan in range(nchan):
                    tbx.progress_bar(
                        [xx, yy, shot, chan], [nx, ny, nshots, nchan],
                        label=['nx', 'ny', 'nshots', 'nchan'])

                    peaks, prop = spike_finder(data[:, xx, yy, shot, chan],
                                               prom=prom)

                    local_width = np.append(local_width, prop['widths'])
                    local_height = np.append(local_height, prop['prominences'])
                    local_peaktime = np.append(local_peaktime, peaks)

            # bin the data here by timing, avoid re-running spike_finder()
            local_stats[:, :, xx, yy] = time_sorted_peaks(
                peaks, local_width, local_height, nt, window=window)
            global_width = np.append(global_width, local_width)
            global_height = np.append(global_height, local_height)
            global_peaktime = np.append(global_peaktime, local_peaktime)

    return global_width, global_height, global_peaktime, local_stats
    # peak width, peak height


def sort_peaktime(ptime, t1=0, t2=-1):
    arg = np.where((ptime >= t1) & (ptime < t2))
    return arg[0], ptime[arg]


def time_sorted_peaks(peaks, data1, data2, nt, window=5000):
    # peaks is the array containing time positions of the peaks (in px)
    bins = int(np.ceil(nt/window))
    mean_arr1 = np.empty(bins)
    mean_arr2 = np.empty(bins)
    count_arr = np.empty(bins)
    for ii in range(bins):
        arg, pdata = sort_peaktime(peaks, ii*window, (ii+1)*window)
        if len(peaks) == 0:
            count_arr[ii], mean_arr1[ii], mean_arr2[ii] = 0, 0, 0
        else:
            count_arr[ii] = len(arg)
            mean_arr1[ii] = np.mean(data1[arg])
            mean_arr2[ii] = np.mean(data2[arg])
    return [count_arr, mean_arr1, mean_arr2]
