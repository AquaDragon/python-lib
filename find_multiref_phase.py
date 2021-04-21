'''
NAME:           find_multiref_phase.py  (IDL:find_multiref_phase.pro)
AUTHOR:         swjtang
DATE:           20 Apr 2021
DESCRIPTION:    Finds the phase between signals to be used in conditional
                averaging. Only for XY planes.
INPUTS:         data = A formatted numpy array in order of (nt,nx,ny,nshots,
                       nchan)
------------------------------------------------------------------------------
to reload a module:
import importlib
importlib.reload(<module>)
------------------------------------------------------------------------------
'''
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
import scipy

import lib.toolbox as tbx
import lib.errorcheck as echk


def correlate_multi_signals(data, lagarray, passarray, trange=None,
                            filterflag=0):
    # Set trange
    if trange is None:
        trange = [5000, 17500]

    nt, nx, ny, nshots, nchan = data.shape
    t1, t2 = echk.check_trange(nt, trange[0], trange[1])

    copyarr = np.empty([nt, nx, ny, nshots, nchan])
    corr_arr = np.zeros(data[t1:t2, :, :, 0, :].shape)

    for xx, yy, ichan in product(range(nx), range(ny), range(nchan)):
        tbx.progress_bar([xx, yy, ichan], [nx, ny, nchan],
                         label=['nx', 'ny', 'nchan'],
                         header='Conditionally averaging data...')
        temp = np.zeros(t2-t1)
        ii = 0
        for ss in range(nshots):
            if passarray[xx, yy, ss, ichan] == 0:
                lag = int(lagarray[xx, yy, ss, ichan])

                # only filter if there is a need to extract the shot
                if filterflag == 1:
                    copyarr[:, xx, yy, ss, ichan] = tbx.filterfreq(
                        data[:, xx, yy, ss, ichan], time, ftype='high',
                        f0=1, width=1)
                    temp += copyarr[t1+lag:t2+lag, xx, yy, ss, ichan]
                else:
                    temp += data[t1+lag:t2+lag, xx, yy, ss, ichan]
                ii += 1
        if ii != 0:
            corr_arr[:, xx, yy, ichan] = temp/ii
    return(t1, t2, corr_arr)


def find_multiref_phase(data, trange=None, ref=None, dbshot=None, **kwargs):
    ''' ----------------------------------------------------------------------
    Function which determines the lag time of all other shots in the array
    with respect to a reference shot.
    INPUTS:     data = 1D array of the fixed B-dot data.
    OPTIONAL:   trange = The start and end time of data to be correlated.
                         If unspecificed, it uses the entire data set.
                         Computationally intensive, so set reasonable values.
                ref    = The index (ix, iy, ishot) of the reference shot. If
                         unspecified, use the 1st shot of the entire dataset.
                dbshot = The index (ix, iy, ishot, ichan) used for debugging.
                **kwargs gets passed into lagtime()
    '''
    # Set default values for ref and dbshot
    if ref is None:
        ref = [0, 0, 0, 0]

    if dbshot is None:
        dbshot = [0, 1, 1, 0]

    nt, nx, ny, nshots, nchan = data.shape
    rx, ry, rs, rchan = ref[0], ref[1], ref[2], ref[3]

    # Error checking ---------------------------------------------------------
    if (trange is None) or (len(trange) != 2):
        print('!!! Unrecognized trange values. Requires trange=[t1, t2]')
        t1, t2 = 0, nt
    else:
        t1, t2 = echk.check_trange(nt, trange[0], trange[1])
    print('trange = {0}, {1}'.format(t1, t2))
    # ------------------------------------------------------------------------

    lagarr = np.empty([nx, ny, nshots, nchan])    # correlated data
    passarr = np.zeros([nx, ny, nshots, nchan])   # determine if shot skipped

    if 'db' not in kwargs:
        for xx, yy, ss, ichan in product(range(nx), range(ny), range(nshots),
                                         range(nchan)):
            # Execute for loop
            tbx.progress_bar([xx, yy, ss, ichan], [nx, ny, nshots, nchan],
                             label=['nx', 'ny', 'nshots', 'nchan'])
            ref1 = data[t1:t2, rx, ry, rs, ichan]
            sig1 = data[t1:t2, xx, yy, ss, ichan]

            laginfo = lagtime(ref1, sig1, **kwargs)
            if laginfo['error'] == 0:
                lagarr[xx, yy, ss, ichan] = laginfo['xlag']
            elif laginfo['error'] == 1:
                passarr[xx, yy, ss, ichan] = 1

    else:
        ref = data[t1:t2, rx, ry, rs, rchan]
        sig = data[t1:t2, dbshot[0], dbshot[1], dbshot[2], dbshot[3]]
        temp = lagtime(ref, sig, **kwargs)
        return 0, 0

    # display a figure for visualization
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(t1, t2), ref1-np.average(ref1))
    plt.plot(range(t1-int(lagarr[xx, yy, ss, ichan]), t2-int(lagarr[xx, yy,
             ss, ichan])), sig1-np.average(sig1))
    plt.legend(['Reference signal', 'Last signal in data'])

    # rejection percentage
    p_reject = (passarr == 1).sum() / passarr.size
    print('Shot rejection rate = {0:.2f}%'.format(p_reject*100))
    if p_reject > 0.5:
        print('Reference shot is lousy! Choose another one.')

    return lagarr, passarr


def lagtime(sig1, sig2, tmax=None, plot=0, threshold=0.6, quiet=0):
    '''
    OPTIONAL:   tmax      = The maximum number of pixels to search for the
                            correlation length. Default value is the entire
                            signal length.
                plot      = Show plots used for debugging.
                threshold = Value between 0 and 1. Used to select peaks above
                            certain level.
    '''
    # ERROR CHECKING
    if len(sig1) != len(sig2):
        print('!!! [lagtime] Length of two inputs are different!')
        return None

    if tmax is None:
        tmax = len(sig1)  # set default value of search range

    npoints = len(sig1)    # Number of points

    # Subtract the mean from the data
    sig1 = sig1 - np.mean(sig1)
    sig2 = sig2 - np.mean(sig2)

    # Perform cross-correlation
    corr = np.array(tbx.c_correlate(sig2, sig1))

    # FIND THE LOCATION OF ALL THE PEAKS
    if np.amax(corr) <= threshold:
        tbx.qprint(quiet, '!!! Max correlation less than set threshold (= '
                   '{0})'.format(threshold))
        if plot != 0:
            plt.plot(corr)
        return {
            'xlag':  None,
            'error': 1
            }
    else:
        peaks, props = scipy.signal.find_peaks(corr, height=threshold)
        xpeaks_0 = peaks - int(npoints/2)
        ypeaks_0 = props['peak_heights']

        condition = np.where((xpeaks_0 < tmax/2) & (xpeaks_0 > -tmax/2))
        xpeaks = xpeaks_0[condition]
        ypeaks = ypeaks_0[condition]

        if len(xpeaks) == 0:
            tbx.qprint(quiet, '!!! No peaks found with specified condition')
            return {
                'xlag':  None,
                'error': 1
                }

    # LOCATE MAXIMUM CROSS-CORRELATION PEAK, DETERMINE LAG-TIME
    ind = np.argmax(ypeaks)
    xlag = xpeaks[ind]

    # RETURN THIS OUTPUT ---------------------------------------
    test = {
        'xlag':   xlag,
        'xpeaks': xpeaks,
        'ypeaks': ypeaks,
        'corr':   corr,
        'error':  0
    }

    if plot != 0:
        plt.figure(figsize=(8, 4.5))
        plt.plot(np.arange(npoints)-int(npoints/2), corr)
        plt.plot(xpeaks, ypeaks, 'rx')
        plt.title('Correlation plot (debug mode)', fontsize=25)

    return test
