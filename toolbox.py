'''
NAME:           toolbox.py
AUTHOR:         swjtang
DATE:           20 Apr 2021
DESCRIPTION:    A toolbox of commonly used functions.
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
from scipy.io.idl import readsav    # for read_IDL_sav
import scipy.signal


'''----------------------------------------------------------------------------
                VISUALIZATION
-------------------------------------------------------------------------------
'''


def progress_bar(cur_arr, tot_arr, label=None, header=''):
    ''' ----------------------------------------------------------------------
    Prints a output/progress bar for jupyter.
        cur_arr = A list of indices of the current progess.
        tot_arr = A list of indices indicating the end of progress.
        label   = A list of strings used to label the indices.
        header  = A string placed at the start of the progress bar.
    '''
    def convert_type(var):
        if isinstance(var, (int, str)):
            return np.array([var])
        else:
            return np.array(var)

    cur_arr, tot_arr = convert_type(cur_arr), convert_type(tot_arr)

    if cur_arr.shape != tot_arr.shape:
        print('!!! progress_bar: lists of unequal length')
    else:
        if label is None:
            prlabel = ''
        else:
            label = convert_type(label)
            prlabel = '(' + '/'.join([str(ii) for ii in label]) + ') = '
        # indices start from 0
        prcurrent = '(' + '/'.join([str(ii+1) for ii in cur_arr]) + ')'
        prtotal = '(' + '/'.join([str(ii) for ii in tot_arr]) + ')'

        print('\r{0} {1}{2} of {3}'.format(header, prlabel, prcurrent,
              prtotal), end='')
        if prcurrent == prtotal:
            print('')


def show_progress_bar(*args, **kwargs):
    # ALIAS FOR progress_bar
    progress_bar(*args, **kwargs)


def prefig(figsize=(16, 4.5), labelsize=35, ticksize=25, xlabel='x',
           ylabel='y'):
    ''' ----------------------------------------------------------------------
    Preamble to create a figure with appropriate labels.
        figsize             = plot dimensions in [width,height]
        labelsize, ticksize = font sizes of labels indicated in name
        xlabel, ylabel      = text label values
    '''
    fig = plt.figure(figsize=figsize)
    plt.xlabel(xlabel, fontsize=labelsize)
    plt.ylabel(ylabel, fontsize=labelsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    return fig


def get_line_profile(x, y, data, axis='x', value=0):
    ''' ----------------------------------------------------------------------
    Extract line profile at (axis=value) from 2D plot.
        x, y = 1D arrays of x/y-axis values
        data = 2D array with dimensions equal to len(x),len(y)
        axis  = choose 'x' or 'y'
        value = the value to cut the data at
    '''
    if axis == 'y':
        y_arg = value_locate_arg(y, value)
        print('Extracting x-axis at y = {0:.2f}...'.format(y[y_arg]))
        return data[:, y_arg]
    elif axis == 'x':
        x_arg = value_locate_arg(x, value)
        print('Extracting y-axis at x = {0:.2f}...'.format(x[x_arg]))
        return data[x_arg, :]
    else:
        print('Axis not specified, input x or y')


'''---------------------------------------------------------------------------
                FILTERING
------------------------------------------------------------------------------
'''


def filter_bint(data, dt=1, mrange=None, axis=0, quiet=0):
    ''' ----------------------------------------------------------------------
    A function used to integrate B-dot. Assumes first dimension is time.
        (IDL: filter_bint.pro)
    INPUTS:
        data    = n-dimensional array containing data to be integrated
    OPTIONAL:
        dt      = Time between two adjacent data points
        mrange  = 2-element array indicating start and stop indices of array
                  to take the mean.
    '''
    qprint(quiet, 'Integrating B-dot data...', end=' ')
    mean_val = np.mean(data, axis=axis)
    if mrange is not None:
        if len(mrange) == 2:
            mean_val = np.mean(data[int(mrange[0]):int(mrange[1]), ...],
                               axis=axis)
    qprint(quiet, 'Mean complete, taking cumulative sum...')
    bint_data = np.cumsum(data-mean_val, axis=axis)*dt
    qprint(quiet, 'Done!')
    # mean_val is broadcast into the dimensions of data, requiring the first
    # dimension to be time or the trailing axes won't align.
    return bint_data


def filterfreq(data, time, ftype=0, f0=0, width=1.0, debug=0, frange=None):
    ''' ----------------------------------------------------------------------
    A function to apply filters to a data set (non-FFT).
        (IDL: filterfreq.pro)
    '''
    # Set default frange
    if frange is None:
        frange = [0, 50]

    fftarr = np.fft.fft(data, norm='ortho')
    freqarr = np.fft.fftfreq(len(time), time[1]-time[0])
    freqarr /= 1e3    # [kHz]

    # determine filter type ------------------------------------------------
    if ftype == 0:
        output = fftarr
        tempfilter = [1 for ii in range(len(fftarr))]
    elif ftype in [1, 'low', 2, 'high']:    # high/low-pass
        cutoff = value_locate_arg(freqarr, f0)
        tempfilter = np.exp(-(freqarr-f0)**2/(2*width**2))

        tempfilter[:cutoff] = 1  # compute as low-pass
        for ii in range(int(len(tempfilter)/2)+1):
            tempfilter[-ii] = tempfilter[ii]

        if ftype in [2, 'high']:
            tempfilter = [1-ii for ii in tempfilter]  # high-pass
    elif ftype in [3, 'gaussian']:
        tempfilter = np.exp(-(freqarr-f0)**2/(2*width**2))
        for ii in range(int(len(tempfilter)/2)+1):
            tempfilter[-ii] = tempfilter[ii]

    tempfilter = np.array(tempfilter)
    data_filtered = np.fft.ifft(np.array(fftarr)*tempfilter, norm='ortho')

    # plot graphs for debug ------------------------------------------------
    if debug != 0:
        def norm_data(data):
            temp = abs(data)
            temp /= np.max(temp)
            return temp

        norm_fftarr = norm_data(fftarr)
        norm_filtered = norm_fftarr * tempfilter

        # sorted indices (for negative frequencies)
        sort_ind = np.argsort(freqarr)

        fig = plt.figure(figsize=(8, 9))
        # plot the FFT graphs ----------------------------------------------
        ax1 = fig.add_subplot(211)
        plt.plot(freqarr[sort_ind], norm_fftarr[sort_ind])
        plt.plot(freqarr[sort_ind], norm_filtered[sort_ind], alpha=0.7)

        plt.title('[Debug] FFT filter range', fontsize=18)
        plt.xlabel('Frequency [kHz]', fontsize=20)
        plt.ylabel('Amplitude', fontsize=20)

        x1, x2 = frange[0], frange[1]
        x1_ind = value_locate_arg(freqarr, x1)
        x2_ind = value_locate_arg(freqarr, x2)
        y2 = np.amax(abs(norm_fftarr[x1_ind:x2_ind]))
        plt.xlim([x1, x2])
        plt.ylim([0, 1.05*y2])
        plt.plot(freqarr[sort_ind], tempfilter[sort_ind]*y2, 'g--', alpha=0.5)
        plt.legend(['original', 'filtered', 'filter (norm to graph)'],
                   fontsize=14)

        # plot the data graphs ----------------------------------------------
        ax2 = fig.add_subplot(212)
        plt.plot(np.array(time)*1e3, data)
        plt.plot(np.array(time)*1e3, np.real(data_filtered))

        plt.title('[Debug] Data with/without filter', fontsize=18)
        plt.xlabel('Time [ms]', fontsize=20)
        plt.ylabel('Amplitude', fontsize=20)
        plt.legend(['original', 'filtered'], fontsize=14)

        plt.tight_layout()
        # fig.savefig('temp2.png', bbox_inches='tight')

    return np.real(data_filtered)


def remove_outliers(data, drange=None):
    ''' ----------------------------------------------------------------------
    Looks for outliers in the data and removes them
        data = array of dimensions (nt,nx,ny,nshot,nchan)
        drange = 2 element array, remove data that does not fall within range.
    '''
    # Set default drange
    if drange is None:
        drange = [0, 1]
    # copy array so that original does not get overwritten
    temp = copy.copy(data)
    index = np.where((temp < drange[0]) | (temp > drange[1]))
    temp[index] = None
    return temp


def smooth(data, nwindow=351, polyn=2, **kwargs):
    ''' ----------------------------------------------------------------------
    Wrapper for smoothing function using Savitzky-Golay filter. See
    scipy.signal.savgol_filter() documentation for info.
        data    = data to be smoothed
        nwindow = Length of the filter window. Must be a positive odd
                  integer
        polyn   = Order of the polynomial used to fit the samples.
                  Must be less than nwindow.
    '''
    # If nwindow is even, make it odd
    if np.mod(nwindow, 2) == 0:
        nwindow += 1
    return scipy.signal.savgol_filter(data, nwindow, polyn, **kwargs)


''' --------------------------------------------------------------------------
                FFT ROUTINES
------------------------------------------------------------------------------
'''


def average_fft(data, time, dt=1, axis=0, avgflag=1):
    ''' --------------------------------------------------------------------------
    FFT then average over dimensions in a multi-dimensional dataset.
        (IDL: avgfft.pro)
        data = array of dimensions (nt,nx,ny,nshot,nchan)
        time = array of time values\
        axis = the index of the time axis on the data (does not get averaged)
        avgflag = Set to any other value to disable averaging. Default is 1.
    '''
    ndim = len(data.shape)
    if len(time) != 0:
        dt = time[1]-time[0]    # if time array or dt exists, set dt value
    freqarr = np.fft.fftfreq(data.shape[axis], dt)
    sort_ind = np.where(freqarr >= 0)  # returns only positive frequencies

    print('Calculating FFTs...', end=' ')
    fftarr = np.fft.fft(data, axis=axis, norm='ortho')
    if avgflag == 1:
        print('Averaging FFTs...', end=' ')
        fftavg = np.mean(abs(fftarr), axis=tuple([ii for ii in range(ndim) if
                         (ii != axis)]))
        print('Done!')
        return fftavg[sort_ind], freqarr[sort_ind]
    else:
        print('Done!')
        return fftarr[sort_ind], freqarr[sort_ind]


def fft(*args, **kwargs):
    # ALIAS of average_fft
    return average_fft(*args, **kwargs)


def plot_fft(data, freqarr, frange=None, units='kHz', title='set title=',
             figsize=(8, 4.5), fname=None, save=None, ylim=None):
    ''' ----------------------------------------------------------------------
    Plots the FFT of a given dataset.
    INPUTS:
        data    = The FFT spectra of the data to be plotted
        freqarr = The frequency array of the corresponding FFT
                  Input frequencies in kHz.
    OPTIONAL:
        frange  = Range of data to show
        units   = The units of the frequency array
        title   = The title of the plot to be displayed
        fname   = The file name to be displayed as a subtitle
        save    = The directory to save the final image
    '''
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18

    fig = plt.figure(figsize=figsize)
    plt.plot(freqarr, data)

    if fname is not None:
        plt.suptitle(title, y=1.015, fontsize=24)
        plt.title(fname, y=1.01, fontsize=10)
    else:
        plt.title(title, fontsize=24)

    plt.xlabel('Frequency [' + units + ']', fontsize=20)
    plt.ylabel('Amplitude', fontsize=20)

    if frange is not None:   # sets the plot range
        x1, x2 = frange[0], frange[1]
        x1_ind = value_locate_arg(freqarr, x1)
        x2_ind = value_locate_arg(freqarr, x2)
        if x1_ind == x2_ind:
            print('!!! Plot range too small! Check units?')
        else:
            if ylim is not None:
                y2 = ylim
            else:
                y2 = np.amax(data[x1_ind:x2_ind])
            plt.xlim([x1, x2])
            # scales the plot so that the max peak is visible
            plt.ylim([0, 1.05*y2])
    else:
        x1_ind, x2_ind = None, None

    if save is not None:
        finalsvpath = check_save_filepath(save, 'image')
        fig.savefig(finalsvpath, bbox_inches='tight')
        print('File saved to = '+finalsvpath)

    temp = {
        'x1': x1_ind, 'x2': x2_ind    # provide the indices for the freq range
    }
    return temp


def fft_peak_find(fftdata, freqarr, frange=None, plot=0):
    ''' ----------------------------------------------------------------------
    Finds the most prominent peak frequency in the FFT by slicing.
        fftdata = The FFT spectra of the data
        freqarr = The frequency array of the corresponidng FFT
        frange  = Range of data to slice to find the peak
        plot    = (optional) Set to any value to disable plot
    '''
    # Set default frange
    if frange is None:
        frange = [0, 1]

    ii = value_locate_arg(freqarr, frange[0])
    jj = value_locate_arg(freqarr, frange[1])

    if plot == 1:
        prefig()
        plt.plot(freqarr[ii:jj], fftdata[ii:jj])

    argmax = np.argmax(fftdata[ii:jj])
    peak_f = freqarr[ii+argmax]
    print(peak_f)
    return peak_f


''' --------------------------------------------------------------------------
                MATH & CALCULATIONS
------------------------------------------------------------------------------
'''


def c_correlate(sig1, sig2):
    ''' ----------------------------------------------------------------------
    Calculates the NORMALIZED cross-correlation Pxy(L) as a function of lag L.
        (IDL: c_correlate.pro)
    '''
    c12 = np.correlate(sig1, sig2, 'same')
    c11 = np.correlate(sig1, sig1)  # auto-correlation of signal 1
    c22 = np.correlate(sig2, sig2)  # auto-correlation of signal 2

    return c12/np.sqrt(c11*c22)


def value_locate(array, value):
    ''' ----------------------------------------------------------------------
    Finds the nearest value in an array and returns the value.
        (IDL: value_locate.pro)
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def value_locate_arg(array, value):
    ''' ----------------------------------------------------------------------
    Finds the nearest value in an array and returns the argument.
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def rungeKutta(x0, y0, x, h):
    ''' ----------------------------------------------------------------------
    4th order Runge-Kutta. Finds value of y for a given x using step size h
    and initial value y0 at x0.
        (IDL: RK4.pro)
    '''
    # Count number of iterations using step size or step height h
    n = (int)((x - x0)/h)
    # Iterate for number of iterations
    y = y0
    for i in range(1, n + 1):
        # Apply Runge-Kutta Formulas to find next value of y
        k1 = h * dydx(x0, y)
        k2 = h * dydx(x0 + 0.5*h, y + 0.5*k1)
        k3 = h * dydx(x0 + 0.5*h, y + 0.5*k2)
        k4 = h * dydx(x0 + h, y + k3)

        # Update next value of y
        y = y + (1.0 / 6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # Update next value of x
        x0 = x0 + h
    return y


def check_save_filepath(save, file_type):
    ''' ----------------------------------------------------------------------
    Checks if a specified filepath is valid
        save      = The path to the intended save file
        file_type = The type of file intended to be saved as
    '''
    if save in [None, '']:
        return ''
    else:
        chkdir, chkfile = os.path.split(save)
        chkname, chkext = os.path.splitext(chkfile)

        if chkdir == '':                        # check if directory specified
            svdir = './'
        elif os.path.isdir(chkdir) is False:    # check if directory exists
            print('!!! [save=] Specified directory does not exist. Saving to'
                  ' current directory.')
            svdir = './'
        else:
            svdir = ''
            if chkdir[0] != '.':
                svdir += '.'
            svdir += chkdir
            if chkdir[-1] != '/':
                svdir += '/'

        # checks the type of file to be saved as
        if file_type == 'image':
            valid_ext = ['png', 'jpg', 'gif', 'bmp', 'pdf']
        elif file_type == 'video':
            valid_ext = ['mp4', 'mov']  # add more

        # check if valid extension, if not save with first valid extension
        if chkext == '':
            svfile = chkfile + '.' + valid_ext[0]
        elif chkext[1:] in valid_ext:
            svfile = chkfile
        else:
            svfile = 'temp.' + valid_ext[0]
            print('!!! [save=] Invalid image extension! Saving as ' + svfile)

        return svdir + svfile


''' --------------------------------------------------------------------------
                AUXILLARY FUNCTIONS
------------------------------------------------------------------------------
'''


def read_IDL_sav(fdir, fname):
    ''' ----------------------------------------------------------------------
    Reads IDL .sav files.
        fdir  = The file directory
        fname = The name of the file
    '''
    ff = readsav(fdir+fname)
    print('Reading... {0}'.format(fdir+fname))
    print(' -> Keys: {0}'.format(list(ff.keys())))
    return ff


def port_to_z(port, p0=None):
    ''' ----------------------------------------------------------------------
    Returns position of port w.r.t fixed port p0.
        port  = The port position in cm
        p0    = If this is set, the port position will be with respect to
                this port.
    '''
    if p0 is None:
        return -31.95*(port)+1693.35
    else:
        return -31.95*(port-p0)


def qprint(quiet, *args, **kwargs):
    ''' ----------------------------------------------------------------------
    A print function with a switch to suppress print outputs.
        quiet  = If set to anything other than zero it will not print
    '''
    if quiet == 0:
        print(*args, **kwargs)


def read_ascii(fname):
    ''' ----------------------------------------------------------------------
    A function to read ascii files from the Tektronic scope and load it into
    an array. Output arrays are time and data.
    '''
    data = np.loadtxt(fname, skiprows=5, delimiter=',')
    return data[:, 0], data[:, 1]


def savefig(*args, **kwargs):
    ''' ----------------------------------------------------------------------
    matplotlib's savefig, but define bbox_inches='tight' so that the axes does
    not get cropped.
    '''
    return plt.savefig(*args, **kwargs, bbox_inches='tight')
