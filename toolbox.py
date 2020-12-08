'''
NAME:           toolbox.py
AUTHOR:         swjtang  
DATE:           01 Dec 2020
DESCRIPTION:    A toolbox of commonly used functions.
'''
import numpy as np
import os, copy, scipy.signal
from matplotlib import pyplot as plt
'''----------------------------------------------------------------------------
to reload module:
import importlib
importlib.reload(<module>)
-------------------------------------------------------------------------------
DESCRIPTION:    (Visualization) Plots the FFT of a given dataset.
INPUTS:         data    = The FFT spectra of the data to be plotted
                freqarr = The frequency array of the corresponidng FFT
                frange  = (optional) Range of data to show
                units   = (optional) The units of the frequency array
                title   = (optional) The title of the plot to be displayed
                fname   = (optional) The file name to be displayed as a subtitle
                save    = (optional) The directory to save the final image
'''
def plot_fft(data, freqarr, frange=None, units='kHz', title='<insert title>',  \
        fname=None, save=None):
    plt.rcParams['xtick.labelsize']=18
    plt.rcParams['ytick.labelsize']=18

    fig = plt.figure(figsize=(8,4.5))
    plt.plot(freqarr, data)

    if fname != None: 
        plt.suptitle(title, y=1.015, fontsize=24)
        plt.title(fname, y=1.01, fontsize=10)
    else:
        plt.title(title, fontsize=24)

    plt.xlabel('Frequency [' + units + ']', fontsize=20)
    plt.ylabel('Amplitude', fontsize=20)

    if frange != None:   # sets the plot range
        x1, x2 = frange[0], frange[1]
        x1_ind, x2_ind = value_locate_arg(freqarr, x1), value_locate_arg(freqarr, x2)
        if x1_ind == x2_ind: print('!!! Plot range too small! Check units?')
        else:
            y2 = np.amax(data[x1_ind:x2_ind])
            plt.xlim([x1,x2])
            plt.ylim([0,1.05*y2])  #scales the plot so that the max peak is visible
    else:
        x1_ind, x2_ind = None, None

    if save != None:
        finalsvpath=check_save_filepath(save, 'image')

        fig.savefig(finalsvpath, bbox_inches='tight')
        print('File saved to = '+finalsvpath)

    temp ={
        'x1': x1_ind, 'x2': x2_ind    # provide the indices for the freq range
    }
    return temp

''' ---------------------------------------------------------------------------
DESCRIPTION:    (Visualization) Prints a output/progress bar for jupyter.
INPUTS:         cur_arr = A list of Indices of the current progess
                tot_arr = A list of indices indicating the end of progress,
                label   = A list of strings used to label the indices.
                header  = A string placed at the start of the progress bar.
'''
### wrapper function/alias
def show_progress_bar(cur_arr, tot_arr, label=None, header=''):
    progress_bar(cur_arr, tot_arr, label=label, header=header)

def progress_bar(cur_arr, tot_arr, label=None, header=''):
    def convert_type(var):
        if isinstance(var, (int,str)): 
            return np.array([var])
        else: 
            return np.array(var)
    cur_arr, tot_arr = convert_type(cur_arr), convert_type(tot_arr)

    if cur_arr.shape != tot_arr.shape: 
        print('!!! progress_bar: lists of unequal length')
    else:
        if label == None:
            prlabel = ''
        else:
            label = convert_type(label)
            prlabel = '(' + '/'.join([str(ii) for ii in label]) + ') = '
        prcurrent = '(' + '/'.join([str(ii+1) for ii in cur_arr]) + ')' #indices start from 0
        prtotal   = '(' + '/'.join([str(ii) for ii in tot_arr])   + ')'

        print('\r{0} {1}{2} of {3}'.format(header,prlabel,prcurrent,prtotal), end='')
        if prcurrent == prtotal: print('')


''' ---------------------------------------------------------------------------
IDL EQUIV:      prefig.pro
DESCRIPTION:    (Visualization) Preamble to create a figure with appropriate labels.
INPUTS:         figsize             = plot dimensions in [width,height]
                labelsize, ticksize = font sizes of labels indicated in name
                xlabel, ylabel      = text label values
'''
def prefig(figsize=[16,4.5], labelsize=40, ticksize=25, xlabel='x', ylabel='y'):
    fig = plt.figure(figsize=figsize)
    plt.xlabel(xlabel, fontsize=labelsize)
    plt.ylabel(ylabel, fontsize=labelsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    return fig

''' ---------------------------------------------------------------------------
IDL EQUIV:      filter_bint.pro
DESCRIPTION:    (Filter) A function used to integrate B-dot. Assumes first 
                dimension is time.
INPUTS:         data       = n-dimensional array containing data to be integrated
                dt         = (optional) time between two adjacent data points
                meanrange  = (optional) 2-element array indicating start and 
                             stop indices of array to take the mean. 
'''
def filter_bint(data, dt=None, meanrange=None):
    print('Integrating B-dot data...', end=' ')
    mean_val = np.mean(data, axis=0)
    if meanrange != None:
        if len(meanrange) == 2 : 
            mean_val = np.mean(data[int(meanrange[0]):int(meanrange[1]),...], axis=0)
    if dt == None: dt=1

    print('Done!')
    # mean_val is broadcast into the dimensions of data, requiring the first dimension
    # to be time or the trailing axes won't align.
    return np.cumsum(data-mean_val, axis=0)*dt

''' ---------------------------------------------------------------------------
IDL EQUIV:      filterfreq.pro
DESCRIPTION:    (Filter) A function to apply filters to a data set (non-FFT).
INPUTS: 
'''
def filterfreq(data, time, ftype=0, f0=0, width=1.0, debug=0, frange=[0,50]):
    fftarr  = np.fft.fft(data, norm='ortho')
    freqarr = np.fft.fftfreq(len(time), time[1]-time[0])
    freqarr /= 1e3    # [kHz]

    #### determine filter type ------------------------------------------------------
    if ftype == 0:
        output = fftarr
        tempfilter = [1 for ii in range(len(fftarr))]
    elif ftype in [1,'low',  2,'high']:    #high/low-pass
        cutoff = value_locate_arg(freqarr, f0)
        tempfilter = np.exp(-(freqarr-f0)**2/(2*width**2))

        tempfilter[:cutoff]=1  # compute as low-pass
        for ii in range(int(len(tempfilter)/2)+1):
            tempfilter[-ii] = tempfilter[ii]

        if ftype in [2, 'high']: tempfilter = [1-ii for ii in tempfilter]  # high-pass
    elif ftype in [3,'gaussian']:
        tempfilter = np.exp(-(freqarr-f0)**2/(2*width**2))
        for ii in range(int(len(tempfilter)/2)+1):
            tempfilter[-ii] = tempfilter[ii]

    tempfilter    = np.array(tempfilter)
    data_filtered = np.fft.ifft(np.array(fftarr)*tempfilter, norm='ortho')

    #### plot graphs for debug ------------------------------------------------------
    if debug != 0:
        def norm_data(data):
            temp = abs(data)
            temp /= np.max(temp)
            return temp

        norm_fftarr   = norm_data(fftarr)
        norm_filtered = norm_fftarr * tempfilter

        sort_ind = np.argsort(freqarr)    # sorted indices (for negative frequencies)

        fig = plt.figure(figsize=(8,9))
        #### plot the FFT graphs ----------------------------------------------
        ax1 = fig.add_subplot(211)
        plt.plot(freqarr[sort_ind], norm_fftarr[sort_ind])
        plt.plot(freqarr[sort_ind], norm_filtered[sort_ind], alpha=0.7)

        plt.title('[Debug] FFT filter range', fontsize=18)
        plt.xlabel('Frequency [kHz]', fontsize=20)
        plt.ylabel('Amplitude', fontsize=20)

        x1, x2 = frange[0], frange[1]
        x1_ind, x2_ind = value_locate_arg(freqarr, x1), value_locate_arg(freqarr, x2)
        y2 = np.amax(abs(norm_fftarr[x1_ind:x2_ind]))
        plt.xlim([x1,x2])
        plt.ylim([0,1.05*y2])
        plt.plot(freqarr[sort_ind], tempfilter[sort_ind]*y2, 'g--', alpha=0.5)
        plt.legend(['original', 'filtered', 'filter (norm to graph)'], fontsize=14)

        #### plot the data graphs ----------------------------------------------
        ax2 = fig.add_subplot(212)
        plt.plot(np.array(time)*1e3, data)
        plt.plot(np.array(time)*1e3, np.real(data_filtered))

        plt.title('[Debug] Data with/without filter', fontsize=18)
        plt.xlabel('Time [ms]', fontsize=20)
        plt.ylabel('Amplitude', fontsize=20)
        plt.legend(['original', 'filtered'], fontsize=14)

        plt.tight_layout()
        #fig.savefig('temp2.png', bbox_inches='tight')

    return np.real(data_filtered)

''' ---------------------------------------------------------------------------
DESCRIPTION:    (Filter) looks for outliers in the data and removes them
INPUTS:         data = array of dimensions (nt,nx,ny,nshot,nchan)
                drange = 2 element array, remove data that does not fall within range
'''
def remove_outliers(data, drange=[0,1]):
    temp  = copy.copy(data)    # copy array so that original does not get overwritten
    index = np.where((temp < drange[0])|(temp > drange[1]))
    temp[index] = None
    return temp

''' ---------------------------------------------------------------------------
DESCRIPTION:    (Filter) Wrapper for smoothing function using Savitzky-Golay filter
                See scipy.signal.savgol_filter() documentation for info
INPUTS:         data    = data to be smoothed
                nwindow = Length of the filter window. Must be a positive odd integer
                polyn   = Order of the polynomial used to fit the samples. 
                          Must be less than nwindow.
'''
def smooth(data, nwindow=351, polyn=2):
    return scipy.signal.savgol_filter(data, nwindow, polyn)

''' ---------------------------------------------------------------------------
IDL EQUIV:      avgfft.pro
DESCRIPTION:    (Math) Averages the FFT over all other dimensions in the dataset.
INPUTS:         data = array of dimensions (nt,nx,ny,nshot,nchan)
'''
def average_fft(data, time, axis=0):
    print('Averaging FFTs...', end=' ')
    ndim = len(data.shape)
    fftarr  = np.fft.fft(data, axis=axis, norm='ortho')
    fftavg  = np.mean(abs(fftarr), axis=tuple([ii for ii in range(ndim) if (ii != axis)]))
    freqarr = np.fft.fftfreq(len(time), time[1]-time[0])

    sort_ind = np.where(freqarr >= 0)  # returns only positive frequencies

    print('Done!')
    return fftavg[sort_ind], freqarr[sort_ind]

''' ---------------------------------------------------------------------------
IDL EQUIV:      c_correlate.pro
DESCRIPTION:    (Math) Calculates the NORMALIZED cross-correlation Pxy(L) as a 
                function of lag L.
'''
def c_correlate(sig1, sig2):
    c12 = np.correlate(sig1, sig2, 'same')
    c11 = np.correlate(sig1, sig1)  # auto-correlation of signal 1
    c22 = np.correlate(sig2, sig2)  # auto-correlation of signal 2

    return [ii/np.sqrt(c11*c22) for ii in c12]

''' ---------------------------------------------------------------------------
IDL EQUIV:      value_locate.pro
DESCRIPTION:    (Math) Finds the nearest value in an array.
'''
def value_locate(array, value):
    array = np.asarray(array)
    idx   = (np.abs(array - value)).argmin()
    return array[idx]

def value_locate_arg(array, value):
    array = np.asarray(array)
    idx   = (np.abs(array - value)).argmin()
    return idx

''' ---------------------------------------------------------------------------
IDL EQUIV:      RK4.pro
DESCRIPTION:    (Math) 4th order Runge-Kutta
'''
# Finds value of y for a given x using step size h 
# and initial value y0 at x0. 
def rungeKutta(x0, y0, x, h): 
    # Count number of iterations using step size or 
    # step height h 
    n = (int)((x - x0)/h)  
    # Iterate for number of iterations 
    y = y0 
    for i in range(1, n + 1): 
        # Apply Runge Kutta Formulas to find next value of y
        k1 = h * dydx(x0, y) 
        k2 = h * dydx(x0 + 0.5*h, y + 0.5*k1) 
        k3 = h * dydx(x0 + 0.5*h, y + 0.5*k2) 
        k4 = h * dydx(x0 + h    , y + k3) 
  
        # Update next value of y 
        y = y + (1.0 / 6.0)*(k1 + 2*k2 + 2*k3 + k4) 
  
        # Update next value of x 
        x0 = x0 + h 
    return y 

''' ---------------------------------------------------------------------------
DESCRIPTION:    (Auxillary) Checks if a specified filepath is valid
INPUTS:         save      = The path to the intended save file
                file_type = The type of file intended to be saved as
'''
def check_save_filepath(save, file_type):
    if save in [None, '']: return ''
    else:
        chkdir, chkfile = os.path.split(save)
        chkname, chkext = os.path.splitext(chkfile)

        if chkdir == '': svdir ='./'            # check if directory specified
        elif os.path.isdir(chkdir) == False:    # check if directory exists
            print('!!! [save=] Specified directory does not exist. Saving to current directory.')
            svdir ='./'  
        else:
            svdir = ''
            if chkdir[0] != '.': svdir += '.'
            svdir += chkdir
            if chkdir[-1] != '/': svdir += '/'

        # checks the type of file to be saved as
        if file_type == 'image': valid_ext = ['png', 'jpg', 'gif', 'bmp', 'pdf']
        elif file_type == 'video': valid_ext = ['mp4', 'mov'] # add more

        # check if valid extension, if not save with first valid extension
        if chkext == '': svfile = chkfile + '.' + valid_ext[0]
        elif chkext[1:] in valid_ext: svfile = chkfile
        else:
            svfile = 'temp.'+ valid_ext[0]
            print('!!! [save=] Invalid image extension! Saving as ' + svfile)
            
        return svdir + svfile

''' ---------------------------------------------------------------------------
DESCRIPTION:    (Auxillary) Returns position of port wrt fixed port p0
INPUTS:         port  = The port position in cm
                p0    = If this is set, the port position will be with respect to this port.
'''     
def port_to_z(port, p0=None):
    if p0 == None: return -31.95*(port)+1693.35
    else: return -31.95*(port-p0)

''' ---------------------------------------------------------------------------
DESCRIPTION:    (Auxillary) A switch function used to suppress print outputs
INPUTS:         quiet  = If set to anything other than zero it will not print
                text   = The string to be printed. Unlike print(), it only accepts one input.
'''
def qprint(quiet, text):
    if quiet == 0: print(text)