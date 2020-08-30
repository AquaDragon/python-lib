'''
IDL CONJUGATE:  find_multiref_phase.pro
NAME:        	find_multiref_phase.py
AUTHOR:      	swjtang  
DATE:        	18 Jul 2020
DESCRIPTION:    (Analysis) Only for XY planes.
INPUTS:			data = A formatted numpy array in order of (nt,nx,ny,nshots,nchan)
'''
import numpy          as np
import lib.toolbox    as tbx
import lib.errorcheck as echk
from matplotlib import pyplot as plt
from scipy import signal

''' to reload a module :    
import importlib
importlib.reload(<module>)
-------------------------------------------------------------------------------
'''
def correlate_multi_signals(data, lagarray, passarray, trange=[5000,17500], filterflag=0):
    nt, nx, ny, nshots, nchan = data.shape
    t1, t2 = echk.check_trange(nt, trange[0], trange[1])
    
    copyarr  = np.empty([nt,nx,ny,nshots,nchan])
    corr_arr = np.zeros(data[t1:t2,:,:,0,:].shape)

    for xx in range(nx):
        for yy in range(ny):
            for ichan in range(nchan):
                temp = np.zeros(t2-t1)
                ii = 0
                for ss in range(nshots):
                    tbx.show_progress_bar([xx+1,yy+1,ss+1,ichan+1], [nx,ny,nshots,nchan], \
                                          label=['nx','ny','nshots','nchan'], \
                                          header='Conditionally averaging data...')
                    if passarray[xx,yy,ss,ichan] == 0:
                        lag = int(lagarray[xx,yy,ss,ichan])

                        # only filter if there is a need to extract the shot
                        if filterflag == 1 :
                            copyarr[:,xx,yy,ss,ichan] = tbx.filterfreq(data[:,xx,yy,ss,ichan], \
                                time, ftype='high', f0=1, width=1)
                            temp += copyarr[t1+lag:t2+lag,xx,yy,ss,ichan]
                        else:
                            temp += data[t1+lag:t2+lag,xx,yy,ss,ichan]
                        ii += 1
                if ii != 0: corr_arr[:,xx,yy,ichan]= temp/ii
    return(t1, t2, corr_arr)


#### ------------------------------------------------------------------------------------
def find_multiref_phase(data, time, ref=[0,0,0], trange=None, arg2pi=None, db=0, dbshot=[4,5,0,1]):
    ## Extract parameters from inputs ------------------------------------------------
    nt, nx, ny, nshots, nchan = data.shape              # data dimensions
    rx, ry, rs                = ref[0], ref[1], ref[2]  # reference shot ID (x,y,shot)
    
    ## Error checking ----------------------------------------------------------------
    if (trange == None) or (len(trange) != 2):
        print('!!! Unrecognized trange values. Requires trange=[t1,t2]')
        t1, t2 = 0, nt
    else:
        t1, t2 = echk.check_trange(nt, trange[0], trange[1])
    print('trange = {0}, {1}'.format(t1,t2))

    ## start -------------------------------------------------------------------------
    lagarr  = np.empty([nx,ny,nshots,nchan])   # the array to store the correlated data
    passarr = np.zeros([nx,ny,nshots,nchan])   # the array to determine if a shot should be skipped

    if db == 0:
        for xx in range(nx):
            for yy in range(ny):
                for ss in range(nshots):
                    for ichan in range(nchan):
                        tbx.show_progress_bar([xx+1,yy+1,ss+1, ichan+1], [nx,ny,nshots, nchan], \
                                          label=['nx', 'ny', 'nshots', 'nchan'])
                        refsig = data[t1:t2, rx, ry, rs, ichan]
                        sig1   = data[t1:t2, xx, yy, ss, ichan]

                        laginfo  = determine_lag_time(refsig, sig1, arg2pi=arg2pi)
                        if laginfo['error'] == 0  :   lagarr[xx,yy,ss,ichan]  = laginfo['xlag']
                        elif laginfo['error'] == 1 :  passarr[xx,yy,ss,ichan] = 1
    else:
        ref  = data[t1:t2, rx, ry, rs, dbshot[3]]
        sig  = data[t1:t2, dbshot[0],dbshot[1],dbshot[2],dbshot[3]]
        temp = determine_lag_time(ref, sig, arg2pi=arg2pi, db=db)

        return 0,0

    
    # display a figure for visualization
    plt.figure(figsize=(8,4.5))
    plt.plot(range(t1,t2), refsig-np.average(refsig))
    plt.plot(range(t1-int(lagarr[xx,yy,ss,ichan]), t2-int(lagarr[xx,yy,ss,ichan])), sig1-np.average(sig1))
    plt.legend(['Reference signal', 'Last signal in data'])
    
    # rejection percentage
    p_reject = (passarr==1).sum() / passarr.size
    print('Reference shot rejection rate = {0:.2f}%'.format(p_reject*100))
    if p_reject > 0.5: print('Reference shot is lousy! Choose another one.')

    return lagarr, passarr


#### ------------------------------------------------------------------------------------
def determine_lag_time(sig1, sig2, arg2pi=None, db=0, threshold=0.5):
    # ERROR CHECKING -----------------------------------------------------------
    if len(sig1) != len(sig2):
        print('!!! [find_multiref_phase] Length of two inputs are different!')
        return None
    if arg2pi == None: arg2pi= len(sig1)  #set default value of search range

    # PREPARING DATA: subtracting mean ----------------------------------------
    sig1 = sig1 - np.average(sig1)
    sig2 = sig2 - np.average(sig2)

    # CONSTRUCT THE LAG ARRAY -------------------------------------------------
    npoints = len(sig1)
    xarr    = [ii - npoints/2 for ii in range(npoints)]

    # PERFORM CROSS-CORRELATION -----------------------------------------------
    yarr = np.array(tbx.c_correlate(sig2, sig1))

    # FIND THE LOCATION OF ALL THE PEAKS
    if np.amax(yarr) <= threshold: 
        if db != 0: 
            print('Debug: Max correlation < threshold (={0})'.format(threshold))
            plt.plot(yarr)
        return {'error': 1}
    else:
        peak_find = signal.find_peaks(yarr.flatten(), height=threshold)
        xpeaks_0  = np.array([xarr[int(ii)] for ii in peak_find[0]])
        ypeaks_0  = np.array(peak_find[1]['peak_heights'])

        condition = np.where(abs(xpeaks_0) < arg2pi)
        xpeaks    = xpeaks_0[condition]
        ypeaks    = ypeaks_0[condition]

        if len(xpeaks)==0: 
            if db != 0: print('Debug: no correlation peaks found')
            return {'error': 1}

    # LOCATE MAXIMUM CROSS-CORRELATION PEAK, DETERMINE LAG-TIME ---------------
    ind  = np.argmax(ypeaks)
    xlag = xpeaks[ind] 

    # RETURN THIS OUTPUT ------------------------------------------------------
    test ={
        'xlag'  : xlag,

        'xpeaks': xpeaks,
        'ypeaks': ypeaks,
        'xarr'  : xarr,
        'yarr'  : yarr,
        'error' : 0
    }

    if db != 0: 
        plt.figure(figsize=(8,4.5))
        plt.plot(xarr, yarr)
        plt.plot(xpeaks, ypeaks, 'rx')
        plt.title('Correlation plot (debug mode)', fontsize=25)

    return test