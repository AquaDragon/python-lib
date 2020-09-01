'''
NAME:           complexity_entropy.py
AUTHOR:         swjtang  
DATE:           31 Aug 2020
DESCRIPTION:    Translated CH plane program written in IDL
INPUTS:         data = A formatted numpy array with dimensions (nt)
                dim  = dimensionality (usually 3 to 7)
'''
import matplotlib.pyplot as plt
import numpy             as np
import lib.toolbox       as tbx
from scipy.integrate import odeint
'''----------------------------------------------------------------------------
to reload module:    
import importlib
importlib.reload(<module>)
-------------------------------------------------------------------------------
'''
#### calculate permutation probability distribution
def permutation_pj(data, dim=5):
    npoints     = len(data)-dim+1
    d_factorial = np.math.factorial(dim)
    temp = np.zeros(d_factorial)
    ## partition the data
    part_data  = np.array([data[ii:ii+dim] for ii in np.arange(0,npoints)])

    ## calculate permutation for each partition
    bp_data    = [np.array_str(np.argsort(jj)) for jj in part_data[:-2]]

    ## count unique permutation values
    label_arr, count_arr = np.unique(bp_data, return_counts=True)
    temp[0:len(count_arr)] = count_arr
    return np.array(temp)

#### --------------------------------------------------------------------------
#### calculate Shannon entropy
def shannon_entropy(data):
    return np.sum([-pj * np.log(pj) for pj in data if pj>0])

#### --------------------------------------------------------------------------
#### calculate Bandt-Pompe entropy H and Jensen-Shannon complexity C_JS
def calculate_ch(data, dim=5):
    npoints  = len(data)-dim+1
    df       = np.math.factorial(dim)   #d factorial

    data_P = permutation_pj(data,dim=dim)/npoints
    S_P    = shannon_entropy(data_P)
    S_Pe   = np.log(df)             ## max S = S[P_e]
    
    bp_entropy = S_P/S_Pe
    
    ##### to reduce computation time, calculate C_JS using the arrays from above
    data_Pe  = np.array([1/df for ii in np.arange(df)])
    data_mid = np.array([(aa+bb)/2 for aa, bb in zip(data_Pe, data_P)])
    
    S_mid = shannon_entropy(data_mid)
    
    denom = (df + 1)/df * np.log(df+1) - 2*np.log(2*df) + np.log(df)
    js_complexity = -(2*S_mid - S_P - S_Pe)/denom * bp_entropy
    
    return js_complexity, bp_entropy

#### --------------------------------------------------------------------------
#### produces the minimum and maximum CH curves for plotting
def minmax_CH_plot(dim=5, npoints=100):
    df = np.math.factorial(dim)   #d factorial
    
    ### uniform probability
    uniform_prob    = [1/df for ii in np.arange(df)]
    uniform_entropy = - np.sum([ii*np.log(ii) for ii in uniform_prob]) / np.log(2)
        
    ### normalization const for disequilibrium
    q0 = -2/ ((df+1)/df * np.log(df+1) - 2*np.log(2*df) + np.log(df))  * np.log(2)
    
    ## MIN COMPLEXITY 
    ## probability looks like: 1 element with probability pmax, others have probability  
    ## (1-pmax)/(Nf-1) and pmax can vary from 1./N to 1 -> each one of these will correspond to  
    ## a different entropy H. And for each of these H's one can calculate the complexity. 
    ## See also Maggs' paper.
    pmax0 = [ii/df * (1- 1/df) + 1/df for ii in np.arange(df)]
    
    ## alog(2) factors cancel out
    Hmin_arr = [-1/np.log(df) * (jj*np.log(jj) + (1-jj)*np.log((1-jj)/(df-1)) ) for jj in pmax0]
    
    ## now calculate Cmin, need sum_probabity -> one element will have (pmax+1/Nf)/2 others will 
    ## have ((1-pmax)/(Nf-1)+1./Nf)/2
    pmax_sum0 = [1/2 * (jj            + 1/df) for jj in pmax0]
    pmin_sum0 = [1/2 * ((1-jj)/(df-1) + 1/df) for jj in pmax0]
    entropy_min  = [ii * np.log(df)/np.log(2) for ii in Hmin_arr]
    sum_entropy0 = [-(aa*np.log(aa) + (df-1)*bb*np.log(bb)) / np.log(2) for aa, bb in \
                    zip(pmax_sum0, pmin_sum0)]
    shanDmin     = [aa - bb/2 - uniform_entropy/2 for aa,bb in zip(sum_entropy0, entropy_min)]
    Cmin_arr     = [q0*aa*bb for aa,bb in zip(shanDmin, Hmin_arr)]
    
    ## MAX COMPLEXITY (n=1, m varies from 1 to N-1)
    ## probabilities as in Martin et al. (2006), (Eq. 31)
    npmax = 200
    Hmax  = np.zeros([df-1, npmax])
    Cmax  = np.zeros([df-1, npmax])
    
    ## m states with probability zero, 1 state with probability pmax, 
    ## Nf-m-1 states with equal probability, i.e., (1-pmax)/(Nf-m-1)
    for mm in np.arange(1,df-1):
        pmax1      = [ii/npmax * (1-1/(df-mm)) +  1/(df-mm) for ii in np.arange(npmax)]
        Hmax[mm,:] = [-1/np.log(df) * ( jj * np.log(jj) + (1-jj)*np.log((1-jj)/(df-mm-1)) ) 
                      for jj in pmax1]
        entropy_max = [jj * np.log(df)/np.log(2) for jj in Hmax[mm,:]]
        
        ## now calculate Cmax, sum probability, one element with probability (pmax+1./Nf)/2, 
        ## m elements with probability 1./2/Nf and Nf-m-1 elements with probability 
        ## ( (1-pmax)/(Nf-m-1) + 1./Nf ) / 2. 
        pmax_sum1 = [1/2 * (jj               + 1/df) for jj in pmax1]
        pmin_sum1 = [1/2 * ((1-jj)/(df-mm-1) + 1/df) for jj in pmax1]
        pzero_sum = (1/df)/2
        sum_entropy1 = [-(aa*np.log(aa) + (df-mm-1)*bb*np.log(bb) + \
                        mm*pzero_sum*np.log(pzero_sum))/np.log(2) \
                        for aa,bb in zip(pmax_sum1,pmin_sum1)]
        shanDmax     = [aa - bb/2 - uniform_entropy/2 for aa, bb in zip(sum_entropy1, entropy_max)]
        Cmax[mm,:]   = [q0*aa*bb for aa,bb in zip(shanDmax, Hmax[mm,:])]
    
    Hmax_arr = Hmax[:,0]
    Cmax_arr = Cmax[:,0]
    jjj = np.argsort(Hmax_arr)
    
    return Hmin_arr, Cmin_arr, Hmax_arr[jjj], Cmax_arr[jjj]

def plot_CH_plane(dim=5, blank=0):
    Hmin,Cmin,Hmax,Cmax= minmax_CH_plot(dim=dim, npoints=100)

    fig = tbx.prefig(figsize=[16,9], xlabel='BP entropy $H$', ylabel='JS complexity $C_{JS}$')
    plt.title('Complexity$-$Entropy causality plane ($d=${0})'.format(dim), fontsize=30)
    plt.plot(Hmin,Cmin, 'c', linewidth=2)
    plt.plot(Hmax,Cmax, 'c', linewidth=2)

    if blank != 0:
        print('\r Calculating CH for Hénon map...            ', end='')
        C_henon, H_henon = henon_map(10000, dim=dim)
        plt.plot(H_henon, C_henon, 'D', markersize=12, fillstyle='none', label='Hénon map')

        print('\r Calculating CH for Logistic map...         ', end='')
        C_logis, H_logis = logistic_map(10000, dim=dim)
        plt.plot(H_logis, C_logis, '^', markersize=15, fillstyle='none', label='Logistic map')

        print('\r Calculating CH for Ricker population map...', end='')
        C_ricker, H_ricker = ricker_map(10000, dim=dim)
        plt.plot(H_ricker, C_ricker, 's', markersize=15, fillstyle='none', label='Ricker population map')

        print('\r Calculating CH for Gingerbreadman map...   ', end='')
        C_gbman, H_gbman = gbman_map(10000, dim=dim)
        plt.plot(H_gbman, C_gbman, '+', markersize=15, fillstyle='none', label='Gingerbreadman map')

        print('\r Calculating CH for sine wave...            ', end='')
        C_sine, H_sine = sine_wave(10000, dim=dim)
        plt.plot(H_sine, C_sine, 'x', markersize=15, fillstyle='none', label='Sine wave')

        print('\r Calculating CH for Lorenz attractor...     ', end='')
        C_lorenz, H_lorenz = lorenz_attractor(sigma=10, beta=8/3, rho=28, dim=dim)
        plt.plot(H_lorenz, C_lorenz, '8', markersize=15, fillstyle='none', label='Lorenz attractor')
        
        print('\r Calculating CH for double pendulum...      ', end='')
        C_dbpd, H_dbpd = double_pendulum(m=1, l=1, g=10, dim=dim)
        plt.plot(H_dbpd, C_dbpd, '*', markersize=15, fillstyle='none', label='Double pendulum')
        
        C_fBm, H_fBm = fBm_gen(dim=dim)
        plt.plot(H_fBm, C_fBm, '.', label='fractional Brownian motion (fBm)')

    plt.legend(fontsize=15)
    return fig

#### --------------------------------------------------------------------------
#### Some mathematical maps of chaotic systems
#### --------------------------------------------------------------------------
#### HÉNON MAP OF CLASSICAL ATTRACTOR (a=1.4, b=0.3)
def henon_map(npoints, x0=1, y0=1, a=1.4, b=0.3, dim=5):
    xarr, yarr = [x0], [y0]   # initial values
    while npoints > 1:
        xnext = 1 - a*(xarr[-1])**2 + yarr[-1]
        ynext = b * xarr[-1]
        xarr.append(xnext)
        yarr.append(ynext)
        npoints -= 1
    return calculate_ch(xarr, dim=dim)

#### --------------------------------------------------------------------------
#### LOGISTIC MAP
def logistic_map(npoints, r=3.875, x0=0.5, dim=5):
    ## check bifurcation diagram, but r between 3.56995 to 4 is chaotic
    xarr = [x0]   # initial value
    while npoints > 1:
        xnext = r * xarr[-1] * (1-xarr[-1])
        xarr.append(xnext)
        npoints -= 1
    return calculate_ch(xarr, dim=dim)

#### --------------------------------------------------------------------------
#### RICKER POPULATION MAP
def ricker_map(npoints, c=20, x0=0.1, dim=5):
    ## check bifurcation diagram, but r between 3.56995 to 4 is chaotic
    xarr = [x0]   # initial value
    while npoints > 1:
        xnext = c * xarr[-1] * np.exp(-xarr[-1])
        xarr.append(xnext)
        npoints -= 1
    return calculate_ch(xarr, dim=dim)

#### --------------------------------------------------------------------------
#### GINGERBREADMAN MAP
def gbman_map(npoints, x0=1.4, y0=3.0, dim=5):
    ## check bifurcation diagram, but r between 3.56995 to 4 is chaotic
    xarr, yarr = [x0], [y0]   # initial value
    while npoints > 1:
        xnext = 1 - yarr[-1] + np.abs(xarr[-1])
        ynext = xarr[-1]
        xarr.append(xnext)
        yarr.append(ynext)
        npoints -= 1
    return calculate_ch(xarr, dim=dim)

#### --------------------------------------------------------------------------
#### SINE WAVE
def sine_wave(npoints, cycles=5, dim=5):
    xarr = [np.sin(cycles*aa*2*np.pi/npoints) for aa in np.arange(npoints)]
    return calculate_ch(xarr, dim=dim)

#### --------------------------------------------------------------------------
#### LORENZ ATTRACTOR
#### https://scipython.com/blog/the-lorenz-attractor/
def lorenz_deriv(X, t, sigma=10, beta=8/3, rho=28):
    xx, yy, zz = X
    xderiv = sigma*(yy - xx)
    yderiv = xx*(rho-zz) - yy
    zderiv = xx*yy - beta*zz
    return xderiv, yderiv, zderiv

def lorenz_attractor(sigma=10, beta=8/3, rho=28, dim=5): ##show=0
    tmax, n    = 2000, 10000
    x0, y0, z0 = 0, 1, 0.5
    t = np.linspace(0, tmax, n)
    f = odeint(lorenz_deriv, (x0,y0,z0), t, args=(sigma,beta,rho))
    
    # if show != 0:
    #   x,y,z = f.T
    #   # Plot the Lorenz attractor using a Matplotlib 3D projection
    #   fig = plt.figure(figsize=[9,9])
    #   ax  = fig.gca(projection='3d')

    #   # Make the line multi-coloured by plotting it in segments of length s which
    #   # change in colour across the whole time series.
    #   s = 10
    #   c = np.linspace(0,1,n)
    #   for i in range(0,n-s,s):
    #       ax.plot(x[i:i+s+1], y[i:i+s+1], z[i:i+s+1], color=(1,c[i],0), alpha=0.4)

    #   # Remove all the axis clutter, leaving just the curve.
    #   ax.set_axis_off()

    return calculate_ch(f[:,0], dim=dim)

#### --------------------------------------------------------------------------
#### DOUBLE PENDULUM
def double_pendulum_deriv(X, t, m=1, l=1, g=10):
    thx0, thy0, thdx0, thdy0 = X
    thxd1 = 6.0/(m*l**2) * (2*thdx0 - 3*np.cos(thx0-thy0)*thdy0)/(16-9*(np.cos(thx0-thy0))**2)
    thyd1 = 6.0/(m*l**2) * (8*thdy0 - 3*np.cos(thx0-thy0)*thdx0)/(16-9*(np.cos(thx0-thy0))**2)
    pxd1 = -0.5*m*l**2  * (thxd1*thyd1*np.sin(thx0-thy0) + 3*g/l*np.sin(thx0))
    pyd1 = -0.5*m*l**2  * (-thxd1*thyd1*np.sin(thx0-thy0) + g/l*np.sin(thy0))
    return thxd1, thyd1, pxd1, pyd1

def double_pendulum(m=1, l=1, g=10, dim=5):
    tmax, n = 2000, 10000
    thx0, thy0, thdx0, thdy0 = np.pi/2, np.pi/2, 0, 0
    t = np.linspace(0, tmax, n)
    f = odeint(double_pendulum_deriv, (thx0, thy0, thdx0, thdy0), t, args=(m, l, g))

    x = [np.sin(aa) + np.sin(bb) for aa, bb in zip(f[:,0], f[:,1])]
    # y = [np.sin(aa) + np.sin(bb) for aa, bb in zip(f[:,0], f[:,1])]

    # c = np.linspace(0,1,len(aaa[:,0]))
    # for ii in range(len(aaa[:,0])):
    #   plt.plot(x[ii],-y[ii], '.', color=(1,c[ii],0))

    return calculate_ch(x, dim=dim)

#### --------------------------------------------------------------------------
#### FRACTIONAL BROWNIAN MOTION (fBm)
def fBm_gen(dim=5):
    nhe     = 500                                # number of Hurst exponents to generate
    he      = [ii/(nhe-1) for ii in range(nhe)]  # define a range of hurst exponents
    siglen  = 2056                               # length of the fBm signals

    freq = [((ii/2)+1)*2*np.pi/siglen for ii in range(siglen)] # angular frequency 0 to pi
    data = np.empty([nhe,2])

    for ii in range(nhe):
        tbx.show_progress_bar([ii+1], [nhe], label=['Hurst exponents'], header='Generating fBm ')
        h2   = 2*he[ii]
        sdf  = np.array([(2*jj)**(-1-h2) for jj in freq])
        sdf0 = sdf[0]
        sdf  = np.append(sdf, sdf0)

        wr  = [np.random.uniform(0,1) for jj in range(siglen)]
        v   = np.array(np.zeros(siglen), dtype=complex) # prep FFT
        mid = int(siglen/2)
        v[0]            = np.sqrt(sdf0 * wr[0])
        v[1:mid]        = [np.sqrt(0.5*sdf[jj])* np.complex(wr[2*jj-1],wr[2*jj]) for jj in range(1, mid)]
        v[mid]          = np.sqrt(sdf[mid])* wr[siglen-1]
        v[mid+1:siglen] = [np.conj(v[siglen-jj]) for jj in range(mid+1, siglen)]

        tout       = np.real(np.fft.fft(v))
        data[ii,:] = calculate_ch(tout, dim=dim)

    c_arr, h_arr = data.T
    index        = np.argsort(h_arr)

    return c_arr[index], h_arr[index]