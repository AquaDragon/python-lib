'''
NAME:           plasma_params.py
AUTHOR:         swjtang
DATE:           15 Jan 2021
DESCRIPTION:    A list of plasma parameter functions
'''
import numpy as np
import os
import scipy.constants as const
from matplotlib import pyplot as plt


def I_KS(Bz, a=3.81, L=11):
    ''' KRUSKAL-SHAFRANOV CURRENT LIMIT
        Bz = magnetic field [G]
        a  = flux rope radius [cm]
        L  = flux rope length [m]
    '''
    return (2*np.pi*a*1e-2)**2 * (Bz*1e-4) / (const.mu_0*L)


def v_alfven(bfield, den=1e9, amu=4.002602):
    ''' ALFVEN VELOCITY
        bfield = magnetic field [G]
        den    = plasma density [cm-3]
    '''
    return 2.18e9*bfield/np.sqrt(amu*den)   # [m/s]
