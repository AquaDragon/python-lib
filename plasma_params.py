'''
NAME:           plasma_params.py
AUTHOR:         swjtang  
DATE:           31 Aug 2020
DESCRIPTION:    A list of plasma parameter functions
'''
import numpy as np
import os
import scipy.constants as const
from matplotlib import pyplot as plt

''' ---------------------------------------------------------------------------
QUANTITY:   KRUSKAL-SHAFRANOV CURRENT LIMIT
INPUTS:     Bz = magnetic field [G]
            a  = flux rope radius [cm]
            L  = flux rope length [m]
'''
def I_KS(Bz, a=3.81, L=11):
    return (2*np.pi*a*1e-2)**2 * (Bz*1e-4) / (const.mu_0*L)

''' ---------------------------------------------------------------------------
QUANTITY:   ALFVEN VELOCITY
INPUTS:     bfield = magnetic field [G]
            den    = plasma density [cm-3]
'''
def v_alfven(bfield, den=1e9, amu=4.002602):
    return 2.18e9*bfield/np.sqrt(amu*den)   # [m/s]
