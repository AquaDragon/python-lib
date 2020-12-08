'''
NAME:           rope_rotation.py
AUTHOR:         swjtang  
DATE:           03 Dec 2020
DESCRIPTION:    Functions used for the rope rotation model.
'''
import numpy as np
import os, copy, scipy.signal
from matplotlib import pyplot as plt
'''----------------------------------------------------------------------------
to reload module:
import importlib
importlib.reload(<module>)
-------------------------------------------------------------------------------
DESCRIPTION:    A source is located at (xs, ys) and is observed from (xo, yo).
				Determine the distance and angle from source to observer. (2D)
INPUTS:         xs, ys = Coordinates for the source
                xo, yo = Coordinates for the observer
'''
def source_to_obsv(xs,ys,xo,yo):
    ro  = np.sqrt(xo**2 + yo**2)
    rs  = np.sqrt(xs**2 + ys**2)
    tho = np.angle(np.complex(xo,yo))  #theta_obsv
    ths = np.angle(np.complex(xs,ys))
        
    r     = np.sqrt(ro**2 + rs**2 - 2*ro*rs*np.cos(tho-ths))
    angle = np.angle(np.complex(ro*np.sin(tho) - rs*np.sin(ths), \
    			ro*np.cos(tho) - rs*np.cos(ths) ))
    return r, angle