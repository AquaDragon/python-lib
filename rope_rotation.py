'''
NAME:           rope_rotation.py
AUTHOR:         swjtang
DATE:           20 Apr 2021
DESCRIPTION:    Functions used for the rope rotation model.
-------------------------------------------------------------------------------
to reload module:
import importlib
importlib.reload(<module>)
-------------------------------------------------------------------------------
'''
import copy
import os
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt

import lib.toolbox as tbx


def source_to_obsv(xs, ys, xo, yo):
    ''' ----------------------------------------------------------------------
    A source is located at (xs, ys) and is observed from (xo, yo). Determine
    the distance and angle from source to observer. (2D)
        xs, ys = Coordinates for the source
        xo, yo = Coordinates for the observer
    '''
    ro = np.sqrt(xo**2 + yo**2)
    rs = np.sqrt(xs**2 + ys**2)
    tho = np.angle(np.complex(xo, yo))  # theta_obsv
    ths = np.angle(np.complex(xs, ys))  # theta_source

    r = np.sqrt(ro**2 + rs**2 - 2*ro*rs*np.cos(tho-ths))
    angle = np.angle(np.complex(ro*np.sin(tho) - rs*np.sin(ths),
                     ro*np.cos(tho) - rs*np.cos(ths)))
    return r, angle


'''
-------------------------------------------------------------------------------
                EIGENFUNCTION MODULES
-------------------------------------------------------------------------------
'''


def get_circle_points(r=1, center=None, Lmode=5, phase=0):
    ''' ----------------------------------------------------------------------
    Given the radius and center of circle, find coordinates of L points on a
    circle.
        r      = Radius of circle
        center = 2D coordinate for the center of the circle
        Lmode  = Number of points, also azimuthal mode number
        phase  = Adds a phase in units of omega*t
    '''
    if center is None:
        center = [0, 0]
    xarr, yarr = [], []    # arrays for coordinates on a circle
    for th0 in np.linspace(0, 2*np.pi, Lmode, dtype='float', endpoint=False):
        angle = th0 + phase
        xarr = np.append(xarr, r*np.cos(angle)+center[0])
        yarr = np.append(yarr, r*np.sin(angle)+center[1])
    return xarr, yarr


def eigenfunction(r, a=2, c=3.5, Lmode=1):
    ''' ----------------------------------------------------------------------
    Define the radial eigenfunction to be used in the calculation.
        r = radius from center of cylinder

    # EIGENFUNCTION NEEDS TO DEPEND ON L, LARGER L, SMALLER AMPLITUDE!!!!
    '''
    return np.exp(-(r-c)**2/a**2)


def get_potential_eigen(x=[], y=[], eigen=[], center=[0, 0], Lmode=1,
                        display=0, phase=[0]):
    ''' ----------------------------------------------------------------------
    Plot the potential created by summing the radial eigenfunction.
    INPUTS:
        center  = 2D coordinate for the center of the circle.
        Lmode   = Azimuthal mode number
        phase   = A 1D array of phases used in potential calculation.
    OPTIONAL:
        x, y    = 1D arrays that contain position information. If not set,
                  default values are used.
        eigen   = A 2D array that contains the map of the radial
                  eigenfunction. Uses the eigenfunction in this program is noy
                  specified.
        display = Set to 1 to show the plot of radial eigenfunction.
    '''
    if len(x) == 0:
        x = np.linspace(-10, 10, 100)
    if len(y) == 0:
        y = np.linspace(-10, 10, 200)
    xmesh, ymesh = np.meshgrid(x, y, indexing='ij')

    th_arr = np.angle((xmesh-center[0])+(ymesh-center[1])*1j)
    r_arr = np.hypot(xmesh-center[0], ymesh-center[1])
    if len(eigen) == 0:
        eigen_arr = eigenfunction(r_arr, Lmode=Lmode)  # use default values
    else:
        eigen_arr = eigen

    if display == 1:
        plot_radial_eigen(x, y, eigen_arr)

    if len(phase) == 1:
        pot_L = eigen_arr * np.exp(1j*(Lmode*th_arr - phase))
    else:
        pot_L = np.array([eigen_arr * np.exp(1j*(Lmode*th_arr - ph))
                         for ph in phase])
        # dim(pot_L) is (nphase, nx, ny)

    return x, y, pot_L  # pot_L is complex


''' --------------------------------------------------------------------------
    Functions to help visualize by making contourf plots.
------------------------------------------------------------------------------
'''


def plot_radial_eigen(x, y, eigen_arr, save=1):
    # Display radial eigenfunction contour plot
    tbx.prefig(figsize=(6, 6), xlabel='x [cm]', ylabel='y [cm]')
    plt.title('radial eigenfunction', fontsize=20)
    plt.xticks(np.arange(min(x), max(x)+1, 5))
    plt.yticks(np.arange(min(y), max(y)+1, 5))
    plt.contourf(x, y, np.transpose(eigen_arr), 100)
    if save == 1:
        print('plot_radial_eigen image saved.')
        plt.savefig('./img/potential_radial-eigen_contour.png',
                    bbox_inches="tight")


def plot_potential_eigen(x, y, pot_L, Lmode=1, save=1):
    pot_arr = np.real(pot_L)
    # Display electric potential of a single azimuthal mode
    tbx.prefig(figsize=[6, 6], xlabel='x [cm]', ylabel='y [cm]')
    plt.title('electric potential for azimuthal mode L={0}'.format(Lmode),
              fontsize=20, y=1.02)
    plt.title('L={0}'.format(Lmode), fontsize=40, y=1.02)
    plt.xticks(np.arange(min(x), max(x)+1, 5))
    plt.yticks(np.arange(min(y), max(y)+1, 5))
    plt.contourf(x, y, np.transpose(pot_arr), 100)
    if save == 1:
        print('plot_potential_eigen image saved.')
        plt.savefig('./img/potential_eigen_L{0}.png'.format(Lmode),
                    bbox_inches="tight")
