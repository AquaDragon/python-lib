'''
NAME:           coord_transform_180E_XYdrive.py
AUTHOR:         swjtang
DATE:           15 Jan 2021
DESCRIPTION:
    Coordinate transform of the XY probe drive on the 180E ICP machine.
    - Assume: Machine zero is when drive coordinates are zero.
    - Assume: The X and Y drives slide independently (no machine correction).
    - Measure: Length of the machine zero to the pivot (i.e. L1) and the
    length of the pivot to the XY drive center (i.e. L2) in cm.
    - Inputs: 1D X- and Y-drive coordinate arrays
'''
import numpy as np
from array import array
from matplotlib import pyplot as plt


def op(xarr, yarr, length1, length2, debug=0):
    nx = len(xarr)
    ny = len(yarr)
    y = [[(length1+xarr[ii])/length2*yarr[jj] for jj in range(ny)]
         for ii in range(nx)]
    x = [[-((length1+xarr[ii])*np.sqrt(1-(yarr[jj]/length2)**2)-length1)
          for jj in range(ny)] for ii in range(nx)]

    theta = [[np.arcsin(yarr[jj]/length2) for jj in range(ny)]
             for ii in range(nx)]

    if debug is not 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Machine coordinate system', fontsize=18)
        ax.set_xlabel('X [cm]', fontsize=18)
        ax.set_ylabel('Y [cm]', fontsize=18)
        ax.set_aspect('equal')

        for ii in range(nx):
            for jj in range(ny):
                plt.plot([x[ii][jj]], [y[ii][jj]], 'ro')
    else:
        return {'x': x, 'y': y, 'nx': nx, 'ny': ny, 'theta': theta}
