'''
NAME:           animate_quiver.py
AUTHOR:         swjtang
DATE:           15 Jan 2021
DESCRIPTION:    Plots an animated quiver plot
INPUTS:             xx,yy    = positional inputs array[nx], array[ny]
                    data_U,V = vector inputs with array[ny,nx,nt]
                    time     = the input array of time values
OPTIONAL PARAMS:    cmap     = color map used to plot the arrows
                    nstep    = the number of frames skipped per iteration
                    ptitle   = the title of the movie
                    svfolder/svfile = the folder and file name of the saved
                                      video
                    qv_scale = length of the arrows, smaller value = larger
                               arrows
------------------------------------------------------------------------------
to reload module:
import importlib
importlib.reload(<module>)
------------------------------------------------------------------------------
'''
import numpy as np
from matplotlib import animation, cm, pyplot as plt

import lib.fname_tds as fn


def animate_quiver(xx, yy, data_U, data_V, time, cmap=cm.OrRd, nstep=100,
                   fid=None, ptitle=None, svfolder='./videos/',
                   svfile='temp_quiver.mp4', qv_scale=3, unit='(unit)'):
    fig = plt.figure(figsize=(10, 10))  # , constrained_layout=True)

    yalign = 0.94
    if ptitle is not None:
        fig.suptitle(str(ptitle), y=yalign, fontsize=15)
    elif fid is not None:
        fig.suptitle('file {0}, {1}'.format(fid, fn.fname_tds(fid, old=0,
                     full=0)), y=yalign, fontsize=15)

    ax = fig.add_subplot(111)

    # animation parameters
    nframes = len(time)//nstep

    data_mag = [np.sqrt(u**2+v**2) for u, v in zip(data_U, data_V)]
    global_max = np.amax(np.absolute(data_mag))

    # animation function
    def generate_frame(i):
        tt = int(nstep*i)
        ms_time = time[tt]*1e3
        ax.clear()

        frame_mag = [np.sqrt(u**2+v**2) for u, v in zip(data_U[tt, :, :],
                     data_V[tt, :, :])]
        local_max = np.amax(np.absolute(frame_mag))
        sc = global_max/local_max
        frame = ax.quiver(xx, yy, data_U[tt, :, :]*sc, data_V[tt, :, :]*sc,
                          frame_mag, scale=global_max/qv_scale,
                          scale_units='xy', cmap=cmap)

        # 1. Data has some value X.
        # 2. Define sc, the scaling factor which all data need to be scaled by
        #    so that arrows are proportion to global maximum. (data = X*sc)
        # 3. Define scale=global_max, the scaling factor of the data. Data is
        #    divided by this number so that the unit length is the scale_unit.
        # 4. => For a unit length displayed in quiver, the value is X*sc.

        ax.set_title('t = {0:.2f} ms [{1}], unit length = {2:.2f} {3}'
                     .format(ms_time, tt, global_max, unit), fontsize=20)
        ax.set_xlabel('X [cm]', fontsize=28)
        ax.set_ylabel('Y [cm]', fontsize=28)
        ax.tick_params(labelsize=20)
        ax.set_aspect('equal')

        print('\r Generating frame {0}/{1} ({2:.2f}%)...'.format(i+1, nframes,
              (i+1)/nframes*100), end='')

    anim = animation.FuncAnimation(fig, generate_frame, interval=50,
                                   frames=nframes)
    anim.save(svfolder+svfile)

##############################################################################
# ax.quiverkey(frame, X=0.3, Y=0.9, U=1000, \
#     label='Quiver key length = {0:.2f}'.format(global_max), labelpos='E')

# quiverkey explanation

# 5. No idea what quiverkey is doing, its bugged
