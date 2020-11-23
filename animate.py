'''
NAME:           animate.py
AUTHOR:         swjtang  
DATE:           23 Nov 2020
DESCRIPTION:    Plots an animated contour plot with colorbars
SYNTAX:
'''
import numpy as np
from matplotlib import animation, cm, pyplot as plt
'''----------------------------------------------------------------------------
to reload module:
import importlib
importlib.reload(<module>)
-------------------------------------------------------------------------------
'''
def animate(xx, yy, data, time, level=0.8, nstep=10, cmap=cm.viridis,\
    stitle=None, unit='??', sdir='./videos/', sname='temp_contourf.mp4'):
    fig = plt.figure(figsize=[12,9])
    ax  = fig.add_subplot(111)
    
    if stitle != None:
        fig.suptitle(str(stitle), fontsize=20)

    ### contourf levels -----------------------------------
    vmin   = level*np.amin(data)
    vmax   = level*np.amax(data)
    levels = np.linspace(vmin, vmax, 50, endpoint=True)

    ### generate frame 0
    frame  = ax.contourf(xx, yy, data[0,...], vmax=vmax, vmin=vmin, \
        levels=levels, cmap=cmap)
    
    ### colorbar options ----------------------------------
    cb = plt.colorbar(frame, ax=ax)
    cb.ax.tick_params(labelsize=20)
    cb.ax.set_title('{0}'.format(unit), fontsize=26)

    ### animation parameters
    nframes = len(time)//nstep

    # animation function
    def generate_frame(i):
        tt = i*nstep
        ax.clear()
        frame  = ax.contourf(xx, yy, data[tt,...], vmax=vmax, vmin=vmin, \
            levels=levels, cmap=cmap)
        ax.set_title('t = {0:.2f} ms [{1}]'.format(time[tt]*1e3, tt), fontsize=18)
        ax.set_xlabel('X [cm]', fontsize=30)
        ax.set_ylabel('Y [cm]', fontsize=30)
        ax.tick_params(labelsize=20)
        ax.set_aspect('equal')
        
        print('\r', 'Generating frame {0}/{1} ({2:.2f}%)...'\
            .format(i+1, nframes, (i+1)/nframes*100), end='')

    anim = animation.FuncAnimation(fig, generate_frame, frames=nframes, interval=50)
    anim.save(sdir+sname)