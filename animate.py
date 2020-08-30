'''
NAME:        	animate.py
AUTHOR:      	swjtang  
DATE:        	31 May 2019
DESCRIPTION: 	Plots an animated contour plot with colorbars
SYNTAX:
'''
import numpy as np
from matplotlib import animation, cm, pyplot as plt

def animate(xx, yy, data, time, fname='temp', nstep=10, cmap=cm.GnBu):
	fig = plt.figure(figsize=(15,10))#, constrained_layout=True)
	fig.suptitle(str(fname), fontsize=20)
	ax  = fig.add_subplot(111)
	#ay  = fig.add_subplot(412)
	#az  = fig.add_subplot(413)

	vmin   = 0.6*np.amin(data)
	vmax   = 0.6*np.amax(data)
	levels = np.linspace(vmin, vmax, 30, endpoint=True)

	frame  = ax.contourf(xx, yy, data[:,:,0], vmax=vmax, vmin=vmin, levels=levels, cmap=cmap)
	cbar   = fig.colorbar(frame, ax=ax)
	cbar.ax.tick_params(labelsize=16)
	cbar.set_label('Magnitude', fontsize=16)

	### animation parameters
	nframes = len(time)//nstep

	# animation function
	def generate_frame(i):
	    ax.clear()
	    frame  = ax.contourf(xx, yy, data[:,:,nstep*i], vmax=vmax, vmin=vmin, levels=levels, cmap=cmap)
	    #ay.contourf(xx, yy, data[:,:,nstep*i], vmax=vmax, vmin=vmin, levels=levels, cmap=cmap)
	    ax.set_title('t = '+"{:.4f}".format(time[nstep*i]*1e6)+' us ['+format(i)+']', fontsize=18)
	    ax.set_xlabel('X [cm]', fontsize=18)
	    ax.set_ylabel('Y [cm]', fontsize=18)
	    ax.tick_params(labelsize=16)
	    ax.set_aspect('equal')
	    
	    print('\r', 'Generating frame '+format(i)+'/'+format(nframes)+' ('+ \
	          "{:.2f}".format(i/nframes*100)+'%)...', end='')

	anim = animation.FuncAnimation(fig, generate_frame, interval=50, frames=nframes)


	temp=fname.find('.')
	anim.save(fname[0:temp]+'.mp4')