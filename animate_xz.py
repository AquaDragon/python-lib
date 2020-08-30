'''
NAME:        	animate_w.py
AUTHOR:      	swjtang  
DATE:        	08 Feb 2019
DESCRIPTION: 	Plots an animated contour plot on the XZ plane
SYNTAX:
'''
import numpy as np
from matplotlib import animation, cm, pyplot as plt

from lib.read_hdf5_v2 import readhdf5

def animate(fname, dirw='', cmap=cm.GnBu):
	dataset = readhdf5(dirw+fname)

	pos  = dataset['pos']
	time = dataset['time']
	bx   = dataset['ch1']
	by   = dataset['ch2']
	bz   = dataset['ch3']

	nx = dataset['nx']
	ny = dataset['ny']
	nt = len(time)

	xx = dataset['xx']
	yy = dataset['yy']
	yy = 20*np.sin(np.deg2rad(yy))  #list of y positions

	btot = [np.sqrt(bx[i]**2 + by[i]**2 + bz[i]**2) for i in range(len(bx))]
	data = np.reshape(btot, (ny, nx, nt), order='C')  #reshape is equivalent to reform

	fig = plt.figure(figsize=(15,10))#, constrained_layout=True)
	fig.suptitle(str(fname), fontsize=20)
	ax  = fig.add_subplot(331)
	ay  = fig.add_subplot(332)
	az  = fig.add_subplot(333)
	atot= fig.add_subplot(312, rowspan=2)

	vmin   = 0.6*np.amin(data)
	vmax   = 0.6*np.amax(data)
	levels = np.linspace(vmin, vmax, 30, endpoint=True)

	frame  = atot.contourf(xx, yy, data[:,:,0], vmax=vmax, vmin=vmin, levels=levels, cmap=cmap)
	cbar   = fig.colorbar(frame, ax=atot)
	cbar.ax.tick_params(labelsize=16)
	cbar.set_label('Magnitude', fontsize=16)

	### animation parameters
	nstep   = 100                      # frames are taken at every dt step size
	nframes = len(time)//nstep

	# animation function
	def generate_frame(i):
		atot.clear()
		atot.contourf(xx, yy, data[:,:,nstep*i], vmax=vmax, vmin=vmin, levels=levels, cmap=cmap)
		atot.set_title('t = '+"{:.4f}".format(time[nstep*i]*1e6)+' us ['+format(i)+']', fontsize=18)
		atot.set_xlabel('X [cm]', fontsize=18)
		atot.set_ylabel('Y [cm]', fontsize=18)
		atot.tick_params(labelsize=16)
		atot.set_aspect('equal')

		ax.contourf(xx, yy, data[:,:,nstep*i], vmax=vmax, vmin=vmin, levels=levels, cmap=cmap)
		ax.set_aspect('equal')

		ay.contourf(xx, yy, data[:,:,nstep*i], vmax=vmax, vmin=vmin, levels=levels, cmap=cmap)
		ay.set_aspect('equal')

		az.contourf(xx, yy, data[:,:,nstep*i], vmax=vmax, vmin=vmin, levels=levels, cmap=cmap)
		az.set_aspect('equal')

		print('\r', 'Generating frame '+format(i+1)+'/'+format(nframes)+' ('+ \
			"{:.2f}".format((i+1)/nframes*100)+'%)...', end='')

	anim = animation.FuncAnimation(fig, generate_frame, interval=50, frames=nframes)


	temp=fname.find('.')
	anim.save(fname[0:temp]+'.mp4')
	plt.show()