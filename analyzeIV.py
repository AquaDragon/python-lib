'''
NAME:           	analyzeIV.py
AUTHOR:         	swjtang  
DATE:           	29 Aug 2020
DESCRIPTION:    	A routine to calculate temperature, plasma potential and 
					density from an IV curve.
REQUIRED INPUTS:    
OPTIONAL INPUTS:	res  = [Ohm] value of resistor used 
					area = [cm^2] effective area of the probe
					gain_curr = current gain (if not implemented on the scope)
					gain_volt = voltage gain (if not implemented on the scope)
'''
import lib.toolbox as tbx
import numpy as np
import os, importlib
import scipy.constants as const
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

''' to reload a module :    
import importlib
importlib.reload(<module>)
-------------------------------------------------------------------------------
'''
def analyzeIV(voltage, current, res=1, area=1, gain_curr=1, gain_volt=1, lim=[0.25,0.75,0.8], \
	noplot=0, quiet=0 ):
	### constants -------------------------------------------------------------
	amu = const.physical_constants['atomic mass constant'][0]
	### program parameters ----------------------------------------------------
	sm        = 200    # isat index
	flag_flip = 0      # indicator to check if current was flipped
	### -----------------------------------------------------------------------

	### get values of the voltage and current
	isat = np.mean(current[:sm])
	volt = gain_volt*tbx.smooth(voltage, nwindow=351, polyn=2)
	curr = gain_curr/res*tbx.smooth(current-isat, nwindow=351, polyn=2)

	### checks if current was flipped (convention is electron current positive)
	if np.mean(curr[:100]) > np.mean(curr[-100:]):
		if quiet == 0:
			print('!!! Current input was flipped')
		flag_flip    = 1
		curr         = [-ii for ii in curr]

	if noplot == 0 :
		tbx.prefig(figsize=[16,4.5], xlabel='Potential [V]', ylabel='Current [A]')
		plt.plot(volt, curr, linewidth=2)

	### define points on the IV curve for analysis
	cmin, cmax = np.amin(curr), np.amax(curr)
	y25 = lim[0]*(cmax-cmin) + cmin   ### 25% of range (not percentile)
	y75 = lim[1]*(cmax-cmin) + cmin   ### 75% of range (not percentile)
	y95 = lim[2]*(cmax-cmin) + cmin 

	arg25 = tbx.value_locate_arg(curr, y25)
	arg75 = tbx.value_locate_arg(curr, y75)
	arg95 = tbx.value_locate_arg(curr, y95)
	
	### analysis: transition region
	x1 = np.arange(arg25, arg75)
	y1 = curr[arg25:arg75]
	try:
		poly_arg1 = np.polyfit(x1,y1,2)
		poly1     = np.poly1d(poly_arg1)
	except:
		return reject(volt, curr, reason='unable to fit polynomial to transition region', quiet=quiet)

	newx1 = np.arange(arg25,arg95)
	out1  = poly1(newx1)

	if noplot == 0 :
		plt.plot(volt[newx1], out1, '--', label='transition')

	### analysis: exponential region
	folding = 1
	while folding < 4:
		yfold   = y25/(np.exp(folding))   # value # folding lengths below y25
		argfold = tbx.value_locate_arg(curr, yfold)
		x2 = np.arange(argfold, arg25)
		y2 = curr[argfold:arg25]

		try:
			poly_der1 = np.poly1d([poly_arg1[1],poly_arg1[2]*2])  ## derivative of poly1
			etest = y25/poly_der1(arg25)
			popt, pcov = curve_fit(f_exp, x2, y2, p0=(y25,etest/x2[-1],0))
			out2 = f_exp(x2, *popt)
			if noplot == 0 :
				plt.plot(volt[x2], out2, '--', label='exponential')
			break

		except:
			folding +=1
			if quiet == 0: print('\rextending folding length to {0}...'.format(folding), end='')
	else:
		return reject(volt, curr, reason='exponential region failed to converge', quiet=quiet)

	### estimate the electron temperature, popt is in indices need to convert
	index_Te  = int(1/popt[1])
	if index_Te+arg25 >= len(volt) or index_Te < 0: 
		return reject(volt, curr, reason='garbage electron temperature', quiet=quiet)
	plasma_Te = 2 / (volt[index_Te+arg25] - volt[arg25])   
	v_thermal = 4.19e7*np.sqrt(abs(plasma_Te))  ## [cm/s]
	if quiet == 0:
		print('electron Temp    = {0:.3f}     [eV]'.format(plasma_Te))

	### analysis: electron saturation region
	x3 = np.arange(arg95, len(volt))
	y3 = curr[arg95:]
	poly_arg3 = np.polyfit(x3,y3,1)
	poly3 = np.poly1d(poly_arg3)

	newx3 = np.arange(arg75, len(volt))    ### extrapolate backwards 
	out3  = poly3(newx3)

	if noplot == 0 :
		plt.plot(volt[newx3], out3, '--', label='electron saturation')

	### the intersection of out1 and out3 is the plasma potential
	coeff = poly_arg1 - np.append(0, poly_arg3)
	roots = np.roots(coeff)

	plasma_pot = None
	for ii in np.unique(np.real(roots)):   ### should only have one root that satisfy this
		### note: if root is imaginary, the real root is the point of closest approach
		if (ii < len(curr)) & (ii > 0):
			if volt[int(ii)] > 0: 
				plasma_pot = volt[int(ii)]
				plasma_den = curr[int(ii)] / (const.e* area*v_thermal)
				plasma_iden = isat / (const.e * area *v_thermal * np.sqrt(const.m_e/(amu*4.0026)))
				if quiet == 0:
					print('Plasma potential = {0:.3f}    [V]'.format(plasma_pot))
					print('Plasma density   = {0:.3e} [cm-3]'.format(plasma_den))
					print('Plasma density(i)= {0:.3e} [cm-3]'.format(plasma_iden))

	if plasma_pot == None:   ### check if convergent solution/root exists
		return reject(volt, curr, reason='no plasma potential found', quiet=quiet)

	### the last entry is a parameter to indicate if a curve was rejected, 0 = no, 1 = yes
	params = [plasma_Te, plasma_pot, plasma_den, plasma_iden, 0]

	### plot output
	if noplot == 0 :
		plt.legend(fontsize=20)
		plt.ylim(cmin-0.1*abs(cmax-cmin), cmax+0.1*abs(cmax-cmin))

	return volt, curr, params

def reject(volt, curr, reason='unspecified reason', quiet=0):
	if quiet == 0 : print('\n!!! Reject curve: {0}'.format(reason))
	return volt, curr, [0, 0, 0, 0, 1]

def f_exp(x,a,b,c):
	return a * np.exp(b*x) + c