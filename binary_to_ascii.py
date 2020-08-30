# -*- coding: utf-8 -*-
"""
Attempt to read Agilent E5100A network analyzer (10kHz-180 MHz) binary data file and write the contents into a .csv file
The user should store only 'MAIN' array and 'SUB' array, typically in real and imaginary format but any format will be read in
I don't know how to get frequency information from the binary file; I think we would need to save ALL instead of DATA ONLY

Patrick Pribyl - Sep'16; based on C++ version originally written Feb 2009
Updated 14 Aug 2019 - swjtang
"""

import numpy as np
import scipy.constants as const
import struct
import sys
from glob import glob
from .toolbox import qprint

def unpack_record(fdata, verbose=False) -> (np.array, int):
	"""
	worker for read_data()
	Parse a data record from the input data
		Each record is of the form
		    unsigned x                 ???
		    unsigned short npts        number of doubles in array
		    unsigned y;                ???
		    double data[npts];         data array
	Returns a numpy array with the data, and the number of bytes of the input data that were processed
	"""
	pos = 0
	x = struct.unpack('=L', fdata[pos:pos+4])[0]       # L = unsigned long
	pos += 4
	npts = struct.unpack('=H', fdata[pos:pos+2])[0]     # H = unsigned short
	pos += 2
	y = struct.unpack('=L', fdata[pos:pos+4])[0]       # L = unsigned long
	pos += 4
	data = struct.unpack('='+str(npts)+'d', fdata[pos:pos+npts*8])    # d = double
	pos += npts*8
	if verbose:
		print('  x =', x, '   y =', y, 'but their meaning is unknown')
		print('npts=',npts)
		print('data=',data[0:4],'...')
	return np.array(data), pos


def read_data(ifn:str, ch=2) -> (np.array, np.array):
	""" read the 'MAIN' and 'SUB' arrays from the input file named in the ifn argument """

	with open(ifn, 'rb') as f:
		fdata = f.read()   # haha can't be more than 1.44 MB just read it all

	a = struct.unpack('=d', fdata[0:8])[0]      # d = double precision
	b = struct.unpack('=d', fdata[8:16])[0]

	error_count = 0
	if a != 10000  or  b != 180000000:
		print("file format may be incorrect due to unexpected first 16 bytes")
		print("...continuing anyway  (a=",a," b=",b,")", sep='')
		error_count = 1

	HEADER_SIZE = 0x126

	"""
	The file consists of a header followed by 4 data records in this order:
	   - LOGMAG CH1 (1)  - "MAIN_ARRAY"
	   - LOGMAG CH2 (3)  - "MAIN_ARRAY"
	   - PHASE CH1 (2)   - "SUB_ARRAY"
	   - PHASE CH2 (4)   - "SUB_ARRAY"
	All four channels are saved even if they are inactive on the scope.
	"""
	pos = HEADER_SIZE
	ch1_logmag, n = unpack_record(fdata[pos:])

	pos += n
	ch2_logmag, n = unpack_record(fdata[pos:])
	
	pos += n
	ch1_phase, n = unpack_record(fdata[pos:])

	pos += n
	ch2_phase, n = unpack_record(fdata[pos:])

	if ch==1: MAIN_array, SUB_array = ch1_logmag, ch1_phase
	else    : MAIN_array, SUB_array = ch2_logmag, ch2_phase

	if MAIN_array.size != SUB_array.size:
		print("This is probably a bad file: the data array sizes are unequal:", MAIN_array.size, "!=", SUB_array.size)
		if error_count == 0:
			print("continuing anyway, but saving the smaller number of points")
		else:
			print("(not processed)")
			return np.zeros(1), np.zeros(1)

	return MAIN_array, SUB_array

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# defining function for writing the csv files
# usage: OUTPUT = b2a(file=<filename>, options=options)
# f0/f1 - start/stop sweep frequency on VNA, r - helmholtz coil radius, g - amplifier gain
def b2a(file=None, f0=10e3, f1=5e6, r=5.4e-2, g=10, ch=2, output=0, quiet=0):
	Re = np.zeros(1)
	Im = np.zeros(1)

	if file == None:
		print('Usage:')
		print('   python E5100A_Reader.py <sumdumfile>')
		print('')
		print('Needs one or more filenames on the command line, (Wildcards are OK)' )
		print('Appends ".csv" to the filename and writes each file')
		print('e.g. ')
		print('    python "<filename>" *.dat')
		print('will write a bunch of files ending in .dat.csv')
		print('')
		print('In the agilent E5100 network analyzer setup, save binary data, MAIN ARRAY and SUB ARRAY only')
		return(None)
	else:
		ifn = glob(file)[0]

		Re,Im = read_data(ifn, ch=ch)
		ofn = ifn[:-4] + '.csv'
		npts = min(Re.size, Im.size)
		if npts <= 1:
			qprint(quiet, 'skipping file "{0}"'.format(ifn))
			#continue
		qprint(quiet, 'writing file "{0}" with {1} entries'.format(ofn, npts))

		freqarr = np.linspace(f0,f1,num=len(Re), dtype='float')
		warr    = 2*np.pi*freqarr

		area = 10**(Re/20) / (32 * (4/5)**1.5) * r/ (g*const.mu_0*warr)  #m^2

		if output != 0:
			htext = 'freq [Hz],LOGMAG [dB],PHASE [deg],Area NA [m2]'
			np.savetxt(ofn, np.transpose((freqarr,Re,Im, area)), delimiter=",", header=htext)

		## function returns data
		temp = {
			'freq'   : freqarr,
			'logmag' : Re,
			'phase'  : Im,
			'area'   : area
		}

		qprint(quiet, '... done')
		return(temp)