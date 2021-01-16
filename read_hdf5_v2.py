'''
The normal structure of an hdf5 file created by 180E lab data acquisition
system is HDF5 File -->
|-> Group 'Acquisition' --> Group 'LeCroy_scope' -->
|      Datasets 'Channel1', 'Channel2', 'Channel3', 'Channel4', 'Headers',
        'LeCroy_scope_Setup_Arrray', 'time'
|-> Group 'Controls' --> Group 'Positions' --> Group 'positions_setup_array'
|-> Group 'Meta' --> Group 'Python' --> Group 'Files' --> Group 'Files' -->
      Datasets 'Data_Run_GUI.py', 'LeCroy_Scope.py', 'Motor_Control_2D.py'

FUNCTION:       readhdf5
AUTHOR:         swjtang (modified) / Yuchen Qian (original)
DATE:           15 Jan 2021
DESCRIPTION:    Reads hdf5 file data created in the 180E lab into NumPy arrays.
INPUTS:         path = The path of the hdf5 file
RETURNS:
        -ch1, ch2, ch3, ch4-
         Numpy arrays
         Data of each channel, usually in the dimension of (time, positions,
          # of shots)
        -pos-
         Numpy array
         Array of positions set up
        -time-
         Numpy array
         Array of time recorded by the scope
        -attributes-
         String
         Descriptions of each channel
'''
import h5py
import numpy as np


def readhdf5(path):

    # Open the hdf5 file
    f = h5py.File(path, 'r')

    # Initialize the returned data array
    array = []

    # Get Acquisition group
    if 'Acquisition' in list(f.keys()):
        acq = f['Acquisition']
    else:
        print('Error: Acquisition group does not exist. No acquired data.')
        return

    # Get Control group under Acquisition and obtain data positions
    if 'Control' in list(f.keys()):
        con = f['Control']
        try:
            pos = np.array(con['Positions']['positions_setup_array'])
        except KeyError:
            pos = []
            print('Position array is empty.')
    else:
        print('Error: Control group does not exist. No data positions.')
        pos = []

    # Get Lecroy_scope group under Acquisition
    if 'LeCroy_scope' in list(acq.keys()):
        scope = acq['LeCroy_scope']
    else:
        print('Lecroy_scope group does not exist.')
        return

    # Get dataset from Lecroy_scope group and translate into arrays
    # The output data of each channel is a numpy array with dimensions
    #  pos * time_step
    # The exact data at a specific position at a specific time is
    #  chx[pos][time_step]
    if 'time' in list(scope.keys()):
        time = np.array(scope['time'])
        nt = (len(time))
    else:
        time = []
        nt = 1
        print('Time array is empty. Problem may occur when reshaping.')

    # Find the shape of the data array #
    dim = pos.shape
    dim = dim + (nt,)

    if 'Channel1' in list(scope.keys()):
        ch1 = np.array(scope['Channel1'])
        ch1 = ch1.reshape(dim)
        if 'description' in list(scope['Channel1'].attrs):
            ch1attr = scope['Channel1'].attrs['description']
        else:
            ch1attr = ''
    else:
        ch1 = []
        ch1attr = ''
        print('Channel 1 is empty.')

    if 'Channel2' in list(scope.keys()):
        ch2 = np.array(scope['Channel2'])
        ch2 = ch2.reshape(dim)
        if 'description' in list(scope['Channel2'].attrs):
            ch2attr = scope['Channel2'].attrs['description']
        else:
            ch2attr = ''
    else:
        ch2 = []
        ch2attr = ''
        print('Channel 2 is empty.')

    if 'Channel3' in list(scope.keys()):
        ch3 = np.array(scope['Channel3'])
        ch3 = ch3.reshape(dim)
        if 'description' in list(scope['Channel3'].attrs):
            ch3attr = scope['Channel3'].attrs['description']
        else:
            ch3attr = ''
    else:
        ch3 = []
        ch3attr = ''
        print('Channel 3 is empty.')

    if 'Channel4' in list(scope.keys()):
        ch4 = np.array(scope['Channel4'])
        ch4 = ch4.reshape(dim)
        if 'description' in list(scope['Channel4'].attrs):
            ch4attr = scope['Channel4'].attrs['description']
        else:
            ch4attr = ''
    else:
        ch4 = []
        ch4attr = ''
        print('Channel 4 is empty.')

    # Descriptions attached to the dataset
    attributes = 'Ch1: ' + ch1attr + '\nCh2: ' + ch2attr + '\nCh3: '
    + ch3attr + '\nCh4: ' + ch4attr

    # Reorganize positions to generate x,y arrays (need a check for
    #  incomplete datasets)
    if len(pos) != 0:
        temp = [x[1] for x in pos]
        ny = temp.count(temp[1])
        nx = int(len(temp)/ny)

        xx = temp[0:nx]  # list of x positions

        temp = [x[2] for x in pos]
        yy = temp[::nx]  # list of y positions
    else:
        nx = 0
        ny = 0
        xx = []
        yy = []

    # Returns the data in a data structure
    data_struct = {'ch1': ch1, 'ch2': ch2, 'ch3': ch3, 'ch4': ch4,
                   'pos': pos, 'time': time, 'attributes': attributes,
                   'nx': nx, 'ny': ny, 'xx': xx, 'yy': yy}
    return(data_struct)

# -------------- for testing ---------------------#
if __name__ == '__main__':
    path = '/Users/TheOne/Desktop/180/1-18 1-40, wide range.hdf5'
    ch1, ch2, ch3, ch4, pos, time, attr = readhdf5(path)
    print(ch1.shape)
    print(ch2.shape)
    print(ch3.shape)
    print(ch4.shape)
    print('----------------------------')
    print(pos.shape)
