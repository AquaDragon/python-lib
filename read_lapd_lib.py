'''
NAME:           read_lapd_lib.py
AUTHOR:         swjtang
DATE:           22 Sep 2020
DESCRIPTION:    A library of helper functions used to read the hdf5 file from LAPD
INPUTS:         
'''
import h5py, re
import numpy                  as np
import lib.toolbox            as tbx
from matplotlib import pyplot as plt

#### Check devices used to write data file (auxillary function) -------------------------
def check_devices(fname):
    devices = ['6K Compumotor', 'SIS crate', 'SIS 3302', 'SIS 3305', \
               'SIS 3301', 'n5700', 'NI_XZ', 'NI_XYZ']
    
    ff = h5py.File(fname, 'r')

    # check if device exists from the name
    active_list = list(ff['Raw data + config'].keys())
    name_check = [(item in active_list) for item in devices]

    if name_check[1]:    # [1] is for SIS crate, check if active
        crate_name = '/Raw data + config/SIS crate/'
        crate_list = list(ff[crate_name].values())
        crate_subgroup_names = get_subgroup_names(fname, crate_name)
        # print(crate_subgroup_names)

        for ii in range(len(crate_subgroup_names)):
            config_name = crate_name + crate_subgroup_names[ii]
            config_list = list(ff[config_name].values())
            config_subgroup_names = get_subgroup_names(fname, config_name)

            # search for all config names starting with 'SIS crate' and outputs digitizer number
            temp = [re.match("SIS crate (\S*)", ii)[1] for ii in config_subgroup_names]
            all_sis_config = ['SIS '+ii for ii in set(temp)]
            config_check   = [(item in all_sis_config) for item in devices]

            name_check     = [any([ii, jj]) for ii, jj in zip(name_check, config_check)] #update

    return name_check


#### ------------------------------------------------------------------------------------
#### These are wrapper functions for the motionlist -------------------------------------
def lapd_6k_config(fname, motionid=0, quiet=0):
    return lapd_motionlist_config(fname, motionid=motionid, module='6K Compumotor', quiet=quiet)

def lapd_ni_xz_config(fname, motionid=0, quiet=0):
    return lapd_motionlist_config(fname, motionid=motionid, module='NI_XZ', quiet=quiet)

def lapd_ni_xyz_config(fname, motionid=0, quiet=0):
    return lapd_motionlist_config(fname, motionid=motionid, module='NI_XYZ', quiet=quiet)

#### Actual read motionlist is here -------------------------------------------
def lapd_motionlist_config(fname, motionid=0, module='6K Compumotor', quiet=0):
    ff = h5py.File(fname, 'r')
    
    if module == '6K Compumotor': motion_group_name = '/Raw data + config/6K Compumotor/'
    elif module == 'NI_XZ'      : motion_group_name = '/Raw data + config/NI_XZ/'  ###
    elif module == 'NI_XYZ'     : motion_group_name = '/Raw data + config/NI_XYZ/' ###
    motion_subgroup_names = get_subgroup_names(fname, motion_group_name)
    motion_dataset_names  = get_dataset_names(fname, motion_group_name)

    tbx.qprint(quiet, 'Reading data from motion list {0}'.format(motion_dataset_names[motionid]))

    motion_data = ff[motion_group_name+motion_dataset_names[motionid]] ## k:motionid
    motion_keys = list(motion_data.dtype.fields.keys()) #for troubleshooting

    motion_list_shot  = motion_data['Shot number']
    motion_list_x     = motion_data['x']
    motion_list_z     = motion_data['z']
    motion_list_theta = motion_data['theta']

    if module == 'NI_XZ': 
        motion_list_y   = [0]
        motion_list_phi = [0]
    else: 
        motion_list_y   = motion_data['y']
        motion_list_phi = motion_data['phi']

    if module == '6K Compumotor': 
        motion_list_r     = [0]
        motion_list_probe = motion_data['Probe name']
        motion_list_list  = motion_data['Motion list']
    else:
        motion_list_r     = motion_data['r']  ###
        motion_list_probe = motion_data['Configuration name'] ###
        motion_list_list  = motion_data['motion_index']       ###

    n_motion_lists = len(list(set(motion_list_list))) # set checks for unique elements
    n_probes       = len(list(set(motion_list_probe)))
    ## skipped error checking with multiple probes or multiple motion lists

    ## Get motion list parameters ---------------------------------------------
    ll = list(ff[motion_group_name].keys())
    if ll[0] == 'Run time list': ll0 = ll[1]
    else: ll0 = ll[0]
    motion_param_name = motion_group_name + ll0

    nx = int(ff[motion_param_name].attrs['Nx'])
    if module == '6K Compumotor':
        dx = ff[motion_param_name].attrs['Delta x']
        x0 = ff[motion_param_name].attrs['Grid center x']
        ny = int(ff[motion_param_name].attrs['Ny'])
        dy = ff[motion_param_name].attrs['Delta y']
        y0 = ff[motion_param_name].attrs['Grid center y']
        nz, dz, z0 = 1, 0, 0
    elif module in ['NI_XZ', 'NI_XYZ']:
        if module =='NI_XZ':
            ny, dy, y0 = 1, 0, 0
        elif module =='NI_XYZ':
            ny = int(ff[motion_param_name].attrs['Ny'])
            dy = ff[motion_param_name].attrs['dy']
            y0 = ff[motion_param_name].attrs['y0'] ###
        dx = ff[motion_param_name].attrs['dx'] ###
        x0 = ff[motion_param_name].attrs['x0'] ###
        nz = int(ff[motion_param_name].attrs['Nz']) ###
        dz = ff[motion_param_name].attrs['dz'] ###
        z0 = ff[motion_param_name].attrs['z0'] ###
        probe_name = ff[motion_param_name].attrs['probe_name'] ###

    nxnynz = nx*ny*nz
    nshots = len(motion_list_shot) // nxnynz

    geom = 'unknown'
    if (nx>1  and ny>1  and nz>1 )  : geom = 'xyz-volume'
    elif (nx>1  and ny>1  and nz==1): geom = 'xy-plane'
    elif (nx>1  and ny==1 and nz>1 ): geom = 'xz-plane'
    elif (nx==1 and ny>1  and nz>1 ): geom = 'yz-plane'
    elif (nx>1  and ny==1 and nz==1): geom = 'x-line'
    elif (nx==1 and ny>1  and nz==1): geom = 'y-line'
    elif (nx==1 and ny>1  and nz>1 ): geom = 'z-line'
    elif (nx==1 and ny==1 and nz==1): geom = 'point'

    if len(motion_list_shot) % nxnynz != 0:    # error checking for incomplete datasets
        tbx.qprint(quiet, '!!! This is a dataset from an incomplete datarun.')
        geom = 'incomplete'
    
    ## Format motion lists (ignore theta, phi) --------------------------------
    if (geom=='xyz-volume'):
        temp = np.reshape(motion_list_x, (nshots,nx,ny,nz), order='F')
        x = temp[0,:,1,1]
        temp = np.reshape(motion_list_y, (nshots,nx,ny,nz), order='F')
        y = temp[0,1,:,1]
        temp = np.reshape(motion_list_z, (nshots,nx,ny,nz), order='F')
        z = temp[0,1,1,:]

    elif (geom=='xy-plane'):
        temp = np.reshape(motion_list_x, (nshots,nx,ny), order='F')
        x = temp[0,:,1]
        temp = np.reshape(motion_list_y, (nshots,nx,ny), order='F')
        y = temp[0,1,:]
        z = [motion_list_z[0]]

    elif (geom=='xz-plane'):
        temp = np.reshape(motion_list_x, (nshots,nx,nz), order='F')
        x = temp[0,:,1]
        temp = np.reshape(motion_list_z, (nshots,nx,nz), order='F')
        z = temp[0,1,:]
        y = [motion_list_y[0]]

    elif (geom=='yz-plane'):
        temp = np.reshape(motion_list_y, (nshots,ny,nz), order='F')
        y = temp[0,:,1]
        temp = np.reshape(motion_list_z, (nshots,ny,nz), order='F')
        z = temp[0,1,:]
        x = [motion_list_x[0]]
    
    ## x line, y-line unverified
    elif (geom=='x-line'):
        temp = np.reshape(motion_list_x, (nshots,nx), order='F')
        x = temp[0,:]
        y = [motion_list_y[0]]
        z = [motion_list_z[0]]
    
    elif (geom=='y-line'):
        x = [motion_list_x[0]]
        temp = np.reshape(motion_list_y, (nshots,ny), order='F')
        y = temp[0,:]
        z = [motion_list_z[0]]

    elif (geom=='z-line'):
        x = [motion_list_x[0]]
        y = [motion_list_y[0]]
        temp = np.reshape(motion_list_z, (nshots,nz), order='F')
        z = temp[0,:]       
    
    elif (geom=='point'):
        x = [motion_list_x[0]]
        y = [motion_list_y[0]]
        z = [motion_list_z[0]]

    elif (geom=='incomplete'):
        nx, ny, nz = 1, 1, 1
        nshots = len(motion_list_shot)
        x = [motion_list_x[0]]
        y = [motion_list_y[0]]
        z = [motion_list_z[0]]

    output = {'nx': nx, 'ny': ny, 'nz': nz, 'nshots': nshots, \
              'x': x, 'y': y, 'z':z, 'geom': geom}
    return output


#### ------------------------------------------------------------------------------------
#### SIS Crate Global (extract digitizer data from SIS crate) ---------------------------
def lapd_siscrate_config(fname, quiet=0):
    siscrate_name = '/Raw data + config/SIS crate/'
    siscrate_subgroup_names = get_subgroup_names(fname, siscrate_name)
    siscrate_dataset_names = get_dataset_names(fname, siscrate_name)

    # select first config file
    sisconfig_name = siscrate_name + siscrate_subgroup_names[0]  

    temp = {
        'sis3820_info': get_sis3820_info(fname, sisconfig_name, quiet=quiet) ,
        'sis3302_info': get_sis3302_config(fname, sisconfig_name, quiet=quiet),
        'sis3305_info': get_sis3305_config(fname, sisconfig_name, quiet=quiet)
    }

    return temp

#### SIS 3820 is the timing digitizer located in the SIS crate ----------------
def get_sis3820_info(fname, sisconfig_name, quiet=0):
    ff = h5py.File(fname, 'r')

    sis3820_name = sisconfig_name + '/SIS crate 3820 configurations[0]'

    # clock information and strings -------------------------------------------
    strings_clock_odd_outputs  = ['Clock', 'Start/stop']
    strings_clock_even_outputs = ['Clock', 'Start/stop']
    strings_clock_source = ['Internal 100 MHz (delay locked loop)', \
                '2nd Internal 100 MHz (U580)', 'External Clock', 'VME Key Clock']
    strings_clock_mode   = ['Double clock',  'Straight clock', '1/2 clock', \
                '1/4 clock', 'User-specified divider']

    clock_odd_output  = strings_clock_odd_outputs[ff[sis3820_name].attrs['Odd outputs']]
    clock_even_output = strings_clock_even_outputs[ff[sis3820_name].attrs['Even outputs']]
    clock_source      = strings_clock_source[ff[sis3820_name].attrs['Clock source']]
    clock_mode        = strings_clock_mode[ff[sis3820_name].attrs['Clock mode']]
    
    global_clock_tick = 1e-8 #sec
    clock_delay = global_clock_tick * ff[sis3820_name].attrs['Delay']

    output = {
        'clock_delay'     : clock_delay,
        'clock_mode'      : clock_mode,
        'clock_source'    : clock_source
    }
    return output


#### Get SIS 3302/3305 board information ------------------------------------------------
# These are wrapper functions that will call the get_sis33xx_config function
def get_sis3302_config(fname, sisconfig_name, quiet=0):
    return get_sisboard_config(fname, sisconfig_name, sid=3302, quiet=quiet)

def get_sis3305_config(fname, sisconfig_name, quiet=0):
    return get_sisboard_config(fname, sisconfig_name, sid=3305, quiet=quiet)

#### Actual read sisboard config code is here ---------------------------------
def get_sisboard_config(fname, sisconfig_name, sid=0, quiet=0):
    # error checking
    if sid not in [3302, 3305]: 
        print("invalid board id") # internal error message
        return None

    ff = h5py.File(fname, 'r')

    # look for both 3302 and 3305 boards --------------------------------------
    sis_board_types    = list(ff[sisconfig_name].attrs['SIS crate board types'])
    sis_config_indices = list(ff[sisconfig_name].attrs['SIS crate config indices'])
    sis_slot_numbers   = list(ff[sisconfig_name].attrs['SIS crate slot numbers'])

    sis3302_config_indices, sis3302_slot_numbers = [], []
    sis3305_config_indices, sis3305_slot_numbers = [], []
    for ii in range(len(sis_board_types)):
        if sis_board_types[ii] == 2: 
            sis3302_config_indices.append(sis_config_indices[ii])
            sis3302_slot_numbers.append(sis_slot_numbers[ii])
        if sis_board_types[ii] == 3: 
            sis3305_config_indices.append(sis_config_indices[ii])
            sis3305_slot_numbers.append(sis_slot_numbers[ii])

    # start looking at individual boards --------------------------------------
    if sid==3302:
        sis_board_numbers  = [int((ii-5)/2 +1)  for ii in sis3302_slot_numbers]
        sis_config_indices = sis3302_config_indices
    if sid==3305:
        sis_board_numbers  = [int((ii-13)/2 +1) for ii in sis3305_slot_numbers]
        sis_config_indices = sis3305_config_indices

    nboards = len(sis_board_numbers)
    if nboards > 0 :
        tbx.qprint(quiet, 'SIS {0} boards used = {1}'.format(sid, sis_board_numbers))
    else :
        tbx.qprint(quiet, 'No '+str(sid)+' boards used.')
        return None
    
    # start extracting config data --------------------------------------------
    temp    = np.empty(nboards, dtype=object)
    for ii in range(nboards):
        iboard  = sis_board_numbers[ii]-1
        iconfig = sis_config_indices[ii]

        board_config_name  = 'SIS crate '+str(sid)+' configurations['+str(iconfig)+']'
        board_config_group = sisconfig_name + '/' + board_config_name
        # tbx.qprint(quiet, iboard, iconfig, board_config_name)
        # tbx.qprint(quiet, list(ff[board_config_group].attrs))

        ### Hardware digitization (different for 3302 and 3305) ---------------
        if sid == 3302:
            global_clock_tick = 1e-8
            sample_averaging  = 2**ff[board_config_group].attrs['Sample averaging (hardware)']
            dt                = global_clock_tick * float(sample_averaging)

            bandwidth = None ## 3305 parameter

        if sid == 3305:
            bw_index_to_bw    = [1e9, 1.8e9]  # in Hz
            mode_index_to_dt  = [8e-10, 4e-10, 2e-10]  # in sec, (4chan,2chan,1chan per FPGA)

            channel_mode = ff[board_config_group].attrs['Channel mode']
            dt           = mode_index_to_dt[channel_mode]
            bw_mode      = ff[board_config_group].attrs['Bandwidth']
            bandwidth    = bw_index_to_bw[bw_mode]

            sample_averaging = None ## 3302 parameter

        tbx.qprint(quiet, 'SIS '+str(sid)+' effective clock rate: Board '+str(iboard+1) + \
                ' = '+ str(1/(1e6*dt)) + ' MHz')

        nt             = ff[board_config_group].attrs['Samples']
        time_acquired  = nt * dt #sec
        shots_averaged = ff[board_config_group].attrs['Shot averaging (software)']

        # there are 8 channels in each board --------------------------------------
        channel_number   = np.empty([8], dtype=object)
        channels_enabled = np.empty([8], dtype=object)
        comments         = np.empty([8], dtype=object)
        dc_offset        = np.empty([8], dtype=object)
        data_type        = np.empty([8], dtype=object)

        def config_attrs(attribute, sid, ichan):
            if sid == 3302: return ff[board_config_group].attrs[attribute+' '+str(ichan+1)]
            if sid == 3305:
                fpga_id, fpga_ch = get_sis3305_fpga_id(ichan)  # helper function
                if attribute == 'Ch':
                    return ff[board_config_group].attrs['FPGA '+str(fpga_id)\
                        +' '+attribute+' '+str(fpga_ch)] + (fpga_id-1)*4 
                else:
                    return ff[board_config_group].attrs['FPGA '+str(fpga_id)\
                        +' '+attribute+' '+str(fpga_ch)]
        
        for jj in range(8):
            channel_number[jj]   = config_attrs('Ch', sid, jj) 
            channels_enabled[jj] = str_decode(config_attrs('Enabled', sid, jj))
            comments[jj]         = str_decode(config_attrs('Comment', sid, jj))
            data_type[jj]        = str_decode(config_attrs('Data type', sid, jj))

            if sid == 3302: dc_offset[jj] = config_attrs('DC offset', sid, jj)
            else          : dc_offset[jj] = None
        
        temp[ii] = {
            'board_number'    : iboard+1,

            'dt'              : dt,
            'nt'              : nt,
            'time_acquired'   : time_acquired,
            'shots_averaged'  : shots_averaged,
            'sample_averaging': sample_averaging,  # 3302 parameter
            'bandwidth'       : bandwidth,         # 3305 parameter

            'channel_number'  : list(channel_number),
            'channels_enabled': list(channels_enabled),
            'data_type'       : list(data_type),
            'comment'         : list(comments),
            'dc_offset'       : list(dc_offset)    # 3302 parameter
        }

    # returns a list with only enabled channels -------------------------------
    key_test = ['board_number', 'dt', 'nt', 'time_acquired', 'shots_averaged', \
        'sample_averaging', 'bandwidth', 'channel_number', 'data_type', 'comment', \
        'dc_offset']
    chan_info_temp = {key: [] for key in key_test}

    for jboard in temp:
        for jj in range(8):
            if jboard['channels_enabled'][jj] == 'TRUE':
                chan_info_temp['board_number'].append(jboard['board_number'])
                chan_info_temp['dt'].append(jboard['dt'])
                chan_info_temp['nt'].append(jboard['nt'])
                chan_info_temp['time_acquired'].append(jboard['time_acquired'])
                chan_info_temp['shots_averaged'].append(jboard['shots_averaged'])
                chan_info_temp['sample_averaging'].append(jboard['sample_averaging'])
                chan_info_temp['bandwidth'].append(jboard['bandwidth'])
                chan_info_temp['channel_number'].append(jboard['channel_number'][jj])
                chan_info_temp['data_type'].append(jboard['data_type'][jj])
                chan_info_temp['comment'].append(jboard['comment'][jj])
                chan_info_temp['dc_offset'].append(jboard['dc_offset'][jj])
    
    chan_info_temp['nchan'] = len(chan_info_temp['channel_number'])
    
    return chan_info_temp


#### ------------------------------------------------------------------------------------
#### These are wrapper functions to read each individual shot ---------------------------
def read_sis3302_shot(fname, config_name, iboard=1, ichan=1, index=0):
    temp = read_sisboard_shot(fname, 3302, config_name, iboard, ichan, index)
    return temp

def read_sis3305_shot(fname, config_name, iboard=1, ichan=1, index=0):
    temp = read_sisboard_shot(fname, 3305, config_name, iboard, ichan, index)
    return temp

#### Actual shot reading code is here -----------------------------------------
def read_sisboard_shot(fname, sisid, config_name, iboard=1, ichan=1, index=0):
    config_subgroup_list = get_subgroup_names(fname, config_name)

    if sisid == 3302:
        islot = iboard*2 + 3

        dataset_config_name  = config_name + config_subgroup_list[0]+ \
            ' [Slot '+str(islot)+': SIS 3302 ch '+str(ichan)+']'
    elif sisid == 3305:
        islot = iboard*2 + 11
        fpga_id, fpga_ch = get_sis3305_fpga_id(ichan)  #helper function

        dataset_config_name  = config_name + config_subgroup_list[0]+ \
            ' [Slot '+str(islot)+': SIS 3305 FPGA '+str(fpga_id)+' ch '+str(fpga_ch)+']'

    # read file data
    ff = h5py.File(fname, 'r')
    ll = ff[dataset_config_name]  # data contained as (nshots, dt)

    dt = ll.shape[1]  # extract size of time dimension

    temp = np.zeros([1, dt])
    ll.read_direct(temp, np.s_[index,:])  # read slice of data

    # convert digitizer indices to voltage values 
    if sisid == 3302  : return [ii*7.7241166e-5-2.531 for ii in temp]
    # 3305: 10 bits (0 to 1023) and 2 Volt range
    elif sisid == 3305: return [-ii*(2 / (2**10 - 1)) for ii in temp]
    else              : return None

#### ------------------------------------------------------------------------------------
#### Read MSI information ---------------------------------------------------------------
def read_msi_info(fname):
    config_list       = '/MSI/'
    msi_subgroup_list = get_subgroup_names(fname, config_list)
        # ['Discharge', 'Gas pressure', 'Heater', 'Interferometer array', 'Magnetic field']

    ## define a function that unpacks the MSI information into a dict
    def unload_dataset_info(fname, name):
        dataset_list = get_dataset_names(fname, name)

        info_arr = {}
        ff = h5py.File(fname, 'r')
        for ii in list(ff[name].attrs):
            info_arr[ii] = ff[name].attrs[ii]              ## unload root

        for jj in dataset_list:
            info_arr[jj] = np.array(ff[name + jj +'/'])    ## unload sub-directories

        return info_arr

    ### return discharge MSI information ------------------------------------------------
    discharge_info = unload_dataset_info(fname, '/MSI/Discharge/')
    
    ### create time array for easy analysis
    dt = discharge_info['Timestep']
    nt = len(discharge_info['Discharge current'][0,:])
    discharge_info['time'] = np.arange(nt)*dt + discharge_info['Start time']

    ### return interferometer MSI information -------------------------------------------
    intf_name          = '/MSI/Interferometer array/'
    intf_subgroup_list = get_subgroup_names(fname, intf_name)

    ii = intf_subgroup_list[0]
    intf0_name = intf_name + ii +'/'
    intf_dataset_list  = get_dataset_names(fname, intf0_name)
        # ['Interferometer summary list', 'Interferometer trace']
    #print(intf_subgroup_list, intf_dataset_list)

    ### return bfield MSI information ---------------------------------------------------
    bfield_info = unload_dataset_info(fname, '/MSI/Magnetic field/')
    print(bfield_info)
    print(bfield_info['Magnetic field profile'])

    ### incomplete ----------------------
    temp = {
        'discharge': discharge_info,
        'bfield'   : bfield_info
    }
    return temp


#### ------------------------------------------------------------------------------------
#### Helper functions used frequently for siscrate and hdf5 manipulation ----------------
def get_sis3305_fpga_id(ichan):
    chan_to_fpga_id   = [1,1,1,1,2,2,2,2]
    chan_to_fpga_chan = [1,2,3,4,1,2,3,4]
    return chan_to_fpga_id[ichan], chan_to_fpga_chan[ichan]

def get_subgroup_names(fname, group_name):
    ff = h5py.File(fname, 'r')
    ll = list(ff[group_name].keys())  # keys are used to grab the object

    # values are used to check for type (this get list of objects (groups, datasets))
    lval = list(ff[group_name].values())  
    subgroup_names = []
    for ii in range(len(lval)):
        item = lval[ii]
        if isinstance(item, h5py.Group): subgroup_names.append(ll[ii])
    return subgroup_names

def get_dataset_names(fname, group_name):
    ff = h5py.File(fname, 'r')
    ll = list(ff[group_name].keys())  # keys are used to grab the object

    # values are used to check for type (this get list of objects (groups, datasets))
    lval = list(ff[group_name].values())  
    dataset_names = []
    for ii in range(len(lval)):
        item = lval[ii]
        if isinstance(item, h5py.Dataset): dataset_names.append(ll[ii])
    return dataset_names

def str_decode(item):
    return str(item.decode('utf-8'))

def get_description(fname):
    ff = h5py.File(fname, 'r')
    aa = ff['Raw data + config']
    return aa.attrs['Description'].decode("utf-8")




#### CODE DUMP
    # discharge_name         = '/MSI/Discharge/'
    # discharge_dataset_list = get_dataset_names(fname, discharge_name)
    #     # ['Cathode-anode voltage', 'Discharge current', 'Discharge summary']

    # discharge_info = {}
    # for ii in list(ff[discharge_name].attrs):
    #     discharge_info[ii] = ff[discharge_name].attrs[ii]

    # for jj in discharge_dataset_list:
    #     discharge_info[jj] = np.array(ff[discharge_name + jj +'/'])
#---
    # bfield_name          = '/MSI/Magnetic field/'
    # bfield_dataset_list = get_dataset_names(fname, bfield_name)
        # ['Magnet power supply currents', 'Magnetic field profile', 'Magnetic field summary']

    # bfield_info = {}
    # for ii in list(ff[bfield_name].attrs):
    #     bfield_info[ii] = ff[bfield_name].attrs[ii]

    # for jj in bfield_dataset_list:
    #     bfield_info[jj] = np.array(ff[bfield_name + jj +'/'])