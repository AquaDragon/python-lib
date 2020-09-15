'''
NAME:           read_lapd_data.py
AUTHOR:         swjtang
DATE:           15 Sep 2020
DESCRIPTION:    functionally similar to read_lapd_data_newsis_ver4.pro (IDL)
INPUTS:         
'''
import h5py, re
import numpy             as np
import lib.toolbox       as tbx
import lib.read_lapd_lib as lapdlib  # import functions from library

def read_lapd_data(fname, daqconfig=0, rchan=None, rshot=None,\
    xrange=None, yrange=None, zrange=None, nstep=1, sisid=None, tchannum=0, motionid=0, quiet=0):

    # check devices being used in the data acquisition ----------------------------------
    device_check = lapdlib.check_devices(fname)

    # check motion list devices ---------------------------------------------------------
    if device_check[0]:  # 6K Compumotor
        motion_data = lapdlib.lapd_6k_config(fname, motionid=motionid, quiet=quiet)
        tbx.qprint(quiet, 'Reading motion list from module:  6K Compumotor')
    elif device_check[6]:  # NI_XZ
        motion_data = lapdlib.lapd_ni_xz_config(fname, motionid=motionid, quiet=quiet)
        tbx.qprint(quiet, 'Reading motion list from module:  NI_XZ')
    elif device_check[7]: # NI_XYZ
        motion_data = lapdlib.lapd_ni_xyz_config(fname, motionid=motionid, quiet=quiet)
        tbx.qprint(quiet, 'Reading motion list from module:  NI_XYZ')
        daqconfig   = 4
    else:
        motion_data = {'nx': 1, 'ny': 1, 'nz': 1, 'nshots': 1, \
                       'x':[0], 'y':[0], 'z':[0], 'geom': 'unknown?'}
        tbx.qprint(quiet, 'No motion list module detected.')

    nx     = motion_data['nx']
    ny     = motion_data['ny']
    nz     = motion_data['nz']
    nshots = motion_data['nshots']
    x      = motion_data['x']
    y      = motion_data['y']
    z      = motion_data['z']
    geom   = motion_data['geom']

    # check if SIS crate active -----------------------------------------------------------
    if device_check[1]:
        siscrate_info = lapdlib.lapd_siscrate_config(fname, quiet=quiet)
        en_flag_3302 = (siscrate_info['sis3302_info'] != None)
        en_flag_3305 = (siscrate_info['sis3305_info'] != None)

        # function to print the list of channels ------------------------------
        def print_channels(sisboard_info, sisid):
            boardlist = sisboard_info['board_number']
            chanlist  = sisboard_info['channel_number']
            dtypelist = sisboard_info['data_type']
            nchan     = sisboard_info['nchan']
            if nchan > 0:
                tbx.qprint(quiet, '---------- SIS {0} list of enabled channels ({1}) ----------'\
                    .format(sisid, nchan))
                for ii in range(nchan):
                    te = [ii, boardlist[ii], chanlist[ii], dtypelist[ii]]
                    tbx.qprint(quiet, '[{0:2}] Board {1}, Channel {2}: {3}'.format(te[0], te[1], te[2], te[3]))
            
            return boardlist, chanlist, dtypelist, nchan
        # ---------------------------------------------------------------------
        # Checks if SIS3302 / SIS3305 is active AND HAS ENABLED CHANNELS.
        # This step is to print the channels.
        if device_check[2] & en_flag_3302:
            sis3302_temp = siscrate_info['sis3302_info']
            bl3302, cl3302, dtl3302, nch3302 = print_channels(sis3302_temp, 3302)
        if device_check[3] & en_flag_3305:
            sis3305_temp = siscrate_info['sis3305_info']
            bl3305, cl3305, dtl3305, nch3305 = print_channels(sis3305_temp, 3305)

        # The rest of the program can only read variables from one SIS board.
        # Checks if user has chosen a SIS board to read or chosen a board that has no
        # enabled channels. Otherwise, 3302 is the board that gets read by default
        # followed by 3305. If no boards are active, program ends and returns None.
        if sisid not in [3302,3305] or \
           (sisid == 3302) & (en_flag_3302!=True) or \
           (sisid == 3305) & (en_flag_3305!=True):
            if en_flag_3302  : sisid = 3302
            elif en_flag_3305: sisid = 3305
            else             : sisid = None

        tbx.qprint(quiet, '-------------------------------------------------------------')
        if sisid == 3302:
            sisboard_info = sis3302_temp
            boardlist, chanlist, dtypelist, nchan = bl3302, cl3302, dtl3302, nch3302
            tbx.qprint(quiet, 'Reading SIS 3302...')
        elif sisid == 3305:
            sisboard_info = sis3305_temp
            boardlist, chanlist, dtypelist, nchan = bl3305, cl3305, dtl3305, nch3305
            tbx.qprint(quiet, 'Reading SIS 3305...')
        else:
            tbx.qprint(quiet, 'None of the SIS 3302/3305 boards have any enabled channels!')
            return None

        # determine sampling rate of the board (use first board) --------------
        dt = sisboard_info['dt']
        nt = sisboard_info['nt']
        if len(np.unique(dt)) > 1 : tbx.qprint(quiet, 'Different boards have different sampling rates!')
        if len(np.unique(nt)) > 1 : tbx.qprint(quiet, 'Different boards have different number of samples!')
        if len(np.unique(dt)) > 1 or len(np.unique(nt)) > 1 : 
            tbx.qprint(quiet, 'Choosing sampling time of channel with index '+str(tchannum)+'...')
        time = [ii*dt[tchannum] for ii in range(nt[tchannum])]
    else: 
        return None  # if SIS crate not active end the program

    #### now read the data --------------------------------------------------------------
    # the following checks the user input for the range of values to be read ------------

    # some helper functions ---------------------------------------------------
    # this chooses the intersection between the default [0,nx] and a range given by user input.
    # sometimes the user range exceeds the range in the data
    def range_picker(r_default, r_user):
        r_user = np.sort(r_user)
        # returns default range if user range is out of bounds
        if r_user[1] < r_default[0] or r_user[0] > r_default[1]:
            return r_default
        # otherwise returns the intersecting range
        else: 
            return [np.amax([r_default[0], r_user[0]]), np.amin([r_default[1], r_user[1]])]

    # this checks the user input for a single value, a value range or for no/invalid input
    def check_user_range_input(max_value, user_input):
        if user_input is None     : return [0, max_value-1]
        elif len(user_input) == 1 : return [user_input[0], user_input[0]]
        elif len(user_input) == 2 : return range_picker([0, max_value-1], user_input)
        else                      : return [0, max_value-1]
    
    xrange = check_user_range_input(nx, xrange)
    yrange = check_user_range_input(ny, yrange)
    zrange = check_user_range_input(nz, zrange)
    rshot  = check_user_range_input(nshots, rshot)

    # each channel to be read has to be individually specified (only for channels)
    if rchan is None: rchan = [0]    # by default read channel 0
    else: 
        temp = np.unique(rchan)
        rchan = temp[np.where(temp < nchan)]

    tbx.qprint(quiet, '-------------------------------------------------------------')
    tbx.qprint(quiet, 'Data geometry = '+geom)
    tbx.qprint(quiet, 'Read Channels = '+'   '.join([str(ii) for ii in rchan]))
    tbx.qprint(quiet, 'Shot range    = '+str(rshot[0])+' to '+str(rshot[1]))
    tbx.qprint(quiet, 'X value range = '+str(xrange[0])+' to '+str(xrange[1]))

    if geom == 'xz-plane':
        tbx.qprint(quiet, 'Z value range = '+str(zrange[0])+' to '+str(zrange[1]))
    elif geom == 'xyz-volume':
        tbx.qprint(quiet, 'Y value range = '+str(yrange[0])+' to '+str(yrange[1]))
        tbx.qprint(quiet, 'Z value range = '+str(zrange[0])+' to '+str(zrange[1]))
    else:
        tbx.qprint(quiet, 'Y value range = '+str(yrange[0])+' to '+str(yrange[1]))

    # store original values
    ntt, nxx, nyy, nzz, nchann, nshotss = nt, nx, ny, nz, nchan, nshots

    x = x[xrange[0]:xrange[1]+1]
    y = y[yrange[0]:yrange[1]+1]
    z = z[zrange[0]:zrange[1]+1]

    nt, nx, ny, nz, nchan = len(time), len(x), len(y), len(z), len(rchan)
    nshots = rshot[1]-rshot[0]+1

    # (insert memory calculation here)

    def create_dataset(case):
        if case is 0      : return np.zeros([nt,nx,ny,nshots,nchan])        #<< standard lapd data run
        elif case in [1,2]: return np.zeros([nt,nx,ny,nshots,nchan,nstep])  #<< extra steps
        elif case is 3    : return np.zeros([nt,nshots,nchan,nstep])        #<< no xy motion, no extra steps
        elif case is 4    : return np.zeros([nt,nx,ny,nz,nshots,nchan])     #<< standard 3D data run
        else              : return 0

    def label_dataset(case):
        if case is 0       : return '(nt, nx, ny, nshots, nchan)'
        elif case in [1,2] : return '(nt, nx, ny, nshots, nchan, nstep)'
        elif case is 3     : return '(nt, nshots, nchan, nstep)'
        elif case is 4     : return '(nt, nx, ny, nz, nshots, nchan)'
        else               : return '()'
    dataset = create_dataset(daqconfig)
    hlabel  = label_dataset(daqconfig)

    #ignore option for SIS 3301

    data_group_name = '/Raw data + config/SIS crate/'
    data_subgroup_names = lapdlib.get_subgroup_names(fname, data_group_name)
    data_dataset_names  = lapdlib.get_dataset_names(fname, data_group_name)

    # tbx.qprint(quiet, data_subgroup_names, data_dataset_names)

    def get_shot_index(daqconfig, ishot=0, ix=0, iy=0, iz=0, istep=0):
        #data loop >> nshots -> xmotion -> ymotion 
        if daqconfig == 0  : return ishot + nshotss*(ix + nxx*iy)
        #data loop >> nshots -> xmotion -> ymotion -> extra variable steps 
        elif daqconfig == 1: return ishot + nshotss*(ix + nxx*(iy + nyy*istep))
        #data loop >> nshots -> extra variable steps -> xmotion -> ymotion
        elif daqconfig == 2: return ishot + nshotss*(istep + nstep*(ix + nxx*iy))
        #data loop >> nshots
        elif daqconfig == 3: return ishot + nshotss*istep
        #data loop >> nshots -> xmotion -> ymotion -> zmotion
        elif daqconfig == 4: return ishot + nshotss*(ix + nxx*iy + nxx*nyy*iz)
        else               : return 0

    # start reading in data
    for iix in range(xrange[0], xrange[1]+1):
        for iiy in range(yrange[0], yrange[1]+1):
            for iiz in range(zrange[0], zrange[1]+1):
                for iishot in range(rshot[0], rshot[1]+1):
                    iindex = get_shot_index(0, ishot=iishot, ix=iix, iy=iiy)

                    for jj in range(len(rchan)):
                        tbx.show_progress_bar([iix-xrange[0]+1, iiy-yrange[0]+1, iiz-zrange[0]+1, \
                            iishot-rshot[0]+1, jj+1], [nx, ny, nz, nshots, len(rchan)], \
                            ['xx', 'yy', 'zz', 'shots', 'chan'])

                        iiboard, iichan  = boardlist[rchan[jj]], chanlist[rchan[jj]]
                        if sisid == 3302:
                            temp = lapdlib.read_sis3302_shot(fname, data_group_name, \
                                iboard=iiboard, ichan=iichan, index=iindex)
                        elif sisid == 3305:
                            temp = lapdlib.read_sis3305_shot(fname, data_group_name, \
                                iboard=iiboard, ichan=iichan, index=iindex)

                        if daqconfig == 0:
                            dataset[:, iix-xrange[0], iiy-yrange[0], \
                                iishot-rshot[0], jj] = np.array(temp)
                        elif daqconfig in [1, 2]:
                            dataset[:, iix-xrange[0], iiy-yrange[0], \
                                iishot-rshot[0], jj, iistep] = np.array(temp)
                        elif daqconfig == 3:
                            dataset[:, iishot-rrange[0], jj, iistep] = np.array(temp)
                        elif daqconfig == 4:
                            dataset[:, iix-xrange[0], iiy-yrange[0], iiz-zrange[0], \
                                iishot-rshot[0], jj] = np.array(temp)

    tbx.qprint(quiet, '!!! '+hlabel+' = '+str(dataset.shape))

    temp ={
        'data'     : dataset,

        'x'        : x,
        'y'        : y,
        'z'        : z,
        'time'     : time,
        'dt'       : dt,

        'chanid'   : chanlist,
        'channame' : dtypelist,
        'desc'     : lapdlib.get_description(fname)
    }
    return temp