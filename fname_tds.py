'''
NAME:           fname_tds.pro
AUTHOR:         swjtang
DATE:           03 Jan 2021
DESCRIPTION:    Returns filenames of TDS .hdf5 files with a single call number
INPUTS:         callnumber = the id of the file to be read
                full       = (optional) returns the filename only if set to 0
'''
import socket

def fname_tds(callnumber, full=1, old=0):
    if socket.gethostname() == 'midas': 
        if old == 1 : datafolder = '/data_old'
        else        : datafolder = '/data' 
    else: 
        if old == 1 : datafolder = '/data'
        else        : datafolder= ' /data_new'

    # list of directories that store TDS file data
    dir1 = '/BAPSF_Data/LaB6_Cathode/HighFreq_FluxRope/'
    dir2 = '/BAPSF_Data/LaB6_Cathode/HighFreq_FluxRope2/'
    dir3 = '/BAPSF_Data/LaB6_Cathode/TDS_FluxRopes/'
    dir4 = '/BAPSF_Data/TDS/May_18/'
    dir5 = '/BAPSF_Data/TDS/18_July/'
    dir6 = '/BAPSF_Data/TDS/18_Nov/'
    dir7 = '/BAPSF_Data/TDS/19_Apr/'
    dir8 = '/BAPSF_Data/TDS/19_July/'
    dirA = '/data_old/BAPSF_Data/LaB6_Cathode/FluxRopes_2_14/'

    testdir = '/BAPSF_Data/LaB6_Cathode/Single5cmFluxRope/'

    filenames = {
        'test': [testdir, 'I75_V100_Bz330_B24_B45_M35_Plane_2014_05_24_926AM.hdf5'],
    #### DATARUN (HighFreq_FluxRope1): DEC 2016 - JAN 2017 ------------------------------
        101: [dir1, 'Line_NoFR_EBp40_12_15.hdf5'],
        102: [dir1, 'Line_EBp40_12_14.hdf5'],
            # SINGLE CHANNEL: 1x51 plane, 100 shots per position
        103: [dir1, 'XZ_12_15.hdf5'],
        104: [dir1, 'XZ_12_16.hdf5'],
            # files 103/104: 7 channels, 30 shots per position
            # file103: 41x41 plane
            # file104: 61x41 plane
            # CHANNELS= 0-2: Ex,Ey,Ez mov , 3-5 Ex,Ey,Ez ref, 6= ???
        105: [dir1, 'XZ_12_20.hdf5'],  ## incomplete run
            # CHANNELS: 0-5: Vx+/Vx-/Vy+/Vy-/Vz+/Vz- ref, 6: By ref, 
            # 8-13: Vx+/Vx-/Vy+/Vy-/Vz+/Vz- mov
            # 51x41 plane
        106: [dir1, 'XY_1_3.hdf5'],
            # 17 channels, 21x21 plane, 30 shots per position
            # CHANNELS (50 MHz): 0-5: Vx+/Vx-/Vy+/Vy-/Vz+/Vz- mov
            # 6-8: Bx/By/Bz ref, 9-11: Bx/By/Bz mov
            # 12-17: Vx+/Vx-/Vy+/Vy-/Vz+/Vz- ref
            # cant really look at the B-field since the single turn loops are damaged
        107: [dir1, 'XY_1_4.hdf5'],
        108: [dir1, 'XY_1_5.hdf5'],
            # files 107/108: 17 channels, 41x41 plane, 30 shots per position
            # 107 is a small area near one rope edge
        109: [dir1, '01-Efield-plane-NoFR-p33.hdf5'],
        110: [dir1, '02-Efield-plane-NoFR-p33-330G.hdf5'],
    #### DATARUN (HighFreq_FluxRope2): 01 MAY - 05 MAY 2017 -----------------------------
        201: [dir2, 'Plane_EP35_B29_May2.hdf5'],
            # 8 channels: 0=LaB6 current, 1=diode, 2-4=Bx,By,Bz ref, 5-7=Bx,By,Bz mov 29
            # use motionid=2
        202: [dir2, 'Plane_V32_V33_V35_May4.hdf5'],
            # 7 channels# 0=diode, 1=Vmov 32, 2=Vref 33, 3-5=Vx,Vy,Vz 35, 6=LaB6 current
            # USE TCHANNUM=1 for file202
        203: [dir2, 'Lang_line_port42.hdf5'],
    #### DATARUN (TDS_FluxRopes): 28 AUG - 08 SEP 2017 ----------------------------------
    # all data was digitized at 3.125 MHz (=100 MHz/32) unless otherwise specified
    # there might have been a 3dB attenuator on the current channel
    # so multiply current by: 10^0.3 (3dB attenuation) * 2 (50 ohm impedance) 
    #                           * 2.5e3 (conversion 2.5kA/mV)
        301: [dir3, '01_Bplane31.hdf5'],     # B-field data plane
            # channel: 0-2=Bx,By,Bz, 3=LaB6 current, 4=LaB6 voltage, 5=Diode28
            # 41x41 plane, 15 shots per position
        303: [dir3, '03_Bline31_refprobe.hdf5'],
        304: [dir3, '04_Bplane31_refprobe.hdf5'],
        306: [dir3, '06_Bplane31_refprobe_coherent.hdf5'],
            # channel: 0=LaB6 current, 1= LaB6 voltage, 2-4=Bx,By,Bz mov, 5-7=Bx,By,Bz fix37
            # 03 Bline: 1x41 line at x=0, 15 shots per position
            # 04 & 06 Bplane: 41x41 plane, 15 shots per position
        305: [dir3, '05_BrefprobeTest.hdf5'],
            # to read this data override 'lapd_6k_configuration.pro L175 with [ny=1]'
            # tim set a motion list on the run but destroyed the probe motion command
        307: [dir3, '07_TDSpotentialA1_perp_move3.hdf5'],
            # channel: 0=LaB6 current, 1=LaB6 voltage, 2= TDS_A2_fix_L, 3-4=TDS_A1_mov_LRperp,
            #          5-7= Bx,By,Bz fix37
            # 07 plane: 41x41 plane, 15 shots per position
        308: [dir3, '08_TDSpotentialA1_parallel_Bplane31.hdf5'],
            # 08 plane: 41x41 plane, 15 shots per position (both)
            # board 1 : 0=LaB6 current, 1=LaB6 voltage, 2= TDS_A2_fix_L
            #           3-4=TDS_A1_mov_LRparallel,  5-7= Bx,By,Bz fix21
            # board 2 : 8-10=Bx,By,Bz mov31
        309: [dir3, '09_TDSpotentialA1_parallel_Bplane31_Isatsweep29.hdf5'],
            # 09 plane: 41x41 plane, 15 shots per position (all 3)
            # board 1 (3.125MHz): 0=LaB6 current (3dB), 1=LaB6 voltage, 2= TDS_A2_fix_L, 
            #                     3-4=TDS_A1_mov_LRparallel,  5-7= Bx,By,Bz fix21
            # board 2 (3.125MHz): 8-10=Bx,By,Bz mov31, 11=V_sweep(L), 12=I_sweep(L), 13=Isat(R)
    #### DATARUN (May_18): 14 - 21 MAY 2018 ---------------------------------------------
        401: [dir4, '01_Bdot45_Dipole30_plane.hdf5'],   # bfield plane & dipole plane
            # board 1: 0-2: Bx,By,Bz mov45, 3-5: Bx,By,Bz fix33     (digitized 3.125MHz)
            # board 2: 6: B1dipole30 left, 7: B1dipole30 right  (digitized 50MHz)
            # 01 plane: 41x41 plane, 15 shots per position (0.75cm spacing)
        402: [dir4, '02a_Dipole30_line.hdf5'],          # line sweep of potentials
            # board 2 (digitized 50MHz): 0: B1dipole30 left, 7: B1dipole30 right
            # 02 line sweep: 41x3 plane, 10 shots per position 
            #   0.75cm spacing per line, 41 points, 3 lines at -3, 0 and 3 cm
        403: [dir4, '02b_Dipole30_line_noLaB6.hdf5'],   # same but no LaB6 cathode

        # 404-408: line sweep of potentials, see how spikes vary with discharge current
        # same conditions as 402
        404: [dir4, '04_Dipole30_line_varyI_1360A.hdf5'],
        405: [dir4, '05_Dipole30_line_varyI_1215A.hdf5'],
        406: [dir4, '06_Dipole30_line_varyI_1125A.hdf5'],
        407: [dir4, '07_Dipole30_line_varyI_985A.hdf5'],
        408: [dir4, '08_Dipole30_line_varyI_838A.hdf5'],

        410: [dir4, '10_Dipole30_line_varyI_705A.hdf5'],
        411: [dir4, '11_Dipole30_line_varyI_565A.hdf5'],
        412: [dir4, '12_Dipole30_line_varyI_442A.hdf5'],
        413: [dir4, '13_Dipole30_line_varyI_305A.hdf5'],
        414: [dir4, '14_Dipole30_line_varyI_240A.hdf5'],
        415: [dir4, '15_Dipole30_line_varyI_140A.hdf5'],

        409: [dir4, '09_Bdot37_Dipole30_plane.hdf5'],
            # board 1 (digitized 3.125MHz): 0-2: Bx,By,Bz mov37, 3-5: Bx,By,Bz fix33
            # board 2 (digitized 50MHz): 6: B1dipole30 left, 7: B1dipole30 right
            # 09 plane: 41x41 plane, 15 shots per position (0.75cm spacing)

        # new flux rope parameters 1000G field
        420: [dir4, '20_Dipole30_EB27_FR1000G_test.hdf5'],
        421: [dir4, '21_Bdot37_Dipole30_EB27.hdf5'],
            # 21 plane: 41x41 plane, 15 shots per position (0.75cm spacing)
            # board 1 (digitized 3.125MHz): 0-2: Bx,By,Bz mov37, 3-5: Bx,By,Bz fix33    
            #                               6-7: Port 43/28 diode (10K Ohm) 
            # board 2 (digitized 50MHz): 8-13: Vx+/Vx-/Vy+/Vy-/Vz+/Vz- EB mov27  
            # board 3 (digitized 50MHz): 14: B1dipole30 left mov, 15: B1dipole30 right mov
        422: [dir4, '22_Dipole30_EB27_FR800G_test.hdf5'],
        423: [dir4, '23_Dipole30_EB27_FR1000G.hdf5'],
        424: [dir4, '24_Dipole30_EB27_FR800G.hdf5'],
        425: [dir4, '25_Bdot37_Dipole30_EB27_FR1000G_plane.hdf5'],
            # 25 plane: 41x41 plane, 15 shots per position (0.75cm spacing)
            # board 1 (digitized 3.125MHz): 0-2: Bx,By,Bz mov37, 3-5: Bx,By,Bz fix33    
            #                               6: Interferometer32, 7: UV light32 (He2+) 
            # board 2 (digitized 50MHz): 8-13: Vx+/Vx-/Vy+/Vy-/Vz+/Vz- EB mov27  
            # board 3 (digitized 50MHz): 14: B1dipole30 left mov, 15: B1dipole30 right mov
            #                            16: B3dipole31 left, 17: B3dipole31 right
            #                            18: LaB6 current, 19:LaB6 voltage
            # board 4 (digitized 3.125MHz): 20: Port 33 diode (450 Ohm) 
            #                               21-22: Port 43/28 diode (10K Ohm)
        # bdot amplifier changed from 10x to 100x
        # 26-33 plane: 41x3 plane, 15 shots per position (0.75cm spacing in X, 3cm spacing in Y)
            # board 1 (digitized 3.125MHz): 0: Interferometer32, 1: UV light32 (He2+)
            # board 2 (digitized 50MHz): 2-7: Vx+/Vx-/Vy+/Vy-/Vz+/Vz- EB mov27
            # board 3 (digitized 50MHz): 8-9: Isat+1/Isat-4 Mach32
            #                            10: B3dipole31 left, 11: B3dipole31 right
            #                            12: LaB6 current, 13:LaB6 voltage, 14: Vfloat Mach32
            # board 4 (digitized 3.125MHz): 15: Port 33 diode (450 Ohm) 
            #                               16-17: Port 43/28 diode (10K Ohm)
        426: [dir4, '26_Mach32_EB27_FR1000G_bank200V.hdf5'],                #}
        427: [dir4, '27_Mach32_EB27_FR1000G_bank175V.hdf5'],                #}
        428: [dir4, '28_Mach32_EB27_FR1000G_bank150V.hdf5'],                #}
        429: [dir4, '29_Mach32_EB27_FR1000G_bank125V.hdf5'],                #}
        430: [dir4, '30_Mach32_EB27_FR1000G_bank100V.hdf5'],                #}
        431: [dir4, '31_Mach32_EB27_FR1000G_bank75V.hdf5'],                 #}
        432: [dir4, '32_Mach32_EB27_FR1000G_bank150V(restart).hdf5'],       #}
        433: [dir4, '33_Mach32_EB27_FR1000G_bank175V(restart).hdf5'],       #}

        434: [dir4, '34_Bdot37_Mach32_EB27_FR800G_bank200V.hdf5'],
            # !!! Mach32 (Ch 14) failure at (x,y)=(29,7)
            # 34 plane: 41x41 plane, 15 shots per position (0.75cm spacing)
            # board 1 (digitized 3.125MHz): 0-2: Bx,By,Bz mov37, 3-5: Bx,By,Bz fix33
            #                               6: Interferometer32, 7: UV light32 (He2+)               
            # board 2 (digitized 50MHz): 8-13: Vx+/Vx-/Vy+/Vy-/Vz+/Vz- EB mov27  
            # board 3 (digitized 50MHz): 14-15: Isat+1/Isat-4 Mach32
            #                            16: B3dipole31 left, 17: B3dipole31 right
            #                            18: LaB6 current, 19:LaB6 voltage, 20: Vfloat Mach32
            # board 4 (digitized 3.125MHz): 21: Port 33 diode (450 Ohm) 
            #                               22-23: Port 43/28 diode (10K Ohm)
        436: [dir4, '36_Bdot37_Mach32_EB27_FR500G_bank200V_planeII.hdf5'],
            # 36 plane: 41x41 plane, 15 shots per position (0.75cm spacing)
            # board 1 (digitized 3.125MHz): 0-2: Bx,By,Bz mov37, 3-5: Bx,By,Bz fix33    
            #                               6: Interferometer32, 7: UV light32 (He2+) 
            # board 2 (digitized 50MHz): 8-13: Vx+/Vx-/Vy+/Vy-/Vz+/Vz- EB mov27  
            # board 3 (digitized 50MHz): 14-15: Isat+1/Isat-4 Mach32
            #                            16: B3dipole31 left, 17: B3dipole31 right
            #                            18: LaB6 current, 19:LaB6 voltage, 20: Vfloat Mach32
            # board 4 (digitized 3.125MHz): 21: Port 33 diode (450 Ohm) 
            #                               22-23: Port 43/28 diode (10K Ohm)
        #### INCOMPLETE DATA ------------------------------------------------------------
        435: [dir4, '35_Bdot37_Mach32_EB27_FR500G_bank200V.hdf5'],
                # !!! INCOMPLETE PLANE (stopped after 12/41 lines)
    #### DATARUN (18_July): 07 - 20 JUL 2018 --------------------------------------------
        # 44x planes: 41x41 plane, 15 shots per position (0.75cm spacing)
        437: [dir5, 'A1_amplifier_test 2018-07-18.hdf5'],
        438: [dir5, 'A2_amplifier_test 2018-07-18.hdf5'],

        #bdot planes
        440: [dir5, '01_Bmov35_41_45_Bfix30_32 2018-07-07.hdf5'],
            # board 1 (3.125MHz):   0-2: Bx,By,Bz fix32 (edge), 3-5: Bx,By,Bz fix30 (center)
            # board 2 (3.125MHz):   6-8: Bx,By,Bz mov45, 9-11: Bx,By,Bz mov41
            # board 3 (3.125MHz):   12-14: Bx,By,Bz mov35
            # board 4 (3.125MHz):   15: LaB6 current, 16: LaB6 voltage, 17: Port 28 Diode
        441: [dir5, '02_Bmov33_41_43_Bfix30_32 2018-07-09.hdf5'],
            # board 1 (3.125MHz):   0-2: Bx,By,Bz fix32 (edge), 3-5: Bx,By,Bz fix30 (center)
            # board 2 (3.125MHz):   6-8: Bx,By,Bz mov43, 9-11: Bx,By,Bz mov41
            # board 3 (3.125MHz):   12-14: Bx,By,Bz mov33
            # board 4 (3.125MHz):   15: LaB6 current, 16: LaB6 voltage, 17: Port 28 Diode
        442: [dir5, '03_Bmov29_37_Bfix30_32 2018-07-10.hdf5'],
            # board 1 (3.125MHz):   0-2: Bx,By,Bz fix32 (edge), 3-5: Bx,By,Bz fix30 (center)
            # board 2 (3.125MHz):   6-8: Bx,By,Bz mov37
            # board 3 (3.125MHz):   9-11: Bx,By,Bz mov29
            # board 4 (3.125MHz):   12: LaB6 current, 13: LaB6 voltage, 14: Port 28 Diode
        443: [dir5, '04_Bmov29_37_39_Bfix30_32 2018-07-17.hdf5'],
            # board 1 (3.125MHz):   0-2: Bx,By,Bz fix32 (edge), 3-5: Bx,By,Bz fix30 (center)
            # board 2 (3.125MHz):   6-8: Bx,By,Bz mov39, 9-11: Bx,By,Bz mov37
            # board 3 (3.125MHz):   12-14: Bx,By,Bz mov29
            # board 4 (3.125MHz):   15: LaB6 current, 16: LaB6 voltage, 17: Port 28 Diode
        444: [dir5, '05_Bmov19_25_27_Bfix30_32 2018-07-18.hdf5'],
            # board 1 (3.125MHz):   0-2: Bx,By,Bz fix32 (edge), 3-5: Bx,By,Bz fix30 (center)
            # board 2 (3.125MHz):   6-8: Bx,By,Bz mov27, 9-11: Bx,By,Bz mov25
            # board 3 (3.125MHz):   12-14: Bx,By,Bz mov19
            # board 4 (3.125MHz):   15: LaB6 current, 16: LaB6 voltage, 17: Port 28 Diode
        #linesweeps with varying parameters
        450: [dir5, '11_EB34_Bmov27_1000G_940A.hdf5'],
            #ch11 died at pos 8, shot 12
        451: [dir5, '12_EB34_Bmov27_800G_770A.hdf5'],
        452: [dir5, '13_EB34_Bmov27_600G_470A.hdf5'],
        453: [dir5, '14_EB34_Bmov27_400G_395A.hdf5'],
        454: [dir5, '15_EB34_Bmov27_1000G_940A_plane.hdf5'],
        455: [dir5, '16_EB34_Bmov27_1000G_725A.hdf5'],
        456: [dir5, '17_EB34_Bmov27_1000G_475A.hdf5'],
            # board 1 (3.125MHz):   0-2: Bx,By,Bz fix32 (edge), 3-5: Bx,By,Bz fix30 (center)
            # board 2 (3.125MHz):   6-8: Bx,By,Bz mov27
            # board 3 (3.125MHz):   9: Port28 Diode, 10-11: Vx+/Vx- EB34
            # board 4 (3.125MHz):   12: LaB6 current, 13: LaB6 voltage
    #### DATARUN (18_Nov): 13 NOV 2018 - ?? ---------------------------------------------
    #### see excel file for details (500-550)
        501: [dir6, '01_Bmov_40_37_Vfloat_34.hdf5'],
        502: [dir6, '02_Bmov_40_37_Vfloat_34.hdf5'],
        503: [dir6, '03_Bmov_45_37_Vfloat_34_short.hdf5'],
        504: [dir6, '03b_Bmov_45_37_Vfloat_34_short.hdf5'],
        505: [dir6, '04_Bmov_45_37_Vfloat_34.hdf5'],
        506: [dir6, '05_Bmov27_EB31.hdf5'],
        507: [dir6, '06_Bmov27_EB31_short.hdf5'],

        510: [dir6, '10_line_Bmov27_EB31_1290A_2018-11-20_16.45.23.hdf5'],
        511: [dir6, '11_line_Bmov27_EB31_1440A_2018-11-20_17.23.45.hdf5'],
        512: [dir6, '12_line_Bmov27_EB31_900A_2018-11-20_18.53.36.hdf5'],
        513: [dir6, '13_line_Bmov27_EB31_540A_2018-11-20_19.40.46.hdf5'],

        521: [dir6, '21_line_Bmov27_EB31_1440A_2018-11-20_18.10.25.hdf5'],

        530: [dir6, '30_Bmov_33_27_EB31_2018-11-20_21.44.33.hdf5'],
        531: [dir6, '31_Bmov_33_27_EB31_short_2018-11-21_14.20.28.hdf5'],
        532: [dir6, '32_Bmov_33_27_EB31_1010A_2018-11-21_18.00.40.hdf5'],
        533: [dir6, '33_Bmov_33_27_EB31_610A_2018-11-22_13.07.49.hdf5'],

    #### DATARUN (19_Apr): 08 - 14 APR 2019 ---------------------------------------------
    #### see excel file for details (500-550)
        551: [dir7, '01_Bmov_43_31.hdf5'],
        552: [dir7, '02_Bmov_43_31_line.hdf5'],
        553: [dir7, '03_Bmov_43_31_line_2.hdf5'],
        554: [dir7, '04_TDSdipole38_37_XZ.hdf5'],
        555: [dir7, '05_Bmov_43(2).hdf5'],
        556: [dir7, '06_TDSdipole38_37_XYZ.hdf5'],    ## incomplete data run
        557: [dir7, '07_TDSdipole38_37_XYZ_1kG.hdf5'],
        558: [dir7, '08_Bmov_43_31.hdf5'],   ## Bdot corrupted by high freq noise 10x amplifier
        559: [dir7, '09_TDSdipole38_37_XYZ_500G.hdf5'],
        560: [dir7, '10_Bmov_43_line.hdf5'],
        561: [dir7, '11_Bmov_43_31_500G.hdf5'],
        562: [dir7, '12_TDSdipole38_37_XYZ_500G_lite.hdf5'],
        563: [dir7, '13_Bmov_43_31_500G_lite_restart.hdf5'],

    #### DATARUN (19_July): 08 - 20 JUL 2019 --------------------------------------------
    #### see excel file for details (600)
        601: [dir8, '01_Bmov33_39_Bfix32_41.hdf5'],
        602: [dir8, '02_Bmov31_37_Bfix32_42.hdf5'],
        603: [dir8, '03_Bmov29_31_Bfix32_42.hdf5'],
        '603a': [dir8, '03_amplifier pickup.hdf5'],
        604: [dir8, '04_Bmov35_37_Bfix32_42.hdf5'],
        '604a': [dir8, '04_amplifier pickup.hdf5'],
        605: [dir8, '05_Bmov41_45_Bfix32_42.hdf5'],
        606: [dir8, '06_Bmov27_43_Bfix32_42.hdf5'],
        '606a': [dir8, '06_amplifier pickup.hdf5'],
        607: [dir8, '07_Bmov25_31_Bfix32_42.hdf5'],
        621: [dir8, '21_Mach35_41_Bfix32_41.hdf5'],
        622: [dir8, '22_Mach33_39_Bfix32_42.hdf5'],
        623: [dir8, '23_Mach31_37_Bfix32_42.hdf5'],
        624: [dir8, '24_Mach29_45_Bfix32_42.hdf5'],
        625: [dir8, '25_Mach27_43_Bmov31_Bfix32_42_aborted.hdf5'],

    #### OLD DATARUN (FluxRopes_2_14) ----------------------------------------------------
    ## setup runs
        'A01': [dirA, 'Emissive_probe_setup 2014-01-30 17.33.32.hdf5'],
        'A02': [dirA, 'Emissive_probe_setup.hdf5'],
        'A03': [dirA, 'xline_HotSweeps_p19.hdf5'],
        'A04': [dirA, 'xline_ColdSweeps_p44.hdf5'],
        'A05': [dirA, 'xline_ColdSweeps_p19.hdf5'],
        'A06': [dirA, 'xline_HotSweeps_p44.hdf5'],
    ## the planes of interest with emissive probe
        'A10': [dirA, 'Plane_Bp40_Ep30_51x51.hdf5'],
        'A11': [dirA, 'Plane_Bp36_Ep31.hdf5'],
        'A12': [dirA, 'Plane_Bp34_Ep32.hdf5'],
        'A13': [dirA, 'Plane_Bp30_Ep33.hdf5'],   # good data set, not req to sub mean for B
        'A14': [dirA, 'Plane_Bp28_Ep33.hdf5'],
        'A15': [dirA, 'Plane_Bp24_Ep34.hdf5'],
        'A16': [dirA, 'Plane_Bp42_Ep27.hdf5'],
        'A17': [dirA, 'Plane_Bp22_Ep19.hdf5'],
        'A18': [dirA, 'Plane_Bp20_Ep44.hdf5'],
        'A19': [dirA, 'Plane_Ep28.hdf5'],
        'A20': [dirA, 'Plane_Bp26_Ep29.hdf5']
    }

    if filenames.get(callnumber, '') != '':
        subdir, fname = filenames.get(callnumber, '')
        if full == 0         : return fname
        elif subdir in [dirA]: return subdir + fname
        else                 : return datafolder + subdir + fname
    else:
        print('Error: File not found.', callnumber)