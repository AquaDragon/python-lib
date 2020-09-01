'''
NAME:           errorcheck.py
AUTHOR:         swjtang  
DATE:           31 Aug 2020
DESCRIPTION:    A list of error checking functions
'''
import numpy as np
import os
from matplotlib import pyplot as plt
''' ---------------------------------------------------------------------------
DESCRIPTION:    Checks if the input values t1 and t2 fall within range of the
                dataset length. Otherwise, set the bounds as t1 or t2.
INPUTS:         nt = (int) length of the dataset
                t1 = (int) left bound value to check
                t2 = (int) right bound value to check
'''
def check_trange(nt,t1,t2):
    if (t1 < 0) or (t1 > nt): t1 = 0
    if (t2 < 0) or (t2 > nt): t2 = nt
    return t1,t2