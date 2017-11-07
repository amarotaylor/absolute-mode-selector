import numpy as np
import sys
from scipy.stats import beta
from scipy.stats import fisher_exact
from itertools import compress
import gzip
import random
import pandas as pd
import matplotlib
import pickle


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_member(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, np.nan) for itm in a]

def remove_small_centromere_segments(het_table):
    centromere_positions = [125000001, 1718373143, 1720573143, 1867507890,
                            1869607890, 1984214406, 1986714406, 2101066301,
                            2102666301, 2216036179, 2217536179, 2323085719, 2326285719, 2444417111,
                            2446417111, 2522371864, 2524171864, 2596767074, 2598567074, 2683844322,
                            2685944322, 339750621, 342550621, 2744173305, 2746073305, 2792498825, 2794798825,
                            2841928720,
                            2844428720, 580349994, 583449994, 738672424, 740872424, 927726700, 930026700, 1121241960,
                            1123541960, 1291657027, 1293557027, 1435895690, 1438395690, 1586459712, 1588159712]