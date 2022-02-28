import h5py
import numpy as np
import pandas as pd
dataset = h5py.File('thor_flight107_studentgnssdeniednavfilter_2014_05_04.h5', 'r')
print(dataset.keys())
GPS_TOW = dataset['GPS_TOW']
print(GPS_TOW)
gps = list(GPS_TOW)
print(gps)
gps = np.array(GPS_TOW)
gps = gps[0]
print(gps)
