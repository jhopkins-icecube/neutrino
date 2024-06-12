import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_process import *

# Reading simulation data from hdf5 file
hdf5 = '/home/bread/Documents/projects/neutrino/data/hdf5/NuMu_genie_149999_030000_level6.zst_cleanedpulses_transformed_IC19.hdf5'
hdf5_load = HDF5Loader(hdf5, 'output_label_names', 'labels')
hdf5_df = hdf5_load.get_df()

# Reading original dom positions from dat file
dat = '/home/bread/Documents/projects/neutrino/data/DOM_Position_GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.dat'
df = pd.read_csv(dat, delimiter=" ", header=None, names=['String', 'Count', 'X', 'Y', 'Z'], skiprows=[0])


# Plotting
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df['X'], df['Y'], df['Z'], s=1, alpha=1)
ax.scatter(hdf5_df['X'], hdf5_df['Y'], hdf5_df['Z'], s=1, alpha=1)
plt.savefig('/home/bread/Documents/projects/neutrino/data/fig/DomPositionAndHits.png')
plt.show()