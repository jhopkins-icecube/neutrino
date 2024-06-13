import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_process import *

# Reading simulation data from hdf5 file
hdf5 = '/home/bread/Documents/projects/neutrino/data/hdf5/NuMu_genie_149999_030000_level6.zst_cleanedpulses_transformed_IC19.hdf5'
hdf5_df = get_hdf5(hdf5, 'output_label_names', 'labels')

# Reading original dom positions from dat file
dat = '/home/bread/Documents/projects/neutrino/data/DOM_Position_GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.dat'
df = pd.read_csv(dat, delimiter=" ", header=None, names=['String', 'Count', 'X', 'Y', 'Z'], skiprows=[0])

# Reading low energy database file
db = '/home/bread/Documents/projects/neutrino/data/db/oscNext_genie_level5_v02.00_pass2.141122.000000.db'
pulse, truth = get_db(db)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df['X'], df['Y'], df['Z'], s=1, alpha=1)
ax.scatter(pulse['dom_x'], pulse['dom_y'], pulse['dom_z'], s=1, alpha=1)
plt.savefig('/home/bread/Documents/projects/neutrino/data/fig/DomPositionAndHits.png')
plt.show()