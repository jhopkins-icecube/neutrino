import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_process import *

def pre_process():
    # Loading low energy database and attempting to clean it
    path = '/home/bread/Documents/projects/neutrino/data/db/oscNext_genie_level5_v02.00_pass2.141122.000000.db'
    loading = DBLoader(path)
    pulse, truth = loading.get_db()

    # Begin processing data
    process = PulseDataProcessing((pulse,truth))

    # Zero pad events to make them the same length then sublist by
    # event number.
    process.zero_pad()
    process.sublist()

    print(type(process.pulse_shape))
    print(process.pulse_shape)

    # Finalizing pulse and truth data
    pulse = process.pulse
    truth = np.array(truth['inelasticity'])

    # Set up CNN model
    model = process.get_model()
    model.summary()

    x_train = pulse[:-250]


def sublist(data) -> None:
        """
        Creates a list of events with each event having a list of data. 
        """

        events = [
            [
                [
                    row['dom_x'],
                    row['dom_y'],
                    row['dom_z'],
                    # row['dom_time'],
                    row['charge']
                ]
                for _, row in group.iterrows()
            ]
            for _, group in data.groupby('event_no')
        ]
        
        return events


@dataclass
class DOMAttributes():
    zenith: float
    azimuth: float
    time: float
    track: float


class DOMGraph():

    def __init__(self) -> None:
        self.node = None
        self.edge = None

    def add_node(self, x, y, z):
        self.node.append((x, y, z))

    def add_edge(self, x, y, z):
        self.edge.append((x, y, z))


def k_nearest_neighbors():
    # Loading low energy database
    path = '/home/bread/Documents/projects/neutrino/data/db/oscNext_genie_level5_v02.00_pass2.141122.000000.db'
    loading = DBLoader(path)
    pulse, truth = loading.get_db()
    
    events = sublist(pulse)

    event = events[1]

    print(len(event))

    nodes = {}

    for dom in event:
        r = 0
        for position in dom:
            r += position**2

        if np.sqrt(r) < distances[-1]:
            distances[-1] = np.sqrt(r)
            distances = np.sort(distances)
            nodes.append(dom)


    print(distances)
    print(nodes)



def check_energy_distribution():
    # Reading simulation data from hdf5 file
    hdf5 = '/home/bread/Documents/projects/neutrino/data/hdf5/NuMu_genie_149999_030000_level6.zst_cleanedpulses_transformed_IC19.hdf5'
    hdf5_load = HDF5Loader(hdf5, 'output_label_names', 'labels')
    hdf5_df = hdf5_load.get_df()

    plt.title("Energy Distribution")
    plt.hist(hdf5_df['Energy'], bins=50, label='Truth', alpha=0.6, color='deepskyblue')
    plt.legend()
    plt.show()


def charge_rolling_percentage():
    # Loading low energy database 
    path = '/home/bread/Documents/projects/neutrino/data/db/oscNext_genie_level5_v02.00_pass2.141122.000000.db'
    loading = DBLoader(path)
    pulse, truth = loading.get_db()

    


if __name__ == "__main__":

    check_energy_distribution()