import keras
import numpy as np
import pandas as pd
import constants as c
import tensorflow as tf
import matplotlib.pyplot as plt

from joblib import Memory
from data_process import *
from matplotlib.ticker import AutoMinorLocator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# Caching
cachedir = './cache'
memory = Memory(cachedir, verbose=0)


def zero_pad_util(group, max_len):
  event_length = len(group)
  padding = max_len - event_length
  if padding > 0:
      padded = group.reindex(group.index.tolist() + list(range(group.index[-1] + 1, group.index[-1] + 1 + padding)))
  else:
      padded = group
  return padded.fillna(0)


def zero_pad(pulse):
    """
    Zero pads events to make each event the same length. 
    """

    max_len = pulse.groupby('event_no').size().max()
    padded = pd.concat([zero_pad_util(group, max_len) for _, group in pulse.groupby('event_no')])

    padded = padded.reset_index(drop=True)

    mask = padded['event_no'] != 0
    padded['event_no'] = padded['event_no'].mask(~mask).ffill().astype(int)

    return padded


def sublist(pulse):
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
        for _, group in pulse.groupby('event_no')
    ]
    
    return events


@memory.cache
def get_data():

    X, y = feature_adding()
    
    X = zero_pad(X)

    X = sublist(X)

    return np.asarray(X), y


def pre_process():
    # Loading low energy database and attempting to clean it
    path = '/home/bread/Documents/projects/neutrino/data/db/oscNext_genie_level5_v02.00_pass2.141122.000000.db'
    pulse, truth = get_db(path)

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
    pulse, truth = get_db(path)
    
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
    hdf5_df = get_hdf5(hdf5, 'output_label_names', 'labels')

    plt.title("Energy Distribution")
    plt.hist(hdf5_df['Energy'], bins=50, label='Truth', alpha=0.6, color='deepskyblue')
    plt.legend()
    plt.show()


def feature_adding():
    # Loading low energy database 
    path = '/home/bread/Documents/projects/neutrino/data/db/oscNext_genie_level5_v02.00_pass2.141122.000000.db'
    pulse,truth = get_db(path)

    # print(pulse)

    # print(pulse['charge'])

    change_array = pulse['charge'].pct_change(periods=1).dropna()

    pulse = pulse[1:]
    pulse['charge change'] = change_array

    return pulse, truth


def neural_network(X, y, X_scaler=RobustScaler(), y_scaler=StandardScaler(), scale_x=True, scale_y=False, early=False, save=True, log=False):

    if scale_x:
        X = X_scaler.fit_transform(X)
    if scale_y:
        y = y_scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = keras.Sequential()

    callbacks = []

    if early:
         
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            min_delta=1e-20, 
            patience=10, 
            verbose=1, 
            mode='min', 
            restore_best_weights=True
        ) 

        callbacks.append(early_stop)

    # dim_val = len(X[0])
    # model.add(layers.Dense(dim_val, shape=np.shape(X[0]),activation='sigmoid'))

    # relu, exponential, sigmoid -- options
    model.add(keras.Input(shape=np.shape(X[0])))
    model.add(layers.Dense(128, activation='sigmoid'))
    model.add(layers.Dense(128, activation='sigmoid'))
    model.add(layers.Dense(64, activation='sigmoid'))
    model.add(layers.Dense(32, activation='sigmoid'))
    model.add(layers.Dense(16, activation='sigmoid'))
    model.add(layers.Dense(8, activation='sigmoid'))
    model.add(layers.Dense(4, activation='sigmoid'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])

    history = model.fit(
        X_train, 
        y_train, 
        validation_split=0.30, 
        epochs=200, 
        batch_size=256, 
        verbose=2,
        callbacks=callbacks
    )

    print(f"Model Eval: {model.evaluate(X_test, y_test)}")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    if log:
        ax.set_yscale('log')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.title('Model Mean Absolute Error')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if save:
        plt.savefig('/home/bread/Documents/projects/neutrino/data/fig/nn_mae_history.png')
    plt.show()

    prediction = model.predict(X_test)
    prediction = [i[0] for i in prediction]

    plt.hist2d(y_test, prediction, bins=30)
    plt.xlabel('True Inelasticity')
    plt.ylabel('Predicted Inelasticity')
    plt.title('Using Neural Network to Predict Neutrino Ineslasticity - 2d Histogram')
    plt.xlim([0,1])
    plt.ylim([0,1])  
    if save:
        plt.savefig('/home/bread/Documents/projects/neutrino/data/fig/nn_inelasticity_2d.png')
    plt.show()


if __name__ == "__main__":

    hdf5_df = cat_hdf5(c.path_array, 'output_label_names', 'labels')

    y = np.array(hdf5_df['Cascade']/hdf5_df['Energy'])

    hdf5_df = hdf5_df.drop(['Cascade', 'Energy', 'Flavor'], axis=1)

    X = np.array(hdf5_df)

    neural_network(X, y)
