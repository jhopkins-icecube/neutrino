import os
import time
import keras
import hashlib
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from joblib import Memory
from data_process import *
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler


# Timing
start_time = time.time()

# Cachine
cachedir = './cache'
memory = Memory(cachedir, verbose=0)

# Scaler
scaler = RobustScaler()

# Annoying
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def get_file_path(index_value):
  if index_value > 9:
    return f'~/Documents/projects/neutrino/data/db/oscNext_genie_level5_v02.00_pass2.141122.0000{index_value}.db'
  else:
    return f'~/Documents/projects/neutrino/data/db/oscNext_genie_level5_v02.00_pass2.141122.00000{index_value}.db'


def get_max_charge(group):
  """
  Get max charge for event
  """
  max_charge = group['charge'].idxmax()
  
  return group.loc[max_charge, ["dom_x", "dom_y", "dom_z"]]


def get_db(index_value=None, file_path='data/db/oscNext_genie_level5_v02.00_pass2.141122.000001.db') -> tuple[pd.DataFrame, pd.DataFrame]:
  """
  Read db file into pandas df
  """
  if index_value is not None:
      if index_value > 9:
          file_path = f'~/Documents/projects/neutrino/data/db/oscNext_genie_level5_v02.00_pass2.141122.0000{index_value}.db'
      else:
          file_path = f'~/Documents/projects/neutrino/data/db/oscNext_genie_level5_v02.00_pass2.141122.00000{index_value}.db'
  
  file_path = '/home/bread/Documents/projects/neutrino/data/db/oscNext_genie_level5_v02.00_pass2.141122.000000.db'

  connect = sqlite3.connect(file_path)
  pulse = pd.read_sql('SELECT * FROM SRTTWOfflinePulsesDC', connect)
  truth = pd.read_sql('SELECT * FROM truth', connect)

  return truth, pulse


def zero_pad_event(event_group, max_event_length):
  """
  Pads based on the event with the longest length
  """
  event_length = len(event_group)
  padding = max_event_length - event_length
  if padding > 0:
      padded_event = event_group.reindex(event_group.index.tolist() + list(range(event_group.index[-1] + 1, event_group.index[-1] + 1 + padding)))
  else:
      padded_event = event_group
  
  return padded_event.fillna(0)


def faster_grouping(df):
  grouped = df.groupby('event_no')[['dom_x', 'dom_y', 'dom_z', 'charge']].apply(list).tolist()
  
  return [[item for item in sublist if item not in df.columns] for sublist in grouped]

# Function to get db, pad, and scalar transform
@memory.cache
def zero_pad(self) -> None:
        """
        Zero pads events to make each event the same length. 
        """

        max_len = self.pulse.groupby('event_no').size().max()
        padded = pd.concat([zero_pad(group, max_len) for _, group in self.pulse.groupby('event_no')])

        padded = padded.reset_index(drop=True)

        mask = padded['event_no'] != 0
        padded['event_no'] = padded['event_no'].mask(~mask).ffill().astype(int)



hdf5 = '/home/bread/Documents/projects/neutrino/data/hdf5/NuMu.zst_cleaned_transformed_cascade.hdf5'
hdf5_load = HDF5Loader(hdf5, 'output_label_names', 'labels')
hdf5_df = hdf5_load.get_df()


hdf5_df = hdf5_df.drop(['Cascade'], axis=1)
X = np.array(hdf5_df.drop('Energy', axis=1))

y = hdf5_df['Energy']

X = scaler.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = keras.Sequential()

# #(~345, 4) size data
# # model.add(Dense(512, input_dim=4, activation='sigmoid'))

# def run_network(X, y, early, layers):


#   early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', # Monitor the validation loss
#                               min_delta=1e-20, # Minimum change to qualify as an improvement
#                               patience=10, # Number of epochs with no improvement after which training will be stopped
#                               verbose=1, # Print messages when training is stopped
#                               mode='min', # Mode for monitoring the validation loss
#                               restore_best_weights=True) # Restore the best model weights after training


# dim_val = len(X[0])
# model.add(layers.Dense(dim_val, input_dim=len(X[0]), activation='sigmoid'))

# # model.add(layers.Dense(300, activation='relu'))
# model.add(layers.Dense(4, activation='relu'))
# model.add(layers.Dense(1, activation='exponential'))

# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# history = model.fit(X_train, y_train, 
#                       validation_split=0.20, 
#                       epochs=200, 
#                       batch_size=8, 
#                       verbose=0) 
#                       #callbacks=[early_stop])

# print(model.evaluate(X_test, y_test))

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# plt.plot(history.history['mae'])
# plt.plot(history.history['val_mae'])
# # ax.set_yscale('log')
# ax.xaxis.set_minor_locator(AutoMinorLocator())
# plt.title('Model Mean Absolute Error')
# plt.ylabel('Mean Absolute Error')
# plt.xlabel('Epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('/home/bread/Documents/projects/neutrino/data/fig/nn_mae_history.png')
# plt.show()

# prediction = model.predict(X_test)
# prediction = [i[0] for i in prediction]


# plt.hist2d(y_test, prediction, bins=100)
# # plt.xlim([0,2])
# # plt.ylim([0,2])
# plt.xlabel('Truth')
# plt.ylabel('Predicted')
# plt.title('Predicting Energy')
# # plt.savefig('/home/bread/Documents/projects/neutrino/data/fig/nn_energy_2d.png')
# plt.show()

# plt.title("Truth vs Predicted Inelasticity Using NN")
# plt.hist(y_test, bins=50, label='Truth', alpha=0.6, color='olive')
# plt.hist(prediction, bins=50, label='Prediction', alpha=0.6, color='deepskyblue')
# plt.legend()
# # plt.savefig('/home/jeremy/Documents/Neutrino/data/fig/nn_energy_1d.png')
# plt.show()




# TODO Fix compress info
def get_db_compress(index_value=None, file_path='./data/db/oscNext_genie_level5_v02.00_pass2.141122.000001.db'):
  """
  Compress event from 2d data to 1d data with inelasticity as output
  """
  
  if index_value is not None:
    truth, pulse = get_db(index_value)
  else:
     truth, pulse = get_db(file_path)

  X = pulse[['charge', 'dom_x', 'dom_y', 'dom_z', 'dom_number', 'event_no']]
  
  charge = pulse[["charge", "event_no"]]
  charge_sum = charge.groupby('event_no')[['charge']].sum().reset_index()

  X = X.groupby('event_no')[['dom_x', 'dom_y', 'dom_z', 'dom_number', 'charge']].apply(get_max_charge).reset_index()
  
  X['charge'] = charge_sum["charge"]

  X['energy'] = truth['energy']
  X = X[X.energy < 15]

  y = X['energy']

  X = X.drop(['event_no', 'energy'], axis=1)

  X = np.array(X)
  y = np.array(y)


  return X, y


# TODO Fix Decision Tree Regression
# ===================================================================================

# X, y = pad_data(1)

# print(len(X))
# print(len(y))

X1 = X[-50:]
X = X[:-50]
y1 = y[-50:]
y = y[:-50]

# print(X1)
# print(y1)
# print(X)
# print(y)

# model = DecisionTreeRegressor()
# model.fit(X, y)

# prediction = model.predict(X1)
# # prediction = [i[0] for i in prediction]

# # y1 = [i[0] for i in y1]

# plt.title("Truth vs Predicted Inelasticity Using Decision Tree Regression")
# plt.hist2d(y1, prediction, bins=25)
# plt.xlabel('Truth')
# plt.ylabel('Prediction')
# # plt.savefig('./fig/DT_2d.png')
# plt.show()

# plt.title("Truth vs Predicted Inelasticity Using Decision Tree Regression")
# plt.hist(y1, bins=40, label='Truth', alpha=0.6, color='olive')
# plt.hist(prediction, bins=40, label='Prediction', alpha=0.6, color='deepskyblue')
# plt.legend()
# plt.savefig('./fig/DT_1d.png')
# plt.show()

# TODO fix random forrest - less accurate than decision tree and nn
#=============================================================

# X, y = pad_data(1)

# print(len(X))
# print(len(y))

# X1 = X[-1500:]
# X = X[:-1501]
# y1 = y[-1500:]
# y = y[:-1501]

model = RandomForestRegressor(n_estimators=40, random_state=30)
model.fit(X, y)

prediction = model.predict(X1)

plt.hist2d(y1, prediction, bins=25)
plt.show()

plt.title("Truth vs Predicted Inelasticity Using Random Forest")
plt.hist(y, bins=1, label='Truth', alpha=0.6, color='olive')
plt.hist(prediction, bins=1, label='Prediction', alpha=0.6, color='deepskyblue')
plt.legend()
plt.show()



# Making large cv and plotting
#===================================================================================
# fig = plt.figure(figsize=[7,4])
# ax = fig.add_subplot(111, projection='3d')

# X1, y1 = get_db(0)

# for i in range(1, 10):
#   xi, yi = get_db(i)
#   X1 = pd.concat([X1, xi], axis=0, ignore_index=True)
#   y1 = pd.concat([y1, yi], axis=0, ignore_index=True)

# X1.to_csv('X.csv', index=False)
# y1.to_csv('y.csv', index=False)

# ax.scatter(X1['dom_z'], X1['charge'], y1)
# ax.set_xlabel('dom_z')
# ax.set_ylabel('charge')
# ax.set_zlabel('inelasticity')
# plt.show()

# ax.cla()

# ax.scatter(X['dom_x'], X['dom_y'], X['dom_z'])
# plt.show()

# TODO pad data
#============================================================
# db = "./DB/oscNext_genie_level5_v02.00_pass2.141122.000002.db"

# connect = sqlite3.connect(db)
# truth = pd.read_sql('SELECT * FROM truth', connect)
# pulse = pd.read_sql('SELECT * FROM SRTTWOfflinePulsesDC', connect)

# # truth = truth[truth.inelasticity != 1.0]
# truth = truth[truth.inelasticity < 1.0]

# inelas = truth['inelasticity']
# energy = truth['energy']
# energy_cascade = truth['energy_cascade']

# # X = pulse[["charge", "dom_x", "dom_y", "dom_z", "event_no"]]
# # y = truth[['inelasticity', 'event_no']]
# # X = pd.merge(X, y, on='event_no', how='left')

# # X = X[X.inelasticity < 1.0]
# # y = X["inelasticity"]
# # X = X.drop("inelasticity", axis=1)
# # X = X.drop("event_no", axis=1)
# # X = X.drop('charge', axis=1)

# # X = X.round()

# fig = plt.figure(figsize=[7,4])
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(energy, energy_cascade, inelas)
# ax.set_xlabel('energy')
# ax.set_ylabel('energy_cascade')
# ax.set_zlabel('inelas')
# ax.set_xlim(-200, 700)
# ax.set_ylim(-200, 700)
# ax.set_zlim(-2, 2)
# plt.show()

# Timing
#===============================
end_time = time.time()
execution_time = end_time - start_time
print(f"Time: {execution_time:.2f}s")