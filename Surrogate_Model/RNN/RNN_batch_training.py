import pandas as pd
import matplotlib.pyplot as plt
import pickle
import wntr
import wntr.metrics.economic as economics
import numpy as np
import pandas as pd
import pdb
import os
import itertools

import sys
sys.path.append('../../Code/')
sys.path.append('../')
from testWN import testWN as twm

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


from RNN_tools import get_model, keras2casadi
from surrogate_model_training_data import get_data

model_path = './models/'

cluster_labels = pd.read_json('cluster_labels_dt1h.json')
pressure_factor = pd.read_json('pressure_factor_dt1h.json')
n_clusters = 30


# Get results:
data_path = '/home/ffiedler/tubCloud/Shared/WDN_SurrogateModels/_RESULTS/150sim/'
file_list = os.listdir(data_path)
file_list = [data_path+file_i for file_i in file_list if '.pkl' in file_i]
nn_input_list, nn_output_list = get_data(file_list, 0, cluster_labels, pressure_factor,
                                         narx_input=False, narx_output=False, return_lists=True)


# ## Normalize Data:
input_scaling = pd.concat(nn_input_list).abs().max()
input_scaling.loc[input_scaling.abs() < 1e-5] = 1e-5

output_scaling = pd.concat(nn_output_list).abs().max()
output_scaling.loc[output_scaling.abs() < 1e-5] = 1e-5


nn_input_list = [nn_in_i/input_scaling for nn_in_i in nn_input_list]
nn_output_list = [nn_out_i/output_scaling for nn_out_i in nn_output_list]


input_scaling = input_scaling.to_numpy().reshape(1, -1)
output_scaling = output_scaling.to_numpy().reshape(1, -1)

train_data_param = {
    'input_scaling': input_scaling,
    'output_scaling': output_scaling
}
with open(model_path+'train_data_param.pkl', 'wb') as f:
    pickle.dump(train_data_param, f)

# ## RNN I/O structure
n_datasets = len(file_list)
seq_length = 20  # length of sequence for ANN training

n_data = nn_input_list[0].shape[0]
n_seq = n_data // seq_length - 1

# create sequences from the data:
X = []
Y = []
for data_i in range(n_datasets):
    X_train = nn_input_list[data_i].to_numpy()
    Y_train = nn_output_list[data_i].to_numpy()
    for offset in range(n_seq):
        for data_sample in range(n_seq):
            start_ind = data_sample*seq_length+offset
            X.append(X_train[start_ind:start_ind + seq_length, :])
            Y.append(Y_train[start_ind:start_ind + seq_length, :])

X = np.stack(X, axis=0)  # [m, seq_length, nx]
Y = np.stack(Y, axis=0)  # [m, seq_length, ny]

batch_size = n_seq*n_datasets
nx = X.shape[2]
ny = Y.shape[2]


""" Setup batch training: """
n_units_dense_list = [20, 50, 80]
n_layer_dense_list = [2, 3, 4, 5]
n_activation_list = [20, 50, 80]

combinations = list(itertools.product(n_units_dense_list, n_layer_dense_list, n_activation_list))
comb_df = pd.DataFrame(combinations, columns=['n_units', 'n_layers', 'n_activations'])
# Add space for loss and validation loss:
comb_df = comb_df.append(pd.DataFrame(columns=['loss', 'epochs']), sort=False)

for i, case_i in comb_df.iterrows():
    print('------------------------------------')
    print(i)
    print(case_i)
    print('------------------------------------')

    stateful = True
    n_layer = int(case_i['n_layers'])  # hidden layer - output layer (linear) is automatically added
    model_param = {}
    model_param['RNN_param'] = {}
    model_param['n_units'] = (n_layer) * [int(case_i['n_units'])]
    model_param['p_dropout'] = None  # Placeholder. Not yet used.
    model_param['activation'] = (n_layer) * ['tanh']

    model_param['RNN_param']['RNN_type'] = 'LSTM'
    model_param['RNN_param']['n_activation'] = int(case_i['n_activations'])
    model_param['RNN_param']['recurrent_dropout'] = 0
    model_param['RNN_param']['dropout'] = 0.
    model_param['RNN_param']['implementation'] = 2

    model = get_model(model_param, nx, ny, batch_size=batch_size,
                      seq_length=seq_length, stateful=stateful)

    model.compile(loss='mse', optimizer='Adam')

    cb_reset = keras.callbacks.LambdaCallback(on_epoch_end=model.reset_states())
    cb_stopping = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-7, patience=10, mode='min')
    history = model.fit(X, Y, callbacks=[cb_reset], batch_size=batch_size,
                        epochs=5000, verbose=1, shuffle=False)

    loss = history.history['loss'][-1]
    comb_df.loc[i, 'loss'] = loss
    comb_df.loc[i, 'epochs'] = len(history.history['loss'])
    model_name = '{:03d}_model_LSTM'.format(i)
    model.save(model_path+model_name+'.h5')

    try:
        comb_df.to_pickle("./models/model_overview.pkl")
        comb_df.to_excel("./models/model_overview.xlsx")
    except:
        print('couldnt save overview')
