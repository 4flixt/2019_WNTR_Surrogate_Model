import pandas as pd
import matplotlib.pyplot as plt
import pickle
import wntr
import numpy as np
import pandas as pd
import pdb
import os

import sys
sys.path.append('../')
from testWN import testWN as twm
from surrogate_model_training_data import get_data

from tensorflow import keras
import itertools


def get_model(n_in, n_out, n_layer, n_units, l1_regularizer=0, **kwargs):

    model_param = {}
    model_param['n_in'] = n_in
    model_param['n_out'] = n_out
    model_param['n_units'] = (n_layer)*[n_units]
    model_param['activation'] = (n_layer) * ['tanh']

    inputs = keras.Input(shape=(model_param['n_in'],))

    layer_list = [inputs]


    for i in range(len(model_param['n_units'])-1):
        layer_list.append(
            keras.layers.Dense(model_param['n_units'][i],
                               activation=model_param['activation'][i],
                               kernel_regularizer=keras.regularizers.l1(l=l1_regularizer)
                               )(layer_list[i])
        )

    outputs = keras.layers.Dense(model_param['n_out'],
                                 activation='linear')(layer_list[-1])

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model



n_units_list = [40, 70, 100]
n_layer_list = [1, 2, 3, 4]
l_1_list = [0]
training_data_list = [
    'training_setup_narx4.pkl',
    'training_setup_narx5.pkl',
    'training_setup_narx6.pkl'
]

combinations = list(itertools.product(training_data_list, n_units_list, n_layer_list, l_1_list))



comb_df = pd.DataFrame(combinations, columns=['training_data', 'n_units', 'n_layers', 'l1_regularizer'])
# Add space for loss and validation loss:
comb_df = comb_df.append(pd.DataFrame(columns=['loss', 'val_loss']),sort=False)

data_path = './training_data/'
model_path = './models/'


for i,case_i in comb_df.iterrows():
    print(case_i['n_layers'])
    file_i = data_path + case_i['training_data']
    with open(file_i, 'rb') as f:
        data = pickle.load(f)

    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']

    n_in = X_train.shape[1]
    n_out = Y_train.shape[1]
    pdb.set_trace()
    model = get_model(n_in, n_out, int(case_i['n_layers']), int(case_i['n_units']), case_i['l1_regularizer'])


    optim = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-8, patience=50, mode='min')
    model.compile(optimizer=optim,
                loss='mse')

    history = model.fit(X_train.to_numpy(),
                    Y_train.to_numpy(),
                    batch_size=2000,
                    epochs=3,
                    validation_data=(X_test.to_numpy(),Y_test.to_numpy()),
                    callbacks=[callback])

    loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    pdb.set_trace()
    comb_df.loc[i,'loss'] = loss
    comb_df.loc[i,'val_loss'] = val_loss

    model_name  = '{:03d}_model'.format(i)
    keras.models.save_model(
        model,
        model_path+model_name,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
