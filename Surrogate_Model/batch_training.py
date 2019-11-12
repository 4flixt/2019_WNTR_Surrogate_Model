import pandas as pd
import pickle
import numpy as np
import pandas as pd
import pdb
import os
import sys

import tensorflow as tf
from tensorflow import keras
import itertools
from sklearn.model_selection import train_test_split

sys.path.append('../Code/')
from testWN import testWN as twm
from surrogate_model_training_data import get_data


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


n_units_list = [30, 50, 70]
n_layer_list = [1, 2, 3, 4]
l_1_list = [0]
filtering_list = [
    'yes',
    'no'
]

combinations = list(itertools.product(filtering_list, n_units_list, n_layer_list, l_1_list))


comb_df = pd.DataFrame(combinations, columns=['filtering', 'n_units', 'n_layers', 'l1_regularizer'])
# Add space for loss and validation loss:
comb_df = comb_df.append(pd.DataFrame(columns=['loss', 'val_loss']), sort=False)


model_path = './models/'

cluster_labels = pd.read_json('cluster_labels_dt1h.json')
pressure_factor = pd.read_json('pressure_factor_dt1h.json')
n_clusters = 30


data_path = '/home/ffiedler/tubCloud/Shared/WDN_SurrogateModels/_RESULTS/150sim_1hourSampling/'
file_list = os.listdir(data_path)
file_list = [data_path+file_i for file_i in file_list if '.pkl' in file_i]


nn_input, nn_output = get_data(file_list, 0, cluster_labels, pressure_factor, narx_input=False)


for i, case_i in comb_df.iterrows():
    try:
        print('------------------------------------')
        print(i)
        print(case_i)
        print('------------------------------------')

        if 'yes' in case_i['filtering']:
            output_neg_value_filter = (nn_output < 0).any(axis=1)
            nn_output_i = nn_output.loc[~output_neg_value_filter]
            nn_input_i = nn_input.loc[~output_neg_value_filter]

            input_neg_value_filter = (nn_input_i < 0).any(axis=1)
            nn_output_i = nn_output_i.loc[~input_neg_value_filter]
            nn_input_i = nn_input_i.loc[~input_neg_value_filter]

            batch_size = 500
        else:
            batch_size = 5000

        # ## Normalize Data:
        input_scaling = nn_input_i.abs().max()
        input_scaling.loc[input_scaling.abs() < 1e-5] = 1e-5

        output_scaling = nn_output_i.abs().max()
        output_scaling.loc[output_scaling.abs() < 1e-5] = 1e-5

        nn_input_scaled = nn_input_i/input_scaling
        nn_output_scaled = nn_output_i/output_scaling

        X_train, X_test, Y_train, Y_test = train_test_split(nn_input_scaled, nn_output_scaled, test_size=0.2, shuffle=False)

        train_data_param = {
            'input_scaling': input_scaling,
            'output_scaling': output_scaling
        }

        if 'yes' in case_i['filtering']:
            with open(model_path+'train_data_param_filtered.pkl', 'wb') as f:
                pickle.dump(train_data_param, f)
        else:
            with open(model_path+'train_data_param_unfiltered.pkl', 'wb') as f:
                pickle.dump(train_data_param, f)

        n_in = 46
        n_out = 42

        model = get_model(n_in, n_out, int(case_i['n_layers']), int(case_i['n_units']), case_i['l1_regularizer'])

        optim = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-8, patience=50, mode='min')
        model.compile(optimizer=optim,
                      loss='mse')

        history = model.fit(X_train.to_numpy(),
                            Y_train.to_numpy(),
                            batch_size=batch_size,
                            epochs=5000,
                            validation_data=(X_test.to_numpy(), Y_test.to_numpy()),
                            callbacks=[callback])

        loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        comb_df.loc[i, 'loss'] = loss
        comb_df.loc[i, 'val_loss'] = val_loss

        model_name = '{:03d}_model_newInput'.format(i)
        model.save(model_path+model_name+'.h5')
    except:
        print('couldnt train for case {}'.format(i))

    try:
        comb_df.to_pickle("./models/model_overview.pkl")
        comb_df.to_excel("./models/model_overview.xlsx")
    except:
        print('couldnt save overview')

comb_df.to_pickle("./models/model_overview.pkl")
comb_df.to_excel("./models/model_overview.xlsx")
