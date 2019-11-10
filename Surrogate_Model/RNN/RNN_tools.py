import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pickle
import pdb
import os
from tensorflow import keras
try:
    from casadi import *
except:
    print('casadi could not be imported')


# TODO : Bias propagation is not properly implemented for all cases
# TODO : GRU is not checked


class backend:
    def __init__(self, backend_name='numpy'):
        if 'numpy' in backend_name:
            self.matmul = np.matmul
            self.relu = lambda x: np.maximum(x, 0)

        if 'casadi' in backend_name:
            self.matmul = mtimes
            self.relu = lambda x: fmax(x, 0)

        self.tanh = np.tanh
        self.sigmoid = lambda x: np.exp(x)/(np.exp(x)+1)
        self.linear = lambda x: x


def generate_dropout_mask(ones, rate, count=1):
    """Sets entries to zero at random, while scaling the entire tensor."""
    return count*[ones*(np.random.rand(*ones.shape) > rate)*1/(1-rate)]


class GRU:
    # weights = model_cell.get_weights()
    # GRU_class = GRU(units=3, kernel=weights[0],
    #                 recurrent_kernel=weights[1],
    #                 bias=weights[2])
    #
    # x0 = x_train[seq_length]
    # c = get_hidden_states(model_cell, c0, x_train[:seq_length])
    #
    # GRU_class.predict(x0,c)

    def __init__(self, units, kernel,
                 recurrent_kernel,
                 bias,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 implementation=1,
                 reset_after=False,
                 backend_name='numpy'):

        # Numpy or Casadi syntax for functions such as tanh, sigmoid and matrix multiplication.
        self.K = backend(backend_name)

        self.units = units

        self.kernel = kernel
        self.recurrent_kernel = recurrent_kernel
        self.bias = bias.reshape(1, -1)  # row vector is required for casadi. numpy doesnt care.

        self.activation = getattr(self.K, activation)
        self.recurrent_activation = getattr(self.K, recurrent_activation)
        self.use_bias = use_bias
        self.implementation = implementation
        self.reset_after = reset_after
        self.state_size = self.units
        self.output_size = self.units

        # update gate
        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        # reset gate
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:, self.units:self.units * 2]
        # new gate
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

        if not self.reset_after:
            self.input_bias, self.recurrent_bias = self.bias, None

        if self.use_bias:
            # bias for inputs (slicing to maintain dimension as 1xm)
            self.input_bias_z = self.input_bias[:, :self.units]
            self.input_bias_r = self.input_bias[:, self.units: self.units * 2]
            self.input_bias_h = self.input_bias[:, self.units * 2:]
            # bias for hidden state - just for compatibility with CuDNN
            if self.reset_after:
                self.recurrent_bias_z = self.recurrent_bias[:, :self.units]
                self.recurrent_bias_r = self.recurrent_bias[:, self.units: self.units * 2]
                self.recurrent_bias_h = self.recurrent_bias[:, self.units * 2:]

    def predict(self, x, c):
        h_tm1 = c
        inputs = x
        if self.implementation == 1:
            inputs_z = inputs_r = inputs_h = inputs

            x_z = self.K.matmul(inputs_z, self.kernel_z)
            x_r = self.K.matmul(inputs_r, self.kernel_r)
            x_h = self.K.matmul(inputs_h, self.kernel_h)
            if self.use_bias:
                x_z = x_z + self.input_bias_z
                x_r = x_r + self.input_bias_r
                x_h = x_h + self.input_bias_h

            h_tm1_z = h_tm1_r = h_tm1_h = h_tm1

            recurrent_z = self.K.matmul(h_tm1_z, self.recurrent_kernel_z)
            recurrent_r = self.K.matmul(h_tm1_r, self.recurrent_kernel_r)
            if self.reset_after and self.use_bias:
                recurrent_z = recurrent_z + self.recurrent_bias_z
                recurrent_r = recurrent_r + self.recurrent_bias_r

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = self.K.matmul(h_tm1_h, self.recurrent_kernel_h)
                if self.use_bias:
                    recurrent_h = recurrent_h + self.recurrent_bias_h
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = self.K.matmul(r * h_tm1_h, self.recurrent_kernel_h)

            hh = self.activation(x_h + recurrent_h)

        elif self.implementation == 2:

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                matrix_inner = self.K.matmul(h_tm1, self.recurrent_kernel)
                if self.use_bias:
                    matrix_inner = K.bias_add(matrix_inner, self.recurrent_bias)
            else:
                # hidden state projected separately for update/reset and new
                matrix_inner = self.K.matmul(h_tm1,
                                             self.recurrent_kernel[:, :2 * self.units])

            recurrent_z = matrix_inner[:, :self.units]
            recurrent_r = matrix_inner[:, self.units: 2 * self.units]

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            if self.reset_after:
                recurrent_h = r * matrix_inner[:, 2 * self.units:]
            else:
                recurrent_h = self.K.matmul(r * h_tm1,
                                            self.recurrent_kernel[:, 2 * self.units:])

            hh = self.activation(x_h + recurrent_h)

        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh

        return h


class LSTM:

    def __init__(self, units, kernel, recurrent_kernel, bias,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 recurrent_dropout=0.,
                 use_bias=True,
                 implementation=2,
                 backend_name='numpy'):

        self.K = backend(backend_name)
        self.units = units

        self.kernel = kernel
        self.recurrent_kernel = recurrent_kernel
        self.bias = bias.reshape(1, -1)  # row vector is required for casadi. numpy doesnt care
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))

        self.activation = getattr(self.K, activation)
        self.recurrent_activation = getattr(self.K, recurrent_activation)
        self.use_bias = use_bias
        self.implementation = implementation
        self.state_size = (self.units, self.units)
        self.output_size = self.units

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = (
            self.recurrent_kernel[:, self.units: self.units * 2])
        self.recurrent_kernel_c = (
            self.recurrent_kernel[:, self.units * 2: self.units * 3])
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:, :self.units]
            self.bias_f = self.bias[:, self.units: self.units * 2]
            self.bias_c = self.bias[:, self.units * 2: self.units * 3]
            self.bias_o = self.bias[:, self.units * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None

    def predict(self, x, h, c):
        inputs = x
        h_tm1 = h
        c_tm1 = c

        rec_dp_mask = self._recurrent_dropout_mask = generate_dropout_mask(
            np.ones(h_tm1.shape),
            self.recurrent_dropout,
            count=4)

        if self.implementation == 1:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs
            x_i = self.K.matmul(inputs_i, self.kernel_i)
            x_f = self.K.matmul(inputs_f, self.kernel_f)
            x_c = self.K.matmul(inputs_c, self.kernel_c)
            x_o = self.K.matmul(inputs_o, self.kernel_o)
            if self.use_bias:
                # Manually propagating bias for casadi if evaluated with multiple inputs x.
                x_i = x_i + np.repeat(self.bias_i, x_i.shape[0], axis=0)
                x_f = x_f + np.repeat(self.bias_f, x_f.shape[0], axis=0)
                x_c = x_c + np.repeat(self.bias_c, x_c.shape[0], axis=0)
                x_o = x_o + np.repeat(self.bias_o, x_o.shape[0], axis=0)

            if 0 < self.recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
            i = self.recurrent_activation(x_i + self.K.matmul(h_tm1_i,
                                                              self.recurrent_kernel_i))
            f = self.recurrent_activation(x_f + self.K.matmul(h_tm1_f,
                                                              self.recurrent_kernel_f))
            c = f * c_tm1 + i * self.activation(x_c + self.K.matmul(h_tm1_c,
                                                                    self.recurrent_kernel_c))
            o = self.recurrent_activation(x_o + self.K.matmul(h_tm1_o,
                                                              self.recurrent_kernel_o))
        else:

            z = self.K.matmul(inputs, self.kernel)

            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]

            z += self.K.matmul(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z = z.T + self.bias.T
                z = z.T

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]
            z3 = z[:, 3 * self.units:]

            i = self.recurrent_activation(z0)
            f = self.recurrent_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.recurrent_activation(z3)

        h = o * self.activation(c)

        return h, h, c


class Dense:

    def __init__(self, units, kernel, bias, activation=None, use_bias=True, backend_name='numpy'):
        self.K = backend(backend_name)

        self.units = units
        self.kernel = kernel
        self.bias = bias.reshape(1, -1)  # row vector is required for casadi. numpy doesnt care
        self.activation = getattr(self.K, activation)
        self.use_bias = use_bias

    def predict(self, inputs):
        output = self.K.matmul(inputs, self.kernel)
        if self.use_bias:
            output = output.T + self.bias.T
            output = output.T
        if self.activation is not None:
            output = self.activation(output)
        return output


def keras2casadi(weights, config, inputs, backend_name='numpy', dropout=False, weights_ind=0):
    # inputs[0] -> x
    # inputs[1] -> h
    # inputs[2] -> c
    for layer in config:
        if 'LSTM' in layer['class_name']:
            units = layer['config']['units']
            kernel = weights[weights_ind]
            weights_ind += 1
            recurrent_kernel = weights[weights_ind]
            weights_ind += 1
            bias = weights[weights_ind]  # TODO: What if no_bias option?
            weights_ind += 1
            activation = layer['config']['activation']
            recurrent_activation = layer['config']['recurrent_activation']
            if dropout:
                recurrent_dropout = layer['config']['recurrent_dropout']
            else:
                recurrent_dropout = 0
            use_bias = True
            implementation = layer['config']['implementation']

            casadi_layer = LSTM(units, kernel, recurrent_kernel, bias,
                                activation=activation,
                                recurrent_activation=recurrent_activation,
                                recurrent_dropout=recurrent_dropout,
                                use_bias=use_bias,
                                implementation=implementation,
                                backend_name=backend_name)
            inputs[0], inputs[1], inputs[2] = casadi_layer.predict(inputs[0], inputs[1], inputs[2])

        if 'Dense' in layer['class_name']:
            units = layer['config']['units']
            kernel = weights[weights_ind]
            weights_ind += 1
            bias = weights[weights_ind]
            weights_ind += 1
            activation = layer['config']['activation']

            casadi_layer = Dense(units, kernel, bias, activation, backend_name=backend_name)

            inputs[0] = casadi_layer.predict(inputs[0])

        if 'Dropout' in layer['class_name']:
            dropout_rate = layer['config']['rate']
            dropout_mask = generate_dropout_mask(np.ones(inputs[0].shape), dropout_rate)
            inputs[0] = dropout_mask[0]*inputs[0]

        if 'TimeDistributed' in layer['class_name']:
            inputs, weights_ind = keras2casadi(
                weights, [layer['config']['layer']], inputs, backend_name=backend_name, weights_ind=weights_ind)

    return inputs, weights_ind


def get_model(model_param, nx, ny, batch_size, seq_length, stateful=True, return_sequences=True):
    n_a = model_param['RNN_param']['n_activation']
    if 'recurrent_dropout' in model_param['RNN_param'].keys():
        rc_do = model_param['RNN_param']['recurrent_dropout']
    else:
        print('No value selected for recurrent dropout, choosing rc_do=0.')
        rc_do = 0
    if 'dropout' in model_param['RNN_param'].keys():
        do = model_param['RNN_param']['dropout']
        print('No value selected for dropout, choosing do=0.')
    else:
        do = 0

    implementation_mode = model_param['RNN_param']['implementation']

    model = keras.models.Sequential()
    if 'SimpleRNN' in model_param['RNN_param']['RNN_type']:
        model.add(keras.layers.SimpleRNN(n_a, input_shape=(seq_length, nx), batch_size=batch_size,
                                         return_sequences=return_sequences, recurrent_dropout=rc_do, stateful=stateful))
    elif 'GRU' in model_param['RNN_param']['RNN_type']:
        model.add(keras.layers.GRU(n_a, input_shape=(seq_length, nx), recurrent_activation='sigmoid', batch_size=batch_size,
                                   return_sequences=return_sequences, stateful=stateful, recurrent_dropout=rc_do, implementation=implementation_mode))
    elif 'LSTM' in model_param['RNN_param']['RNN_type']:
        model.add(keras.layers.LSTM(n_a, input_shape=(seq_length, nx), recurrent_activation='sigmoid', batch_size=batch_size,
                                    return_sequences=return_sequences, stateful=stateful, recurrent_dropout=rc_do, dropout=do, implementation=implementation_mode))

    if not 'p_dropout' in model_param.keys() or model_param['p_dropout'] is None:
        p_dropout = (len(model_param['n_units']))*[0]
    else:
        p_dropout = model_param['p_dropout']

    if len(p_dropout) != len(model_param['n_units']):
        raise Exception(
            'p_dropout in model_param was not selected according to the number of layers')

    for units, activation, dropout in zip(model_param['n_units'], model_param['activation'], p_dropout):
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(units, activation=activation)))
        model.add(keras.layers.Dropout(rate=dropout))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(ny, activation='linear')))
    return model


def create_data(train_data_param, A, B, C, D=0, mode=0, seed=None):

    if seed:
        np.random.seed(seed)

    x_arr = []
    u_arr = []
    t_arr = []

    nx = A.shape[0]
    nu = B.shape[1]
    # Initial values:
    # ---------------------------------
    # timer for last reset of control input (one for each channel)
    x = np.zeros((nx, 1))
    u = np.zeros((nu, 1))
    t = 0
    t_reset = np.array([t, t]).reshape(nu, 1)
    stop_phase_counter = 0
    t_stop_phase_init = 0
    while t < train_data_param['t_end']:
        if mode == 0:
            # Boolean vector : Should control input be resetted?
            reset = (t-t_reset > train_data_param['t_u_min'])*(t-t_reset + np.random.rand(nu, 1)*(
                train_data_param['t_u_max']-train_data_param['t_u_min']) > train_data_param['t_u_max'])
            # Candidate for new control input:
            u_candidate = (-0.5+np.random.rand(nu, 1))*train_data_param['u_amp']
            # New control input (if resetted):
            u = (reset*u_candidate+(1-reset)*u)
            # Reset timer (if resetted):
            t_reset = reset*t+(1-reset)*t_reset
        elif mode == 1:
            u1_candidate = train_data_param['u_amp']/2*np.sin(t*0.2*(2+np.sin(0.1*t)))
            u2_candidate = train_data_param['u_amp']/2*np.cos(t*0.2*(2+np.cos(0.1*t)))
            u = np.stack((u1_candidate, u2_candidate)).reshape(nu, 1)

        # Control input to zero multiple times
        if train_data_param['num_stop_phases'] > 0 and stop_phase_counter*train_data_param['t_end']/train_data_param['num_stop_phases'] <= t:
            u = np.zeros((nu, 1))
            if t-t_stop_phase_init >= train_data_param['stop_phase_duration']:
                stop_phase_counter += 1
        else:
            t_stop_phase_init = t

        # Update states (x) and time (t):
        x = np.matmul(A, x)+np.matmul(B, u)
        t += train_data_param['dt']

        # Save results:
        x_arr.append(x)
        u_arr.append(u)
        t_arr.append(t)

    x_arr = np.concatenate(x_arr, axis=1)
    u_arr = np.concatenate(u_arr, axis=1)
    t_arr = np.array(t_arr)

    return x_arr, u_arr, t_arr
