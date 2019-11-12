#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2018 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.
#

from casadi import *
from casadi.tools import *
import numpy as NP
import core_do_mpc
import pickle
import pdb
import pandas as pd

from tensorflow import keras


def dense_nn(weights, config, nn_in):
    """
    Forward Propagation through a Neural Network with weights and config from keras (as imported with TF 2.0+)

    weights and config are lists that can be retrieved from keras models with the get_weights, get_config method on layers
    activation can be part of the dense layer or added as an extra layer.

    p_dropout is the probability that weights will be set to zero. By defaul p_dropout is zero (scalar) and will not be executed.
    If p_dropout is supplied it must have the same length as the amount of dropout layers in the ANN.
    """
    k = 0
    for layer_i in config['layers']:
        if 'Dense' in layer_i['class_name']:
            nn_in = nn_in@weights[k] + weights[k+1].reshape(1, -1)
            # Two weights for every dense layer
            k += 2
        if 'activation' in layer_i['config'].keys():
            if 'linear' in layer_i['config']['activation']:
                nn_in = nn_in
            if 'tanh' in layer_i['config']['activation']:
                nn_in = tanh(nn_in)
            else:
                print('Activation not currently supported.')
    return nn_in


def model():
    """
    --------------------------------------------------------------------------
    template_model: Load the neural network system model
    --------------------------------------------------------------------------
    """
    nn_model_path = './model/002_man_4x80/'
    nn_model_name = '002_man_4x80.h5'

    keras_model = keras.models.load_model(nn_model_path+nn_model_name)
    print('----------------------------------------------------')
    print('Loaded Keras model with the following architecture:')
    print(keras_model.summary())

    weights = keras_model.get_weights()
    config = keras_model.get_config()

    with open(nn_model_path+'train_data_param_unfiltered.pkl', 'rb') as f:
        train_data_param = pickle.load(f)

    print('----------------------------------------------------')
    print('Loaded input and output scaling for the Keras model')
    input_scaling = train_data_param['input_scaling']
    output_scaling = train_data_param['output_scaling']

    # with open(nn_model_path+'io_scheme.pkl', 'rb') as f:
    #     io_scheme = pickle.load(f)

    # Check if Keras and casadi return the same thing:
    # nn_in_test = np.ones((1, 46))
    # keras_out = keras_model.predict(nn_in_test)
    # cas_out = dense_nn(weights, config, nn_in_test)
    # print('Difference between CasADi and Keras Neural Network evaluation (with input all ones).')
    # print(keras_out-cas_out)
    # pdb.set_trace()

    # Load the cluster labels and the pressure factors and determine the smallest factor for each cluster.
    # For a given mean value of the normalized cluster pressure, this will determine the smallest physical pressure in the cluster
    cluster_labels = pd.read_json(nn_model_path+'cluster_labels_dt1h.json')
    pressure_factor = pd.read_json(nn_model_path+'pressure_factor_dt1h.json')

    jun_cl_press_fac_min = pressure_factor.groupby(cluster_labels.loc['pressure_cluster'], axis=1).min()

    """
    --------------------------------------------------------------------------
    template_model: define uncertain parameters, states and controls as symbols
    --------------------------------------------------------------------------
    """
    # Define the uncertainties as CasADi symbols
    alpha = SX.sym("alpha")
    beta = SX.sym("beta")

    # Define the differential states as CasADi symbols
    _x = struct_symMX([
        entry('tank_press_T3'),
        entry('tank_press_T1'),
        entry('tank_press_T7'),
        entry('tank_press_T6'),
        entry('tank_press_T5'),
        entry('tank_press_T2'),
        entry('tank_press_T4'),
    ])

    # Define the disturbances as CasADi symbols
    # These are time-varying parameters that change at each step of the prediction and at each sampling time of the MPC controller.
    _d = MX.sym('jun_cl_demand_sum', 30)

    # Define the algebraic states as CasADi symbols

    # Define the control inputs as CasADi symbols

    _u = struct_symMX([
        entry('head_pump_PU2'),
        entry('head_pump_PU5'),
        entry('head_pump_PU6'),
        entry('head_pump_PU8'),
        entry('head_pump_PU10'),
        entry('PRValve_V1'),
        entry('PRValve_V45'),
        entry('PRValve_V47'),
        entry('TCValve_V2'),
    ])

    """
    --------------------------------------------------------------------------
    template_model: define algebraic and differential equations
    --------------------------------------------------------------------------
    """
    nn_in_sym = vertcat(_x, _u, _d)
    nn_in_sym_scaled = nn_in_sym/input_scaling.to_numpy()

    nn_out_sym_scaled = dense_nn(weights, config, nn_in_sym.T)
    nn_out_sym = nn_out_sym_scaled.T*output_scaling.to_numpy()

    dtank_press = nn_out_sym[:7]
    pump_energy = nn_out_sym[7:12]
    jun_cl_press_mean_norm = nn_out_sym[12:]  # Mean value of the normalized cluster pressures
    jun_cl_press_min = nn_out_sym[12:]*jun_cl_press_fac_min.to_numpy().reshape(-1, 1)

    _xdot = _x+dtank_press

    _p = vertcat(alpha, beta)

    _z = []

    _tv_p = vertcat(_d)

    """
    --------------------------------------------------------------------------
    template_model: initial condition and constraints
    --------------------------------------------------------------------------
    """
    x0 = _x(1)

    # Bounds on the states. Use "inf" for unconstrained states
    x_lb = _x(0)
    x_ub = _x(20)

    # Bounds on the control inputs. Use "inf" for unconstrained inputs
    u_lb = _u(0)
    u_ub = _u(np.inf)
    u0 = _u(0)

    # Scaling factors for the states and control inputs. Important if the system is ill-conditioned
    x_scaling = _x(1)
    u_scaling = _u(1)

    # Other possibly nonlinear constraints in the form cons(x,u,p) <= cons_ub
    # Define the expresion of the constraint (leave it empty if not necessary)
    cons = vertcat(50-jun_cl_press_min,
                   -pump_energy
                   )
    # Define the lower and upper bounds of the constraint (leave it empty if not necessary)
    cons_ub = NP.array([np.zeros((35, 1))])

    # Activate if the nonlinear constraints should be implemented as soft constraints
    soft_constraint = 0
    # Penalty term to add in the cost function for the constraints (it should be the same size as cons)
    penalty_term_cons = NP.array([])
    # Maximum violation for the constraints
    maximum_violation = NP.array([0])

    # Define the terminal constraint (leave it empty if not necessary)
    cons_terminal = vertcat()
    # Define the lower and upper bounds of the constraint (leave it empty if not necessary)
    cons_terminal_lb = NP.array([])
    cons_terminal_ub = NP.array([])

    """
    --------------------------------------------------------------------------
    template_model: cost function
    --------------------------------------------------------------------------
    """
    # Define the cost function
    # Lagrange term
    lterm = pump_energy
    # lterm =  - C_b
    # Mayer term
    mterm = 0
    # mterm =  - C_b
    # Penalty term for the control movements
    rterm = _u(0)

    # For compliance with do_mpc:
    # TODO: Check if necessary.
    # _x = vertcat(_x)
    # _u = vertcat(_u)

    """
    --------------------------------------------------------------------------
    template_model: pass information (not necessary to edit)
    --------------------------------------------------------------------------
    """
    model_dict = {'x': _x, 'u': _u, 'rhs': _xdot, 'p': _p, 'z': _z, 'x0': x0, 'x_lb': x_lb, 'x_ub': x_ub, 'u0': u0, 'u_lb': u_lb, 'u_ub': u_ub, 'x_scaling': x_scaling, 'u_scaling': u_scaling, 'cons': cons,
                  "cons_ub": cons_ub, 'cons_terminal': cons_terminal, 'cons_terminal_lb': cons_terminal_lb, 'tv_p': _tv_p, 'cons_terminal_ub': cons_terminal_ub, 'soft_constraint': soft_constraint, 'penalty_term_cons': penalty_term_cons, 'maximum_violation': maximum_violation, 'mterm': mterm, 'lterm': lterm, 'rterm': rterm}

    model = core_do_mpc.model(model_dict)

    return model
