import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy
from scipy.linalg import block_diag
from casadi import *
from casadi.tools import *
import pdb
import warnings
import pickle
import pandas as pd
from tensorflow import keras


class go_mpc:
    def __init__(self, n_horizon,  nn_model_path, nn_model_name, cluster_labels, pressure_factor, min_control, max_control, cas_verbose=True):

        self.n_horizon = n_horizon

        self.create_model(nn_model_path, nn_model_name, cluster_labels, pressure_factor, min_control, max_control)
        self.create_optim(cas_verbose)

    def create_model(self, nn_model_path, nn_model_name, cluster_labels, pressure_factor, min_control, max_control):
        """
        --------------------------------------------------------------------------
        template_model: Load the neural network system model
        --------------------------------------------------------------------------
        """

        keras_model = keras.models.load_model(nn_model_path+nn_model_name+'.h5')
        print('----------------------------------------------------')
        print('Loaded Keras model with the following architecture:')
        print(keras_model.summary())

        weights = keras_model.get_weights()
        config = keras_model.get_config()

        with open(nn_model_path+nn_model_name+'_train_data_param.pkl', 'rb') as f:
            train_data_param = pickle.load(f)

        print('----------------------------------------------------')
        print('Loaded input and output scaling for the Keras model')
        input_scaling = train_data_param['input_scaling']
        output_scaling = train_data_param['output_scaling']

        jun_cl_press_fac_min = pressure_factor.groupby(cluster_labels.loc['pressure_cluster'], axis=1).min()
        n_cluster = int(cluster_labels.loc['pressure_cluster'].max() +1)

        """
        --------------------------------------------------------------------------
        model: define parameters, states and controls as symbols
        --------------------------------------------------------------------------
        """
        # States struct (optimization variables):
        self.x = x = struct_symMX([
            entry('tank_press', shape=(7, 1)),
        ])

        # Input struct (optimization variables):
        self.u = u = struct_symMX([
            entry('head_pump', shape=(5, 1)),
            entry('PRValve', shape=(3, 1)),
            entry('TCValve', shape=(1, 1)),
        ])

        # time-varying parameter struct (parameters for optimization problem):
        self.tvp = tvp = struct_symMX([
            entry('jun_cl_demand_sum', shape=(n_cluster, 1)),
            entry('u_prev', struct=u),
        ])

        self.n_x = n_x = x.shape[0]
        self.n_u = n_u = u.shape[0]
        self.n_tvp = n_tvp = tvp.shape[0]

        # Fixed parameters:
        self.p_set = p_set = struct_symMX([
            entry('dummy_1'),
        ])

        """
        --------------------------------------------------------------------------
        model: define difference equations
        --------------------------------------------------------------------------
        """
        self.x_next = x_next = struct_MX(x)

        nn_in_sym = vertcat(x, u, tvp['jun_cl_demand_sum'])
        nn_in_sym_scaled = nn_in_sym/input_scaling.to_numpy()

        nn_out_sym_scaled = dense_nn(weights, config, nn_in_sym_scaled.T)
        nn_out_sym = nn_out_sym_scaled.T*output_scaling.to_numpy()

        dtank_press = nn_out_sym[:7]
        pump_energy = nn_out_sym[7:12]
        jun_cl_press_mean_norm = nn_out_sym[12:]  # Mean value of the normalized cluster pressures
        jun_cl_press_min = nn_out_sym[12:]*jun_cl_press_fac_min.to_numpy().reshape(-1, 1)

        x_next['tank_press'] = x['tank_press']+dtank_press

        # Create casadi functions:
        self.x_next_fun = Function('state_equation', [x, u, tvp, p_set], [x_next])

        """
        --------------------------------------------------------------------------
        model: define constraints
        --------------------------------------------------------------------------
        """
        # Softconstraint slack variables:
        self.eps = eps = struct_symMX([
            entry('tank_press_lb', shape=(7, 1)),
            entry('jun_cl_press_min', shape=(n_cluster,1)),
            entry('pump_energy', shape=(5,1))
        ])

        # For states
        self.x_lb = x(-1e-1)
        # From INP file:
        max_tank_level = np.array([6.75, 6.5, 5, 5.5, 4.5, 5.9, 4.7])+1e-3
        self.x_ub = x(max_tank_level)

        # Do not change bounds for soft constraint slack variables
        self.eps_lb = eps(0)
        self.eps_ub = eps(np.inf)

        # Terminal constraints
        self.x_terminal_lb = self.x_lb
        self.x_terminal_ub = self.x_ub

        # Inputs
        self.u_lb = u(min_control)
        self.u_ub = u(max_control)

        # Further (non-linear) constraints:
        self.nl_cons = struct_MX([
            entry('tank_press_lb',    expr=self.x['tank_press']+self.eps['tank_press_lb']),
            entry('jun_cl_press_min', expr=jun_cl_press_min+self.eps['jun_cl_press_min']),
            entry('pump_energy',      expr=pump_energy+self.eps['pump_energy'])
        ])

        self.nl_ub = self.nl_cons(np.inf)
        self.nl_lb = self.nl_cons(-np.inf)

        self.nl_lb['jun_cl_press_min'] = 0
        self.nl_lb['pump_energy'] = 0
        self.nl_lb['tank_press_lb'] = 2

        self.nl_cons_fun = Function('nl_cons', [x, u, tvp, p_set, eps], [self.nl_cons])

        """
        --------------------------------------------------------------------------
        model: define cost function
        --------------------------------------------------------------------------
        """
        #lterm = sum1(x.cat-2)**2  # +sum1((jun_cl_press_min-50)**2)
        lterm = sum1(pump_energy)/100 + 1e6*sum1((eps.cat)**2)
        mterm = 0
        # Penalize changes in the control input from t_k to t_k+1:
        self.rterm_factor = 1e-2

        self.lterm_fun = Function('lterm', [x, u, tvp, p_set, eps], [lterm])
        self.mterm_fun = Function('mterm_fun', [x], [mterm])

    def create_optim(self, cas_verbose=True):
        """
        --------------------------------------------------------------------------
        MHE: create optimization problem
        --------------------------------------------------------------------------
        """

        # Create struct for optimization variables:
        self.obj_x = obj_x = struct_symMX([
            entry('x', repeat=self.n_horizon+1, struct=self.x),
            entry('u', repeat=self.n_horizon, struct=self.u),
            entry('eps', repeat=self.n_horizon, struct=self.eps)
        ])

        # Number of optimization variables:
        self.n_x_optim = self.obj_x.shape[0]

        # Create struct for optimization parameters:
        self.obj_p = obj_p = struct_symMX([
            entry('x_0', struct=self.x),
            entry('tvp',    repeat=self.n_horizon, struct=self.tvp),
            entry('p_set', struct=self.p_set)
        ])

        self.mpc_obj_aux = struct_MX(struct_symMX([
            entry('nl_cons', repeat=self.n_horizon, struct=self.nl_cons)
        ]))

        self.lb_obj_x = obj_x(-np.inf)
        self.ub_obj_x = obj_x(np.inf)

        # Initialize objective function and constraints
        obj = 0
        cons = []
        cons_lb = []
        cons_ub = []

        # Initial condition:
        cons.append(obj_x['x', 0]-obj_p['x_0'])
        cons_lb.append(np.zeros((self.x.shape[0], 1)))
        cons_ub.append(np.zeros((self.x.shape[0], 1)))

        # Note:
        # X = [x_0, x_1, ... , x_(N+1)]         -> n_horizon+1 elements
        # U = [u_0, u_1, ... , u_N]             -> n_horizon elements

        for k in range(self.n_horizon):
            # Add constraints for state equation:
            x_next = self.x_next_fun(obj_x['x', k], obj_x['u', k], obj_p['tvp', k], obj_p['p_set'])

            cons.append(x_next-obj_x['x', k+1])
            cons_lb.append(np.zeros((self.x.shape[0], 1)))
            cons_ub.append(np.zeros((self.x.shape[0], 1)))

            nl_cons_k = self.nl_cons_fun(obj_x['x', k], obj_x['u', k], obj_p['tvp', k], obj_p['p_set'], obj_x['eps', k])
            cons.append(nl_cons_k)
            cons_lb.append(self.nl_lb)
            cons_ub.append(self.nl_ub)
            self.mpc_obj_aux['nl_cons', k] = nl_cons_k

            obj += self.lterm_fun(obj_x['x', k], obj_x['u', k], obj_p['tvp', k], obj_p['p_set'], obj_x['eps', k])

            # U regularization:
            if k == 0:
                obj += self.rterm_factor*sum1(((obj_x['u', k]-obj_p['tvp', k+1, 'u_prev'])**2)/self.u_ub)
            else:
                obj += self.rterm_factor*sum1(((obj_x['u', k]-obj_x['u', k-1])**2)/self.u_ub)

            self.lb_obj_x['x', k] = self.x_lb
            self.ub_obj_x['x', k] = self.x_ub

            self.lb_obj_x['u', k] = self.u_lb
            self.ub_obj_x['u', k] = self.u_ub

            self.lb_obj_x['eps', k] = self.eps_lb
            self.ub_obj_x['eps', k] = self.eps_ub

        obj += self.mterm_fun(obj_x['x', self.n_horizon])

        self.lb_obj_x['x', self.n_horizon] = self.x_terminal_lb
        self.ub_obj_x['x', self.n_horizon] = self.x_terminal_ub

        cons = vertcat(*cons)
        self.cons_lb = vertcat(*cons_lb)
        self.cons_ub = vertcat(*cons_ub)

        """
        --------------------------------------------------------------------------
        MHE: Casadi options for optimization and create optimization class
        --------------------------------------------------------------------------
        """
        optim_opts = {}
        optim_opts["expand"] = False
        optim_opts["ipopt.linear_solver"] = 'ma27'
        # NOTE: this could be passed as parameters of the optimizer class
        optim_opts["ipopt.max_iter"] = 200
        # optim_opts["ipopt.ma27_la_init_factor"] = 50.0
        # optim_opts["ipopt.ma27_liw_init_factor"] = 50.0
        # optim_opts["ipopt.ma27_meminc_factor"] = 10.0
        optim_opts["ipopt.tol"] = 1e-6
        if not cas_verbose:
            optim_opts["ipopt.print_level"] = 0
            optim_opts["ipopt.sb"] = 'yes'
            optim_opts["print_time"] = 0

        # Create casadi optimization object:
        nlp = {'x': vertcat(obj_x), 'f': obj, 'g': cons, 'p': vertcat(obj_p)}
        self.S = nlpsol('S', 'ipopt', nlp, optim_opts)

        self.aux_fun = Function('aux_fun', [self.obj_x, self.obj_p], [self.mpc_obj_aux])

        # Create copies of these structures with numerical values (all zero):
        self.obj_x_num = self.obj_x(0)
        self.obj_p_num = self.obj_p(0)
        self.obj_aux_num = self.mpc_obj_aux(0)

    def solve(self):
        """
        Solves the optimization problem for the given intial condition and parameter set.
        Populates the self.obj_x_num object, which has the same structure as the self.obj_x object
        and can be used to conveniently query parts of the solution.

        .solve() will print the casadi output unless deactivated.
        """
        r = self.S(x0=self.obj_x_num, lbx=self.lb_obj_x, ubx=self.ub_obj_x,  ubg=self.cons_ub, lbg=self.cons_lb, p=self.obj_p_num)
        self.obj_x_num = self.obj_x(r['x'])
        # Values of lagrange multipliers:
        self.lam_g_num = r['lam_g']
        self.solver_stats = self.S.stats()

        self.obj_aux_num = self.mpc_obj_aux(self.aux_fun(self.obj_x_num, self.obj_p_num))


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

    return nn_in
