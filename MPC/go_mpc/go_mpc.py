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
    def __init__(self, n_horizon, cas_verbose=True):

        self.n_horizon = n_horizon

        self.create_model()
        self.create_optim(cas_verbose)

    def create_model(self):
        """
        --------------------------------------------------------------------------
        template_model: Load the neural network system model
        --------------------------------------------------------------------------
        """
        nn_model_path = './model/007_man_5x50_both_datasets_filtered_mpc02/'
        nn_model_name = '007_man_5x50_both_datasets_filtered_mpc02.h5'

        keras_model = keras.models.load_model(nn_model_path+nn_model_name)
        print('----------------------------------------------------')
        print('Loaded Keras model with the following architecture:')
        print(keras_model.summary())

        weights = keras_model.get_weights()
        config = keras_model.get_config()

        with open(nn_model_path+'007_man_5x50_both_datasets_filtered_mpc02_train_data_param.pkl', 'rb') as f:
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
        cluster_labels = pd.read_json(nn_model_path+'cluster_labels_dt1h_both_datasets.json')
        pressure_factor = pd.read_json(nn_model_path+'pressure_factor_dt1h_both_datasets.json')

        jun_cl_press_fac_min = pressure_factor.groupby(cluster_labels.loc['pressure_cluster'], axis=1).min()

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
            entry('jun_cl_demand_sum', shape=(30, 1)),
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

        x_next['tank_press'] = x.cat+dtank_press

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
        ])

        # For states
        self.x_lb = x(0)
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
        self.u_lb = u(0)
        self.u_ub = u(1)
        self.u_ub['head_pump'] = 2
        self.u_ub['PRValve'] = 600
        self.u_ub['TCValve'] = 70

        # Further (non-linear) constraints:
        self.nl_cons = struct_MX([
            entry('tank_press_lb', expr=self.x['tank_press']+self.eps['tank_press_lb']),
            entry('jun_cl_press_min', expr=jun_cl_press_min),
            entry('pump_energy', expr=pump_energy)
        ])

        self.nl_ub = self.nl_cons(np.inf)
        self.nl_lb = self.nl_cons(-np.inf)

        self.nl_lb['jun_cl_press_min'] = 0
        #self.nl_lb['pump_energy'] = 0
        self.nl_lb['tank_press_lb'] = 1

        self.nl_cons_fun = Function('nl_cons', [x, u, tvp, p_set, eps], [self.nl_cons])

        """
        --------------------------------------------------------------------------
        model: define cost function
        --------------------------------------------------------------------------
        """
        lterm = sum1(x.cat-4)**2  # +sum1((jun_cl_press_min-50)**2)
        #lterm = sum1(pump_energy)/100 + 1e4*sum1(eps.cat**2)
        mterm = 0
        self.rterm_factor = 1e-7

        self.lterm_fun = Function('lterm', [x, u, tvp, p_set, eps], [lterm])
        self.mterm_fun = Function('mterm_fun', [x], [mterm])

        self.model_vars = {
            'x_mhe': x,
            'u_mhe': u,
            'tvp': tvp
        }

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


class simulator(go_mpc):
    def __init__(self, n_horizon):

        self.mhe_counter = 0     # must start at zero.

        """
        --------------------------------------------------------------------------
        simulator: Create Optimizer
        --------------------------------------------------------------------------
        """
        # go_mpc creates the model and optimizer object and initilizes
        # the casadi structures that hold the optimal solution.
        go_mpc.__init__(self, n_horizon)

        """
        --------------------------------------------------------------------------
        simulator: Create MHE data structure:
        --------------------------------------------------------------------------
        """
        self.mhe_data = mhe_data(self)

    def config_check(self):
        """
        Configuration check to find missing values and validity of MHE problem + Parameters
        """
        None

    def mpc_step(self):
        if self.mhe_counter == 0:
            self.config_check()
        print('---------------------------------------')
        print('MHE counter: {}'.format(self.mhe_counter))

        x_real_now = self.x_real_now(self.t_0)
        tvp_real_now = self.tvp_real_now(self.t_0)
        t_real_now = self.t_real_now(self.t_0)

        assert tvp_real_now.shape == (self.n_horizon, self.tvp.shape[0]), 'TVP has wrong shape'
        # Store data:
        self.mhe_data.x_meas = np.append(self.mhe_data.x_meas, x_real_now.reshape(1, -1), axis=0)
        self.mhe_data.tvp = np.append(self.mhe_data.tvp, tvp_real_now[0, :].reshape(1, -1), axis=0)
        self.mhe_data.t_mhe = np.append(self.mhe_data.t_mhe, t_real_now.reshape(1, -1), axis=0)

        # Solve MPC
        self.obj_p_num['x_0'] = self.mhe_data.x_meas[-1, :]
        self.obj_p_num['tvp'] = vertsplit(self.mhe_data.tvp[-self.n_horizon:, :])
        # Solve the optimization problem and populate self.obj_x_num with the new solution.
        self.solve()

        # Save solution
        self.mhe_data.u_mhe = np.append(self.mhe_data.u_mhe, horzcat(*self.obj_x_num['u', :]).T.full(), axis=0)

        self.mhe_counter += 1
        self.t_0 += self.t_step

    def t_real_now(self, time_step):
        """
        t_real function. Determines the real time (in seconds) for the current time_step.
        This is useful when an index is used as the timestep instead of a physical time.
        """
        return np.array(self.t_0)

    def u_real_now(self, time_step):
        """
        Input function. Determines the inputs at the current timestep.
        """
        time_ind = np.searchsorted(self.sim_data['t'][0, :], time_step, side='right')

        u_real = self.u(0)  # Get structure defined in model and initialize values to zero.
        u_real['pos_mot_1'] = self.sim_data['u'][0, time_ind]
        u_real['pos_mot_2'] = self.sim_data['u'][1, time_ind]
        # Return as concatenated numpy array:
        u_real = u_real.cat.full()
        # Add noise and bias:
        u_real += self.u_bias
        u_real += self.u_noise_mag*np.random.randn(*u_real.shape)

        return u_real.reshape(self.n_u, 1)

    def y_real_now(self, time_step):
        """
        Output function. Determines the output at the current timestep.
        """
        time_ind = np.searchsorted(self.sim_data['t'][0, :], time_step, side='right')

        y_real = self.y_meas(0)  # Get structure defined in model and initialize values to zero.
        y_real['disc_angle'] = self.sim_data['y'][:, time_ind]
        # Return as concatenated numpy array:
        y_real = y_real.cat.full()
        # Add noise:
        y_real += self.y_noise_mag*np.random.randn(*y_real.shape)

        return y_real.reshape(self.n_y, 1)

    def tvp_real_now(self, time_step):
        """
        tvp function. Determines the time-varying parameters at the current timestep.
        """
        tvp_real = self.tvp(0)
        tvp_real['dummy_1'] = 0
        tvp_real['dummy_2'] = 0

        # Return as concatenated numpy array:
        tvp_real = vertcat(tvp_real).full()

        return tvp_real


class mhe_data:
    def __init__(self, simulator):
        self.simulator = simulator
        # Initialize solution elements as empty arrays of defined dimension.
        self.x_mhe = np.empty((0, simulator.x.shape[0]))
        self.p_mhe = np.empty((0, simulator.p_est.shape[0]))
        self.u_mhe = np.empty((0, simulator.u.shape[0]))
        self.u_meas = np.empty((0, simulator.u_meas.shape[0]))
        self.y_meas = np.empty((0, simulator.y_meas.shape[0]))
        self.y_calc = np.empty((0, simulator.y_calc.shape[0]))
        self.tvp = np.empty((0, simulator.tvp.shape[0]))
        self.t_mhe = np.empty((0, 1))

    def export_to_matlab(self, save_name='mhe_results.mat'):
        print('Storing results to matlab as: {}'.format(save_name))
        # Store all numpy arrays in the name scope of mhe_data.
        matlab_dict = {key: value for key, value in self.__dict__.items() if type(value) == np.ndarray}

        sio.savemat(save_name, matlab_dict)

    def export_cas_struct(self, save_name='mhe_results.pkl'):
        """
        Store result as a pickled casadi structure. Very convenient to work with and smaller file size than .mat.
        The structure can be indexed intuitiveley with the names assigned in the model definition.
        More information on structures and power indexing here: http://casadi.sourceforge.net/v2.0.0/tutorials/tools/structure.pdf

        Note that storing and saving results take langer than with export_to_matlab.
        """
        n = self.t_mhe.shape[0]
        cas_res = struct_symSX([
            entry('x', repeat=n+1, struct=self.simulator.x),
            entry('u', repeat=n, struct=self.simulator.u),
            entry('u_meas', repeat=n, struct=self.simulator.u),
            entry('y', repeat=n, struct=self.simulator.y_meas),
            entry('y_meas', repeat=n, struct=self.simulator.y_meas),
            entry('p_est', repeat=n, struct=self.simulator.p_est),
            entry('t', shape=(n, 1))
        ])
        cas_res_num = cas_res(0)
        cas_res_num['x'] = vertsplit(self.x_mhe)
        cas_res_num['u'] = vertsplit(self.u_mhe)
        cas_res_num['u_meas'] = vertsplit(self.u_meas)
        cas_res_num['y'] = vertsplit(self.y_calc)
        cas_res_num['y_meas'] = vertsplit(self.y_meas)
        cas_res_num['p_est'] = vertsplit(self.p_mhe)
        cas_res_num['t'] = self.t_mhe

        with open(save_name, 'wb') as f:
            pickle.dump(cas_res_num, f)


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
