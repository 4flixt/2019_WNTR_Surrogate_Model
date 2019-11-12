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


class go_mhe:
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

        nn_in_sym = vertcat(x, u, tvp)
        nn_in_sym_scaled = nn_in_sym/input_scaling.to_numpy()

        nn_out_sym_scaled = dense_nn(weights, config, nn_in_sym.T)
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
        self.nl_cons = [
            vertcat(jun_cl_press_min-50,
                    pump_energy
                    )]
        self.nl_ub = np.inf*np.ones((35, 1))
        self.nl_lb = np.zeros((35, 1))

        assert type(self.nl_cons) == list, 'nl_cons must be a list. Can be left empty if not used.'
        self.nl_cons_fun = Function('nl_cons', [x, u, tvp, p_set], self.nl_cons)

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
            entry('p_est', struct=self.p_est)
        ])

        # Number of optimization variables:
        self.n_x_optim = self.obj_x.shape[0]

        # Create struct for optimization parameters:
        self.obj_p = obj_p = struct_symMX([
            entry('x_0', struct=self.x),
            entry('p_0', struct=self.p_est),
            entry('u_meas', repeat=self.n_horizon, struct=self.u_meas),
            entry('y_meas', repeat=self.n_horizon, struct=self.y_meas),
            entry('tvp',    repeat=self.n_horizon, struct=self.tvp),
            entry('p_set', struct=self.p_set)
        ])

        # Initialize objective function and constraints
        obj = 0
        cons = []
        cons_lb = []
        cons_ub = []

        # Arrival cost:
        dx = (obj_p['x_0']-obj_x['x', 0])
        obj += 0.5*dx.T@obj_p['p_set', 'P_x']@dx
        dp = (obj_p['p_0']-obj_x['p_est'])
        obj += 0.5*dp.T@obj_p['p_set', 'P_p']@dp

        # Note:
        # X = [x_0, x_1, ... , x_(N+1)]         -> n_horizon+1 elements
        # U = [u_0, u_1, ... , u_N]             -> n_horizon elements
        # Y = [y_0, y_1, ... , y_N]             -> n_horizon elements

        for k in range(self.n_horizon):
            # Add constraints for state equation:
            x_next = self.x_next_fun(obj_x['x', k], obj_x['u', k], obj_x['p_est'], obj_p['tvp', k], obj_p['p_set'])

            cons.append(x_next-obj_x['x', k+1])
            cons_lb.append(np.zeros((self.x.shape[0], 1)))
            cons_ub.append(np.zeros((self.x.shape[0], 1)))
            if self.nl_cons:  # nl_cons is a list. Empty list == False means no further constraints are added.
                cons.append(self.nl_cons_fun(obj_x['x', k], obj_x['u', k], obj_x['p_est'], obj_p['tvp', k], obj_p['p_set']))
                cons_lb.append(self.nl_lb.reshape(-1, 1))
                cons_ub.append(self.nl_ub.reshape(-1, 1))

            # Add input penalty:
            du = (obj_x['u', k]-obj_p['u_meas', k])
            obj += 0.5*du.T@obj_p['p_set', 'P_u']@du

            # Add measurement penalty:
            y_calc = self.y_calc_fun(obj_x['x', k], obj_x['u', k], obj_x['p_est'], obj_p['tvp', k], obj_p['p_set'])
            dy = (y_calc-obj_p['y_meas', k])
            obj += 0.5*dy.T@obj_p['p_set', 'P_y']@dy

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
        optim_opts["ipopt.max_iter"] = 1000
        optim_opts["ipopt.ma27_la_init_factor"] = 50.0
        optim_opts["ipopt.ma27_liw_init_factor"] = 50.0
        optim_opts["ipopt.ma27_meminc_factor"] = 10.0
        optim_opts["ipopt.tol"] = 1e-6
        if not cas_verbose:
            optim_opts["ipopt.print_level"] = 0
            optim_opts["ipopt.sb"] = 'yes'
            optim_opts["print_time"] = 0

        # Create casadi optimization object:
        nlp = {'x': vertcat(obj_x), 'f': obj, 'g': cons, 'p': vertcat(obj_p)}
        self.S = nlpsol('S', 'ipopt', nlp, optim_opts)
        # Create copies of these structures with numerical values (all zero):
        self.obj_x_num = self.obj_x(0)
        self.obj_p_num = self.obj_p(0)

    def solve(self):
        """
        Solves the optimization problem for the given intial condition and parameter set.
        Populates the self.obj_x_num object, which has the same structure as the self.obj_x object
        and can be used to conveniently query parts of the solution.

        .solve() will print the casadi output unless deactivated.
        """
        r = self.S(x0=self.obj_x_num, ubg=self.cons_ub, lbg=self.cons_lb, p=self.obj_p_num)
        self.obj_x_num = self.obj_x(r['x'])
        # Values of lagrange multipliers:
        self.lam_g_num = r['lam_g']
        self.solver_stats = self.S.stats()


class simulator(go_mhe):
    def __init__(self, n_horizon):

        self.mhe_counter = 0     # must start at zero.

        """
        --------------------------------------------------------------------------
        simulator: Create Optimizer
        --------------------------------------------------------------------------
        """
        # go_mhe creates the model and optimizer object and initilizes
        # the casadi structures that hold the optimal solution.
        go_mhe.__init__(self, n_horizon)

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

    def mhe_step(self):
        if self.mhe_counter == 0:
            self.config_check()
        print('---------------------------------------')
        print('MHE counter: {}'.format(self.mhe_counter))

        u_real_now = self.u_real_now(self.t_0)
        y_real_now = self.y_real_now(self.t_0)
        tvp_real_now = self.tvp_real_now(self.t_0)
        t_real_now = self.t_real_now(self.t_0)

        # Store data:
        self.mhe_data.u_meas = np.append(self.mhe_data.u_meas, u_real_now.reshape(1, -1), axis=0)
        self.mhe_data.y_meas = np.append(self.mhe_data.y_meas, y_real_now.reshape(1, -1), axis=0)
        self.mhe_data.tvp = np.append(self.mhe_data.tvp, tvp_real_now.reshape(1, -1), axis=0)
        self.mhe_data.t_mhe = np.append(self.mhe_data.t_mhe, t_real_now.reshape(1, -1), axis=0)

        if self.mhe_counter >= self.n_horizon-1:
            # Set initial condition as second element of previous solution
            self.obj_p_num['x_0'] = self.obj_x_num['x', 1]
            # Take the last n_horizon measurements and tvp values and set them as parameters.
            self.obj_p_num['u_meas'] = vertsplit(self.mhe_data.u_meas[-self.n_horizon:, :])
            self.obj_p_num['y_meas'] = vertsplit(self.mhe_data.y_meas[-self.n_horizon:, :])
            self.obj_p_num['tvp'] = vertsplit(self.mhe_data.tvp[-self.n_horizon:, :])
            # Solve the optimization problem and populate self.obj_x_num with the new solution.
            self.solve()

        if self.mhe_counter == self.n_horizon-1:
            # If the optimization runs for the first time, store the entire solution of x and u.
            self.mhe_data.x_mhe = np.append(self.mhe_data.x_mhe, horzcat(*self.obj_x_num['x', :]).T.full(), axis=0)
            self.mhe_data.u_mhe = np.append(self.mhe_data.u_mhe, horzcat(*self.obj_x_num['u', :]).T.full(), axis=0)
        elif self.mhe_counter > self.n_horizon-1:
            # If the optimization runs again, store only the most recent value of x and u.
            self.mhe_data.x_mhe = np.append(self.mhe_data.x_mhe, self.obj_x_num['x', -1].T.full(), axis=0)
            self.mhe_data.u_mhe = np.append(self.mhe_data.u_mhe, self.obj_x_num['u', -1].T.full(), axis=0)
        # Store the current values of the estimated parameters at every iteration (the intial guess for the first n_horizon iterations)
        self.mhe_data.p_mhe = np.append(self.mhe_data.p_mhe, self.obj_x_num['p_est'].T.full(), axis=0)

        y_calc_now = self.y_calc_now(self.mhe_counter)
        self.mhe_data.y_calc = np.append(self.mhe_data.y_calc, y_calc_now.reshape(1, -1), axis=0)

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

    def y_calc_now(self, mhe_counter):
        """
        Returns the calculated value of y at each mhe step. This is useful for debugging.
        Uses the casadi function defined in "model"
        """
        if mhe_counter < self.n_horizon-1:
            y_calc_now = np.zeros(self.n_y)
        else:
            x = self.mhe_data.x_mhe[mhe_counter]
            u = self.mhe_data.u_mhe[mhe_counter]
            p_est = self.mhe_data.p_mhe[mhe_counter]
            tvp = self.mhe_data.tvp[mhe_counter]
            p_set = self.obj_p_num['p_set'].full()
            y_calc_now = self.y_calc_fun(x, u, p_est, tvp, p_set).full()

        return y_calc_now

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


gmpc = go_mhe(n_horizon=10)
