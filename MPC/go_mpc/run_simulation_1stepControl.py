#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
#import sklearn
import numpy as np

import time

import sys
sys.path.append('../../Code/')

from testWN import testWN as twm
import wntr
import wntr.network.controls as controls
import wntr.metrics.economic as economics

from casadi import *

import pickle
import random
import pdb

from go_mpc import go_mpc

# %% ::: Loading .inp file
inp_file = '../../Code/c-town_true_network_simplified_controls.inp'
ctown = twm(inp_file)

# %% ::: Setting up time and options for simulation
nDaysSim = 30
nHourDay = 24
simTimeSteps = nDaysSim*nHourDay  # Sampling frequency of 15 min

controlTvary = 8  # Number of time steps for varying controls (if set to 8 -> controls are stable for 2 hours)

ctown.wn.options.time.hydraulic_timestep = 3600  # 1 hr
ctown.wn.options.time.quality_timestep = 3600  # 1 hr
ctown.wn.options.time.report_timestep = 3600
ctown.wn.options.quality.mode = 'AGE'
ctown.wn.options.results.energystr = True
ctown.wn.options.time.duration = 0

# ::: Getting tank elevations
tankEl = []
for tank, name in ctown.wn.tanks():
    tankEl.append(name.elevation)
np.array(tankEl)
nodeNames = ctown.getNodeName()

# ::: Setting upper and lower bounds to control elements
control_components = ctown.wn.pump_name_list + ctown.wn.valve_name_list
min_control = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])  # Lower bondary for controls
max_control = np.array([2., 2., 2., 2., 2., 600.0, 600.0, 600.0, 70.])  # Upper bondary for controls

# Load clustering information:
nn_model_path = './model/002_man_4x80/'
cluster_labels = pd.read_json(nn_model_path+'cluster_labels_dt1h.json')
pressure_factor = pd.read_json(nn_model_path+'pressure_factor_dt1h.json')

# Create controller:
n_horizon = 10
gmpc = go_mpc(n_horizon)

# %% ::: Simulation with updated controls at each time step
for t in range(simTimeSteps):

    # ::: Initializing random seed
    random.seed(t)

    # ::: Loading .inp file from previous step
    if t > 0:
        ctown.wn.reset_initial_values()
        ctown = twm(tempInpFile)
        # Setting simulation options
        ctown.wn.options.time.hydraulic_timestep = 3600  # 1 hr
        ctown.wn.options.time.quality_timestep = 3600  # 1 hr
        ctown.wn.options.time.report_timestep = 3600
        ctown.wn.options.quality.mode = 'AGE'
        ctown.wn.options.results.energystr = True
        ctown.wn.options.time.duration = 0
        ctown.wn.options.time.duration = t*3600

    """
    ---------------------------------------------------
    Forecasting water demand for the next k steps
    ---------------------------------------------------
    """
    startT = t
    addNoise = False
    demand_pred = ctown.forecast_demand_gnoise(n_horizon, startT*ctown.wn.options.time.hydraulic_timestep, ctown.wn.options.time.hydraulic_timestep, addNoise)
    # Cluster demand:
    demand_pred_cl = demand_pred.groupby(cluster_labels.loc['pressure_cluster'], axis=1).sum()

    """
    ---------------------------------------------------
    Get current state:
    ---------------------------------------------------
    """
    if t == 0:
        x0 = 10*np.ones((7, 1))
    else:
        None

    """
    ---------------------------------------------------
    Setup (for current time) and Run controller
    ---------------------------------------------------
    """
    pdb.set_trace()
    # Setup controller for time t:
    gmpc.obj_p_num['x_0'] = x0
    gmpc.obj_p_num['tvp'] = vertsplit(demand_pred_cl.to_numpy())

    gmpc.solve()

    # ::: Adding control for current step
    # ::::::::::::::::::::::::::::::::::::::
    control_vector = np.zeros(len(min_control))
    for el in range(len(control_vector)):
        control_vector[el] = random.uniform(min_control[el], max_control[el])  # TODO: modify. Now it is random

    # ::: Running the simulation
    start_time = time.time()

    # ::::::::::::::::::::::::::::::::::::::
    ctown.control_action(control_components, control_vector, t, ctown.wn.options.time.hydraulic_timestep)

    # ::: Run the simulation up to the current time step
    sim = wntr.sim.EpanetSimulator(ctown.wn)
    results = sim.run_sim()
    results.tankLevels = results.node['head'][nodeNames[0]]-tankEl
    results.energy = economics.pump_energy(results.link['flowrate'], results.node['head'], ctown.wn)

    # ::: Saving simulation output
    tempInpFile = "tempResults/tempInpFile_time%s.inp" % t
    ctown.wn.write_inpfile(tempInpFile)
    with open("tempResults/results_sim_time%s.pkl" % t, "wb") as f:
        pickle.dump(results, f)
        f.close()

    print('Total simulation time: %.3f s' % (time.time()-start_time))
