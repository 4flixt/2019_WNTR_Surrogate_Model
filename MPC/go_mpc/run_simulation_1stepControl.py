#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
#import sklearn
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process

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
nn_model_path = './model/004_man_6x40_both_datasets/'
cluster_labels = pd.read_json(nn_model_path+'cluster_labels_dt1h.json')
pressure_factor = pd.read_json(nn_model_path+'pressure_factor_dt1h.json')

# Create controller:
n_horizon = 10
gmpc = go_mpc(n_horizon)

# Plotting function:


def plot_pred(gmpc, results, time_arr):
    # pdb.set_trace()
    plt.close('all')
    fig, ax = plt.subplots(3, 1)
    results.tankLevels.plot(ax=ax[0], legend=False)
    x = horzcat(*gmpc.obj_x_num['x']).T.full()
    u_pump = horzcat(*gmpc.obj_x_num['u']).T.full()[:, :5]
    p_min = horzcat(*gmpc.obj_aux_num['nl_cons', :, 'jun_cl_press_min']).T.full()
    ax[0].set_prop_cycle(None)
    ax[0].plot(time_arr.reshape(-1, 1), x, '--')
    ax[0].set_xlim(0, time_arr[-1])

    #results.tankLevels.plot(ax=ax[1], legend=False)
    ax[1].plot(time_arr[:-1], u_pump, '--')

    ax[2].plot(time_arr[:-1], p_min, '--')
    plt.show()


"""
---------------------------------------------------
Initialize simlation:
---------------------------------------------------
"""
# control_vector = np.zeros(9)
# # ::::::::::::::::::::::::::::::::::::::
# ctown.control_action(control_components, control_vector, 0, ctown.wn.options.time.hydraulic_timestep)
#
# # ::: Run the simulation up to the current time step
# sim = wntr.sim.EpanetSimulator(ctown.wn)
# results = sim.run_sim()
# results.tankLevels = results.node['head'][nodeNames[0]]-tankEl
# results.energy = economics.pump_energy(results.link['flowrate'], results.node['head'], ctown.wn)

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
    dt_hyd = ctown.wn.options.time.hydraulic_timestep
    addNoise = False
    demand_pred = ctown.forecast_demand_gnoise(n_horizon, startT*dt_hyd, dt_hyd, addNoise)

    time_arr = np.arange(dt_hyd*t, dt_hyd*(t+n_horizon+1), dt_hyd)-dt_hyd
    # Cluster demand:
    demand_pred_cl = demand_pred.groupby(cluster_labels.loc['pressure_cluster'], axis=1).sum()

    """
    ---------------------------------------------------
    Get current state:
    ---------------------------------------------------
    """
    if t == 0:
        x0 = np.array([3, 3, 2.5, 5.2, 1, 0.5, 2.5])
    else:
        x0 = results.tankLevels.iloc[t-1].to_numpy()
        print(results.tankLevels.iloc[t-1])

    """
    ---------------------------------------------------
    Setup (for current time) and Run controller
    ---------------------------------------------------
    """
    # Setup controller for time t:
    gmpc.obj_p_num['x_0'] = x0
    gmpc.obj_p_num['tvp'] = vertsplit(demand_pred_cl.to_numpy())

    gmpc.solve()
    control_vector = gmpc.obj_x_num['u', 0].full().flatten()

    if t >= 1:
        if t >= 2:
            p.terminate()
        p = Process(target=plot_pred, args=(gmpc, results, time_arr))
        p.start()

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
