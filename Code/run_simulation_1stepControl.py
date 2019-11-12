#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
#import sklearn
import numpy as np

import time
from testWN import testWN as twm
import wntr
import wntr.network.controls as controls
import wntr.metrics.economic as economics

import pickle
import random
import inefficient

#%% ::: Loading .inp file
inp_file = '../Code/c-town_true_network_simplified_controls.inp'
ctown = twm(inp_file)

#%% ::: Setting up time and options for simulation
nDaysSim = 30
nHourDay =24
simTimeSteps =nDaysSim*nHourDay # Sampling frequency of 15 min
    
controlTvary = 8 # Number of time steps for varying controls (if set to 8 -> controls are stable for 2 hours)

ctown.wn.options.time.hydraulic_timestep = 3600  # 1 hr
ctown.wn.options.time.quality_timestep = 3600  # 1 hr
ctown.wn.options.time.report_timestep = 3600
ctown.wn.options.quality.mode = 'AGE'
ctown.wn.options.results.energystr = True  
ctown.wn.options.time.duration = 0

# ::: Getting tank elevations
tankEl =[]
for tank, name in ctown.wn.tanks():
    tankEl.append(name.elevation)
np.array(tankEl)
nodeNames = ctown.getNodeName()
    
# ::: Setting upper and lower bounds to control elements
control_components = ctown.wn.pump_name_list + ctown.wn.valve_name_list
min_control = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.]) # Lower bondary for controls
max_control = np.array([2., 2., 2., 2., 2., 600.0, 600.0, 600.0, 70. ]) # Upper bondary for controls

#%% ::: Simulation with updated controls at each time step
for t in range(simTimeSteps):

    # ::: Initializing random seed
    random.seed(t)

    # ::: Loading .inp file from previous step
    if t>0:
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
    
    # ::: Adding control for current step
    # ::::::::::::::::::::::::::::::::::::::
    control_vector = np.zeros(len(min_control))
    for el in range(len(control_vector)):
        control_vector[el] = random.uniform(min_control[el], max_control[el])   #TODO: modify. Now it is random      
    # ::::::::::::::::::::::::::::::::::::::    
    ctown.control_action(control_components, control_vector, t, ctown.wn.options.time.hydraulic_timestep)
    
    # Forecasting water demand for the next k steps
    k = 24 # Time horizon for water demand prediction
    startT = t+1
    demand_pred = ctown.forecast_demand_gnoise(k, startT*ctown.wn.options.time.hydraulic_timestep, ctown.wn.options.time.hydraulic_timestep)
    
    # ::: Running the simulation
    start_time = time.time()
         
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
   