#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
#import sklearn
import numpy as np

import time
from testWN import testWN as twm
import wntr
import wntr.network.controls as controls
import wntr.metrics.economic as economics
import pdb
import pickle
import random


# In[3]:


inp_file = '../Code/c-town_true_network_simplified_controls.inp'
ctown = twm(inp_file)

# # Setting up multiple simulations for AI training

# In[4]:

# ::: Setting up time for simulation
nDaysSim = 7
nHourDay =24
simTimeSteps =nDaysSim*nHourDay*4 # Sampling frequency of 15 min
    
controlTvary = 8 # Number of time steps for varying controls (if set to 8 -> controls are stable for 2 hours)
    
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
    
# ::: Running multiple simulations
for i in range(150): # Number of simulations
    print(i)

    # ::: Initializing random seed
    random.seed(i)
    
    # ::: Resetting initial water networks settings
    ctown = twm(inp_file)
    ctown.wn.reset_initial_values()
              
    # Setting simulation options
    ctown.wn.options.time.hydraulic_timestep = 60*15  # 15 min
    ctown.wn.options.time.quality_timestep = 60*15  # 15 min
    ctown.wn.options.time.report_timestep = 60*15
    ctown.wn.options.quality.mode = 'AGE'
    ctown.wn.options.results.energystr = True  
    ctown.wn.options.time.duration = simTimeSteps*900
  
    # ::: Generating random initial states (tank levels)
    for k, name in ctown.wn.tanks():
        name.init_level =random.uniform(name.min_level, name.max_level)
        
    # ::: Generating random demands
    # Scaling
    ctown.randomlyScaleMultipliers(0.1)  # Input is the maximum possible percentage of change for each pattern value
    #
    # Shifting
    ctown.randomlyShiftMultipliers(3)  # Input is the maximum time shift allowed (in hours)
    
    # ::: Generating initial random controls
    control_vector = np.zeros(len(min_control))
    for el in range(len(control_vector)):
        control_vector[el] = random.uniform(min_control[el], max_control[el])   
    control_matrix = np.tile(control_vector, (simTimeSteps,1))
    #
    # Randomly changing controls every few simulation steps
    for t in range(simTimeSteps): # Iterate over simulation steps
        if (t % controlTvary)==0: 
            temp=np.random.randint(0,len(min_control), size = np.int(np.floor(len(min_control)/2)))
            temp=np.unique(temp)
            for nControl in range(len(temp)):
                control_matrix[t:,temp[nControl]] = random.uniform(min_control[temp[nControl]], max_control[temp[nControl]])
    #
    # Removing all existing controls
    for j in range(len(control_components)): # Iterate over controls
        if np.isin(control_components[j], ctown.wn.control_name_list):
            ctown.wn.remove_control(control_components[j])
    #
    # Add controls to ctown 
    for j in range(len(control_components)): # Iterate over controls
        currComp = ctown.wn.get_link(control_components[j])
        for t in range(simTimeSteps):
            cond = controls.SimTimeCondition(ctown.wn, None,t*900)
            if np.isin(control_components[j], ctown.wn.head_pump_name_list): #is a head pump
                act = controls.ControlAction(currComp, 'setting', control_matrix[t,j])
                act1 = controls.ControlAction(currComp, 'status', 1)
                ctrl = controls.Rule(cond,[act1,act], name="control_components_%s_%s" % (control_components[j],t))
            elif np.isin(control_components[j], ctown.wn.valve_name_list): #is a valve
                act = controls.ControlAction(currComp, 'setting', control_matrix[t,j])
                ctrl = controls.Control(cond,act, name="control_components_%s_%s" % (control_components[j],t))
            ctown.wn.add_control("control_components_%s_%s" % (control_components[j],t), ctrl)

    # ::: Running the simulation
    start_time = time.time()
               
    # Setting up dataframe for saving link settings
    #results_setting=pd.DataFrame(columns=control_components)
         
    # ::: Run the simulation for the next time step
    sim = wntr.sim.EpanetSimulator(ctown.wn)
    results = sim.run_sim()
    results.tankLevels = results.node['head'][nodeNames[0]]-tankEl
    
    # ::: Saving simulation output
    with open("results_sim_%s_try.pkl" % i, "wb") as f:
        pickle.dump(results, f)
        f.close()

    print('Total simulation time: %.3f s' % (time.time()-start_time))
