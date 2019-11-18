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


inp_file = '../Code/c-town_true_network_simplified_2speed.inp'
#inp_file = '../Code/c-town_true_network_initialOpen.inp'
ctown = twm(inp_file)

# # Setting up multiple simulations for AI training

# In[4]:

# ::: Setting up time for simulation
nDaysSim = 30
nHourDay =24
simTimeSteps =nDaysSim*nHourDay # Sampling frequency of 1 hr    
 
# ::: Getting tank elevations
tankEl =[]
for tank, name in ctown.wn.tanks():
    tankEl.append(name.elevation)
np.array(tankEl)
nodeNames = ctown.getNodeName()
  
# ::: Running multiple simulations
for i in range(150): # Number of simulations
    print(i)

    # ::: Initializing random seed
    random.seed(i)
    
    # ::: Resetting initial water networks settings
    ctown = twm(inp_file)
    ctown.wn.reset_initial_values()
              
    # Setting simulation options
    ctown.wn.options.time.hydraulic_timestep = 3600  # 1 hr
    ctown.wn.options.time.quality_timestep = 3600  # 1 hr
    ctown.wn.options.time.report_timestep = 3600
    ctown.wn.options.quality.mode = 'AGE'
    ctown.wn.options.results.energystr = True  
    ctown.wn.options.time.duration = simTimeSteps*3600
  
    # ::: Generating random initial states (tank levels)
    for k, name in ctown.wn.tanks():
        name.init_level =random.uniform(name.min_level, name.max_level)
        
    # ::: Generating random demands
    # Scaling
    ctown.randomlyScaleMultipliers(0.1)  # Input is the maximum possible percentage of change for each pattern value
    #
    # Shifting
    ctown.randomlyShiftMultipliers(3)  # Input is the maximum time shift allowed (in hours)

    # ::: Running the simulation
    start_time = time.time()
               
    # Setting up dataframe for saving link settings
    #results_setting=pd.DataFrame(columns=control_components)
         
    # ::: Run the simulation for the next time step
    sim = wntr.sim.EpanetSimulator(ctown.wn)
    results = sim.run_sim()
    results.tankLevels = results.node['head'][nodeNames[0]]-tankEl
    results.energy = economics.pump_energy(results.link['flowrate'], results.node['head'], ctown.wn)
    
    # ::: Saving simulation output
    with open("results_sim_%s_noControls_2speed.pkl" % i, "wb") as f:
        pickle.dump(results, f)
        f.close()

    print('Total simulation time: %.3f s' % (time.time()-start_time))