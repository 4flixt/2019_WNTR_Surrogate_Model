#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for surrogate modelling of urban water networks
IFAC 2020

Authors: Andrea, Sergio

"""

# %% Surrogate modelling experiment on C-Town network
import os
# os.chdir(path="/Users/aco/tubCloud/Shared/WDN_SurrogateModels/Code/")

# Load modules
import pandas as pd
#import sklearn
import matplotlib as mpl
import matplotlib.pyplot as plt

import time
from testWN import testWN as twm
import wntr
import wntr.network.controls as controls
import pdb
import pickle
# %% ::: Load the calibrated C-Town water network model
inp_file = '../Networks/BWCNdata/c-town_true_network.inp'
#inp_file = '../Networks/Anytown.inp'
#inp_file = '../Networks/CTOWN.INP'
#inp_file = '../Networks/7 KL/KL.inp'
#inp_file = '../Networks/2 Extended Hanoi/Extended Hanoi.inp'
#inp_file = '../Networks/4 ZJ/ZJ.inp'

ctown = twm(inp_file)

# plt.figure()

# %%
# Graphical representation of the network
#plt.rcParams.update({'font.size': 20})
#wntr.graphics.plot_network(ctown.wn, title=ctown.wn.name, node_labels=False, link_labels=False, directed=True)
# fig = mpl.pyplot.gcf()
# fig.set_size_inches(10, 8)
# mpl.rcParams.update(mpl.rcParamsDefault)

# plt.show()
# %% ::: Modify demand patterns

# Get and display demand patterns
#df_pat = pd.DataFrame(columns = ctown.wn.junction_name_list)
# for name, j in ctown.wn.junctions():
#    base = j.demand_timeseries_list[0].base_value
#    pat = j.demand_timeseries_list[0].pattern
#    if pat is not None:
#        df_pat.loc[:,name] = base * j.demand_timeseries_list[0].pattern.multipliers
#
# get nodes with base-demand > 0
#d_juncs = df_pat.dropna(axis=1).columns
#
# display demand patterns
# df_pat.plot(legend=None)

# Varying demand patterns
# Scaling
ctown.randomlyScaleMultipliers(0.1)  # Input is the maximum possible percentage of change for each pattern value

# Shifting
ctown.randomlyShiftMultipliers(3)  # Input is the maximum time shift allowed (in hours)

# Representing modified demand patterns
#df_pat = pd.DataFrame(columns = ctown.wn.junction_name_list)
# for name, j in ctown.wn.junctions():
#    base = j.demand_timeseries_list[0].base_value
#    pat = j.demand_timeseries_list[0].pattern
#    if pat is not None:
#        df_pat.loc[:,name] = base * j.demand_timeseries_list[0].pattern.multipliers
#
# get nodes with base-demand > 0
#d_juncs = df_pat.dropna(axis=1).columns
#
# display demand patterns
# df_pat.plot(legend=None)

# %% ::: SIMULATION
# Simulate hydraulics
nDaysSim = 10
nHoursSim = 24*nDaysSim

# wn.options.time.duration = 3600*24*nDaysSim#*nYearsSim
ctown.wn.options.time.hydraulic_timestep = 60*15*4  # 15 min
ctown.wn.options.time.quality_timestep = 60*5  # 5 min
ctown.wn.options.quality.mode = 'AGE'
ctown.wn.options.results.energystr = True

start_time = time.time()

# TODO: Add controls
ctown.addControls('V45', 'setting', 10., 'T4', '<', 3, 'user_added_1')
ctown.addControls('V45', 'setting', 40., 'T4', '>', 4, 'user_added_2')

for i in range(nHoursSim):
    # Run the simulation for the next time step
    ctown.wn.options.time.duration = 3600*i
    sim = wntr.sim.EpanetSimulator(ctown.wn)
    results = sim.run_sim()

print('Total simulation time: %.3f s' % (time.time()-start_time))
with open("results.pkl", "wb") as f:
    pickle.dump(results, f)

plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(10, 8))

# %% ::: Output plotting
toPlotNames = ctown.getNodeName()

# Plotting demand patterns at nodes
df = results.node['demand']
toPlot = df[toPlotNames[2]]  # junctions
toPlot.plot(legend=None)

# Plotting water quality patterns at nodes
df = results.node['quality']
toPlot = df[toPlotNames[2]]  # junctions
toPlot.plot(legend=None)

# % Plotting median water quality at nodes after the simulation
df = results.node['quality']
df_med = df.median()
wntr.graphics.plot_network(ctown.wn, node_attribute=df_med[toPlotNames[0]], node_cmap='bwr', title='Tanks')  # tanks
wntr.graphics.plot_network(ctown.wn, node_attribute=df_med[toPlotNames[1]], node_cmap='bwr', title='Reservoirs')  # reservoir
wntr.graphics.plot_network(ctown.wn, node_attribute=df_med[toPlotNames[2]], node_cmap='bwr', title='Junctions')  # junctions

plt.show()
