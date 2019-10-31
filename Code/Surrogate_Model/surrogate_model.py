import pandas as pd
import matplotlib.pyplot as plt
import pickle
import wntr
import numpy as np
import pandas as pd
import pdb

import sys
sys.path.append('../')
from testWN import testWN as twm

from tensorflow.compat.v1.keras import Sequential
#from keras.models import Sequential
from tensorflow.compat.v1.keras import layers, regularizers, optimizers


"""
--------------------------------------------------
Get network informations
--------------------------------------------------
"""

inp_file = '../../Networks/BWCNdata/c-town_true_network.inp'
ctown = twm(inp_file)
nw_node_df = pd.DataFrame(ctown.wn.nodes.todict())
nw_link_df = pd.DataFrame(ctown.wn.links.todict())

node_names = ctown.getNodeName()
link_names = ctown.getLinkName()


"""
--------------------------------------------------
Data Pre-Processing: 01
--------------------------------------------------
"""

# Get clusters
cluster_labels = pd.read_json('cluster_labels.json')
n_clusters = 30

# Get results
with open('../results.pkl', 'rb') as f:
    results = pickle.load(f)


""" Junctions """
#                  pressure from Results  | for all junctions | group by the pressure cluster |  create the mean / standard deviation
jun_cl_press_mean = results.node['pressure'][node_names[2]].groupby(cluster_labels.loc['pressure'], axis=1).mean()
jun_cl_press_std = results.node['pressure'][node_names[2]].groupby(cluster_labels.loc['pressure'], axis=1).std()
# index = pd.MultiIndex.from_tuples(list(zip(*[['press_std']*n_clusters, jun_cl_press_std.columns.tolist()])))
# jun_cl_press_std.columns = index

#                  quality from Results  | for all junctions | difference  | group by the quality cluster |  create the mean / standard deviation
dqual_cl_press_mean = results.node['quality'][node_names[2]].diff(axis=0).groupby(cluster_labels.loc['quality'], axis=1).mean()
dqual_cl_press_std = results.node['quality'][node_names[2]].diff(axis=0).groupby(cluster_labels.loc['quality'], axis=1).std()

""" Tanks """

tank_press = results.node['pressure'][node_names[0]]
# Subtract tank elevation from tank head to obtain tank_level
tank_level = results.node['head'][node_names[0]]-nw_node_df[node_names[0]].loc['elevation']

tank_qual = results.node['quality'][node_names[0]]


""" Reservoirs """

reservoir_press = results.node['pressure'][node_names[1]]
reservoir_level = results.node['head'][node_names[1]]-nw_node_df[node_names[1]].loc['elevation']

reservoir_qual = results.node['quality'][node_names[1]]
reservoir_level

""" Pumps """
# Overview over all pumps in nw: nw_link_df[nw_link_df.keys()[nw_link_df.loc['link_type'] == 'Pump']]
# https://wntr.readthedocs.io/en/latest/apidoc/wntr.network.elements.html#wntr.network.elements.HeadPump
# The setting for head pump is alias for normalized speed (usually in the range of 0-1)
head_pump_speed = results.link['setting'][nw_link_df.keys()[nw_link_df.loc['link_type'] == 'Pump']]


""" Valves """
# Overview over all valves in nw: nw_link_df[nw_link_df.keys()[nw_link_df.loc['link_type'] == 'Valve']]
# https://wntr.readthedocs.io/en/latest/apidoc/wntr.network.elements.html#wntr.network.elements.TCValve
# TODO: Only one valve type in the future?
PRValve_dp = results.link['setting'][nw_link_df.keys()[nw_link_df.loc['valve_type'] == 'PRV']]
TCValve_throttle = results.link['setting'][nw_link_df.keys()[nw_link_df.loc['valve_type'] == 'TCV']]


"""
--------------------------------------------------
Data Pre-Processing: 02 - Create states + inputs
--------------------------------------------------
"""
# TODO: Reservoir?
state_dict = {'jun_cl_press_mean': jun_cl_press_mean,
              'jun_cl_press_std': jun_cl_press_std,
              'dqual_cl_press_mean': dqual_cl_press_mean,
              'dqual_cl_press_std': dqual_cl_press_std,
              'tank_press': tank_press,
              'tank_level': tank_level,
              'tank_qual': tank_qual,
              # 'reservoir_press': reservoir_press,
              # 'reservoir_level': reservoir_level,
              # 'reservoir_qual': reservoir_qual,
              }

sys_states = pd.concat(state_dict.values(), axis=1, keys=state_dict.keys())


input_dict = {'head_pump_speed': head_pump_speed,
              'PRValve_dp': PRValve_dp,
              'TCValve_throttle': TCValve_throttle}

sys_inputs = pd.concat(input_dict.values(), axis=1, keys=input_dict.keys())

"""
--------------------------------------------------
Data Pre-Processing: 03 - Neural Network I/O
--------------------------------------------------
"""

dstates = sys_states.diff(axis=0)
dstates_next = sys_states.shift(-1, axis=0)

nn_input_dict = {'sys_states': sys_states,
                 'sys_inputs': sys_inputs}

nn_input = pd.concat(nn_input_dict.values(), axis=1, keys=nn_input_dict.keys())

nn_output = dstates_next

"""
--------------------------------------------------
Neural Network: 02 - Create Model
--------------------------------------------------
"""
n_layer = 3
n_units = 50
l1_regularizer = 0

model_param = {}
model_param['n_in'] = 10
model_param['n_out'] = 10
model_param['n_units'] = (n_layer)*[n_units]
model_param['activation'] = (n_layer) * ['tanh']
model_param['l1_regularizer'] = (n_layer) * [l1_regularizer]


model = Sequential()


for i in range(len(model_param['n_units'])):
    if i == 0:
        input_shape = model_param['n_in']
    else:
        input_shape = model_param['n_units'][i - 1]

    model.add(layers.Dense(units=model_param['n_units'][i],
                           activation=model_param['activation'][i],
                           kernel_regularizer=regularizers.l1(model_param['l1_regularizer'][i])))

model.add(layers.Dense(units=model_param['n_out']))
