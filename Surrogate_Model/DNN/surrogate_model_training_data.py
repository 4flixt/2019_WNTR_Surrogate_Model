import pandas as pd
import matplotlib.pyplot as plt
import pickle
import wntr
import wntr.metrics.economic as economics
import numpy as np
import pandas as pd
import pdb

import sys
sys.path.append('../../WNTR_Model/')
from testWN import testWN as twm


def get_data(file_list, narx_horizon, cluster_labels, pressure_factor, narx_input=True, narx_output=False, return_lists=False, inp_file=None):
    """
    This file is an outsourced implementation from the code in the jupyter notebook "dnn_surrogate_prototyping.ipynb".
    In this notebook we developed and tested the pre-processing pipeline on a single result file from WNTR/Epanet.
    We tested the training process on this limited data source but didn't use the resulting model for control.

    In the jupyter notebook "dnn_surrogate_full_model.ipynb" we use this outsourced implementation to pre-process multiple result files
    and create a large training data base. We furthermore trained the model in "dnn_surrogate_full_model.ipynb"
    and used the resulting models for the control task.
    """

    """
    --------------------------------------------------
    Get network informations
    --------------------------------------------------
    """
    if not inp_file:
        # For legacy reasons.
        inp_file = '../../WNTR_Model/c-town_true_network_simplified_controls.inp'
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

    # For Legacy reasons: Check the name of the pressure clusters in cluster_labels:
    if 'pressure' in cluster_labels.index.tolist():
        cl_ind_press = 'pressure'
    else:
        cl_ind_press = 'pressure_cluster'

    nn_input_list = []
    nn_output_list = []

    for i, file in enumerate(file_list):
        progress = (i+1)/len(file_list)
        print('{:.2f}% complete'.format(100*progress), end="\r")
        # Get results
        with open(file, 'rb') as f:
            results = pickle.load(f)

        """ Junctions """
        # Scale junction pressure
        junction_pressure_scaled = results.node['pressure'][node_names[2]]/pressure_factor.to_numpy()

        jun_cl_press = junction_pressure_scaled.groupby(cluster_labels.loc[cl_ind_press], axis=1)
        jun_cl_press_mean = jun_cl_press.mean()
        jun_cl_press_std = jun_cl_press.std()

        jun_cl_demand = results.node['demand'][node_names[2]].groupby(cluster_labels.loc[cl_ind_press], axis=1)
        jun_cl_demand_sum = jun_cl_demand.sum()

        #             quality from Results  | for all junctions | difference  | group by the quality cluster |  create the mean / standard deviation
        jun_cl_qual = results.node['quality'][node_names[2]].diff(axis=0).groupby(cluster_labels.loc[cl_ind_press], axis=1)
        qual_cl_qual_mean = jun_cl_qual.mean()
        qual_cl_qual_std = jun_cl_qual.std()

        """ Tanks """

        tank_press = results.node['pressure'][node_names[0]]
        # Subtract tank elevation from tank head to obtain tank_level
        tank_level = results.node['head'][node_names[0]]-nw_node_df[node_names[0]].loc['elevation']

        tank_qual = results.node['quality'][node_names[0]]

        """ Reservoirs """

        reservoir_press = results.node['pressure'][node_names[1]]
        reservoir_level = results.node['head'][node_names[1]]-nw_node_df[node_names[1]].loc['elevation']

        reservoir_qual = results.node['quality'][node_names[1]]

        """ Pumps """
        # Overview over all pumps in nw: nw_link_df[nw_link_df.keys()[nw_link_df.loc['link_type'] == 'Pump']]
        # https://wntr.readthedocs.io/en/latest/apidoc/wntr.network.elements.html#wntr.network.elements.HeadPump
        # The setting for head pump is alias for normalized speed (usually in the range of 0-1)
        head_pump_speed = results.link['setting'][nw_link_df.keys()[nw_link_df.loc['link_type'] == 'Pump']]
        head_pump_status = results.link['status'][nw_link_df.keys()[nw_link_df.loc['link_type'] == 'Pump']]
        head_pump_speed_corr = head_pump_speed

        pump_energy = economics.pump_energy(results.link['flowrate'], results.node['head'], ctown.wn)[link_names[0]]
        pump_energy /= 1000
        pump_energy.head(3)

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
        state_dict = {
            'tank_press': tank_press,
        }

        sys_states = pd.concat(state_dict.values(), axis=1, keys=state_dict.keys())

        input_dict = {  # 'head_pump_speed': head_pump_speed,
            'head_pump_speed': head_pump_speed_corr,
            'PRValve_dp': PRValve_dp,
            'TCValve_throttle': TCValve_throttle,
            'jun_cl_demand_sum': jun_cl_demand_sum}

        sys_inputs = pd.concat(input_dict.values(), axis=1, keys=input_dict.keys())

        aux_output_dict = {'pump_energy': pump_energy,
                           'jun_cl_press_mean': jun_cl_press_mean, }

        aux_outputs = pd.concat(aux_output_dict.values(), axis=1, keys=aux_output_dict.keys())

        """
        --------------------------------------------------
        Data Pre-Processing: 03 - Neural Network I/O
        --------------------------------------------------
        """

        nn_input_dict = {'sys_states': sys_states,
                         'sys_inputs': sys_inputs}

        nn_input = pd.concat(nn_input_dict.values(), axis=1, keys=nn_input_dict.keys(), names=['type', 'name', 'index'])

        if narx_input:
            arx_input = []
            for i in range(narx_horizon):
                arx_input.append(nn_input.shift(i, axis=0))

            arx_input = pd.concat(arx_input, keys=np.arange(narx_horizon), names=['NARX', 'type', 'name', 'index'], axis=1)
            nn_input = arx_input

        if narx_output:
            nn_output = nn_input.shift(-1, axis=0)
        else:
            sys_states_next = sys_states.shift(-1, axis=0)
            dsys_states = sys_states.diff(axis=0)
            dsys_states_next = dsys_states.shift(-1, axis=0)
            nn_output_dict = {  # 'sys_states': sys_states_next,
                'sys_states': dsys_states_next,
                'aux_outputs': aux_outputs}

            nn_output = pd.concat(nn_output_dict.values(), axis=1, keys=nn_output_dict.keys())

        # Filter nan:
        output_filter = nn_output.isnull().any(axis=1)
        if output_filter.any():
            nn_input = nn_input[~output_filter]
            nn_output = nn_output[~output_filter]

        input_filter = nn_input.isnull().any(axis=1)
        if input_filter.any():
            nn_input = nn_input[~input_filter]
            nn_output = nn_output[~input_filter]

        nn_input_list.append(nn_input)
        nn_output_list.append(nn_output)

    nn_input_concat = pd.concat(nn_input_list, axis=0)
    nn_output_concat = pd.concat(nn_output_list, axis=0)

    if return_lists:
        return nn_input_list, nn_output_list
    else:
        return nn_input_concat, nn_output_concat


def get_qual_data(file_list, narx_horizon, cluster_labels, pressure_factor, narx_input=True, narx_output=False, return_lists=False, inp_file=None, shift_x=False):
    """
    --------------------------------------------------
    Get network informations
    --------------------------------------------------
    """
    if not inp_file:
        inp_file = '../Code/c-town_true_network_simplified_controls.inp'
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

    n_clusters = 30

    nn_input_list = []
    nn_output_list = []

    for file in file_list:
        # Get results
        with open(file, 'rb') as f:
            results = pickle.load(f)

        """ Junctions """
        junction_qual = results.node['quality'][node_names[2]]

        jun_cl_qual = junction_qual.groupby(cluster_labels.loc['quality'], axis=1)
        jun_cl_qual_max_95 = jun_cl_qual.aggregate(lambda arr: np.percentile(arr, axis=1, q=95))

        jun_cl_demand = results.node['demand'][node_names[2]].groupby(cluster_labels.loc['quality'], axis=1)
        jun_cl_demand_sum = jun_cl_demand.sum()

        """ Tanks """

        tank_press = results.node['pressure'][node_names[0]]
        tank_qual = results.node['quality'][node_names[0]]

        """ Pumps """
        # Overview over all pumps in nw: nw_link_df[nw_link_df.keys()[nw_link_df.loc['link_type'] == 'Pump']]
        # https://wntr.readthedocs.io/en/latest/apidoc/wntr.network.elements.html#wntr.network.elements.HeadPump
        # The setting for head pump is alias for normalized speed (usually in the range of 0-1)
        head_pump_speed = results.link['setting'][nw_link_df.keys()[nw_link_df.loc['link_type'] == 'Pump']]
        head_pump_status = results.link['status'][nw_link_df.keys()[nw_link_df.loc['link_type'] == 'Pump']]
        head_pump_speed_corr = head_pump_speed*head_pump_status

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
        state_dict = {  # 'jun_cl_press_mean': jun_cl_press_mean,
            'jun_cl_qual_max_95': jun_cl_qual_max_95,
            'tank_qual': tank_qual,
        }

        sys_states = pd.concat(state_dict.values(), axis=1, keys=state_dict.keys())

        if shift_x:
            sys_states = sys_states.shift(1, axis=0)

        input_dict = {  # 'head_pump_speed': head_pump_speed,
            'tank_press': tank_press,
            'head_pump_speed': head_pump_speed_corr,
            'PRValve_dp': PRValve_dp,
            'TCValve_throttle': TCValve_throttle,
            'jun_cl_demand_sum': jun_cl_demand_sum}

        sys_inputs = pd.concat(input_dict.values(), axis=1, keys=input_dict.keys())

        """
        --------------------------------------------------
        Data Pre-Processing: 03 - Neural Network I/O
        --------------------------------------------------
        """

        nn_input_dict = {'sys_states': sys_states,
                         'sys_inputs': sys_inputs}

        nn_input = pd.concat(nn_input_dict.values(), axis=1, keys=nn_input_dict.keys(), names=['type', 'name', 'index'])

        if narx_input:
            arx_input = []
            for i in range(narx_horizon):
                arx_input.append(nn_input.shift(i, axis=0))

            arx_input = pd.concat(arx_input, keys=np.arange(narx_horizon), names=['NARX', 'type', 'name', 'index'], axis=1)
            nn_input = arx_input

        if narx_output:
            nn_output = nn_input.shift(-1, axis=0)
        else:
            sys_states_next = sys_states.shift(-1, axis=0)
            dsys_states = sys_states.diff(axis=0)
            dsys_states_next = dsys_states.shift(-1, axis=0)
            nn_output_dict = {  # 'sys_states': sys_states_next,
                'sys_states': dsys_states_next}

            nn_output = pd.concat(nn_output_dict.values(), axis=1, keys=nn_output_dict.keys())

        # Filter nan:
        output_filter = nn_output.isnull().any(axis=1)
        if output_filter.any():
            nn_input = nn_input[~output_filter]
            nn_output = nn_output[~output_filter]

        input_filter = nn_input.isnull().any(axis=1)
        if input_filter.any():
            nn_input = nn_input[~input_filter]
            nn_output = nn_output[~input_filter]

        nn_input_list.append(nn_input)
        nn_output_list.append(nn_output)

    nn_input_concat = pd.concat(nn_input_list, axis=0)
    nn_output_concat = pd.concat(nn_output_list, axis=0)

    if return_lists:
        return nn_input_list, nn_output_list
    else:
        return nn_input_concat, nn_output_concat
