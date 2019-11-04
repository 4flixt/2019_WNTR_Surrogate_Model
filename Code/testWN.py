#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:08:27 2019

@author: ruizhiluo
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import wntr
import functools
import pickle
import wntr.network.controls as controls
import random
from random import uniform as rnd
import pdb


class testWN:
    '''
    this class contains attributes and methods to test the water network of
    c_town
    '''

    def __init__(self, filePath):
        '''
        constructor
        : type filePath: string
        '''
        self.__filePath = filePath
        self.wn = wntr.network.WaterNetworkModel(self.__filePath)
        #self.timeSim = timeSim

    def getNodeName(self):
        '''
        get the node name of .inp file
        : rtype: list contains three different node types
        '''
        node_names = self.wn.node_name_list
        tank_names = [tank for tank in node_names if tank.startswith("T")]
        reservoir_names = [reservoir for reservoir in node_names if reservoir.startswith("R")]
        junction_names = [junction for junction in node_names if junction.startswith("J")]
        return [tank_names, reservoir_names, junction_names]

    def getLinkName(self):
        '''
        get the link name of .ipn file
        : rtype: list contains different link types

        '''

        link_names = self.wn.link_name_list
        pump_names = [pump for pump in link_names if pump.startswith("PU")]
        pipe_names = [pipe for pipe in link_names if pipe.startswith("P") and not(pipe.startswith("PU"))]
        valve_names = [valve for valve in link_names if valve.startswith("V")]

        return [pump_names, pipe_names, valve_names]

    def checkSimuDuration(self, durations):
        if sum(durations)*3600 >= self.wn.options.time.duration:
            print("please re-enter the time durations")
            return False
        else:
            return True

    def WNTRsimulator(self, durations):
        '''
        set mode to DD: demand driven
        : type: int, simulation time
        : type: bool, check if the simulation needs to be restarted
        : type: int, number of time split
        : rtype: WNTR simulation, it contains:
            1. Timestamp
            2. Network Name
            3. Node Results
            4. Link Results

            Link and Node Results are both dictionaries of PANDA dataFrame

        '''
        if self.checkSimuDuration(durations):
            sim_results = []
            for i in range(len(durations)):
                if_add_controls = input('do you want to add rules or controls? Y/N :')
                if if_add_controls == 'Y':
                    link_name = input('please input the link name:')
                    open_or_close = input('set it open or close: (1 for open, 0 for close):')
                    node_name = input('please input the node name:')
                    above_or_below = input('above or below(> or <):')
                    value = input('value:')
                    new_control_name = input('define the name of the new control:')
                    self.addControls(link_name, open_or_close, node_name, above_or_below, value, new_control_name)


#                if if_change_controls == 'Y':
#                    control_name = input('the name of the changing control:')
#                    self.wn.get_control(control_name)
#                    self.wn.remove_control(control_name)
#                    self.addControls(link_name,open_or_close,node_name,above_or_below,value,control_name)

                start = time.time()
                sim_results.append(self.doSimulation(durations[i]))
                end = time.time()
                if i == 0:
                    print("the first {} (0 -> {}) hours simulation completed, it takes: {} s \n".format(durations[i], durations[i], durations[i], round((end-start), 2)))
                else:
                    print("the next {} ({} -> {})hours simulation completed, it takes: {} s \n".format(durations[i], durations[i], durations[i-1]+durations[i], round((end-start), 2)))

                option_remove_control = input('remove the added control?:')
                if option_remove_control == 'Y':
                    self.wn.remove_control(new_control_name)

        else:
            return 0
        self.wn.reset_initial_values()
        return sim_results

    def doSimulation(self, duration):
        self.wn.options.time.duration = duration*3600
        sim = wntr.sim.WNTRSimulator(self.wn, mode="DD")
        sim_results = sim.run_sim()
        return sim_results

    def addControls(self, link_name, attr, attr_value, node_name, above_or_below, cond_value, control_name):
        '''
        : type: link_name: string
        : type: attr: string. Which attribute to change. Typically "status"
        : type: attr_value: 1 or 0, 1 for open and 0 for close
        : type: node_name: string, the name of a node influenced by condition
        : type: above_or_below: string, either larger '>' or smaller '<'
        : type: cond_value: int, above what level and below what level
        : type: control_name: string, the name of the control added
        : rtype: bool, true of false
        '''

        '''
        set the actions of certain link
        '''
        control_link = self.wn.get_link(link_name)
        control_act = controls.ControlAction(control_link, attr, attr_value)

        '''
        set the conditions regarding the control link
        '''
        control_node = self.wn.get_node(node_name)

        control_cond = controls.ValueCondition(control_node, 'level', above_or_below, cond_value)
        new_control = controls.Control(control_cond, control_act)
        self.wn.add_control(control_name, new_control)

    def changeControls(self, control_name):
        control = self.wn.get_control(control_name)
        pass

    def simResultNode(self):
        '''
        get the results of node
        : type: node_name, string
        : rtype: dictionary
            (key-value pair)
            Keys contain Demand, Leak Demand, Pressure and Head
        '''

        node_dict = self.sim_results.node

        '''
        : type: pandas data frame, contains timestamp and values
            values of Demand, Leak Demand, Pressure and Head
        '''
        return node_dict

    def simResultLink(self):
        '''
        : rtype:dictionary
            (key-value pair)
            1. a dictionary (key-value pair)
            2. Keys contain flowrate, velocity and status
            3. status 0 : closed
               status 1 : open
        '''

        link_dict = self.sim_results.link
        return link_dict

    def patternMultipliers(self, pattern_name):
        multipliers = self.wn.get_pattern(pattern_name).multipliers
        return multipliers

    def baseline(self):
        generator = self.wn.junctions()
        baseline = []
        for j_name, j_object in generator:
            baseline.append(j_object.demand_timeseries_list[0].base_value)

        baseline = np.array(baseline).reshape(1, -1)
        return baseline

    def randomlyScaleMultipliers(self, maxChangePerc):
        # Added by Andrea - Randomly modifies the demand multipliers by max +/- maxChangePerc, with a probability of 20%
        for name, j in self.wn.junctions():
            changeRatio = rnd(0, 1)
            if j.demand_timeseries_list[0].pattern is not None:
                if changeRatio >= 0.8:
                    j.demand_timeseries_list[0].pattern.multipliers += (j.demand_timeseries_list[0].pattern.multipliers*np.random.uniform(low=-maxChangePerc, high=maxChangePerc, size=(len(j.demand_timeseries_list[0].pattern.multipliers))))

    def randomlyShiftMultipliers(self, maxChange):
        # Added by Andrea - Randomly modifies the demand multipliers by max +/- maxChange hours, with a probability of 20%
        patNames = self.wn.pattern_name_list
        for pat in patNames:
            changeRatio = rnd(0, 1)
            if changeRatio >= 0.8:
                currPat = self.wn.get_pattern(pat)
                rollRatio = random.randint(-maxChange, maxChange)
                currPat.multipliers = np.roll(currPat.multipliers, rollRatio)

    def changeMultipliers(self, pattern_name, random_gaussian=False, Up_Shift=False, Down_Shift=False):

        multipliers = self.patternMultipliers(pattern_name)
        #multipliers_max = multipliers.max(axis = 0)
        multipliers_min = multipliers.min(axis=0)

        if random_gaussian:
            # every multipliers should be ranged between 0 to 1.
            # add gaussian random noise with zero mean and a suitable variance
            # varaiance can be determined by
            variance = input('the variance of random gaussuian noise:')
            variance = float(variance)
            noise = np.random.normal(0, variance, len(multipliers))
            multipliers += noise
            return multipliers

        if Up_Shift:
            up_parameter = input('shift up to:')
            multipliers += up_parameter
            return multipliers

        if Down_Shift:
            down_parameter = input('shift down to:')
            if down_parameter >= multipliers_min:
                print("down shift bigger than the min value.Re-enter \n")
                down_parameter = input('shift down to:')
            multipliers -= down_parameter

            return multipliers

    def timeRange(self):
        duration = self.wn.options.time.duration
        pattern_timestep = self.wn.options.time.pattern_timestep
        pattern_start = self.wn.options.time.pattern_start
        timeSeries = np.arange(pattern_start, duration+pattern_timestep, pattern_timestep)
        return timeSeries

    def getPressure(self):
        '''
        returns a pandas dataframe
        '''
        return self.sim_results.node['pressure']


class plot_wn(testWN):

    def __init__(self, filePath):
        self.__filePath = filePath
        self.wn = wntr.network.WaterNetworkModel(self.__filePath)

    def plot_junction(self, title):

        node_attribute = self.getNodeName()
        wntr.graphics.plot_network(self.wn, node_attribute[2], None, title, node_size=20)

    def plot_tank(self, title):

        node_attribute = self.getNodeName()
        wntr.graphics.plot_network(self.wn, node_attribute[0], None, title)

    def plot_reservoir(self, title):

        node_attribute = self.getNodeName()
        wntr.graphics.plot_network(self.wn, node_attribute[1], None, title)

    def plot_valve(self, title):

        link_attribute = self.getLinkName()
        wntr.graphics.plot_network(self.wn, None, link_attribute[2], title)

    def plot_pump(self, title):

        link_attribute = self.getLinkName()
        wntr.graphics.plot_network(self.wn, None, link_attribute[0], title)

    def plot_pipe(self, title):

        link_attribute = self.getLinkName()
        wntr.graphics.plot_network(self.wn, None, link_attribute[1], title)

    def plot_pump_curve(self, pump_name):
        '''
        plot the char. curve of pump
        in c town 11 pumps
        '''

        pump = self.wn.get_link(pump_name)
        wntr.graphics.plot_pump_curve(pump)

    def plotPatternMultipliers(self, pattern_name):
        '''
        plot a specific patter in a certain simulation time
        '''
        multipliers = self.patternMultipliers(pattern_name)
        x = self.timeRange()[1:].reshape(-1, 1)/3600
        y = multipliers.reshape(-1, 1)
        plt.figure()
        plt.plot(x, y)
        plt.title("{} Curve".format(pattern_name))
        plt.xlabel("time(h)")
        plt.ylabel("Multipliers")

    def plotPatternChangedMultipliers(self, pattern_name, *args):
        '''
        '''
        x = self.timeRange()[1:].reshape(-1, 1)/3600
        y = np.array([arg for arg in args]).reshape(-1, 1)
        plt.figure()
        plt.plot(x, y)
        plt.title("{} changed curve".format(pattern_name))
        plt.xlabel("time(h)")
        plt.ylabel("Multipliers")

    def plotNode(self, node_key, node_name):
        node_dict = self.simResultNode()
        y = node_dict[node_key].loc[:, node_name].values.reshape(-1, 1)  # pressure, demand, head or leak demand
        x = node_dict[node_key].index.values.reshape(-1, 1)/3600
        plt.figure()
        plt.plot(x, y)
        plt.title("{} at {}".format(node_key, node_name))
        plt.xlabel("time(h)")
        plt.ylabel(node_key)
        plt.show()

    def plotLink(self, link_key, link_name):
        link_dict = self.simResultLink()
        y = link_dict[link_key].loc[:, link_name].values.reshape(-1, 1)  # flowrate, velocity or status
        x = link_dict[link_key].index.values.reshape(-1, 1)/3600
        plt.figure()
        plt.plot(x, y)
        plt.title("{} at {}".format(link_key, link_name))
        plt.xlabel("time(h)")
        plt.ylabel(link_name)
        plt.show()


class export_file(testWN):

    def __init__(self, filePath):
        self.__filePath = filePath
        self.wn = wntr.network.WaterNetworkModel(self.__filePath)


if __name__ == '__main__':
    inp_file = "c-town_true_network.inp"
    c_town = testWN(inp_file)
    c_town_plot = plot_wn(inp_file)
