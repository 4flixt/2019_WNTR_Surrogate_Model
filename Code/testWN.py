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
import wntr.metrics.hydraulic as hydraulics
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



# ::: Added
    def randomlyScaleMultipliers(self, maxChangePerc):
        # Randomly modifies the demand multipliers by max +/- maxChangePerc, with a probability of 20%
        #for name, j in self.wn.junctions():
        patNames = self.wn.pattern_name_list
        for pat in patNames:
            changeRatio = rnd(0, 1)
            if changeRatio >= 0.8:
                currPat = self.wn.get_pattern(pat)
                currPat.multipliers += (currPat.multipliers*np.random.uniform(low=-maxChangePerc, high=maxChangePerc, size=(len(currPat.multipliers))))


    def randomlyShiftMultipliers(self, maxChange):
        # Randomly modifies the demand multipliers by max +/- maxChange hours, with a probability of 20%
        patNames = self.wn.pattern_name_list
        for pat in patNames:
            changeRatio = rnd(0, 1)
            if changeRatio >= 0.8:
                currPat = self.wn.get_pattern(pat)
                rollRatio = random.randint(-maxChange, maxChange)
                currPat.multipliers = np.roll(currPat.multipliers, rollRatio)

    def control_action(self, control_components, control_vector, currTime, timeStepSize):
        '''
        add control action to the current simulation step
        '''
        for j in range(len(control_components)): # Iterate over controls
            currComp = self.wn.get_link(control_components[j])
            cond = controls.SimTimeCondition(self.wn, None, currTime*timeStepSize)
            if np.isin(control_components[j], self.wn.head_pump_name_list): #is a head pump
                act = controls.ControlAction(currComp, 'setting', control_vector[j])
                act1 = controls.ControlAction(currComp, 'status', 1)
                ctrl = controls.Rule(cond,[act1,act], name="control_components_%s_%s" % (control_components[j],currTime))
            elif np.isin(control_components[j], self.wn.valve_name_list): #is a valve
                act = controls.ControlAction(currComp, 'setting', control_vector[j])
                ctrl = controls.Control(cond,act, name="control_components_%s_%s" % (control_components[j],currTime))
            self.wn.add_control("control_components_%s_%s" % (control_components[j],currTime), ctrl)

    def forecast_demand_gnoise(self, k, startT, timeStep, lBound, uBound):
        '''
        forecast node demand starting from time startT for the next k timesteps
        '''
        forecasted_demand = hydraulics.expected_demand(self.wn, start_time =startT, end_time = timeStep*(k-1)+startT, timestep = timeStep)
        # Adding noise
        # noise = np.random.normal(0,np.mean(np.std(forecasted_demand))*0.10,np.shape(forecasted_demand))
        noise = np.random.uniform(lBound,uBound,np.shape(forecasted_demand))
        forecasted_demand = np.multiply(noise,forecasted_demand)
        return forecasted_demand
