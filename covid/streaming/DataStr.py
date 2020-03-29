
"""
Created: Mar 25 2020

@author: Daulet
"""
from sympy.core.cache import *

import time
import threading
import random
from functools import partial
import pandas as pd
import numpy as np
from multiprocessing import Process
import config
import os


import multiprocessing
from multiprocessing import Pool
from network_sim import Network
import os
from covid_simulator_upd import Node
import numpy as np
import random
import csv
import copy
import time
import matplotlib.pyplot as plt
import sys


def process_node(node):
    temp_node = node
    temp_node.stoch_solver()
    return temp_node

def simulate_network(params_node_, params_network, nodes_old, sim_iter, params_old):

    nodes_num = 17

    # load population data
    nodes_population = np.squeeze(np.array([1854556,2039379,738587,869603,633801,652314,1125297,678224,1078362,753804,1378554,872736,794165,1378504,1011511,554519,1981747])) # np.array([100000,200000,300000])

    # transition params
    tr_boxes = params_network[0]
    param_transition_box = np.zeros((17,3))

    for i, way in enumerate(tr_boxes): # air 0 rail 1 road 2
        for j, node in enumerate(way):
            status = int(node)
            param_transition_box[status, i] = 1

    param_transition_leakage = params_network[1]
    param_transition_scale = params_network[2]

    # load transition matrix
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

    transition_railway = list(csv.reader(open(os.path.join(THIS_FOLDER, 'tr_2.csv'))))
    transition_railway = np.array(transition_railway, dtype = np.float32)

    transition_airway = list(csv.reader(open(os.path.join(THIS_FOLDER, 'tr_1.csv'))))
    transition_airway = np.array(transition_airway, dtype = np.float32)

    transition_roadway = list(csv.reader(open(os.path.join(THIS_FOLDER, 'tr_3.csv'))))
    transition_roadway = np.array(transition_roadway, dtype = np.float32)

    transition_matrix_init = (transition_railway + transition_airway + transition_roadway).astype(int)

    tr_table = [transition_airway, transition_railway, transition_roadway]

    for j, tr in enumerate(tr_table):
        for i in range(17):
            tr[i, :] = tr[i, :]*param_transition_box[i,j]
            tr[:, i] = tr[i, :]*param_transition_box[i,j]

    transition_matrix = (transition_railway + transition_airway + transition_roadway).astype(int)
    transition_matrix = 0.5*transition_matrix * (param_transition_scale)

    for i in range(nodes_num):
        for j in range(nodes_num):
            if transition_matrix[i,j] < 0.01:
                transition_matrix[i,j] = transition_matrix_init[i,j]*param_transition_leakage # base data is for 24 days, tran_dt = 1/2

    transition_matrix = transition_matrix.astype(int)

    # create nodes
    params_node = params_node_

    if sim_iter > 0:
        # put values from prev iter
        params_node[10,:] = params_old[10,:]
        params_node[11,:] = params_old[11,:]
        params_node[12,:] = params_old[12,:]
        params_node[14,:] = params_old[14,:]
        
    params_node[11,:] = params_node_[11][0]*np.ones(nodes_num)
    params_node[12,:] = params_node_[12][0]*np.ones(nodes_num)
       
    
    nodes = [Node(params_node[:,i], sim_iter) for i in range(nodes_num)]

    for index, node in enumerate(nodes):
        node.check_init()
        node.create_states()
        node.indexes()
        node.define_state_arr()
        node.create_transitions()

    # create the network
    param_dt = nodes[0].param_dt
    param_dt_transition = 1/2                       # Sampling time for transition in days (1/2 corresponds to 12 hour)
    param_freq_transition  = 12
    param_sim_len = params_node[10]
    param_num_sim = int(param_sim_len[0] / param_dt) + 1       # Number of simulation

    param_static_names = ['Quarantined', 'Severe_Infected']         # States not subject to transition
    param_static_indices1 = [i for i, s in enumerate(nodes[0].states_name) if param_static_names[0] in s]
    param_static_indices2 = [i for i, s in enumerate(nodes[0].states_name) if param_static_names[1] in s]
    param_static_indices = np.squeeze(np.array([param_static_indices1 + param_static_indices2]))

    nodes_state_arr = np.zeros((param_num_sim, nodes_num, nodes[0].param_num_states))
    nodes_network = Network(nodes_num, nodes_population)

    if sim_iter > 0:
        pass
        # put values from prev iter
        nodes = nodes_network.update_node_states(nodes, nodes_old)

    start = time.time()

    pool = Pool()

    for ind in range(param_num_sim):
        nodes = list(pool.map(process_node, nodes))
        for index, node in enumerate(nodes):
            #node.stoch_solver()
            nodes_state_arr[ind, index, :] = node.states_x
        if ind % param_freq_transition == 0 :
            nodes = nodes_network.node_states_transition(nodes, transition_matrix, param_static_indices)      # Update the state of every node due to transition

    end = time.time()

    print("[INFO] Sim.time: {:.4f} sec".format(end - start))
    states_arr_plot = np.zeros((param_num_sim, nodes_num, 7))

    for iter in range(param_num_sim):
        for i_node in range(nodes_num):
            states_arr_plot[iter, i_node, 0] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_inf)
            states_arr_plot[iter, i_node, 1] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_exp)
            states_arr_plot[iter, i_node, 2] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_sin)
            states_arr_plot[iter, i_node, 3] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_qua)
            states_arr_plot[iter, i_node, 4] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_imm)
            states_arr_plot[iter, i_node, 5] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_sus)
            states_arr_plot[iter, i_node, 6] = nodes_state_arr[iter, i_node, -1]
    
     
    #states_arr_plotx = np.zeros((2,17,7))
    #states_arr_plotx[0, :, :] = states_arr_plot[24,:,:]
    #states_arr_plotx[1, :, :] = states_arr_plot[48,:,:]
    pool.close()
    return nodes, states_arr_plot, params_node, transition_matrix


class DataStream(threading.Thread):
    def __init__(self, callbackFunc, running):
        global new_plot
        threading.Thread.__init__(self)
        self.val = 5
        self.running = running
        self.callbackFunc = callbackFunc

    def run(self):

        count = config.sim_len/2
        df = pd.DataFrame()
        nodes_old = []
        state_sus = []
        state_inf = []
        state_sin = []
        state_qua = []
        state_imm = []
        state_dea = []
        try:
            while True:
                #config.new_plot_all = []
                while config.counter != count+1 and config.run_iteration:
                    for i in range(int(config.loop_num)):
                        config.flag_sim = 1

                        config.iteration_over = False
                        config.param_transition_box = []
                        config.param_transition_box.append(config.box1)
                        config.param_transition_box.append(config.box2)
                        config.param_transition_box.append(config.box3)
                        config.box_time.append(config.param_transition_box)
                        
                        config.params_node = np.vstack([config.param_beta_exp, config.param_qr,
                                        config.param_sir, config.param_eps_exp, config.param_eps_qua, config.param_eps_sev,config.param_hosp_capacity,
                                        config.param_gamma_mor1,config.param_gamma_mor2, config.param_gamma_im, config.param_sim_len,
                                        config.param_t_exp, config.param_t_inf, config.param_init_susceptible, config.param_init_exposed])

                        config.params_network = [config.param_transition_box, config.param_transition_leakage, config.param_transition_scale]

                        new_nodes, new_plot, new_params, tr_m = simulate_network(config.params_node, config.params_network, config.nodes_old, config.counter_func, config.params_old)
                        config.nodes_old = new_nodes
                        config.new_plot_all.append(new_plot) # new plot is 2*17*7 matrix not large
                        config.params_old = new_params.copy()
                        config.counter_func +=1
                        self.callbackFunc.doc.add_next_tick_callback(partial(self.callbackFunc.update, False))
                    config.flag_sim = 0
                    print('[INFO] Simulation is finished, press Simulation button for next iteration')
                    config.counter +=1
                    config.run_iteration = False
                    config.iteration_over = True
        except (KeyboardInterrupt, SystemExit):
            print('[INFO] Exiting the program. ')
            pass
