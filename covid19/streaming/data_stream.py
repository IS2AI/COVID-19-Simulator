
"""
Created: Mar 25 2020

@author: Daulet
"""
import os
import sys

import time
import threading
import random
from functools import partial
import pandas as pd
import numpy as np
import csv
import copy
import time

import config
from multiprocessing import Pool
from network_transition import Network
from node_simulation import Node

# process each node using the multiprocessing
def process_node(node):
    temp_node = node
    temp_node.stoch_solver()
    return temp_node

def simulate_network(params_node_, nodes_old, sim_iter, transition_matrix, init_sus):

    nodes_num = 17
    # load population data
    nodes_population = init_sus

    # create nodes
    params_node = params_node_

    params_node[11,:] = params_node_[11][0]*np.ones(nodes_num)
    params_node[12,:] = params_node_[12][0]*np.ones(nodes_num)

    nodes = [Node(params_node[:,i]) for i in range(nodes_num)]

    for index, node in enumerate(nodes):
        node.check_init()
        node.create_states()
        node.indexes()
        node.create_transitions()

    # create the network
    param_dt = nodes[0].param_dt                    # Sampling time for transition
    param_dt_transition = 1/2                       # Sampling time for transition in days (1/2 corresponds to 12 hour)
    param_freq_transition  = 12
    param_sim_len = params_node[10]
    param_num_sim = int(param_sim_len[0] / param_dt)      # Number of simulation

    # Find states not subject to transition
    param_static_names = ['Quarantined', 'Severe_Infected', 'Dead', 'Isolated']
    param_static_indices1 = [i for i, s in enumerate(nodes[0].states_name) if param_static_names[0] in s]
    param_static_indices2 = [i for i, s in enumerate(nodes[0].states_name) if param_static_names[1] in s]
    param_static_indices3 = [i for i, s in enumerate(nodes[0].states_name) if param_static_names[2] in s]
    param_static_indices4 = [i for i, s in enumerate(nodes[0].states_name) if param_static_names[3] in s]

    param_static_indices = np.squeeze(np.array([param_static_indices1 + param_static_indices2 + param_static_indices3 + param_static_indices4]))

    nodes_state_arr = np.zeros((param_num_sim, nodes_num, nodes[0].param_num_states))
    nodes_network = Network(nodes_num, nodes_population)

    if sim_iter > 0:
        if config.is_loaded == True:
            # copy state values from prev iter (loading)
            for index, node in enumerate(nodes):
                node.states_x = np.array(config.last_state_list[index]).astype(np.float)
            config.is_loaded = False
        else:
            # copy values from prev iteration
            nodes = nodes_network.update_node_states(nodes, nodes_old)
    else:
        # state values are initialized from scratch
        pass

    #pool = Pool()
    for ind in range(param_num_sim):
        #nodes = list(pool.map(process_node, nodes))
        for index, node in enumerate(nodes):
            # process node simulation
            node.stoch_solver()
            nodes_state_arr[ind, index, :] = node.states_x
        if ind % param_freq_transition == 0 :
            # Update the state of every node due to transition
            nodes = nodes_network.node_states_transition(nodes, transition_matrix, param_static_indices)

    #pool.close()
    # save the values to buffer
    states_arr_plot = np.zeros((param_num_sim, nodes_num, 8))

    for iter in range(param_num_sim):
        for i_node in range(nodes_num):
            states_arr_plot[iter, i_node, 0] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_inf)
            states_arr_plot[iter, i_node, 1] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_exp)
            states_arr_plot[iter, i_node, 2] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_sin)
            states_arr_plot[iter, i_node, 3] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_qua)
            states_arr_plot[iter, i_node, 4] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_imm)
            states_arr_plot[iter, i_node, 5] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_sus)
            states_arr_plot[iter, i_node, 6] = nodes_state_arr[iter, i_node, -1]
            states_arr_plot[iter, i_node, 7] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_iso)

    return nodes, states_arr_plot, params_node

class DataStream(threading.Thread):
    def __init__(self, callbackFunc, running):
        threading.Thread.__init__(self)
        self.running = running
        self.callbackFunc = callbackFunc

    def run(self):
        try:
            while True:
                # wait for signal from web interface
                if config.run_iteration:    # if RUN Simulation button is pressed
                    #  start simulation
                    print('[INFO] Simulation started.')
                    for i in range(int(config.loop_num)):
                        start = time.time()
                        # set simulation running flag
                        config.flag_sim = 1

                        # save the params to buffer
                        config.box_time = np.dstack((config.box_time, config.param_transition_table))

                        arr_for_save = np.dstack((config.param_init_exposed, config.param_beta_exp, config.param_qr, config.param_sir, config.param_hosp_capacity,
                                                config.param_gamma_mor1, config.param_gamma_mor2, config.param_gamma_im, config.param_eps_exp,
                                                config.param_eps_qua, config.param_eps_sev, config.param_t_exp, config.param_t_inf, config.param_transition_leakage,
                                                 config.param_transition_scale))
                        config.arr_for_save = np.vstack([config.arr_for_save, arr_for_save ])

                        # update the params for simulation
                        config.params_node = np.vstack([config.param_beta_exp, config.param_qr,
                                                        config.param_sir, config.param_eps_exp, config.param_eps_qua, config.param_eps_sev,config.param_hosp_capacity,
                                                        config.param_gamma_mor1,config.param_gamma_mor2, config.param_gamma_im, config.param_sim_len,
                                                        config.param_t_exp, config.param_t_inf, config.param_init_susceptible, config.param_init_exposed])

                        # RUN the netwoork transition
                        new_nodes, new_plot, new_params = simulate_network(config.params_node, config.nodes_old, config.counter_func,
                                                                                    config.transition_matrix, config.param_init_susceptible)
                        config.nodes_old = new_nodes

                        # append the new results to list
                        config.new_plot_all.append(new_plot)
                        self.callbackFunc.doc.add_next_tick_callback(partial(self.callbackFunc.update, False))
                        config.counter_func +=1
                        end = time.time()
                        #print("[INFO] Sim.time: {:.4f} sec".format(end - start))
                        print("[INFO] Step: {}/{}, Elapsed time: {:.4f} sec".format(i+1, int(config.loop_num), end - start))

                    config.flag_sim = 0
                    print('[INFO] Simulation finished, press Run Simulation button for next iteration.')
                    config.run_iteration = False

                # Loading prev experiments
                elif config.load_iteration: # if Load Results button is pressed
                    config.flag_sim = 1
                    # plot results
                    for i in range(1,config.counter_load+1):
                        temp_plot = np.stack((config.new_plot[i,:,:], config.new_plot[i,:,:]),axis=0)
                        config.new_plot_all.append(temp_plot)
                        self.callbackFunc.doc.add_next_tick_callback(partial(self.callbackFunc.update, False))
                        config.counter_func +=1

                    config.flag_sim = 0
                    config.load_iteration = False
                    print('[INFO] Loading the previous results ..')

        except (KeyboardInterrupt, SystemExit):
            print('[INFO] Exiting the program.. ')
            exit
