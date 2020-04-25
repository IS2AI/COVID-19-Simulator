
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

# test the population

def test_population(init_sus, nodes, num_test, prev_rate, test_sens, test_spec):


    if not nodes and not config.last_state_list:
        print('[INFO] Test Error: Run simulation or load previous experiment')

    else:
        print('[INFO] Testing started. ')

        states_x = []
        states_name = []
        t_pos = 0
        t_neg = 0

        if nodes:
            for i in range(config.nodes_num):
                states_x.append(nodes[i].states_x)
            states_name.append(nodes[i].states_name)

        else:
            for i in range(config.nodes_num):
                states_x.append(np.array(config.last_state_list[i]).astype(np.float))
            states_name = config.load_states_name

        # randomly test the population of the node
        states_indices = []
        states_indices_inf = []
        states_indices_rec = []
        states_indices_sev = []

        #states_names_all = ['Susceptible', 'Vaccinated', 'Exposed', 'Quarantined','Infected', 'Isolated', 'Severe_Infected', 'Vaccination_Immunized', 'Maternally_Immunized', 'Recovery_Immunized'
        states_names = ['Susceptible', 'Vaccinated', 'Exposed', 'Infected_', 'Vaccination_Immunized', 'Maternally_Immunized', 'Recovery_Immunized']
        states_names_inf = ['Infected_']
        states_names_rec = ['Susceptible']
        states_names_sev = ['Severe_Infected']

        for k in range(len(states_names)):
            states_indices_temp = [i for i, s in enumerate(states_name[0]) if states_names[k] in s]
            states_indices.extend(states_indices_temp)

        for k in range(len(states_names_inf)):
            states_indices_temp = [i for i, s in enumerate(states_name[0]) if states_names_inf[k] in s]
            states_indices_inf.extend(states_indices_temp)

        for k in range(len(states_names_rec)):
            states_indices_temp = [i for i, s in enumerate(states_name[0]) if states_names_rec[k] in s]
            states_indices_rec.extend(states_indices_temp)

        for k in range(len(states_names_sev)):
            states_indices_temp = [i for i, s in enumerate(states_name[0]) if states_names_sev[k] in s]
            states_indices_sev.extend(states_indices_temp)

        # remove severe_infected state
        states_indices = [x for x in states_indices if x not in states_indices_sev]
        states_indices_inf = [x for x in states_indices_inf if x not in states_indices_sev]

        #print('ss -- ', states_indices, '\n', states_indices_inf,'\n', states_indices_sev)

        # for each node of interest for testing
        for i in range(config.nodes_num):

            # check node has testing
            if num_test[i] > 0:
                tpos_count = 0
                tneg_count = 0
                # find the prevalance_rate
                # later implement switch
                if config.param_prev_mode == 0 :
                    prevalance_rate = prev_rate
                else:
                    prevalance_rate = config.param_prev_auto[i]

                #print('prevalance_rate ' , prevalance_rate)
                # initialize the state probabilities
                states_prob = np.zeros(len(states_indices))
                # find the prob that state can be selected
                for index in range(len(states_indices)):
                    states_prob[index] = int(float(states_x[i][states_indices[index]])) / init_sus[i]
                states_prob /= states_prob.sum()
                #print('sum+', states_prob)
                # sample the num_test population of the node:
                # option 1: randomly
                for k in range(int(num_test[i])):

                    # find the tpos_i
                    is_infected = np.random.choice(2, p = [1 - prevalance_rate, prevalance_rate])

                    # randomly sample the state
                    random_index = np.random.choice(states_indices, p = states_prob)

                    if states_x[i][random_index] > 0:
                        if is_infected:
                            # if random_index not in states_indices_inf:
                            #     # remove from current state
                            #     #print('d', states_x[i][random_index])
                            #     states_x[i][random_index] -= 1
                            #     # add to random infected state
                            #     random_index_inf = np.random.choice(states_indices_inf)
                            #     states_x[i][random_index_inf] += 1
                            # else:
                            #     # leave
                            #     pass
                            tpos_count += 1
                        else:
                            tneg_count += 1
                            # if random_index in states_indices_inf:
                            #     # remove from current state
                            #     states_x[i][random_index]  -= 1
                            #     # add to Recovery/succeptible state
                            #     random_index_rec =  np.random.choice(states_indices_rec)
                            #     states_x[i][random_index_rec]  += 1
                            # else:
                            #     # leave
                            #     pass
                            pass
                    else:
                        print('[INFO] All state values are zero')

                # return updated state_x
                # if nodes:
                #     print('run finish')
                #     #print(nodes[i].states_x)
                #     for j in range(config.nodes_num):
                #         nodes[j].states_x = states_x[j][:]
                #     #print(nodes[i].states_x)
                #
                # else:
                #     print('load finish')
                #     #print(config.last_state_list[i])
                #     config.last_state_list = []
                #     for j in range(config.nodes_num):
                #         temp =  states_x[j][:]
                #         config.last_state_list.append(temp)
                #     #print(config.last_state_list[i])

                t_pos += tpos_count
                t_neg += tneg_count

        d_pos = t_pos
        d_neg = t_neg

        # d_pos = (num_test.sum() * test_spec - num_test.sum() + t_pos) / (test_sens + test_spec - 1)
        # d_neg = (num_test.sum() * test_sens - t_pos) / (test_sens + test_spec - 1)

        true_pos = test_sens * d_pos
        false_pos = (1 - test_spec) * d_neg
        false_neg = (1 - test_sens) * d_pos
        true_neg = test_spec * d_neg

        # round to nearest integer
        d_pos = int(round(d_pos))
        d_neg = int(round(d_neg))
        true_pos = int(round(true_pos))
        false_pos = int(round(false_pos))
        true_neg = int(round(true_neg))
        false_neg = int(round(false_neg))

        t_pos = true_pos + false_pos
        t_neg = false_neg + true_neg

        config.param_test_sum = num_test.sum()
        config.param_t_pos = t_pos
        config.param_t_neg = t_neg
        config.param_d_pos = d_pos
        config.param_d_neg = d_neg
        config.param_true_pos = true_pos
        config.param_true_neg = true_neg
        config.param_false_pos = false_pos
        config.param_false_neg = false_neg

        #print('N= {} \n T+ = {}, T- = {}, D+ = {}, D- = {}, \n TP = {}, TN = {}, FP = {}, FN = {}'.format(num_test.sum(), t_pos, t_neg, d_pos, d_neg, true_pos, true_neg, false_pos, false_neg))
        print('[INFO] Testing finished.')

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

                        # update Prevalence Table
                        for ind in range(17):
                            config.param_prev_auto[ind] = round((config.new_plot_all[-1][:, ind, 0][-1] + config.new_plot_all[-1][:, ind, 2][-1] + config.new_plot_all[-1][:, ind, 7][-1])/config.param_init_susceptible[ind], 6)
                            #config.param_prev_auto[ind] = config.param_init_susceptible[ind]*(i+10)

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

                        # update Prevalence Table
                        for ind in range(17):
                            # print(config.new_plot_all[-1][:, ind, 0][-1], config.new_plot_all[-1][:, ind, 2][-1], config.new_plot_all[-1][:, ind, 7][-1], config.param_init_susceptible[ind])
                            config.param_prev_auto[ind] = round((config.new_plot_all[-1][:, ind, 0][-1] + config.new_plot_all[-1][:, ind, 2][-1] + config.new_plot_all[-1][:, ind, 7][-1])/config.param_init_susceptible[ind], 6)
                            #config.param_prev_auto[ind] = config.param_init_susceptible[ind]

                        self.callbackFunc.doc.add_next_tick_callback(partial(self.callbackFunc.update, False))
                        config.counter_func +=1

                    config.flag_sim = 0
                    config.load_iteration = False
                    print('[INFO] Loading the previous results ..')

                # # Conducting the test
                # elif config.is_test:
                #     config.flag_sim = 1
                #
                #     #test_population(config.param_init_susceptible, config.nodes_old, config.param_test_num, config.param_test_prev, config.param_test_sens, config.param_test_spec)
                #
                #     #new_nodes, new_plot, new_params = simulate_network(...)
                #     #config.nodes_old = new_nodes
                #     #append the new results to list
                #     #config.new_plot_all.append(new_plot)
                #     #self.callbackFunc.doc.add_next_tick_callback(partial(self.callbackFunc.update, False))
                #
                #     config.flag_sim = 0
                #     config.is_test = False

        except (KeyboardInterrupt, SystemExit):
            print('[INFO] Exiting the program.. ')
            pass
