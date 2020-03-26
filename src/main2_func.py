"""
Created: Mar 25 2020

@author: Daulet
"""

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

def process_node(node):
    temp_node = node
    temp_node.stoch_solver()
    return temp_node

def simulate_network(params_node_, params_network, nodes_old, sim_iter, params_old):

    nodes_num = 17

    # load population data
    nodes_population = np.squeeze(np.array([2039376,1854556,738587,869603,633801,652314,1125297,678224,1078362,753804,1378554,872736,794165,1378504,1011511,554519,1981747])) # np.array([100000,200000,300000])


    # transition params
    param_transition_box = params_network[0]
    param_transition_leakage = params_network[1]
    param_transition_scale = params_network[2]

    # load transition matrix
    transition_railway = list(csv.reader(open('../data/tr_2.csv')))
    transition_railway = np.array(transition_railway, dtype = np.float32)

    transition_airway = list(csv.reader(open('../data/tr_1.csv')))
    transition_airway = np.array(transition_airway, dtype = np.float32)

    transition_roadway = list(csv.reader(open('../data/tr_3.csv')))
    transition_roadway = np.array(transition_airway, dtype = np.float32)

    tr_table = [transition_airway, transition_railway, transition_roadway]

    for tr in tr_table:
        for i in range(17):
            tr[i, :] = tr[i, :]*param_transition_box[i,0]
            tr[:, i] = tr[i, :]*param_transition_box[i,0]

    transition_matrix = (transition_railway + transition_airway + transition_roadway).astype(int)
    transition_matrix = 0.5*transition_matrix * param_transition_scale* (1 + param_transition_leakage)

    # create nodes
    params_node = params_node_

    if sim_iter > 0:
        # put values from prev iter
        params_node[10,:] = params_old[10,:]
        params_node[11,:] = params_old[11,:]
        params_node[12,:] = params_old[12,:]
        params_node[14,:] = params_old[14,:]

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

    #pool = Pool()

    start = time.time()
    for ind in range(param_num_sim):

        #nodes = list(pool.map(process_node, nodes))
        for index, node in enumerate(nodes):
            node.stoch_solver()
            nodes_state_arr[ind, index, :] = node.states_x
        if ind % param_freq_transition == 0 :
            nodes = nodes_network.node_states_transition(nodes, transition_matrix, param_static_indices)      # Update the state of every node due to transition

    end = time.time()

    print("Sim.time: {:.4f} sec".format(end - start))
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

    return nodes, states_arr_plot, params_node

if __name__ == '__main__':

    nodes_num = 17

    ####
    # ALL parameters from here

    #    Parameters

    param_beta_exp = 0.2*np.ones(nodes_num)     # Susceptible to exposed transition constant
    param_qr  = 0.02*np.ones(nodes_num)         # Daily quarantine rate (Ratio of Exposed getting Quarantined)
    param_sir  = 0.01*np.ones(nodes_num)        # Daily isolation rate (Ratio of Infected getting Isolated)

    param_eps_exp = 0.7*np.ones(nodes_num)       # Disease transmission rate of exposed compared to the infected
    param_eps_qua = 0.3*np.ones(nodes_num)       # Disease transmission rate of quarantined compared to the infected
    param_eps_sev  = 0.3*np.ones(nodes_num)       # D  isease transmission rate of isolated compared to the infected

    param_hosp_capacity = 3000*np.ones(nodes_num)   # Maximum amount patients that hospital can accommodate

    param_gamma_mor1 = 0.03*np.ones(nodes_num) # Severe Infected (Hospitalized) to Dead transition probability
    param_gamma_mor2 = 0.1*np.ones(nodes_num) # Severe Infected (Not Hospitalized) to Dead transition probability
    param_gamma_im = 0.9*np.ones(nodes_num)      # Infected to Recovery Immunized transition probability

    param_sim_len = 10*np.ones(nodes_num)            # Length of simulation in days

    param_t_exp = 5*np.ones(nodes_num)             # Incubation period (The period from the start of incubation to the end of the incubation state
    param_t_inf = 8*np.ones(nodes_num)             # Infection period (The period from the start of infection to the end of the infection state

    param_init_exposed = 10*np.ones(nodes_num) # np.array([100,100,100])#10*np.ones(nodes_num)

    param_transition_box = np.ones((17,3))
    param_transition_leakage = 0.1
    param_transition_scale = 0.2

    # Init values for nodes constant
    param_init_susceptible = np.squeeze(np.array([2039376,1854556,738587,869603,633801,652314,1125297,678224,1078362,753804,1378554,872736,794165,1378504,1011511,554519,1981747]))

    params_node = np.vstack([param_beta_exp, param_qr,
                    param_sir, param_eps_exp, param_eps_qua, param_eps_sev,param_hosp_capacity,
                    param_gamma_mor1, param_gamma_mor2, param_gamma_im, param_sim_len,
                    param_t_exp, param_t_inf, param_init_susceptible, param_init_exposed])

    params_network = [param_transition_box, param_transition_leakage, param_transition_scale]

    # Run the script

    a  = 5
    b = 0

    nodes_old = []
    params_old = []

    state_sus = []
    state_exp = []
    state_inf = []
    state_sin = []
    state_qua = []
    state_imm = []
    state_dea = []

    for i in range(a):

        new_nodes, new_plot, new_params = simulate_network(params_node, params_network, nodes_old, i, params_old)
        nodes_old = new_nodes
        params_old = new_params.copy()

        state_sus = np.append(state_sus, new_plot[:, b,5])
        state_exp = np.append(state_exp, new_plot[:, b,1])
        state_inf = np.append(state_inf, new_plot[:, b,0])
        state_sin = np.append(state_sin, new_plot[:, b,2])
        state_qua = np.append(state_qua, new_plot[:, b,3])
        state_imm = np.append(state_imm, new_plot[:, b,4])
        state_dea = np.append(state_dea, new_plot[:, b,6])

    #print(state_sus.shape)
    plt.plot(state_sus, label = 'Susceptible')
    plt.plot(state_exp, label = 'Exposed')
    plt.plot(state_inf, label = 'Infected')
    plt.plot(state_sin, label = 'Severe')
    plt.plot(state_qua, label = 'Quarantined')
    plt.plot(state_imm, label = 'Immunized')
    plt.plot(state_dea, label = 'Dead')

    plt.xlabel("Day")
    plt.ylabel("Population")
    plt.legend(loc="upper right")
    plt.show()
