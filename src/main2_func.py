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

nodes_num = 1

# load population data
nodes_population = list(csv.reader(open('../data/population.csv')))
nodes_population = np.squeeze(np.array(nodes_population, dtype = np.int))

# load transition matrix
transition_railway = list(csv.reader(open('../data/tr_1.csv')))
transition_railway = np.array(transition_railway, dtype = np.float32)

transition_airway = list(csv.reader(open('../data/tr_2.csv')))
transition_airway = np.array(transition_airway, dtype = np.float32)

transition_matrix = (transition_railway + transition_airway).astype(int)


####
# ALL parameters from here


#  from Mukhamet Parameters
param_br = 0.0*np.ones(nodes_num)           # Daily birth rate
param_dr = 0.0*np.ones(nodes_num)           # Daily mortality rate except infected people
param_vr = 0.0*np.ones(nodes_num)           # Daily vaccination rate (Ratio of susceptible population getting vaccinated)

param_vir = 0.9*np.ones(nodes_num)          # Ratio of the immunized after vaccination
param_mir = 0.0*np.ones(nodes_num)          # Maternal immunization rate

param_beta_exp = 0.2*np.ones(nodes_num)     # Susceptible to exposed transition constant
param_qr  = 0.02*np.ones(nodes_num)         # Daily quarantine rate (Ratio of Exposed getting Quarantined)
param_beta_inf = 0.0*np.ones(nodes_num)     # Susceptible to infected transition constant
param_sir  = 0.01*np.ones(nodes_num)        # Daily isolation rate (Ratio of Infected getting Isolated)

param_eps_exp = 0.7*np.ones(nodes_num)       # Disease transmission rate of exposed compared to the infected
param_eps_qua = 0.3*np.ones(nodes_num)       # Disease transmission rate of quarantined compared to the infected
param_eps_sev  = 0.3*np.ones(nodes_num)       # D  isease transmission rate of isolated compared to the infected

param_hosp_capacity = 3000*np.ones(nodes_num)   # Maximum amount patients that hospital can accommodate

param_gamma_mor = 0.0*np.ones(nodes_num)     # Infected to Dead transition probability
param_gamma_mor1 = 0.03*np.ones(nodes_num) # Severe Infected (Hospitalized) to Dead transition probability
param_gamma_mor2 = 0.1*np.ones(nodes_num) # Severe Infected (Not Hospitalized) to Dead transition probability
param_gamma_im = 0.9*np.ones(nodes_num)      # Infected to Recovery Immunized transition probability

param_dt = 1/24*np.ones(nodes_num)               # Sampling time in days (1/24 corresponds to one hour)
param_sim_len = 10*np.ones(nodes_num)            # Length of simulation in days

param_t_exp = 5*np.ones(nodes_num)             # Incubation period (The period from the start of incubation to the end of the incubation state
param_t_inf = 8*np.ones(nodes_num)             # Infection period (The period from the start of infection to the end of the infection state
param_t_vac = 3*np.ones(nodes_num)            # Vaccination immunization period (The time to vaccinatization immunization after being vaccinated

#np.random.seed(1)
#param_rand_seed = np.random.randint(low = 1, high = 100, size = 625)

# Init values for nodes

init_susceptible = 10000*np.ones(nodes_num) # np.array([5000,5000,5000]) #[]
init_exposed = 10*np.ones(nodes_num) # np.array([100,100,100])#10*np.ones(nodes_num)
init_quarantined = 0*np.ones(nodes_num)
init_infected = 0*np.ones(nodes_num)
init_isolated = 0*np.ones(nodes_num)
init_vaccination_imm = 0*np.ones(nodes_num)
init_maternally_imm = 0*np.ones(nodes_num)
init_recovery_imm = 0*np.ones(nodes_num)

# Collect the params and inits to a list
params_node = np.vstack([param_br, param_dr, param_vr, param_vir, param_mir, param_beta_exp, param_qr,
            param_beta_inf, param_sir, param_eps_exp, param_eps_qua, param_eps_sev,param_hosp_capacity,
            param_gamma_mor,param_gamma_mor1, param_gamma_mor2, param_gamma_im, param_dt, param_sim_len,
            param_t_exp, param_t_inf, param_t_vac])

inits_node = np.vstack([init_susceptible, init_exposed, init_quarantined, init_infected, init_isolated,
                    init_vaccination_imm, init_maternally_imm, init_recovery_imm])


# for network
param_num_sim = int(param_sim_len[0] / param_dt[0]) + 1       # Number of simulation
param_dt_transition = 1/2            # Sampling time for transition in days (1/2 corresponds to 12 hour)
param_static_types = ['Birth', 'Dead']         # States not subject to transition
param_freq_transition  = int(param_dt_transition/param_dt[0])


####
# Till here we need from sliders

transition_matrix = transition_matrix
params_network = [param_num_sim, param_dt_transition, param_static_types, transition_matrix, nodes_population]

#
def process_node(node):
    temp_node = node
    temp_node.stoch_solver() # Update the state of every node.
    return temp_node

def simulate_network(params_node, inits_node, params_network, nodes_old, sim_iter):

    # Start simulation

    param_num_sim = params_network[0]
    transition_matrix = params_network[3]
    static_states_types = params_network[2]

    params_node = params_node
    inits_node = inits_node

    # create nodes
    nodes = [Node(params_node[:,i], inits_node[:,i]) for i in range(nodes_num)]

    for index, node in enumerate(nodes):
        node.check_init()
        node.create_states()
        node.indexes()
        node.define_state_arr()
        node.create_transitions()

    # create the network
    nodes_network = Network(nodes_num, nodes_population, static_states_types)
    nodes_state_arr = np.zeros((param_num_sim, nodes_num, nodes[0].param_num_states))

    if sim_iter > 0:
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
            print(ind)
            #static_states_indices = [0,1]
            #nodes = nodes_network.node_states_transition(nodes, transition_matrix)      # Update the state of every node due to transition

    end = time.time()

    print("Sim.time: {:.4f} sec".format(end - start))
    states_arr_plot = np.zeros((param_num_sim, nodes_num, 8))

    for iter in range(param_num_sim):
        for i_node in range(nodes_num):
            states_arr_plot[iter, i_node, 0] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_vac)
            states_arr_plot[iter, i_node, 1] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_inf)
            states_arr_plot[iter, i_node, 2] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_exp)
            states_arr_plot[iter, i_node, 3] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_sin)
            states_arr_plot[iter, i_node, 4] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_qua)
            states_arr_plot[iter, i_node, 5] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_imm)
            states_arr_plot[iter, i_node, 6] = nodes_state_arr[iter, i_node, :].dot(nodes[i_node].ind_sus)
            states_arr_plot[iter, i_node, 7] = nodes[i_node].states_x[nodes[0].param_num_states-1].sum()

    return nodes, states_arr_plot

####
# Mukhamet , you need this part to run
nodes_old = []

state_sus = []
state_exp = []
state_inf = []
state_sin = []
state_qua = []
state_imm = []
state_dea = []

a  = 5

for i in range(a):
    new_nodes, new_plot = simulate_network(params_node, inits_node, params_network, nodes_old, i)
    nodes_old = new_nodes

    state_sus = np.append(state_sus, new_plot[:, 0,0])
    state_exp = np.append(state_exp, new_plot[:, 0,1])
    state_inf = np.append(state_inf, new_plot[:, 0,2])
    state_sin = np.append(state_sin, new_plot[:, 0,3])
    state_qua = np.append(state_qua, new_plot[:, 0,4])
    state_imm = np.append(state_imm, new_plot[:, 0,5])
    state_dea = np.append(state_dea, new_plot[:, 0,6])

####

time_arr = np.linspace(0, new_nodes[0].param_num_sim, new_nodes[0].param_num_sim)*new_nodes[0].param_dt*a




plt.plot(state_sus, label = 'Susceptible')
plt.plot(state_exp, label = 'Susceptible')
plt.plot(state_inf, label = 'Susceptible')
plt.plot(state_sin, label = 'Susceptible')
plt.plot(state_qua, label = 'Susceptible')
plt.plot(state_imm, label = 'Susceptible')
plt.plot(state_dea, label = 'Susceptible')


plt.xlabel("Day")
plt.ylabel("Population")
plt.legend(loc="upper right")
plt.show()


'''
if __name__ == '__main__':

    for i in range(3):
        new_nodes, new_plot = simulate_network(params_node, inits_node, params_network, nodes_old, i)
        nodes_old = new_nodes


    time_arr = np.linspace(0, new_nodes[0].param_num_sim, new_nodes[0].param_num_sim)*new_nodes[0].param_dt
    state_sus = new_plot[:, 0,0]
    state_exp = new_plot[:, 0,0]
    state_inf = new_plot[:, 0,0]
    state_sin = new_plot[:, 0,0]
    state_qua = new_plot[:, 0,0]
    state_imm = new_plot[:, 0,0]
    state_dea = new_plot[:, 0,0]

    plt.plot(time_arr, state_sus, label = 'Susceptible')
    plt.plot(time_arr, state_exp, label = 'Exposed')
    plt.plot(time_arr, state_qua, label = 'Quarantined')
    plt.plot(time_arr, state_inf, label = 'Infected')
    plt.plot(time_arr, state_sin, label = 'Severe Infected')
    plt.plot(time_arr, state_imm, label = 'Immunized')
    plt.plot(time_arr, state_dea, label = 'Dead')
    plt.xlabel("Day")
    plt.ylabel("Population")
    plt.legend(loc="upper right")
    plt.show()

'''
