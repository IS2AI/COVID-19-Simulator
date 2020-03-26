import numpy as np
import matplotlib.pyplot as plt
from copy import copy

class Network:
    def __init__(self, nodes_num, nodes_population):

        self.nodes_num = nodes_num
        self.nodes_population = nodes_population
        self.exp_thresh = 1
        self.delta_population = np.zeros((self.nodes_num), dtype = np.int32)

    def update_node_states(self, nodes, states_x_old):
        nodes_new_iter = copy(nodes)

        for index, node in enumerate(nodes_new_iter):
            node.states_x = states_x_old[index].states_x

        return nodes_new_iter

    def node_states_transition(self, nodes, transition_matrix, static_states_indices):
        temp_nodes = copy(nodes)
        for i in range(self.nodes_num):
            for j in range(self.nodes_num):
                # if some transition ha ppens from node_i to node_j
                if transition_matrix[i,j] > 0:
                    exp_i_j = (transition_matrix[i,j]/self.nodes_population[i])*(nodes[i].states_x)
                    # for each state_index calculate the expected value
                    for index in range(nodes[0].param_num_states):
                        # if exp value greater than zero and state is subject to transition
                        if exp_i_j[index] > 0 and index not in static_states_indices:
                            if exp_i_j[index] > self.exp_thresh:
                                #print('mode1')
                                exp_i_j[index] = int(round(exp_i_j[index]))
                            else:
                                random_arr = np.random.uniform(0,1,100)
                                exp_i_j[index] = (random_arr < self.exp_thresh/100).sum()
                            # do transfer from node_i to node_j if in node_i enough population
                            if  nodes[i].states_x[index] > exp_i_j[index]:
                                temp_nodes[i].states_x[index] -= exp_i_j[index]
                                temp_nodes[j].states_x[index] += exp_i_j[index]

        self.update_node_population(transition_matrix)
        # for plotting
        return temp_nodes

    def update_node_population(self, transition_matrix):
        for i in range(self.nodes_num):
            self.delta_population[i] = -np.sum(transition_matrix[i,:]) + np.sum(transition_matrix[:,i])
        self.nodes_population += self.delta_population
