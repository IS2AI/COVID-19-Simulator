'''
nodes_population = np.array([100000,200000,1000000]) #1000000*np.ones(nodes_num, dtype = np.int64)     # Initial population of each node [2039376,1854556,738587,869603,633801,652314,1125297,678224,1078362,753804,1378554,872736,794165,1378504,1011511,554519,1981747]

transition_matrix = np.zeros((nodes_num,nodes_num))           # transition matrix between nodes
transition_matrix[0, 1] = 100
transition_matrix[1, 0] = 100
transition_matrix[0, 2] = 0
transition_matrix[2, 0] = 0
print(transition_matrix)
'''
