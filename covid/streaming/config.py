import numpy as np


#  Initial Parameters for each regions
nodes_num = 17

param_beta_exp = 30.0*np.ones(nodes_num)     # Susceptible to Exposed transition constant
param_qr  = 2.0*np.ones(nodes_num)         # Daily Quarantine rate of the Exposed
param_sir  = 0.35*np.ones(nodes_num)        # Daily Infected to Severe Infected transition rate

param_eps_exp = 100.0*np.ones(nodes_num)       # Disease transmission rate of Exposed compared to Infected
param_eps_qua = 20.0*np.ones(nodes_num)       # Disease transmission rate of Quarantined compared to Infected
param_eps_sev  = 20.0*np.ones(nodes_num)       # Disease transmission rate of Severe Infected compared to Infected

param_hosp_capacity = np.array((280,2395,895,600,650,250,725,100,885,425,1670,300,465,1420,1505,380,300))   # Hospital Capacity

param_gamma_mor1 = 7.0*np.ones(nodes_num) # Severe Infected to Dead transition probability
param_gamma_mor2 = 11.0*np.ones(nodes_num) # Severe Infected to Dead transition probability (Hospital Cap. Exceeded)
param_gamma_im = 90.0*np.ones(nodes_num)      # Infected to Recovery Immunized transition probability

param_sim_len = 1*np.ones(nodes_num)            # Length of simulation (Days)

param_t_exp = 5*np.ones(nodes_num)             # Incubation period (The period from the start of incubation to the end of the incubation state)
param_t_inf = 14*np.ones(nodes_num)             # Infection period (The period from the start of infection to the end of the infection state)

param_init_exposed = 0*np.ones(nodes_num)       # Inital number of Exposed
param_init_susceptible = np.squeeze(np.array([2039379,1854556,738587,869603,633801,652314,1125297,
                            678224,1078362,753804,1378554,872736,794165,1378504,1011511,554519,1981747]))  # Inital number of Susceptible = population

param_transition_scale = 1.0        # Traffic ratio
param_transition_leakage = 0.0      # Leakage ratio

param_save_file = 'foldername'      # Default folder name to save results

# Simulation parameters

max_sim_len = 10000
loop_num = 1
flag_sim = 0
counter = 1
counter_func = 0
run_iteration = False

# Solver arguments
nodes_old = []
params_network = []
params_node = []
params_old = []

# Plot parameters

region = 0
new_plot_all = []
new_nodes_all = []

state_sus = []
state_exp = []
state_inf = []
state_sin = []
state_qua = []
state_imm = []
state_dea = []
newx = []

pstate_sus = []
pstate_exp = []
pstate_inf = []
pstate_sin = []
pstate_qua = []
pstate_imm = []
pstate_dea = []

# File save parameters
header_file_csv = r'Day,Infected,Exposed,Severe Infected,Quarantined,Immunized,Susceptible,Dead,Initial Exposed,Susceptible to Exposed transition constant,Daily Quarantine rate of the Exposed,Daily Infected to Severe Infected transition rate,Hospital Capacity,Severe Infected to Dead transition probability,Severe Infected to Dead transition probability (Hospital Cap. Exceeded),Infected to Recovery Immunized transition probability,Disease transmission rate of Exposed compared to Infected,Disease transmission rate of Quarantined compared to Infected,Disease transmission rate of Severe Infected compared to Infected,Incubation period (Days),Infection  period (Days),Leakage ratio,Traffic ratio,Airway,Railway,Highway'

arr_for_save = np.concatenate((param_beta_exp, param_qr,param_sir, param_hosp_capacity,
                    param_gamma_mor1, param_gamma_mor2, param_gamma_im, param_eps_exp,
                    param_eps_qua, param_eps_sev, param_transition_leakage, param_transition_scale), axis=None)

# transition matrix read from checkbox
box1 = list(range(0, 17))
box2 = list(range(0, 17))
box3 = list(range(0, 17))

param_transition_box = []
param_transition_box.append(box1)
param_transition_box.append(box2)
param_transition_box.append(box3)

box_time = [param_transition_box]

# base transition_matrix
transition_matrix = (0.5*np.array([[0,5100,0,0,0,0,0,0,1042,0,120,0,0,60,0,0,0],
                        [5100,0,0,824,653,668,1280,688,1193,744,1352,826,892,1532,1454,0,240],
                        [0,0,0,0,461,0,0,478,2057,0,180,0,0,0,0,0,0],
                        [0,824,0,0,495,0,0,515,651,0,721,0,0,0,0,0,0],
                        [0,653,461,495,0,614,0,503,539,0,587,0,0,0,526,0,0],
                        [0,668,0,0,614,0,0,449,549,0,0,0,0,0,0,0,0],
                        [0,1280,0,0,0,0,0,0,744,0,60,0,0,0,0,0,0],
                        [0,688,478,515,503,449,0,0,563,0,1034,0,0,0,549,0,0],
                        [1042,1193,2057,651,539,549,744,563,0,659,1769,652,798,1058,705,718,0],
                        [0,744,0,0,0,0,0,0,659,0,102,0,0,0,0,0,0],
                        [120,1352,180,721,587,0,60,1034,1769,102,0,765,801,1077,968,60,240],
                        [0,826,0,0,0,0,0,0,652,0,765,0,0,0,0,0,0],
                        [0,892,0,0,0,0,0,0,798,0,801,0,0,0,0,0,0],
                        [60,1532,0,0,0,0,0,0,1058,0,1077,0,0,0,0,0,0],
                        [0,1454,0,0,526,0,0,549,705,0,968,0,0,0,0,484,900],
                        [0,0,0,0,0,0,0,0,718,0,60,0,0,0,484,0,0],
                        [0,240,0,0,0,0,0,0,0,0,240,0,0,0,900,0,0]])).astype(int)
