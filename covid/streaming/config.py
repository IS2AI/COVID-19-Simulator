import numpy as np
import os
import csv

initial_params = 5
nodes_num = 17
sim_len = 10000
run_iteration = False
region = 0
is_finished = False
nodes_old = []
new_plot_all = []
iteration_over = False
new_nodes_all = []

state_sus = []
state_exp = []
state_inf = []
state_sin = []
state_qua = []
state_imm = []
state_dea = []
newx = []
counter = 1
counter_func = 0

pstate_sus = []
pstate_exp = []
pstate_inf = []
pstate_sin = []
pstate_qua = []
pstate_imm = []
pstate_dea = []

flag_sim = 0

#  Parameters
param_beta_exp = 30.0*np.ones(nodes_num)     # Susceptible to exposed transition constant
param_qr  = 2.0*np.ones(nodes_num)         # Daily quarantine rate (Ratio of Exposed getting Quarantined)
param_sir  = 0.35*np.ones(nodes_num)        # Daily isolation rate (Ratio of Infected getting Isolated)

param_eps_exp = 100.0*np.ones(nodes_num)       # Disease transmission rate of exposed compared to the infected
param_eps_qua = 20.0*np.ones(nodes_num)       # Disease transmission rate of quarantined compared to the infected
param_eps_sev  = 20.0*np.ones(nodes_num)       # D  isease transmission rate of isolated compared to the infected

param_hosp_capacity = np.array((280,2395,895,600,650,250,725,100,885,425,1670,300,465,1420,1505,380,300))   # Maximum amount patients that hospital can accommodate

param_gamma_mor1 = 7.0*np.ones(nodes_num) # Severe Infected (Hospitalized) to Dead transition probability
param_gamma_mor2 = 11.0*np.ones(nodes_num) # Severe Infected (Not Hospitalized) to Dead transition probability
param_gamma_im = 90.0*np.ones(nodes_num)      # Infected to Recovery Immunized transition probability

param_sim_len = 1*np.ones(nodes_num)            # Length of simulation in days

param_t_exp = 5*np.ones(nodes_num)             # Incubation period (The period from the start of incubation to the end of the incubation state
param_t_inf = 14*np.ones(nodes_num)             # Infection period (The period from the start of infection to the end of the infection state

param_init_exposed = 0*np.ones(nodes_num) # np.array([100,100,100])#10*np.ones(nodes_num)

param_transition_leakage = 0.0
param_transition_scale = 1.0

param_dt = 1/24
param_dt_inv = 1/param_dt

param_save_file = 'foldername'

# Init values for nodes constant
param_init_susceptible = np.squeeze(np.array([2039379,1854556,738587,869603,633801,652314,1125297,678224,1078362,753804,1378554,872736,794165,1378504,1011511,554519,1981747]))

params_network = []
params_node = []

params_old = []
testing_var = []

box1 = list(range(0, 17))
box2 = list(range(0, 17))
box3 = list(range(0, 17))

region_status = np.ones((17,3))

loop_num = 2

param_transition_box = []
param_transition_box.append(box1)
param_transition_box.append(box2)
param_transition_box.append(box3)

box_time = [param_transition_box]

header_file_csv = r'Day,Infected,Exposed,Severe Infected,Quarantined,Immunized,Susceptible,Dead,Initial Exposed,Susceptible to Exposed transition constant,Daily Quarantine rate of the Exposed,Daily Infected to Severe Infected transition rate,Hospital Capacity,Severe Infected to Dead transition probability,Severe Infected to Dead transition probability (Hospital Cap. Exceeded),Infected to Recovery Immunized transition probability,Disease transmission rate of Exposed compared to Infected,Disease transmission rate of Quarantined compared to Infected,Disease transmission rate of Severe Infected compared to Infected,Incubation period (Days),Infection  period (Days),Leakage ratio,Traffic ratio,Airway,Railway,Highway'


arr_for_save = np.concatenate((param_beta_exp, param_qr,param_sir, param_hosp_capacity,
    param_gamma_mor1, param_gamma_mor2, param_gamma_im, param_eps_exp,
    param_eps_qua, param_eps_sev, param_transition_leakage, param_transition_scale), axis=None)

init_for_save = []

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
transition_all = list(csv.reader(open(os.path.join(THIS_FOLDER, 'static/base_tr.csv'))))
transition_matrix = 0.5*np.array(transition_all, dtype = np.float32).astype(int)
