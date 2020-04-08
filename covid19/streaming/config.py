import os
import csv
import numpy as np

# Initial Parameters for each region
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
param_t_exp = 5*np.ones(nodes_num)              # Incubation period (The period from the start of incubation to the end of the incubation state)
param_t_inf = 14*np.ones(nodes_num)             # Infection period (The period from the start of infection to the end of the infection state)

param_transition_table = np.ones((17, 3))

param_init_susceptible = np.squeeze(np.array([2039379,1854556,738587,869603,633801,652314,1125297,
                                            678224,1078362,753804,1378554,872736,794165,1378504,1011511,554519,1981747]))  # Inital number of Susceptible = population

param_init_exposed = 0*np.ones(nodes_num)            # Inital number of Exposed
param_transition_scale = 1.0*np.ones(nodes_num)      # Traffic ratio
param_transition_leakage = 0.0*np.ones(nodes_num)    # Leakage ratio

# Buffer parameters
new_plot = []
new_plot_all = []
box_time = param_transition_table
arr_for_save = np.dstack((param_init_exposed, param_beta_exp, param_qr, param_sir, param_hosp_capacity, param_gamma_mor1, param_gamma_mor2,
                        param_gamma_im, param_eps_exp, param_eps_qua, param_eps_sev, param_t_exp, param_t_inf, param_transition_leakage, param_transition_scale))

last_state_list = []

# Simulation parameters
max_sim_len = 10000
loop_num = 1
flag_sim = 0
counter_func = 0
counter_load = 0
run_iteration = False
load_iteration = False
is_loaded = False

# Plot parameters
region = 0
state_sus = []
state_exp = []
state_inf = []
state_sin = []
state_qua = []
state_imm = []
state_dea = []
newx = []

# Solver arguments

nodes_old = []
params_node = []

# File save parameters
last_date = '2020-02-10'
first_date = '2020-02-10'

region_names = ['Almaty', 'Almaty Qalasy', 'Aqmola', 'Aqtobe', 'Atyrau', 'West Kazakhstan', 'Jambyl', 'Mangystau', 'Nur-Sultan', 'Pavlodar', 'Qaraqandy', 'Qostanai',  'Qyzylorda', 'East Kazakhstan', 'Shymkent', 'North Kazakhstan', 'Turkistan']

param_save_file = 'foldername'

header_file_csv = r'Day,Date,Infected,Exposed,Severe Infected,Quarantined,Immunized,Susceptible,Dead,Isolated,Initial Exposed,Susceptible to Exposed transition constant,Daily Quarantine rate of the Exposed,Daily Infected to Severe Infected transition rate,Hospital Capacity,Severe Infected to Dead transition probability,Severe Infected to Dead transition probability (Hospital Cap. Exceeded),Infected to Recovery Immunized transition probability,Disease transmission rate of Exposed compared to Infected,Disease transmission rate of Quarantined compared to Infected,Disease transmission rate of Severe Infected compared to Infected,Incubation period (Days),Infection  period (Days),Leakage ratio,Traffic ratio,Airway,Railway,Highway'

header_file_csv2 = r'Day,Date,Infected,Exposed,Severe Infected,Quarantined,Immunized,Susceptible,Dead,Isolated'

# load the transition_matrix from the csv files
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
transition_railway = list(csv.reader(open(os.path.join(THIS_FOLDER, 'static', 'rail_tr.csv'))))
transition_railway = np.array(transition_railway, dtype = np.float32)

transition_airway = list(csv.reader(open(os.path.join(THIS_FOLDER, 'static', 'air_tr.csv'))))
transition_airway = np.array(transition_airway, dtype = np.float32)

transition_roadway = list(csv.reader(open(os.path.join(THIS_FOLDER, 'static', 'high_tr.csv'))))
transition_roadway = np.array(transition_roadway, dtype = np.float32)

# Initialize base transition_matrix for
transition_matrix_init =   (0.5*(transition_railway + transition_airway + transition_roadway)).astype(int) # transition between nodes happens every 12 hours or 1/2 days
transition_matrix = transition_matrix_init
