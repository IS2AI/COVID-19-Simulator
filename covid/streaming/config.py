import numpy as np

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

loop_num = 1

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

#THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
#transition_all = list(csv.reader(open(os.path.join(THIS_FOLDER, 'base_tr.csv'))))
#transition_matrix = 0.5*np.array(transition_all, dtype = np.float32).astype(int)



#transition_matrix  = np.zeros((17,17)).astype(int)


transition_matrix = np.array([[0,5100,0,0,0,0,0,0,1042,0,120,0,0,60,0,0,0],
[5100,0,0,824,653,668,1280.65387,688,1193,744,1352,826,892,1532,1454,0,240],
[0,0,0,0,461,0,0,478,2057,0,180,0,0,0,0,0,0],
[0,824,0,0,495.770727,0,0,515.2931831,651.0091423,0,721.1088195,0,0,0,0,0,0],
[0,653,461,495.770727,0,614.7228299,0,503.0641343,539.827519,0,587.1576691,0,0,0,526.9360323,0,0],
[0,668,0,0,614.7228299,0,0,449.6609804,549.6524092,0,0,0,0,0,0,0,0],
[0,1280,0,0,0,0,0,0,744.6804923,0,60,0,0,0,0,0,0],
[0,688,478.1294838,515.2931831,503.0641343,449.6609804,0,0,563.055162,0,1034.740959,0,0,0,549.044817,0,0],
[1042,1193.904495,2057.142857,651.0091423,539.827519,549.6524092,744.6804923,563.055162,0,659.985598,1769.668525,652.3054614,798.4859841,1058.226928,705.8264394,718.0896035,0],
[0,744,0,0,0,0,0,0,659.985598,0,102.8571429,0,0,0,0,0,0],
[120,1352.347684,180,721.1088195,587.1576691,0,60,1034.740959,1769.668525,102.8571429,0,765.5568244,801.417991,1077.827238,968.982395,60,240],
[0,826.5226377,0,0,0,0,0,0,652.3054614,0,765.5568244,0,0,0,0,0,0],
[0,892.9674007,0,0,0,0,0,0,798.4859841,0,801.417991,0,0,0,0,0,0],
[60,1532.323611,0,0,0,0,0,0,1058.226928,0,1077.827238,0,0,0,0,0,0],
[0,1454.375169,0,0,526,0,0,549.044817,705.8264394,0,968.982395,0,0,0,0,484.3614783,900],
[0,0,0,0,0,0,0,0,718.0896035,0,60,0,0,0,484.3614783,0,0],
[0,240,0,0,0,0,0,0,0,0,240,0,0,0,900,0,0]])
