function [param, init] = moses_init();
%
% MOSES_INIT() initializes the parameters of the epidemic simulator
%
% Author: Atakan Varol
% Date: January 2016

param.br = 0.0/365;      % Daily birth rate
param.dr = 0.0/365;      % Daily mortality rate except infected people
param.vr = 0.0;         % Daily vaccination rate (Ratio of susceptible population getting vaccinated)

param.vir = 0.9;          % Ratio of the immunized after vaccination
param.mir = 0.0;          % Maternal immunization rate

param.beta_exp = 0.0;     % Susceptible to exposed transition constant
param.qr  = 0.0;         % Daily quarantine rate (Ratio of Exposed getting Quarantined)
param.beta_inf = 0.3;     % Susceptible to infected transition constant
param.ir  = 0.0;         % Daily isolation rate (Ratio of Infected getting Isolated)
param.eps_exp = 0.0;      % Disease transmission rate of exposed compared to the infected
param.eps_qua = 0.0;      % Disease transmission rate of quarantined compared to the infected
param.eps_iso = 0.0;      % Disease transmission rate of isolated compared to the infected

% Check for wrong initialization of the parameters
if param.beta_exp == 0 && param.beta_inf == 0
    error('Both beta_exp and beta_inf cannot be zero.');
elseif param.beta_exp ~= 0 && param.beta_inf ~= 0
    error('Both beta_exp and beta_inf cannnot be non-zero.');
end

param.gamma_mor = 1;      % Infected to Dead transition probability
param.gamma_im = 0.0;       % Infected to Recovery Immunized transition probability

param.dt = 1/24;           % Sampling time in days (1/24 corresponds to one hour)
param.sim_len = 100;       % Length of simulation in days

param.t_exp = 0;          % Incubation period (The period from the start of incubation to the end of the incubation state
param.t_inf = 5;          % Infection period (The period from the start of infection to the end of the infection state
param.t_vac = 0;          % Vaccination immunization period (The time to vaccinatization immunization after being vaccinated

param.n_exp = ceil( param.t_exp / param.dt);
param.n_inf = ceil( param.t_inf / param.dt);
param.n_vac = ceil( param.t_vac / param.dt);

param.save_res = 1;
param.disp_progress = 1;
param.disp_interval = 100;
param.vis_on = 1;                               % Visualize results after simulation
param.rand_seed = rng( mod( now*100000, 1e5) ); % Generate random numbers that are different (Based on system clock)
% param.rand_seed = 110;                        % In case, repeatable random numbers are desired for the stochastic simulation, use a fixed seed.

% Define the initial values for the states
init.susceptible = 490;
init.exposed = 0;
init.quarantined = 0;
init.infected = 10;
init.isolated = 0;
init.vaccination_imm = 0;
init.maternally_imm = 0;
init.recovery_imm = 0;

end