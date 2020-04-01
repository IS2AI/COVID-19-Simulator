function [states]  = moses_stoch_solver(states, trans, param)
%
% STATES_NEXT = MOSES_STOCH_SOLVER(STATES, PARAM) is the main stochastic
% solver.
%
% STATES_NEXT = MOSES_STOCH_SOLVER(STATES, PARAM) firstly computes the expected
% values of each transition in the state machine
% The results are stored in an array containing the expected value and
% source and destination state indexes
%
% Author: Atakan Varol
% Date: January 2016

% Use persisten variables to gain speed
persistent ind_vac ind_inf ind_exp ind_iso ind_qua ind_exp1 ind_expn ind_qua1 ind_quan ind_inf1 ind_infn ind_iso1 ind_ison
if isempty(ind_vac)
    
    temp1 = ['Vaccinated_', num2str(param.n_vac)];
    ind_vac = strcmp(states.name, temp1);
    
    ind_inf = strncmp('Infected_', states.name, 8);
    ind_exp = strncmp('Exposed_', states.name, 7);
    ind_iso = strncmp('Severe_Infected_', states.name, 16);
    ind_qua = strncmp('Quarantined_', states.name, 12);
    
    ind_exp1 = find( strcmp(states.name, 'Exposed_1') == 1);
    ind_expn = find( strcmp(states.name, ['Exposed_', num2str(param.n_exp)] ) == 1);
    
    ind_qua1 = find( strcmp(states.name, 'Quarantined_1') == 1);
    ind_quan = find( strcmp(states.name, ['Quarantined_', num2str(param.n_exp)] ) == 1);
    
    ind_iso1 = find( strcmp(states.name, 'Severe_Infected_1') == 1);
    ind_ison = find( strcmp(states.name, ['Severe_Infected_', num2str(param.n_inf)] ) == 1);
    
    ind_inf1 = find( strcmp(states.name, 'Infected_1') == 1);
    ind_infn = find( strcmp(states.name, ['Infected_', num2str(param.n_inf)] ) == 1);
end

total_pop = sum(states.x(2:end-1)); % Total population is the sum of all states except birth and death.

%% Transition 1 - Birth to Susceptible
expval(1) = total_pop*param.br*(1-param.mir)*param.dt;

%% Transition 2 - Birth to Maternally Immunized
expval(2) = total_pop*param.br*param.mir*param.dt;

%% Transition 3 - Any State except Birth to Dead (Natural Mortality)
count = 2;
for ind = 2 : param.num_states - 1
    count = count + 1;
    expval(count) = states.x(ind)*param.dr*param.dt;
end

%
%% Transition 4 - Susceptible to Vaccinated[1]
if param.vr ~= 0
    count = count + 1;
    expval(count) = states.x(2)*param.vr*param.dt;
end

%% Transition 5 - Vaccinated[i] to Vaccinated[i+1] until i+1 == n_vac
if param.n_vac ~= 0
    for ind = 1 : param.n_vac - 1
        count = count + 1;
        expval(count) = states.x(2+ind)*(1 - param.dr*param.dt);
    end
end

%% Transition 6 - Vaccinated[n_vac] to Vaccination_Immunized
if param.vr ~= 0
    count = count + 1;
    % temp1 = ['Vaccinated_', num2str(param.n_vac)];
    % ind_vac = strcmp(states.name, temp1);
    expval(count) = states.x(ind_vac)*param.vir;
end

%% Transition 7 - Vaccinated[n_vac] to Susceptible
if param.vr ~= 0
    count = count + 1;
    % temp1 = ['Vaccinated_', num2str(param.n_vac)];
    % ind_vac = strcmp(states.name, temp1);
    expval(count) = states.x(ind_vac)*(1 - param.dr*param.dt - param.vir);
end

%% Transition 8 - Susceptible to Exposed[1]
if param.n_exp ~= 0
    count = count + 1;
    % ind_inf = strncmp('Infected_', states.name, 8);
    % ind_exp = strncmp('Exposed_', states.name, 7);
    % ind_iso = strncmp('Severe_Infected_', states.name, 9);
    % ind_qua = strncmp('Quarantined_', states.name, 12)

    temp1 = sum(states.x(ind_inf)) + param.eps_exp*sum(states.x(ind_exp)) + ...
        param.eps_sev*sum(states.x(ind_iso)) + param.eps_qua*sum(states.x(ind_qua));
    expval(count) = states.x(2)*temp1*param.beta_exp*param.dt/total_pop;
end
%% Transition 9 - Susceptible to Infected[1]
count = count + 1;
% ind_inf = strncmp('Infected_', states.name, 8);
% ind_exp = strncmp('Exposed_', states.name, 7);
% ind_iso = strncmp('Severe_Infected_', states.name, 9);
% ind_qua = strncmp('Quarantined_', states.name, 12);
temp1 = sum(states.x(ind_inf)) + param.eps_exp*sum(states.x(ind_exp)) + ...
    param.eps_sev*sum(states.x(ind_iso)) + param.eps_qua*sum(states.x(ind_qua));
expval(count) = states.x(2)*temp1*param.beta_inf*param.dt/total_pop;

%% Transition 10 - Exposed[i] to Exposed[i+1] until i+1 == n_exp
% ind_exp1 = find( strcmp(states.name, 'Exposed_1') == 1);
for ind = 1 : param.n_exp - 1
    count = count + 1;
    expval(count) = states.x(ind_exp1 + ind - 1)*(1 - param.dr*param.dt - param.qr*param.dt );
end

%% Transition 11 - Exposed[n_exp] to Infected[1]
if param.n_exp ~= 0
    count = count + 1;
    expval(count) = states.x(ind_expn)*(1 - param.dr*param.dt);
end

%% Transition 12 - Exposed[i] to Quarantined[i+1] until i+1 == n_exp
for ind = 1: param.n_exp - 1
    count = count + 1;
    expval(count) = states.x(ind_exp1 + ind - 1)*(param.qr*param.dt);
end

%% Transition 13 - Quarantined[i] to Quarantined[i+1] until i+1 == n_exp
% ind_qua1 = find( strcmp(states.name, 'Quarantined_1') == 1);
for ind = 1 : param.n_exp - 1
    count = count + 1;
    expval(count) = states.x(ind_qua1 + ind - 1)*(1 - param.dr*param.dt);
end

%% Transition 14 - Quarantined[n_exp] to Severe_Infected[1]
if param.n_exp ~= 0
    count = count + 1;
    % ind_quan = find( strcmp(states.name, ['Quarantined_', num2str(param.n_exp)] ) == 1);
    expval(count) = states.x(ind_quan)*(1 - param.dr*param.dt);
end

%% Transition 15 - Infected[i] to Infected[i+1] until i+1 == n_inf
% ind_inf1 = find( strcmp(states.name, 'Infected_1') == 1);
for ind = 1:param.n_inf - 1
    count = count + 1;
    expval(count) = states.x(ind_inf1 + ind - 1)*(1 - param.dr*param.dt - param.sir*param.dt);
end

%% Transition 16 - Severe_Infected[i] to Severe_Infected[i+1] until i+1 == n_inf
% ind_iso1 = find( strcmp(states.name, 'Severe_Infected_1') == 1);
for ind = 1:param.n_inf - 1
    count = count + 1;
    expval(count) = states.x(ind_iso1 + ind - 1)*(1 - param.dr*param.dt);
end

%% Transition 17 - Infected[i] to Severe_Infected[i+1] until i+1 == n_inf
for ind = 1: param.n_inf - 1
    count = count + 1;
    expval(count) = states.x(ind_inf1 + ind - 1)*(param.sir*param.dt);
end

%% Transition 18 - Infected[n_inf] to Recovery_Immunized
count = count + 1;
% ind_infn = find( strcmp(states.name, ['Infected_', numtr2str(param.n_inf)] ) == 1);
expval(count) = states.x(ind_infn)*param.gamma_im;

%% Transition 19 - Severe_Infected[n_inf] to Recovery Immunized
count = count + 1;
% ind_ison = find( strcmp(states.name, ['Severe_Infected_', num2str(param.n_inf)] ) == 1);
expval(count) = states.x(ind_ison)*param.gamma_im;

%% Transition 20 - Infected[n_inf] to Susceptible
count = count + 1;
% ind_infn = find( strcmp(states.name, ['Infected_', num2str(param.n_inf)] ) == 1);
expval(count) = states.x(ind_infn)*(1 - param.gamma_mor - param.gamma_im);

%% Transition 21 - Severe_Infected[n_inf] to Susceptible
count = count + 1;
% ind_ison = find( strcmp(states.name, ['Severe_Infected_', num2str(param.n_inf)] ) == 1);
if sum(states.x(ind_iso)) < param.hosp_capacity
    expval(count) = states.x(ind_ison)*(1 - param.gamma_mor1 - param.gamma_im);
else
    expval(count) = states.x(ind_ison)*(1 - param.gamma_mor2 - param.gamma_im);
end

%% Transition 22 - Infected[n_inf] to Dead
count = count + 1;
% ind_infn = find( strcmp(states.name, ['Infected_', num2str(param.n_inf)] ) == 1);
expval(count) = states.x(ind_infn)*param.gamma_mor;

%% Transition 23 - Severe_Infected[n_inf] to Dead
count = count + 1;
% ind_ison = find( strcmp(states.name, ['Severe_Infected_', num2str(param.n_inf)] ) == 1);
if sum(states.x(ind_iso)) < param.hosp_capacity
    expval(count) = states.x(ind_ison)*param.gamma_mor1;
else
    % The mortality rate of the severe infected beyond the hospital capacity 
    % is much higher.
    expval(count) = states.x(ind_ison)*param.gamma_mor2;
end
%% Randomly generate the transition value based on the expected value

num_trans = numel(expval);
for ind = 1 : num_trans
    if expval(ind) < 10
        temp1 = ceil( expval(ind)*10 + eps );    
        dx(ind) = sum( rand( temp1, 1) < (expval(ind)/temp1));
    else
        dx(ind) = round(expval(ind));
    end
end

% Apply the changes for the transitions to the corresponding source and destination states
for ind = 1 : num_trans
    
    sind = trans.source_ind(ind);
    dind = trans.dest_ind(ind);
    
    temp = states.x(sind) - dx(ind);
    
    if sind == 1
        states.x(sind) = temp;
        states.x(dind) = states.x(dind) + dx(ind);
    elseif temp <= 0
        states.x(dind) = states.x(dind) + states.x(sind);
        states.x(sind) = 0;
    else
        states.x(sind) = temp;
        states.x(dind) = states.x(dind) + dx(ind);
    end
    
end

states.dx = dx;

end