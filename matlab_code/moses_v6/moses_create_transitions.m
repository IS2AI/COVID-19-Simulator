function trans = moses_create_transitions(states, param)
%
% TRANS = MOSES_CREATE_TRANSITIONS(STATES, PARAM) defines the simulator transitions.
%
% TRANS = MOSES_CREATE_TRANSITIONS(STATES, PARAM) defines the source state
% and destination state for each transitions. The output is the TRANS
% structure. This structure consists of the following:
%
%   SOURCE: cell array containing transition source state string names
%   DEST: cell array containing transition destination state string names
%   TRANS_TYPE:  is a cell array containing solver option 'Stoch' (Stochastic) or
%   'Deter' (Deterministic) transition type.
%   SOURCE_IND: Indices of the transition source states
%   DEST_IND: Indices of the transition destination states
%
% Author: Atakan Varol
% Date: March 2015


%% Transition 1 - Birth to Susceptible
source{1} = 'Birth';
dest{1} = 'Susceptible';

%% Transition 2 - Birth to Maternally Immunized
source{2} = 'Birth';
dest{2} = 'Maternally_Immunized';

%% Transition 3 - Any State except Birth to Dead (Natural Mortality)
count = 2;
for ind = 2 : param.num_states - 1
    count = count + 1;
    source{count} = states.name{ind};
    dest{count} = 'Dead';
end

%% Transition 4 - Susceptible to Vaccinated[1]
if param.vr ~= 0
    count = count + 1;
    source{count} = 'Susceptible';
    dest{count} = 'Vaccinated_1';
end

%% Transition 5 - Vaccinated[i] to Vaccinated[i+1] until i+1 == n_vac
if param.n_vac ~= 0 
for ind = 1 : param.n_vac - 1
    count = count + 1;
    source{count} = states.name{ 2 + ind};
    dest{count} = states.name{ 3 + ind};
end
end

%% Transition 6 - Vaccinated[n_vac] to Vaccination_Immunized
if param.vr ~= 0
count = count + 1;
temp1 = ['Vaccinated_', num2str(param.n_vac)];
ind_vac = strcmp(states.name, temp1);
source{count} = ['Vaccinated_', num2str(param.n_vac)];
dest{count} = 'Vaccination_Immunized';
end

%% Transition 7 - Vaccinated[n_vac] to Susceptible
if param.vr ~= 0 
count = count + 1;
source{count} = ['Vaccinated_', num2str(param.n_vac)];
dest{count} = 'Susceptible';
end

%% Transition 8 - Susceptible to Exposed[1]
if param.n_exp ~= 0
    count = count + 1;
    source{count} = 'Susceptible';
    dest{count} =  'Exposed_1';
end

%% Transition 9 - Susceptible to Infected[1]
count = count + 1;
source{count} = 'Susceptible';
dest{count} =  'Infected_1';

%% Transition 10 - Exposed[i] to Exposed[i+1] until i+1 == n_exp
for ind = 1 : param.n_exp - 1
    count = count + 1;
    source{count} = ['Exposed_', num2str(ind) ];
    dest{count} = ['Exposed_', num2str(ind+1)];
end

%% Transition 11 - Exposed[n_inc] to Infected[1]
if param.n_exp ~= 0
    count = count + 1;
    source{count} = ['Exposed_', num2str(param.n_exp)];
    dest{count} = 'Infected_1';
end

%% Transition 12 - Exposed[i] to Quarantined[i+1] until i+1 == n_exp
for ind = 1: param.n_exp - 1
    count = count + 1;
    source{count} = ['Exposed_', num2str(ind) ];
    dest{count} = ['Quarantined_', num2str(ind+1)];
end

%% Transition 13 - Quarantined[i] to Quarantined[i+1] until i+1 == n_exp
for ind = 1:param.n_exp - 1
    count = count + 1;
    source{count} = ['Quarantined_', num2str(ind)];
    dest{count} = ['Quarantined_', num2str(ind+1)];
end

%% Transition 14 - Quarantined[n_exp] to Isolated[1]
if param.n_exp ~=0
    count = count + 1;
    source{count} = ['Quarantined_', num2str(param.n_exp)];
    dest{count} = 'Isolated_1';
end

%% Transition 15 - Infected[i] to Infected[i+1] until i+1 == n_inf
for ind = 1:param.n_inf - 1
    count = count + 1;
    source{count} = ['Infected_', num2str(ind)];
    dest{count} = ['Infected_', num2str(ind+1)];
end

%% Transition 16 - Isolated[i] to Isolated[i+1] until i+1 == n_inf
for ind = 1:param.n_inf - 1
    count = count + 1;
    source{count} = ['Isolated_', num2str(ind)];
    dest{count} = ['Isolated_', num2str(ind+1)];
end

%% Transition 17 - Severe_Infected[i] to Severe_Infected[i+1] until i+1 == n_inf
for ind = 1:param.n_inf - 1
    count = count + 1;
    source{count} = ['Severe_Infected_', num2str(ind)];
    dest{count} = ['Severe_Infected_', num2str(ind+1)];
end

%% Transition 18 - Infected[i] to Severe_Infected[i+1] until i+1 == n_inf
for ind = 1: param.n_inf - 1
    count = count + 1;
    source{count} = ['Infected_', num2str(ind) ];
    dest{count} = ['Severe_Infected_', num2str(ind+1)];
end

%% Transition 19 - Isolated[i] to Severe_Infected[i+1] until i+1 == n_inf
for ind = 1: param.n_inf - 1
    count = count + 1;
    source{count} = ['Isolated_', num2str(ind) ];
    dest{count} = ['Severe_Infected_', num2str(ind+1)];
end

%% Transition 20 - Infected[n_inf] to Recovery_Immunized
count = count + 1;
source{count} = ['Infected_', num2str(param.n_inf)];
dest{count} = 'Recovery_Immunized';

%% Transition 21 - Isolated[n_inf] to Recovery Immunized
count = count + 1;
source{count} = ['Isolated_', num2str(param.n_inf)];
dest{count} = 'Recovery_Immunized';

%% Transition 22 - Severe_Infected[n_inf] to Recovery Immunized
count = count + 1;
source{count} = ['Severe_Infected_', num2str(param.n_inf)];
dest{count} = 'Recovery_Immunized';

%% Transition 23 - Infected[n_inf] to Susceptible
count = count + 1;
source{count} = ['Infected_', num2str(param.n_inf)];
dest{count} = 'Susceptible';

%% Transition 24 - Isolated[n_inf] to Susceptible
count = count + 1;
source{count} = ['Isolated_', num2str(param.n_inf)];
dest{count} = 'Susceptible';

%% Transition 25 - Severe_Infected[n_inf] to Susceptible
count = count + 1;
source{count} = ['Severe_Infected_', num2str(param.n_inf)];
dest{count} = 'Susceptible';

%% Transition 26 - Infected[n_inf] to Dead
count = count + 1;
source{count} = ['Infected_', num2str(param.n_inf)];
dest{count} = 'Dead';

%% Transition 27 - Severe_Infected[n_inf] to Dead
count = count + 1;
source{count} = ['Severe_Infected_', num2str(param.n_inf)];
dest{count} = 'Dead';

% Assign state indices to the transition source and destinations
num_trans = numel(source);
source_ind = zeros(num_trans,1);
dest_ind = zeros(num_trans,1);
for ind = 1:num_trans    
    source_ind(ind) = find( strcmp(states.name, source{ind}) == 1);
    dest_ind(ind) = find( strcmp(states.name, dest{ind}) == 1);
end

% Collect the output variables in a struct
trans.source = source;
trans.dest = dest;
trans.source_ind = source_ind;
trans.dest_ind = dest_ind;

end