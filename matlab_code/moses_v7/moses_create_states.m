function [states, param] = moses_create_states(param, init)
%
% [STATES] = MOSES_CREATE_STATES(PARAM , INIT) creates states struct, which
% contains the state vector and a cell array containing the name of the
% states.
%
% Author: Atakan Varol
% Date: January 2016

% birth
% susceptible, vaccinated[1], vaccinated[2], ..., vaccinated[n_vac],
% exposed[1], exposed[2], ..., exposed[n_exp];
% quarantined[1], quarantined[2], ..., quarantined[n_qua];
% infected[1], infected[2], ..., infected[n_inf];
% isolated[1], isolated[2], ..., isolated[n_inf];
% vac_im, mat_im, rec_im,
% dead

states.x = [0, 0];
states.name{1} = 'Birth';           states.type{1} = 'Birth';
states.name{2} = 'Susceptible';     states.type{2} = 'Susceptible';
% Vaccination Satess
states.x = [states.x , zeros(1, param.n_vac )];
count = numel(states.name);
for ind = 1:param.n_vac
    states.name{count + ind} = ['Vaccinated_' , num2str(ind)];
    states.type{count + ind} = 'Susceptible';
end

% Exposed States (Includes both exposed and quarantined)
states.x = [states.x, zeros(1, param.n_exp)];
count = numel(states.name);
for ind = 1:param.n_exp
    states.name{count + ind} = ['Exposed_' , num2str(ind)];
    states.type{count + ind} = 'Exposed';
end
states.x = [states.x, zeros(1, param.n_exp)];
count = numel(states.name);
for ind = 1:param.n_exp
    states.name{count + ind} = ['Quarantined_' , num2str(ind)];
    states.type{count + ind} = 'Exposed';
end

% Infected States (Includes infected, isolated and severe infected)
states.x = [states.x, zeros(1, param.n_inf)];
count = numel(states.name);
for ind = 1:param.n_inf
    states.name{count + ind} = ['Infected_' , num2str(ind)];
    states.type{count + ind} = 'Infected';
end
states.x = [states.x, zeros(1, param.n_inf)];
count = numel(states.name);
for ind = 1:param.n_inf
    states.name{count + ind} = ['Isolated_' , num2str(ind)];
    states.type{count + ind} = 'Infected';
end
states.x = [states.x, zeros(1, param.n_inf)];
count = numel(states.name);
for ind = 1:param.n_inf
    states.name{count + ind} = ['Severe_Infected_' , num2str(ind)];
    states.type{count + ind} = 'Infected';
end

% Add the Immunized and dead states
states.x = [states.x, 0, 0, 0, 0];
count = numel(states.name);
states.name{count+1} = 'Vaccination_Immunized'; states.type{count+1} = 'Immunized';
states.name{count+2} = 'Maternally_Immunized';  states.type{count+2} = 'Immunized';
states.name{count+3} = 'Recovery_Immunized';    states.type{count+3} = 'Immunized';
states.name{count+4} = 'Dead';                  states.type{count+4} = 'Dead';
states.name = states.name;
states.dx = zeros(size(states.x));

param.num_states = numel(states.x);                   % Number of states
param.num_sim = round(param.sim_len/param.dt) + 1;    % Number of simulation time instances

% Assign the initial values of the states
field_list = fields(init);
temp_val = struct2array(init);
for ind = 1:numel( field_list )
    [temp temp_ind] = find( ~cellfun(@isempty, strfind(lower(states.name), field_list{ind} )));
    if ~isempty(temp_ind)
        states.x( temp_ind(1) ) = temp_val(ind);
    end
end

end