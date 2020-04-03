%% Matlab based Open-Source Epidemic Simulator

clc; clear; close all; format compact; clear moses_stoch_solver;  % Clean the environment

% Set the parameters and initialize the state values

% For recreating the simulations in the paper use one of the following lines
% [param, init] = moses_init_sim1;
% [param, init] = moses_init_sim2;
% [param, init] = moses_init_sim3;
% [param, init] = moses_init_sim4;
% [param, init] = moses_init_sim5;
  [param, init] = moses_init_lombardy;

% Create the states vector and assign labels to the states
[states, param] = moses_create_states(param, init);
% Create the transitions between states
trans = moses_create_transitions(states, param);

% Store the results ot the states array (init to gain speed)
states_arr = zeros(param.num_sim, param.num_states);
time_arr = (0:param.num_sim-1)*param.dt;
tic; 
for ind = 1 : param.num_sim   % Main Simulation Loop
    
    % Parameters such as vaccination rate can be modified here for control purposes
      if ind > 53*24 && ind < 70*24
          param.beta_exp = 0.12;
          param.qr = 0;
      end
      if ind >= 70*24
          param.beta_exp = 0.06;
      end   
      if ind >= 80*24
          param.beta_exp = 0.03;
      end
    
    % Run the stochastic solve
    states = moses_stoch_solver(states, trans, param);
    
    % Store the states for visualization purposes
    states_arr(ind,:) = states.x;
    
    if mod(ind, param.disp_interval) == 0
        el_time = round(toc*100)/100; % Elapsed time
        disp([ 'Iteration ', num2str(ind) ,' in ',  num2str(el_time), ' secs.'] );
    end
end

if param.vis_on   % Visualize the Results
    moses_visualize(states_arr, time_arr, states, param);
end

% Save the results
if param.save_res
    fname = ['moses_' num2str(round(now*10000) ) ];
    save(fname, 'states_arr', 'param', 'init');
end