function moses_visualize(states_arr, time_arr, states, param)
%
%
%
% Author: Atakan Varol
% Date: March 2015


f1 = figure(1);
set(f1,'Position',[60 60 1200 700]);
subplot(3,1,1:2)
hold on;

ind_sus = find(strcmp(states.type, 'Susceptible') == 1);
tot_sus = sum(states_arr(:,ind_sus),2);

ind_exp = find(strncmp(states.name, 'Exposed_', 8) == 1 );
tot_exp = sum(states_arr(:,ind_exp),2);

ind_qua = find(strncmp(states.name, 'Quarantined_',12) == 1);
tot_qua = sum(states_arr(:,ind_qua),2);

ind_inf = find(strncmp(states.name, 'Infected_',9) == 1);
tot_inf = sum(states_arr(:,ind_inf),2);

ind_iso = find(strncmp(states.name, 'Severe_Infected_',16) == 1);
tot_iso = sum(states_arr(:,ind_iso),2);

ind_imm = find(strcmp(states.type, 'Immunized') == 1);
tot_imm = sum(states_arr(:,ind_imm),2);

legend_list = {};
if sum(tot_sus) > 0;
    p1 = stairs( time_arr, tot_sus, 'g-', 'LineWidth', 2 );
    legend_list = [legend_list, 'Susceptible'];
end
if sum(tot_exp) > 0;
    p2 = stairs( time_arr, tot_exp, 'm-', 'LineWidth', 2 );
    legend_list = [legend_list, 'Exposed'];
end
if sum(tot_qua) > 0;
    p3 = stairs( time_arr, tot_qua, 'm--', 'LineWidth', 2);
    legend_list = [legend_list, 'Quarantined'];
end
if sum(tot_inf) > 0;
    p4 = stairs( time_arr, tot_inf, 'r-', 'LineWidth', 2 );
    legend_list = [legend_list, 'Infected'];
end
if sum(tot_iso) > 0;
    p5 = stairs( time_arr, tot_iso, 'r--','LineWidth', 2);
    legend_list = [legend_list, 'Severe Infected'];
end
if sum(tot_imm) > 0;
    p6 = stairs( time_arr, tot_imm, 'b-', 'LineWidth',2 );
    legend_list = [legend_list, 'Immunized'];
end
if sum(states_arr(:,end)) > 0;
    p7 = stairs( time_arr, states_arr(:, end), 'k-', 'LineWidth',2 );
    legend_list = [legend_list, 'Dead'];    
end
legend( legend_list, 'FontName','Arial', 'FontSize', 12','FontWeight','Demi','Location','NorthOutside','Orientation','Horizontal');

ylabel('Number of individuals','FontName','Arial', 'FontSize', 12, 'FontWeight', 'Demi');
xlim([0 param.sim_len]);
ylim([0 max(tot_inf)*1.2]);
grid on; box on; 

subplot 313
yyaxis left
stairs( time_arr, states_arr(:, end), 'LineWidth',2 );
ylabel('Num. of Dead', 'FontSize', 12, 'FontWeight', 'Demi');
xlim([0 param.sim_len]);
ylim([0 18000]);
xlabel('Time (days)', 'FontName','Arial', 'FontSize', 12, 'FontWeight', 'Demi');
grid on; box on;
% Real-data for Lombardy from the Italian Government repository
% https://github.com/pcm-dpc/COVID-19/tree/master/dati-regioni
% The first column represents the day starting with 24 Feb 2020
% The second column represents the number of deaths for each day.
Lombardy_data= ...
[0	6
1	9
2	9
3	14
4	17
5	23
6	24
7	38
8	55
9	73
10	98
11	135
12	154
13	267
14	333
15	468
16	617
17	744
18	890
19	966
20	1218
21	1420
22	1640
23	1959
24	2168
25	2549
26	3095
27	3456
28	3776
29	4178
30	4474
31	4861
32	5402
33	5944
34	6360
35	6818
36	7199];
hold on; 
plot(Lombardy_data(:,1) + 33, Lombardy_data(:,2),'r-x');
hold off;
yyaxis right
hold on
stairs( time_arr, tot_iso, 'LineWidth', 2);
plot([0 param.sim_len], [param.hosp_capacity param.hosp_capacity], 'LineWidth',2 );
xlim([0 param.sim_len])
lgd1 = legend('Dead','Real Data', 'Severe Infected', 'Hospital Capacity',  'Location','northwest','Orientation','Vertical');
lgd1.FontSize = 11;
hold off;
end