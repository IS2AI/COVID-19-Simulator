% Run the script and pick the csv result file to generate visualizations

close all; clc; warning off; % Close all figure windows

% Opens the filepicker GUI
[file,filedir] = uigetfile('*.csv');
filedir1 = [filedir,file];
filedir2 = join(filedir1);

% read csv into table and convert table fields into arrays
T = readtable(filedir2,'PreserveVariableNames',1);
filename = split(file, '.'); % get the filename
filename = char(filename(1));
days = table2array(T(:,1)); % simulation length
infected = table2array(T(:,3));
exposed = table2array(T(:,4));
sev_inf = table2array(T(:,5));
quarantined = table2array(T(:,6));
immunized = table2array(T(:,7));
susceptible = table2array(T(:,8));
dead = table2array(T(:,9));
isolated = table2array(T(:,10));

%Create figure for plotting
f1 = figure('Name', filename); 
set(f1,'Position',[60 60 1440 700])
subplot(2,1,1)
hold on;
legend_list = {};

p2 = plot( days, isolated, 'm-', 'LineWidth', 2 );
legend_list = [legend_list, 'Isolated'];

p3 = plot( days, quarantined, 'm--', 'LineWidth', 2);
legend_list = [legend_list, 'Quarantined'];

p4 = plot( days, infected, 'r-', 'LineWidth', 2 );
legend_list = [legend_list, 'Infected'];

p5 = plot( days, sev_inf, 'r--','LineWidth', 2);
legend_list = [legend_list, 'Severe Infected'];

p6 = plot( days, immunized, 'b-', 'LineWidth',2 );
legend_list = [legend_list, 'Immunized'];

p7 = plot( days, dead, 'k-', 'LineWidth',2 );
legend_list = [legend_list, 'Dead'];    

leg1 = legend( legend_list, 'FontName','Arial', 'FontSize', 12','FontWeight','Demi','Location','EastOutside','Orientation','Vertical');

str1 = ['Time (days)'];
xlbl = join(str1);
xlabel(xlbl, 'FontName','Arial', 'FontSize', 12, 'FontWeight', 'Demi');
ylabel('Number of individuals','FontName','Arial', 'FontSize', 12, 'FontWeight', 'Demi');
title( ['COVID-19 Simulation Results for: ', filename]);
xlim([0 days(end)]);
ylim([0 infected(end)*2])
%ytickformat('%.4f')
grid on; box on; 

subplot 212
p1 = plot( days, dead, 'k-', days, sev_inf, 'r-','LineWidth', 2);
xlabel(xlbl, 'FontName','Arial', 'FontSize', 12, 'FontWeight', 'Demi');
ylabel('Number of individuals','FontName','Arial', 'FontSize', 12, 'FontWeight', 'Demi');
legend( {'Dead', 'Severe Infected'}, 'FontName','Arial', 'FontSize', 12','FontWeight','Demi','Location','EastOutside','Orientation','Vertical');
xlim([0 days(end)]);
grid on;





