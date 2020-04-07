% Run the script and GUI for picking file to plot will occur

% Opens the filepicker GUI
[file,filedir] = uigetfile('*.csv');
filedir1 = [filedir,file];
filedir2 = join(filedir1);

% read csv into table and convert table fields into arrays
T = readtable(filedir2,'PreserveVariableNames',1);
filename = split(file, '.'); % get the filename
filename = char(filename(1));
days = table2array(T(:,1)); % simulation length
infected = table2array(T(:,2));
exposed = table2array(T(:,3));
sev_inf = table2array(T(:,4));
quarantined = table2array(T(:,5));
immunized = table2array(T(:,6));
susceptible = table2array(T(:,7));
dead = table2array(T(:,8));



%Create figure for plotting
f1 = figure('Name', filename); 
set(f1,'Position',[60 60 1200 700])
subplot(2,1,1)
hold on;
legend_list = {};

p2 = plot( days, exposed, 'm-', 'LineWidth', 2 );
legend_list = [legend_list, 'Exposed'];

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

leg1 = legend( legend_list, 'FontName','Arial', 'FontSize', 12','FontWeight','Demi','Location','NorthOutside','Orientation','Horizontal');

str1 = ['Time (days)            Simulation for: ', filename];
xlbl = join(str1);
xlabel(xlbl, 'FontName','Arial', 'FontSize', 12, 'FontWeight', 'Demi');
ylabel('Number of individuals','FontName','Arial', 'FontSize', 12, 'FontWeight', 'Demi');
xlim([0 days(end)]);
ylim([0 infected(end)*2])
%ytickformat('%.4f')
grid on; box on; 

subplot 212
p1 = plot( days, susceptible, 'g-', 'LineWidth', 2 );
xlabel(xlbl, 'FontName','Arial', 'FontSize', 12, 'FontWeight', 'Demi');
ylabel('Number of individuals','FontName','Arial', 'FontSize', 12, 'FontWeight', 'Demi');
legend( 'Susceptible', 'FontName','Arial', 'FontSize', 12','FontWeight','Demi','Location','NorthOutside','Orientation','Horizontal');
xlim([0 days(end)]);
%ytickformat('%12.1f')
grid on;