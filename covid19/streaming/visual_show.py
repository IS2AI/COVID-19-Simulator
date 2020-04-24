import os
import csv
import numpy as np
from copy import copy
from tornado import gen
from bokeh.plotting import figure
from bokeh.layouts import gridplot, column, row
from bokeh.io import curdoc,output_file, show
from bokeh.models import (CustomJS, CategoricalColorMapper, Panel, Select, Button, DataTable,DateFormatter, TableColumn,
                            PrintfTickFormatter, Legend, Slider, TextInput, CheckboxGroup, Div, ColumnDataSource, FileInput)
from bokeh.models import (CDSView, ColorBar, CustomJSFilter, GeoJSONDataSource, HoverTool, LinearColorMapper, PrintfTickFormatter)
from bokeh.models.widgets.inputs import DatePicker
from datetime import datetime, date, timedelta
import json
import config
from data_stream import test_population

class Visual:

    def __init__(self, callbackFunc, running):

        self.running = running
        self.callbackFunc = callbackFunc
        # define the sources for plot and map
        self.source = ColumnDataSource(dict(x=[0], sus=[config.param_init_susceptible[config.region]], exp=[config.param_init_exposed[config.region]], inf=[0], sin=[0],
                                        qua=[0], imm=[0], dea=[0], text=[""], mdates = [""]))

        self.sourceJS = ColumnDataSource(dict(text=[]))

        mcallback = CustomJS(args=dict(source=self.source), code="""
            window.data  = source.data

            console.log(source)
        """)
        self.source.js_on_change('change',mcallback)

        self.tools = 'pan, box_zoom, wheel_zoom, reset'
        self.plot_options = dict(plot_width=800, plot_height=600, tools = [self.tools])
        self.updateValue = True
        self.pAll = self.definePlot(self.source)
        self.doc = curdoc()
        self.layout()
        self.prev_y1 = 0

        # initialize the widgets' values
        self.region_names = config.region_names

        self.init_exposed.value = config.param_init_exposed[config.region]
        self.sus_to_exp_slider.value = config.param_beta_exp[config.region]
        self.param_qr_slider.value = config.param_qr[config.region]
        self.param_sir.value = config.param_sir[config.region]
        self.param_hosp_capacity.value = config.param_hosp_capacity[config.region]
        self.param_gamma_mor1.value = config.param_gamma_mor1[config.region]
        self.param_gamma_mor2.value = config.param_gamma_mor2[config.region]
        self.param_gamma_im.value = config.param_gamma_im[config.region]
        self.param_eps_exp.value = config.param_eps_exp[config.region]
        self.param_eps_qua.value = config.param_eps_qua[config.region]
        self.param_eps_sev.value = config.param_eps_sev[config.region]

        self.start_date = date.today()
        # transition_matrix checkbox
        self.box1 = list(range(0, 17))
        self.box2 = list(range(0, 17))
        self.box3 = list(range(0, 17))

    def definePlot(self, source):

        # format the text of the plot
        p1 = figure(**self.plot_options, title='Covid Simulation',  toolbar_location='above')
        p1.yaxis.axis_label = 'Number of people'
        p1.xaxis.axis_label = 'Simulation time (days)'
        p1.xaxis[0].formatter = PrintfTickFormatter(format="%9.0f")
        p1.yaxis[0].formatter = PrintfTickFormatter(format="%9.0f")
        p1.xaxis.major_label_text_font_size = "10pt"
        p1.yaxis.major_label_text_font_size = "10pt"

        p2 = figure(**self.plot_options, title='Number of Susceptible people', toolbar_location='above')
        p2.yaxis.axis_label = 'Number of people'
        p2.xaxis.axis_label = 'Simulation time (days)'
        p2.xaxis[0].formatter = PrintfTickFormatter(format="%9.0f")
        p2.yaxis[0].formatter = PrintfTickFormatter(format="%9.0f")
        p2.xaxis.major_label_text_font_size = "10pt"
        p2.yaxis.major_label_text_font_size = "10pt"

        # format the plot line
        r0 = p2.line(source =source, x='x', y='sus', color='cyan', line_width=1,line_dash='dashed', legend='Susceptible')
        r1 = p2.circle(source=source, x='x', y='sus', color='cyan', size=10, legend='Susceptible')

        r2 = p1.line(source=source, x='x', y='exp',color='gold',line_width=1,line_dash='dotted', legend='Exposed')
        r3 = p1.circle(source=source, x='x', y='exp',color='gold',size=10, legend='Exposed')

        r4 = p1.line(source=source, x='x', y='inf',color='white',line_width=1,line_dash='dotted', legend='Infected')
        r5 = p1.circle(source=source, x='x', y='inf',color='white',size=10, legend='Infected')

        r6 = p1.line(source=source, x='x', y='sin',color='purple',line_width=1,line_dash='dotted', legend='Severe Infected')
        r7 = p1.circle(source=source, x='x', y='sin',color='purple',size=10, legend='Severe Infected')

        r8 = p1.line(source=source, x='x', y='qua',color='lime',line_width=1,line_dash='dotted', legend='Quarantined')
        r9 = p1.circle(source=source, x='x', y='qua',color='lime',size=10, legend='Quarantined')

        r10 = p1.line(source=source, x='x', y='imm',color='deepskyblue',line_width=1,line_dash='dotted', legend='Immunized')
        r11 = p1.circle(source=source, x='x', y='imm',color='deepskyblue',size=10, legend='Immunized')

        r12 = p1.line(source=source, x='x', y='dea',color='red',line_width=1,line_dash='dotted', legend='Dead')
        r13 = p1.circle(source=source, x='x', y='dea',color='red',size=10, legend='Dead')

        legend = Legend(items=[
                                ('Exposed', [r2, r3]),
                                ('Infected', [r4, r5]),
                                ('Severe Infected', [r6, r7]),
                                ('Quarantined', [r8, r9]),
                                ('Immunized', [r10, r11]),
                                ('Dead', [r12, r13])])

        # legends
        p1.legend.click_policy = 'hide'
        p1.background_fill_color = "black"
        p1.background_fill_alpha = 0.8
        p1.legend.location = "top_left"
        p1.legend.background_fill_color = "cyan"
        p1.legend.background_fill_alpha = 0.5
        p1.outline_line_width = 7
        p1.outline_line_alpha = 0.9
        p1.outline_line_color = "black"

        p2.legend.click_policy = 'hide'
        p2.background_fill_color = "black"
        p2.background_fill_alpha = 0.8
        p2.legend.location = "top_left"
        p2.legend.background_fill_color = "cyan"
        p2.legend.background_fill_alpha = 0.5
        p2.outline_line_width = 7
        p2.outline_line_alpha = 0.9
        p2.outline_line_color = "black"

        kz_map_tag = Div(text="""<div id="svg_holder" style="float:left;"> <svg width="780" height="530" id="statesvg"></svg> <div id="tooltip"></div>   </div>""", width=960, height=600)
        kz_map_row = row(kz_map_tag)
        pAll = row(p1, kz_map_row)

        return pAll

    #@gen.coroutine
    def update(self, change_view):

        region_states = dict()
        # obtain the state values
        new_nodes_all = config.new_plot_all
        # construct the array for plotting the states
        newx = [0]
        state_inf = [0]
        state_sus=[config.param_init_susceptible[config.region]]
        state_exp = [config.param_init_exposed[config.region]]
        state_sin = [0]
        state_qua = [0]
        state_imm = [0]
        state_dea = [0]

        tmp_state_inf = [0]
        tmp_state_sus=[config.param_init_susceptible[config.region]]
        tmp_state_exp = [config.param_init_exposed[config.region]]
        tmp_state_sin = [0]
        tmp_state_qua = [0]
        tmp_state_imm = [0]
        tmp_state_dea = [0]

        start_date = self.start_date
        cur_date = (start_date + timedelta(config.counter_func)).strftime("%d %b %Y")
        start_date = self.start_date.strftime("%d %b %Y")

        # for graph
        if new_nodes_all != [] and config.region != 17:
            for i in range(len(config.new_plot_all)):
                state_inf.append(new_nodes_all[i][:, config.region, 0][-1] + new_nodes_all[i][:, config.region, 7][-1])
                state_exp.append(new_nodes_all[i][:, config.region, 1][-1])
                state_sin.append(new_nodes_all[i][:, config.region, 2][-1])
                state_qua.append(new_nodes_all[i][:, config.region, 3][-1])
                state_imm.append(new_nodes_all[i][:, config.region, 4][-1])
                state_sus.append(new_nodes_all[i][:, config.region, 5][-1])
                state_dea.append(new_nodes_all[i][:, config.region, 6][-1])
                newx = config.param_sim_len[0]*(np.arange(config.counter_func+1))

                # for map
                regions_ids = [ lregion for lregion in range(17)]
                for region in regions_ids:
                    if region in region_states:
                        region_states[region]["tmp_state_inf"].append(new_nodes_all[i][:, region, 0][-1]+ new_nodes_all[i][:, region, 7][-1])
                        region_states[region]["tmp_state_sin"].append(new_nodes_all[i][:, region, 2][-1])
                        region_states[region]["tmp_state_exp"].append(new_nodes_all[i][:, region, 1][-1])
                        region_states[region]["tmp_state_qua"].append(new_nodes_all[i][:, region, 3][-1])
                        region_states[region]["tmp_state_imm"].append(new_nodes_all[i][:, region, 4][-1])
                        region_states[region]["tmp_state_sus"].append(new_nodes_all[i][:, region, 5][-1])
                        region_states[region]["tmp_state_dea"].append(new_nodes_all[i][:, region, 6][-1])
                    else:
                        tmp_data = {
                            "tmp_state_inf": [],
                            "tmp_state_sin": [],
                            "tmp_state_exp": [],
                            "tmp_state_qua": [],
                            "tmp_state_imm": [],
                            "tmp_state_sus": [],
                            "tmp_state_dea": []
                            }

                        tmp_data["tmp_state_inf"].append(new_nodes_all[i][:, region, 0][-1]+ new_nodes_all[i][:, region, 7][-1])
                        tmp_data["tmp_state_sin"].append(new_nodes_all[i][:, region, 2][-1])
                        tmp_data["tmp_state_exp"].append(new_nodes_all[i][:, region, 1][-1])
                        tmp_data["tmp_state_qua"].append(new_nodes_all[i][:, region, 3][-1])
                        tmp_data["tmp_state_imm"].append(new_nodes_all[i][:, region, 4][-1])
                        tmp_data["tmp_state_sus"].append(new_nodes_all[i][:, region, 5][-1])
                        tmp_data["tmp_state_dea"].append(new_nodes_all[i][:, region, 6][-1])

                        region_states[region] = tmp_data


        elif new_nodes_all != [] and config.region == 17:
            for i in range(len(config.new_plot_all)):

                state_inf.append(sum(new_nodes_all[i][:, :, 0][-1]) + sum(new_nodes_all[i][:, :, 7][-1]))
                state_exp.append(sum(new_nodes_all[i][:, :, 1][-1]))
                state_sin.append(sum(new_nodes_all[i][:, :, 2][-1]))
                state_qua.append(sum(new_nodes_all[i][:, :, 3][-1]))
                state_imm.append(sum(new_nodes_all[i][:, :, 4][-1]))
                state_sus.append(sum(new_nodes_all[i][:, :, 5][-1]))
                state_dea.append(sum(new_nodes_all[i][:, :, 6][-1]))
                newx = config.param_sim_len[0]*(np.arange(config.counter_func+1))

                regions_ids = [ lregion for lregion in range(17)]
                for region in regions_ids:
                    if str(region) in region_states and type(region_states[region]) is dict:
                        region_states[region]["tmp_state_inf"].append(new_nodes_all[i][:, region, 0][-1] + new_nodes_all[i][:, region, 7][-1])
                        region_states[region]["tmp_state_sin"].append(new_nodes_all[i][:, region, 2][-1])
                        region_states[region]["tmp_state_exp"].append(new_nodes_all[i][:, region, 1][-1])
                        region_states[region]["tmp_state_qua"].append(new_nodes_all[i][:, region, 3][-1])
                        region_states[region]["tmp_state_imm"].append(new_nodes_all[i][:, region, 4][-1])
                        region_states[region]["tmp_state_sus"].append(new_nodes_all[i][:, region, 5][-1])
                        region_states[region]["tmp_state_dea"].append(new_nodes_all[i][:, region, 6][-1])
                    else:
                        tmp_data = {
                            "tmp_state_inf": [],
                            "tmp_state_sin": [],
                            "tmp_state_exp": [],
                            "tmp_state_qua": [],
                            "tmp_state_imm": [],
                            "tmp_state_sus": [],
                            "tmp_state_dea": []
                            }

                        tmp_data["tmp_state_inf"].append(new_nodes_all[i][:, region, 0][-1] + new_nodes_all[i][:, region, 7][-1])
                        tmp_data["tmp_state_sin"].append(new_nodes_all[i][:, region, 2][-1])
                        tmp_data["tmp_state_exp"].append(new_nodes_all[i][:, region, 1][-1])
                        tmp_data["tmp_state_qua"].append(new_nodes_all[i][:, region, 3][-1])
                        tmp_data["tmp_state_imm"].append(new_nodes_all[i][:, region, 4][-1])
                        tmp_data["tmp_state_sus"].append(new_nodes_all[i][:, region, 5][-1])
                        tmp_data["tmp_state_dea"].append(new_nodes_all[i][:, region, 6][-1])

                        region_states[region] = tmp_data

        str_data = json.dumps(region_states, ensure_ascii=False)
        str_mdates = json.dumps([start_date, cur_date],ensure_ascii=False)
        new_data = dict(x=newx, sus=state_sus, exp=state_exp, inf=state_inf, sin=state_sin,
                    qua=state_qua, imm=state_imm, dea=state_dea, text=[str_data]*len(state_imm), mdates=[str_mdates]*len(state_imm))

        self.source.data.update(new_data)

        self.data1 = dict(
                        c0=[(config.transition_matrix[0,i]) for i in range(0,17)],
                        c1=[(config.transition_matrix[1,i]) for i in range(0,17)],
                        c2=[(config.transition_matrix[2,i]) for i in range(0,17)],
                        c3=[(config.transition_matrix[3,i]) for i in range(0,17)],
                        c4=[(config.transition_matrix[4,i]) for i in range(0,17)],
                        c5=[(config.transition_matrix[5,i]) for i in range(0,17)],
                        c6=[(config.transition_matrix[6,i]) for i in range(0,17)],
                        c7=[(config.transition_matrix[7,i]) for i in range(0,17)],
                        c8=[(config.transition_matrix[8,i]) for i in range(0,17)],
                        c9=[(config.transition_matrix[9,i]) for i in range(0,17)],
                        c10=[(config.transition_matrix[10,i]) for i in range(0,17)],
                        c11=[(config.transition_matrix[11,i]) for i in range(0,17)],
                        c12=[(config.transition_matrix[12,i]) for i in range(0,17)],
                        c13=[(config.transition_matrix[13,i]) for i in range(0,17)],
                        c14=[(config.transition_matrix[14,i]) for i in range(0,17)],
                        c15=[(config.transition_matrix[15,i]) for i in range(0,17)],
                        c16=[(config.transition_matrix[16,i]) for i in range(0,17)],)

        self.sourceT.data.update(self.data1)
        self.data_tableT.update()

        # update Results table
        self.dataTest = dict(
                        c00 =  ['Performed, N','Confirmed, T(+)','Not Confirmed, T(-)', 'True Positives, TP', 'False Positives, FP', 'False Negatives, FN', 'True Negatives, TN',
                                'Truly Infected, D(+)', 'Non-Infected, D(-)',],
                        c0=[config.param_test_sum, config.param_t_pos, config.param_t_neg, config.param_true_pos, config.param_false_pos, config.param_false_neg, config.param_true_neg,
                             config.param_d_pos, config.param_d_neg],)

        self.sourceTest.data.update(self.dataTest)
        self.data_tableTest.update()

        self.dataPrev = dict(
                        c0=[config.param_prev_auto[0]],
                        c1=[config.param_prev_auto[1]],
                        c2=[config.param_prev_auto[2]],
                        c3=[config.param_prev_auto[3]],
                        c4=[config.param_prev_auto[4]],
                        c5=[config.param_prev_auto[5]],
                        c6=[config.param_prev_auto[6]],
                        c7=[config.param_prev_auto[7]],
                        c8=[config.param_prev_auto[8]],
                        c9=[config.param_prev_auto[9]],
                        c10=[config.param_prev_auto[10]],
                        c11=[config.param_prev_auto[11]],
                        c12=[config.param_prev_auto[12]],
                        c13=[config.param_prev_auto[13]],
                        c14=[config.param_prev_auto[14]],
                        c15=[config.param_prev_auto[15]],
                        c16=[config.param_prev_auto[16]],)

        self.sourcePrev.data.update(self.dataPrev)
        self.data_tablePrev.update()

    def SelectRegionHandler(self, attr, old, new):
        regions = config.region_names

        for i, region in enumerate(regions):
            if new == region:
                config.region = i
                break
        self.update(True)
        self.slider_update_initial_val(self, old, new)

    def SelectPrevHandler(self, attr, old, new):
        if new == 'Manual':
            config.param_prev_mode = 0
        else:
            config.param_prev_mode = 1
        print(config.param_prev_mode)

    def update_transition_matrix(self):
        nodes_num = 17
        self.param_transition_box = []
        self.param_transition_box.append(self.box1)
        self.param_transition_box.append(self.box2)
        self.param_transition_box.append(self.box3)
        tr_boxes = self.param_transition_box

        param_transition_table = np.zeros((17,3))
        for i, way in enumerate(tr_boxes): # air 0 rail 1 road 2
            for j, node in enumerate(way):
                status = int(node)
                param_transition_table[status, i] = 1
        # load transition matrix
        transition_railway = config.transition_railway.copy()
        transition_airway = config.transition_airway.copy()
        transition_roadway = config.transition_roadway.copy()

        tr_table = [transition_airway, transition_railway, transition_roadway]

        for j, tr in enumerate(tr_table):
            for i in range(17):
                tr[i, :] = tr[i, :]*param_transition_table[i,j]
                tr[:, i] = tr[i, :]*param_transition_table[i,j]

        transition_matrix = 0.5*(transition_railway + transition_airway + transition_roadway)*(config.param_transition_scale[0] )

        for i in range(nodes_num):
            for j in range(nodes_num):
                if transition_matrix[i,j] < 0.01:
                    transition_matrix[i,j] = config.transition_matrix_init[i,j]*config.param_transition_leakage[0] # base data is for 24 days, tran_dt = 1/2

        transition_matrix = transition_matrix.astype(int)

        config.param_transition_table = copy(param_transition_table)
        config.transition_matrix = copy(transition_matrix)
        self.update(False)

    def test_click(self):
        if config.flag_sim == 0:

            config.is_test = True

            test_population(config.param_init_susceptible, config.nodes_old, config.param_test_num, config.param_test_prev, config.param_test_sens, config.param_test_spec)

            self.update(False)
            pass

    def reset_click(self):
        # reset the params
        if config.flag_sim == 0:
            config.new_plot_all = []
            config.counter_func = 0
            config.run_iteration=False

            config.last_state_list = []
            config.nodes_old = []
            config.new_plot = []
            config.is_loaded = False

            self.slider_update_reset(self, 0, 0)
            self.update(False)
            print('[INFO] Resetting the simulation parameters ..')

    def load_click(self):
        # load the previous results
        self.reset_click()
        directory = 'results' + '/' +  config.param_save_file
        fname = directory + '/' + 'Kazakhstan' + '.csv'

        if os.path.isfile(fname):
            with open(fname,"r") as f:
                reader = csv.reader(f,delimiter = ",")
                data = list(reader)
                row_n = len(data)

            # reset
            new_plot = np.zeros((row_n, 17, 8))
            config.box_time = np.zeros((17, 3, row_n))
            config.arr_for_save = np.zeros((row_n, 17, 15))

            # fill the new_plot
            for j in range(config.nodes_num):
                filename =  directory + '/' + config.region_names[j] + '.csv'
                with open(filename, newline='') as csvfile:
                    csvreader = csv.reader(csvfile, delimiter=',')
                    count_row = 0
                    for row in csvreader:
                        if count_row > 0:
                            # states
                            data_states = [(float(item)) for item in row[2:10]]
                            data_states = np.array(data_states)
                            new_plot[count_row,j,:] = data_states[:]
                            # transition
                            data_box = [(float(item)) for item in row[25:28]]
                            data_box = np.array(data_box)
                            config.box_time[j,:, count_row] = data_box[:]
                            # parameters
                            data_arr = [(float(item)) for item in row[10:25]]
                            data_arr = np.array(data_arr)
                            config.arr_for_save[count_row,j,:] = data_arr[:]
                        count_row += 1
                        if count_row == (2):
                            config.last_date = row[1]


            config.counter_load = count_row-1
            config.new_plot = new_plot

            # restore parameters
            config.param_init_exposed = config.arr_for_save[config.counter_func-1,:,0]
            config.param_beta_exp = config.arr_for_save[config.counter_func-1,:,1]
            config.param_qr  = config.arr_for_save[config.counter_func-1,:,2]
            config.param_sir  = config.arr_for_save[config.counter_func-1,:,3]
            config.param_hosp_capacity = config.arr_for_save[config.counter_func-1,:,4]

            config.param_gamma_mor1 = config.arr_for_save[config.counter_func-1,:,5]
            config.param_gamma_mor2 = config.arr_for_save[config.counter_func-1,:,6]
            config.param_gamma_im = config.arr_for_save[config.counter_func-1,:,7]

            config.param_eps_exp = config.arr_for_save[config.counter_func-1,:,8]
            config.param_eps_qua = config.arr_for_save[config.counter_func-1,:,9]
            config.param_eps_sev  = config.arr_for_save[config.counter_func-1,:,10]

            config.param_t_exp = config.arr_for_save[config.counter_func-1,:,11]
            config.param_t_inf = config.arr_for_save[config.counter_func-1,:,12]

            config.param_transition_leakage = config.arr_for_save[config.counter_func-1,:,13]
            config.param_transition_scale = config.arr_for_save[config.counter_func-1,:,14]
            config.param_transition_table = copy(config.box_time[:,:,config.counter_func-1])

            l1 = [i for i, x in enumerate(list(config.param_transition_table[:,0])) if x > 0]
            l2 = [i for i, x in enumerate(list(config.param_transition_table[:,1])) if x > 0]
            l3 = [i for i, x in enumerate(list(config.param_transition_table[:,2])) if x > 0]

            self.checkbox_group1.active = l1
            self.checkbox_group2.active = l2
            self.checkbox_group3.active = l3

            self.slider_update_initial_val(0,0,0)

            filename =  directory + '/' + 'states_x' + '.csv'
            with open(filename,"r") as f:
                reader = csv.reader(f,delimiter = ",")
                r_count = 0
                for row in reader:
                    if r_count < 17:
                        temp = np.array(row)
                        config.last_state_list.append(temp)
                    else:
                        temp = np.array(row)
                        config.load_states_name.append(temp)
                    r_count += 1

            config.is_loaded = True
            # plot graph
            if config.flag_sim == 0:
                config.load_iteration=True

            self.datepicker.value = config.last_date

        else:
            print('[INFO] No such folder to load the results.')

    def run_click(self):
        if config.flag_sim == 0:
            self.update_transition_matrix()
            config.run_iteration=True
            self.update(False)

    def save_file_click(self):

        if config.flag_sim == 0:
            # points*nodes*states
            info = config.header_file_csv
            info2 = config.header_file_csv2

            directory = 'results' + '/' +  config.param_save_file
            if not os.path.exists(directory):
                os.makedirs(directory)

            box_corr = config.box_time
            if config.new_plot_all:
                for j in range(17):
                    filename =  directory + '/' + self.region_names[j] + '.csv'
                    with open(filename, 'w', newline='') as csvfile:
                        data_writer = csv.writer(csvfile, delimiter=',', escapechar=' ', quoting=csv.QUOTE_NONE)
                        #points*nodes*states
                        data_writer.writerow([info])
                        for iter in range(1,config.counter_func+1):
                            one_arr = config.new_plot_all[iter-1] #
                            one_arr_node = one_arr[-1,j,:].astype(int)
                            m = 17

                            curr_date = self.start_date + timedelta(iter-1)
                            one_arr_node = np.append(one_arr_node, (config.arr_for_save[iter,j,0], config.arr_for_save[iter,j,1], config.arr_for_save[iter,j,2],
                                                     config.arr_for_save[iter,j,3], config.arr_for_save[iter,j,4], config.arr_for_save[iter,j,5], config.arr_for_save[iter,j,6],
                                                     config.arr_for_save[iter,j,7], config.arr_for_save[iter,j,8], config.arr_for_save[iter,j,9], config.arr_for_save[iter,j,10],
                                                     config.arr_for_save[iter,j,11], config.arr_for_save[iter,j,12], config.arr_for_save[iter,j,13], config.arr_for_save[iter,j,14],
                                                     box_corr[j,0,iter],box_corr[j,1,iter],box_corr[j,2,iter]))

                            one_arr_node_list = list(one_arr_node)
                            alist = [iter] + [curr_date] + one_arr_node_list
                            data_writer.writerows([alist])

                filename =  directory + '/' + 'Kazakhstan' + '.csv'
                with open(filename, 'w', newline='') as csvfile:
                    data_writer = csv.writer(csvfile, delimiter=',',  escapechar=' ', quoting=csv.QUOTE_NONE)
                    #points*nodes*states
                    data_writer.writerow([info2])
                    for iter in range(1, config.counter_func+1):
                        if config.new_plot_all:
                            one_arr = config.new_plot_all[iter-1]
                            one_arr_node = one_arr[-1,:,:].astype(int)
                            one_arr_node_sum = one_arr_node.sum(axis=0)
                            one_arr_node_list = list(one_arr_node_sum)
                            curr_date = self.start_date + timedelta(iter-1)
                            alist = [iter] + [curr_date] + one_arr_node_list
                            data_writer.writerows([alist])

                # last state save
                filename =  directory + '/' + 'states_x' + '.csv'
                with open(filename, 'w', newline='') as csvfile:
                    data_writer = csv.writer(csvfile, delimiter=',', escapechar=' ', quoting=csv.QUOTE_NONE)
                    nodes_new_iter = copy(config.nodes_old)
                    for index, node in enumerate(nodes_new_iter):
                        node.states_x = nodes_new_iter[index].states_x
                        st_t = copy(node.states_x)
                        st_t = list(st_t)
                        data_writer.writerow(st_t)

                    # save states_name
                    st_t = copy(nodes_new_iter[0].states_name)
                    st_t = list(st_t)
                    data_writer.writerow(st_t)

                print('[INFO] Saving results to .csv format ..')
            else:
                print('[INFO] No data to save.')

    def slider_update_initial_val(self, attr, old, new):

        self.init_exposed.value = config.param_init_exposed[config.region]
        self.sus_to_exp_slider.value = config.param_beta_exp[config.region]
        self.param_qr_slider.value = config.param_qr[config.region]
        self.param_sir.value = config.param_sir[config.region]
        self.param_hosp_capacity.value = config.param_hosp_capacity[config.region]
        self.param_gamma_mor1.value = config.param_gamma_mor1[config.region]
        self.param_gamma_mor2.value = config.param_gamma_mor2[config.region]
        self.param_gamma_im.value = config.param_gamma_im[config.region]
        self.param_eps_exp.value = config.param_eps_exp[config.region]
        self.param_eps_qua.value = config.param_eps_qua[config.region]
        self.param_eps_sev.value = config.param_eps_sev[config.region]
        self.param_t_exp.value = config.param_t_exp[0]
        self.param_t_inf.value = config.param_t_inf[0]
        self.param_tr_leakage.value = config.param_transition_leakage[0]
        self.param_tr_scale.value = config.param_transition_scale[0]

        self.param_test_spec.value = config.param_test_spec
        self.param_test_sens.value = config.param_test_sens
        self.param_test_prev.value = config.param_test_prev

        self.param_text_1.value = str(config.param_test_num[0])
        self.param_text_2.value = str(config.param_test_num[1])
        self.param_text_3.value = str(config.param_test_num[2])
        self.param_text_4.value = str(config.param_test_num[3])
        self.param_text_5.value = str(config.param_test_num[4])
        self.param_text_6.value = str(config.param_test_num[5])
        self.param_text_7.value = str(config.param_test_num[6])
        self.param_text_8.value = str(config.param_test_num[7])
        self.param_text_9.value = str(config.param_test_num[8])
        self.param_text_10.value = str(config.param_test_num[9])
        self.param_text_11.value = str(config.param_test_num[10])
        self.param_text_12.value = str(config.param_test_num[11])
        self.param_text_13.value = str(config.param_test_num[12])
        self.param_text_14.value = str(config.param_test_num[13])
        self.param_text_15.value = str(config.param_test_num[14])
        self.param_text_16.value = str(config.param_test_num[15])
        self.param_text_17.value = str(config.param_test_num[16])

    def slider_update_reset(self, attr, old, new):
        nodes_num =17

        config.param_init_exposed = 0*np.ones(nodes_num)
        config.param_beta_exp = 30.0*np.ones(nodes_num)
        config.param_qr = 2.0*np.ones(nodes_num)
        config.param_sir = 0.35*np.ones(nodes_num)
        config.param_hosp_capacity = np.array((280,2395,895,600,650,250,725,100,885,425,1670,300,465,1420,1505,380,300))
        config.param_gamma_mor1 = 7.0*np.ones(nodes_num)
        config.param_gamma_mor2= 11.0*np.ones(nodes_num)
        config.param_gamma_im = 90.0*np.ones(nodes_num)
        config.param_eps_exp= 100.0*np.ones(nodes_num)
        config.param_eps_qua = 20.0*np.ones(nodes_num)
        config.param_eps_sev = 20.0*np.ones(nodes_num)
        config.param_t_exp = 5*np.ones(nodes_num)
        config.param_t_inf = 14*np.ones(nodes_num)
        config.param_transition_leakage = 0.0*np.ones(nodes_num)
        config.param_transition_scale = 1.0*np.ones(nodes_num)

        # testing
        config.param_test_spec = 0.99
        config.param_test_sens = 0.7
        config.param_test_prev = 0.1
        config.param_prev_auto = np.zeros(nodes_num)

        config.param_test_num = np.zeros(nodes_num, dtype=int)
        config.param_test_sum = 0
        config.param_t_pos = 0
        config.param_t_neg = 0
        config.param_d_pos = 0
        config.param_d_neg = 0
        config.param_true_pos = 0
        config.param_true_neg = 0
        config.param_false_pos = 0
        config.param_false_neg = 0

        self.slider_update_initial_val(self,old, new)

        self.checkbox_group1.active = list(range(0, 17))
        self.checkbox_group2.active = list(range(0, 17))
        self.checkbox_group3.active = list(range(0, 17))
        self.update_transition_matrix()

        config.box_time = copy(config.param_transition_table)
        config.arr_for_save = np.dstack((config.param_init_exposed, config.param_beta_exp, config.param_qr, config.param_sir, config.param_hosp_capacity,
                                config.param_gamma_mor1, config.param_gamma_mor2, config.param_gamma_im, config.param_eps_exp,
                                config.param_eps_qua, config.param_eps_sev, config.param_t_exp, config.param_t_inf, config.param_transition_leakage,
                                 config.param_transition_scale))

        self.datepicker.value = datetime.today()
        self.start_date = datetime.today()


    def handler_beta_exp(self, attr, old, new):
        config.param_beta_exp[config.region]=new

    def handler_param_qr(self, attr, old, new):
        config.param_qr[config.region]=new

    def handler_param_sir(self, attr, old, new):
        config.param_sir[config.region]=new

    def handler_param_eps_exp(self, attr, old, new):
        config.param_eps_exp[config.region]=new

    def handler_param_eps_qua(self, attr, old, new):
        config.param_eps_qua[config.region]=new

    def handler_param_eps_sev(self, attr, old, new):
        config.param_eps_sev[config.region]=new

    def handler_param_hosp_capacity(self, attr, old, new):
        config.param_hosp_capacity[config.region]=new

    def handler_param_gamma_mor1(self, attr, old, new):
        config.param_gamma_mor1[config.region]=new

    def handler_param_gamma_mor2(self, attr, old, new):
        config.param_gamma_mor2[config.region]=new

    def handler_param_gamma_im(self, attr, old, new):
        config.param_gamma_im[config.region]=new

    def handler_param_sim_len(self, attr, old, new):
        config.loop_num=new

    def handler_param_t_exp(self, attr, old, new):
        if config.counter_func < 1:
            config.param_t_exp[0]=new
        else :
            self.slider_update_initial_val(self, old, new)

    def handler_param_t_inf(self, attr, old, new):
        if config.counter_func < 1:
            config.param_t_inf[0]=new
        else:
            self.slider_update_initial_val(self, old, new)

    def handler_init_exposed(self, attr, old, new):
        if config.counter_func < 1:
            config.param_init_exposed[config.region]=new
            self.update(False)
        else:
            self.slider_update_initial_val(self, old, new)

    def handler_param_tr_scale(self, attr, old, new):
        config.param_transition_scale=new*np.ones(config.nodes_num)
        self.update_transition_matrix()

    def handler_param_tr_leakage(self, attr, old, new):
        config.param_transition_leakage=new*np.ones(config.nodes_num)
        self.update_transition_matrix()

    def handler_checkbox_group1(self, new):
        self.box1 = new
        self.update_transition_matrix()

    def handler_checkbox_group2(self, new):
        self.box2 = new
        self.update_transition_matrix()

    def handler_checkbox_group3(self, new):
        self.box3 = new
        self.update_transition_matrix()

    def handler_param_save_file(self, attr, old, new):
        config.param_save_file= str(new)

    def get_date(self, attr, old, new):
        self.start_date = new

    def handler_param_test_spec(self, attr, old, new):
        config.param_test_spec=new

    def handler_param_test_sens(self, attr, old, new):
        config.param_test_sens=new

    def handler_param_test_prev(self, attr, old, new):
        config.param_test_prev=new

    def handler_param_text_1(self, attr, old, new):
        config.param_test_num[0]= str(new)

    def handler_param_text_2(self, attr, old, new):
        config.param_test_num[1]= str(new)

    def handler_param_text_3(self, attr, old, new):
        config.param_test_num[2]= str(new)

    def handler_param_text_4(self, attr, old, new):
        config.param_test_num[3]= str(new)

    def handler_param_text_5(self, attr, old, new):
        config.param_test_num[4]= str(new)

    def handler_param_text_6(self, attr, old, new):
        config.param_test_num[5]= str(new)

    def handler_param_text_7(self, attr, old, new):
        config.param_test_num[6]= str(new)

    def handler_param_text_8(self, attr, old, new):
        config.param_test_num[7]= str(new)

    def handler_param_text_9(self, attr, old, new):
        config.param_test_num[8]= str(new)

    def handler_param_text_10(self, attr, old, new):
        config.param_test_num[9]= str(new)

    def handler_param_text_11(self, attr, old, new):
        config.param_test_num[10]= str(new)

    def handler_param_text_12(self, attr, old, new):
        config.param_test_num[11]= str(new)

    def handler_param_text_13(self, attr, old, new):
        config.param_test_num[12]= str(new)

    def handler_param_text_14(self, attr, old, new):
        config.param_test_num[13]= str(new)

    def handler_param_text_15(self, attr, old, new):
        config.param_test_num[14]= str(new)

    def handler_param_text_16(self, attr, old, new):
        config.param_test_num[15]= str(new)

    def handler_param_text_17(self, attr, old, new):
        config.param_test_num[16]= str(new)

    def layout(self):

        # define text font, colors
        self.text1 = Div(text="""<h1 style="color:blue">COVID-19 Simulator for Kazakhstan</h1>""", width=500, height=50)
        self.text4 = Div(text="""<h1 style="color:blue"> </h1>""", width=900, height=50)

        self.text2 =  Div(text="<b>Select parameters for each region</b>", style={'font-size': '150%', 'color': 'green'},width=350)
        self.text3 =  Div(text="<b>Select global parameters </b>", style={'font-size': '150%', 'color': 'green'}   )
        self.text5 =  Div(text="<b>Change transition matrix</b>", style={'font-size': '150%', 'color': 'green'})

        self.text6 =  Div(text="<b>Select testing parameters</b>", style={'font-size': '150%', 'color': 'green'})
        self.text7 =  Div(text="<b>Enter number of test</b>", style={'font-size': '150%', 'color': 'green'})
        self.text8 =  Div(text="<b>Testing results</b>", style={'font-size': '150%', 'color': 'green'})
        self.text9 =  Div(text="<b>Select prevalence rate</b>", style={'font-size': '150%', 'color': 'green'})

        # select region - dropdown menu
        regions = config.region_names

        initial_region = 'Almaty'
        region_selection = Select(value=initial_region, title=' ', options=regions, width=250, height=15)
        region_selection.on_change('value', self.SelectRegionHandler)

        initial_prev = 'Manual'
        options_prev = ['Auto', 'Manual']
        prev_selection = Select(value=initial_prev, title=' ', options=options_prev, width=250, height=15)
        prev_selection.on_change('value', self.SelectPrevHandler)

        # select parameters - sliders
        self.sus_to_exp_slider = Slider(start=0.0,end=50.0,step=0.5,value=config.param_beta_exp[config.region], title='Susceptible to Exposed transition constant (%)')
        self.sus_to_exp_slider.on_change('value', self.handler_beta_exp)

        self.param_qr_slider = Slider(start=0.0,end=25.0,step=0.25,value=config.param_qr[config.region], title='Daily Quarantine rate of the Exposed (%)')
        self.param_qr_slider.on_change('value', self.handler_param_qr)

        self.param_sir = Slider(start=0.0,end=5.0,step=0.05,value=config.param_sir[config.region], title='Daily Infected to Severe Infected transition rate (%)')
        self.param_sir.on_change('value', self.handler_param_sir)

        self.param_eps_exp = Slider(start=0,end=100,step=1.0,value=config.param_eps_exp[config.region], title='Disease transmission rate of Exposed compared to Infected (%)')
        self.param_eps_exp.on_change('value', self.handler_param_eps_exp)

        self.param_eps_qua = Slider(start=0,end=100,step=1.0,value=config.param_eps_qua[config.region], title='Disease transmission rate of Quarantined compared to Infected (%)')
        self.param_eps_qua.on_change('value', self.handler_param_eps_qua)

        self.param_eps_sev = Slider(start=0,end=100,step=1.0,value=config.param_eps_sev[config.region], title='Disease transmission rate of Severe Infected compared to Infected (%)')
        self.param_eps_sev.on_change('value', self.handler_param_eps_sev)

        self.param_hosp_capacity = Slider(start=0,end=10000,step=1,value=config.param_hosp_capacity[config.region], title='Hospital Capacity')
        self.param_hosp_capacity.on_change('value', self.handler_param_hosp_capacity)

        self.param_gamma_mor1 = Slider(start=0,end=100,step=1.0,value=config.param_gamma_mor1[config.region], title='Severe Infected to Dead transition probability (%)')
        self.param_gamma_mor1.on_change('value', self.handler_param_gamma_mor1)

        self.param_gamma_mor2 = Slider(start=0,end=100,step=1,value=config.param_gamma_mor2[config.region], title='Severe Infected to Dead transition probability (Hospital Cap. Exceeded) (%)')
        self.param_gamma_mor2.on_change('value', self.handler_param_gamma_mor2)

        self.param_gamma_im = Slider(start=0,end=100,step=1,value=config.param_gamma_im[config.region], title='Infected to Recovery Immunized transition probability (%)')
        self.param_gamma_im.on_change('value', self.handler_param_gamma_im)

        self.param_sim_len = Slider(start=1,end=100,step=1,value=config.loop_num, title='Length of simulation (Days)')
        self.param_sim_len.on_change('value', self.handler_param_sim_len)

        self.param_t_exp = Slider(start=1,end=20,step=1,value=config.param_t_exp[0], title='Incubation period (Days) ')
        self.param_t_exp.on_change('value', self.handler_param_t_exp)

        self.param_t_inf = Slider(start=1,end=20,step=1,value=config.param_t_inf[0], title=' Infection  period (Days) ')
        self.param_t_inf.on_change('value', self.handler_param_t_inf)

        self.init_exposed = Slider(start=0,end=100,step=1,value=config.param_init_exposed[config.region], title='Initial Exposed')
        self.init_exposed.on_change('value', self.handler_init_exposed)

        self.param_tr_scale = Slider(start=0.0,end=1,step=0.01,value=config.param_transition_scale[0], title='Traffic ratio')
        self.param_tr_scale.on_change('value', self.handler_param_tr_scale)

        self.param_tr_leakage = Slider(start=0.0,end=1,step=0.01,value=config.param_transition_leakage[0], title='Leakage ratio')
        self.param_tr_leakage.on_change('value', self.handler_param_tr_leakage)

        # testing
        self.param_test_spec = Slider(start=0.0,end=1,step=0.01,value=config.param_test_spec, title='Test specifity')
        self.param_test_spec.on_change('value', self.handler_param_test_spec)

        self.param_test_sens = Slider(start=0.0,end=1,step=0.01,value=config.param_test_sens, title='Test sensitivity')
        self.param_test_sens.on_change('value', self.handler_param_test_sens)

        self.param_test_prev = Slider(start=0.0,end=1,step=0.01,value=config.param_test_prev, title='Prevalence rate (Manual)')
        self.param_test_prev.on_change('value', self.handler_param_test_prev)

        self.param_text_1 = TextInput(value="0", title=regions[0], width = 100)
        self.param_text_1.on_change('value', self.handler_param_text_1)

        self.param_text_2 = TextInput(value="0", title=regions[1], width = 100)
        self.param_text_2.on_change('value', self.handler_param_text_2)

        self.param_text_3 = TextInput(value="0", title=regions[2], width = 100)
        self.param_text_3.on_change('value', self.handler_param_text_3)

        self.param_text_4 = TextInput(value="0", title=regions[3], width = 100)
        self.param_text_4.on_change('value', self.handler_param_text_4)

        self.param_text_5 = TextInput(value="0", title=regions[4], width = 100)
        self.param_text_5.on_change('value', self.handler_param_text_5)

        self.param_text_6 = TextInput(value="0", title=regions[5], width = 100)
        self.param_text_6.on_change('value', self.handler_param_text_6)

        self.param_text_7 = TextInput(value="0", title=regions[6], width = 100)
        self.param_text_7.on_change('value', self.handler_param_text_7)

        self.param_text_8 = TextInput(value="0", title=regions[7], width = 100)
        self.param_text_8.on_change('value', self.handler_param_text_8)

        self.param_text_9 = TextInput(value="0", title=regions[8], width = 100)
        self.param_text_9.on_change('value', self.handler_param_text_9)

        self.param_text_10 = TextInput(value="0", title=regions[9], width = 100)
        self.param_text_10.on_change('value', self.handler_param_text_10)

        self.param_text_11 = TextInput(value="0", title=regions[10], width = 100)
        self.param_text_11.on_change('value', self.handler_param_text_11)

        self.param_text_12 = TextInput(value="0", title=regions[11], width = 100)
        self.param_text_12.on_change('value', self.handler_param_text_12)

        self.param_text_13 = TextInput(value="0", title=regions[12], width = 100)
        self.param_text_13.on_change('value', self.handler_param_text_13)

        self.param_text_14 = TextInput(value="0", title=regions[13], width = 100)
        self.param_text_14.on_change('value', self.handler_param_text_14)

        self.param_text_15 = TextInput(value="0", title=regions[14], width = 100)
        self.param_text_15.on_change('value', self.handler_param_text_15)

        self.param_text_16 = TextInput(value="0", title=regions[15], width = 100)
        self.param_text_16.on_change('value', self.handler_param_text_16)

        self.param_text_17 = TextInput(value="0", title=regions[16], width = 100)
        self.param_text_17.on_change('value', self.handler_param_text_17)

        dumdiv = Div(text='',width=10)
        dumdiv2= Div(text='',width=10)
        dumdiv3= Div(text='',width=200)
        dumdiv3ss= Div(text='',width=120)

        # Buttons
        reset_button = Button(label = 'Reset data', button_type='primary', background = "red")
        save_button_result = Button(label='Save current plot to .csv in directory results/', button_type='primary')
        run_button = Button(label='Run the simulation',button_type='primary')
        load_button = Button(label='Load data from directory results/', button_type='primary')
        test_button = Button(label='Run testing', button_type='primary')

        run_button.on_click(self.run_click)
        reset_button.on_click(self.reset_click)
        save_button_result.on_click(self.save_file_click)
        load_button.on_click(self.load_click)
        test_button.on_click(self.test_click)

        # input folder name
        text_save = TextInput(value="foldername", title="")
        text_save.on_change('value', self.handler_param_save_file)

        # transition matrix - checkbox
        div_cb1 = Div(text = 'Airways', width = 150)
        div_cb2 = Div(text = 'Railways', width = 150)
        div_cb3 = Div(text = 'Highways', width = 150)

        self.checkbox_group1 = CheckboxGroup(labels=regions, active = list(range(0, 17)))
        self.checkbox_group2 = CheckboxGroup(labels=regions, active= list(range(0, 17)))
        self.checkbox_group3 = CheckboxGroup(labels=regions, active= list(range(0, 17)))

        self.checkbox_group1.on_click(self.handler_checkbox_group1)
        self.checkbox_group2.on_click(self.handler_checkbox_group2)
        self.checkbox_group3.on_click(self.handler_checkbox_group3)

        # transition matrix - table
        self.data1 = dict(
                        c00 =  regions,
                        c0= [(config.transition_matrix[0,i]) for i in range(0,17)],
                        c1= [(config.transition_matrix[1,i]) for i in range(0,17)],
                        c2= [(config.transition_matrix[2,i]) for i in range(0,17)],
                        c3=[(config.transition_matrix[3,i]) for i in range(0,17)],
                        c4=[(config.transition_matrix[4,i]) for i in range(0,17)],
                        c5=[(config.transition_matrix[5,i]) for i in range(0,17)],
                        c6=[(config.transition_matrix[6,i]) for i in range(0,17)],
                        c7=[(config.transition_matrix[7,i]) for i in range(0,17)],
                        c8=[(config.transition_matrix[8,i]) for i in range(0,17)],
                        c9=[(config.transition_matrix[9,i]) for i in range(0,17)],
                        c10=[(config.transition_matrix[10,i]) for i in range(0,17)],
                        c11=[(config.transition_matrix[11,i]) for i in range(0,17)],
                        c12=[(config.transition_matrix[12,i]) for i in range(0,17)],
                        c13=[(config.transition_matrix[13,i]) for i in range(0,17)],
                        c14=[(config.transition_matrix[14,i]) for i in range(0,17)],
                        c15=[(config.transition_matrix[15,i]) for i in range(0,17)],
                        c16=[(config.transition_matrix[16,i]) for i in range(0,17)],)

        columns = [
                    TableColumn(field="c00", title=" ",),
                    TableColumn(field="c0", title="Almaty",),
                    TableColumn(field="c1", title="Almaty Qalasy",),
                    TableColumn(field="c2", title="Aqmola",),
                    TableColumn(field="c3", title="Aqtobe",),
                    TableColumn(field="c4", title="Atyrau",),
                    TableColumn(field="c5", title="West Kazakhstan",),
                    TableColumn(field="c6", title="Jambyl",),
                    TableColumn(field="c7", title="Mangystau",),
                    TableColumn(field="c8", title="Nur-Sultan",),
                    TableColumn(field="c9", title="Pavlodar",),
                    TableColumn(field="c10", title="Qaragandy",),
                    TableColumn(field="c11", title="Qostanai",),
                    TableColumn(field="c12", title="Qyzylorda",),
                    TableColumn(field="c13", title="East Kazakhstan",),
                    TableColumn(field="c14", title="Shymkent",),
                    TableColumn(field="c15", title="North Kazakhstan",),
                    TableColumn(field="c16", title="Turkistan",),]

        self.sourceT = ColumnDataSource(self.data1)
        self.data_tableT = DataTable(source=self.sourceT, columns=columns, width=1750, height=500, sortable = False)

        # results of the testing
        self.dataTest = dict(
                        c00 =  ['Performed, N','Confirmed, T(+)','Not Confirmed, T(-)', 'True Positives, TP', 'False Positives, FP', 'False Negatives, FN', 'True Negatives, TN',
                                'Truly Infected, D(+)', 'Non-Infected, D(-)',],
                        c0=[config.param_test_sum, config.param_t_pos, config.param_t_neg, config.param_true_pos, config.param_false_pos, config.param_false_neg, config.param_true_neg,
                             config.param_d_pos, config.param_d_neg],)

        columnsTest = [
                    TableColumn(field="c00", title=" ",),
                    TableColumn(field="c0", title=" ",),]

        self.sourceTest = ColumnDataSource(self.dataTest)
        self.data_tableTest = DataTable(source=self.sourceTest, columns=columnsTest, width=400, height=300, sortable = False)

        # prevalance_rate
        self.dataPrev = dict(
                        c00=['Prevalence rate (Auto)'],
                        c0=[config.param_prev_auto[0]],
                        c1=[config.param_prev_auto[1]],
                        c2=[config.param_prev_auto[2]],
                        c3=[config.param_prev_auto[3]],
                        c4=[config.param_prev_auto[4]],
                        c5=[config.param_prev_auto[5]],
                        c6=[config.param_prev_auto[6]],
                        c7=[config.param_prev_auto[7]],
                        c8=[config.param_prev_auto[8]],
                        c9=[config.param_prev_auto[9]],
                        c10=[config.param_prev_auto[10]],
                        c11=[config.param_prev_auto[11]],
                        c12=[config.param_prev_auto[12]],
                        c13=[config.param_prev_auto[13]],
                        c14=[config.param_prev_auto[14]],
                        c15=[config.param_prev_auto[15]],
                        c16=[config.param_prev_auto[16]],)

        columnsPrev = [
                    TableColumn(field="c00", title=" ",),
                    TableColumn(field="c0", title="Almaty",),
                    TableColumn(field="c1", title="Almaty Qalasy",),
                    TableColumn(field="c2", title="Aqmola",),
                    TableColumn(field="c3", title="Aqtobe",),
                    TableColumn(field="c4", title="Atyrau",),
                    TableColumn(field="c5", title="West Kazakhstan",),
                    TableColumn(field="c6", title="Jambyl",),
                    TableColumn(field="c7", title="Mangystau",),
                    TableColumn(field="c8", title="Nur-Sultan",),
                    TableColumn(field="c9", title="Pavlodar",),
                    TableColumn(field="c10", title="Qaragandy",),
                    TableColumn(field="c11", title="Qostanai",),
                    TableColumn(field="c12", title="Qyzylorda",),
                    TableColumn(field="c13", title="East Kazakhstan",),
                    TableColumn(field="c14", title="Shymkent",),
                    TableColumn(field="c15", title="North Kazakhstan",),
                    TableColumn(field="c16", title="Turkistan",),]

        self.sourcePrev = ColumnDataSource(self.dataPrev)
        self.data_tablePrev = DataTable(source=self.sourcePrev, columns=columnsPrev, width=2200, height=80, sortable = False)


        # select start date - calendar
        self.datepicker = DatePicker(title="Starting date of simulation", min_date=datetime(2015,11,1), value=datetime.today())
        self.datepicker.on_change('value',self.get_date)

        # place the widgets on the layout

        sliders_1 = column(self.init_exposed, self.sus_to_exp_slider, self.param_qr_slider, self.param_sir)
        sliders_2 = column(self.param_hosp_capacity, self.param_gamma_mor1, self.param_gamma_mor2, self.param_gamma_im)
        sliders_0 = column(self.param_eps_exp, self.param_eps_qua, self.param_eps_sev)

        sliders = row(sliders_1, dumdiv3ss, sliders_2, dumdiv3, sliders_0)

        sliders_3 = row(self.param_t_exp, self.param_t_inf, self.param_sim_len,self.datepicker,)
        text2 = Div(text="""<h1 style='color:black'>   issai.nu.edu.kz/episim </h1>""", width = 500, height = 100)
        text_footer_1 = Div(text="""<h3 style='color:green'> Developed by ISSAI Researchers : Askat Kuzdeuov, Daulet Baimukashev, Bauyrzhan Ibragimov, Aknur Karabay, Almas Mirzakhmetov, Mukhamet Nurpeiissov and Huseyin Atakan Varol </h3>""", width = 1500, height = 10)
        text_footer_2 = Div(text="""<h3 style='color:red'> Disclaimer : This simulator is a research tool. The simulation results will show general trends based on entered parameters and initial conditions  </h3>""", width = 1500, height = 10)
        text_footer = column(text_footer_1, text_footer_2)
        text = column(self.text1, text2)

        draw_map_js = CustomJS(code=""" uStates.draw("#statesvg", currRegionData, tooltipHtml); """)
        run_button.js_on_click(draw_map_js)

        layout_t = row(save_button_result, text_save, load_button)
        buttons = row(reset_button,run_button, layout_t)

        reg1 = row(self.text2, region_selection)

        buttons = column(buttons, reg1)

        params =  column(sliders, self.text3, sliders_3, self.text5)

        sliders_4 = column(self.param_tr_scale, self.param_tr_leakage)
        check_table = row(column(div_cb1,self.checkbox_group1), column(div_cb2,self.checkbox_group2), column(div_cb3,self.checkbox_group3), sliders_4)
        check_trans = row(self.data_tableT)

        ###
        dummy_div = Div(text=""" """, height=25);
        dummy_div11 = Div(text=""" """, height=5);
        layout = column(self.pAll, buttons)
        layout = column (layout, dummy_div11, params, check_table)

        layout = column (layout, check_trans, self.text4)

        sliders_test = column(self.param_test_spec, self.param_test_sens)
        layout_num_test = row(self.param_text_1, self.param_text_2, self.param_text_3, self.param_text_4, self.param_text_5, self.param_text_6,
                                self.param_text_7, self.param_text_8, self.param_text_9, self.param_text_10, self.param_text_11, self.param_text_12,
                                self.param_text_13, self.param_text_14, self.param_text_15, self.param_text_16, self.param_text_17)

        layout = column (layout, self.text6, sliders_test, row(self.text9, prev_selection))
        layout = column(layout, dummy_div11, column(self.param_test_prev,dummy_div11), self.data_tablePrev, self.text7,layout_num_test, column(test_button, self.text8, self.data_tableTest) )


        layout = column (layout,self.text4)     # text_footer

        self.doc.title = 'ISSAI Covid-19 Simulator'
        self.doc.add_root(layout)
