from bokeh.plotting import figure
from bokeh.layouts import gridplot, column, row
from bokeh.io import curdoc
from tornado import gen
import numpy as np
from bokeh.models import CustomJS, CategoricalColorMapper, Panel, Select, Button, DataTable,DateFormatter, TableColumn, PrintfTickFormatter, Legend, Slider, TextInput, CheckboxGroup, Div, ColumnDataSource
from covid_simulator_upd import Node
import config
import os
import csv
import copy
from datetime import date
from random import randint
from PIL import Image
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, DataTable, DateFormatter, TableColumn
from bokeh.models import (CDSView, ColorBar, ColumnDataSource,
                          CustomJS, CustomJSFilter,
                          GeoJSONDataSource, HoverTool,
                          LinearColorMapper, Slider,PrintfTickFormatter)

import csv
import pandas as pd
import json

#import geopandas as gpd
#df_kz = gpd.read_file('data_geomap/KAZ_adm1.shp')
#geosource = GeoJSONDataSource(geojson = df_kz.to_json())

class Visual:
    def __init__(self, callbackFunc, running):
        self.text1 = Div(text="""<h1 style="color:blue">COVID-19 Simulator for Kazakhstan</h1>""", width=500, height=50) # Text to be displayed at the top of the webpage
        self.text2 = Div(text="""<h1 style="color:blue">Select parameters for each region</h1>""", width=500, height=10) # Text to be displayed at the top of the webpage
        self.text3 = Div(text="""<h1 style="color:blue">Select global parameters </h1>""", width=900, height=10) # Text to be displayed at the top of the webpage
        self.text4 = Div(text="""<h1 style="color:blue"> </h1>""", width=900, height=50) # Text to be displayed at the top of the webpage
        self.text4rr = Div(text="""<h1 style="color:blue"> </h1>""", width=200, height=10) # Text to be displayed at the top of the webpage

        self.text5 = Div(text="""<h1 style="color:blue"> Change transition matrix </h1>""", width=900, height=10) # Text to be displayed at the top of the webpage
        self.text6 = Div(text="""<h1 style="color:blue">Select global parameters </h1>""", width=900, height=10) # Text to be displayed at the top of the webpage
        self.text7 = Div(text="""<h1 style="color:blue">Save current results to file </h1>""", width=900, height=10) # Text to be displayed at the top of the webpage


        self.running = running
        self.callbackFunc = callbackFunc
        self.source = ColumnDataSource(dict(x=[0], sus=[config.param_init_susceptible[config.region]], exp=[config.param_init_exposed[config.region]], inf=[0], sin=[0],
                                        qua=[0], imm=[0], dea=[0], text=[""]))

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
        self.region_names = ['Almaty', 'Almaty Qalasy', 'Aqmola', 'Aqtobe', 'Atyrau', 'Batys Qazaqstan', 'Jambyl', 'Mangystau', 'Nur-Sultan', 'Pavlodar', 'Qaraqandy', 'Qostanai',
                            'Qyzylorda', 'Shygys Qazaqstan', 'Shymkent', 'Soltustik Qazaqstan', 'Turkistan']

        self.init_exposed.value = config.param_init_exposed[config.region]
        self.sus_to_exp_slider.value = config.param_beta_exp[config.region]
        self.param_qr_slider.value = config.param_qr[config.region]
        self.param_sir.value = config.param_sir[config.region]
        self.param_hosp_capacity.value = config.hosp_cap_init[config.region]
        self.param_gamma_mor1.value = config.param_gamma_mor1[config.region]
        self.param_gamma_mor2.value = config.param_gamma_mor2[config.region]
        self.param_gamma_im.value = config.param_gamma_im[config.region]
        self.param_eps_exp.value = config.param_eps_exp[config.region]
        self.param_eps_qua.value = config.param_eps_qua[config.region]
        self.param_eps_sev.value = config.param_eps_sev[config.region]

    def set_initial_params(self, params):
        global initial_params
        config.initial_params = params

    def definePlot(self, source):
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        img_nu  = Image.open(os.path.join(THIS_FOLDER, 'nu_logo.png')).convert('RGBA')
        img_issai = Image.open(os.path.join(THIS_FOLDER, 'issai_logo.png')).convert('RGBA')
        
        x_nu, y_nu = img_nu.size
        img_nu_plot = np.empty((x_nu, y_nu), dtype=np.uint32)
        img_nu_view = img_nu_plot.view(dtype=np.uint8).reshape((y_nu, x_nu, 4))
        img_nu_view[:,:,:]=np.flipud(np.asarray(img_nu))
        
        dim_nu = max(x_nu, y_nu)
        dim_nu_y = min(x_nu, y_nu)
        p_nu = figure(x_range=(0,dim_nu), y_range=(0,dim_nu_y), height=250, width=479)
        p_nu.image_rgba(image=[img_nu_view], x=0, y=0, dw=x_nu, dh=y_nu)
        p_nu.axis.visible = False
        p_nu.axis.visible = False
        p_nu.toolbar.logo = None
        p_nu.toolbar_location = None
        
        x_is, y_is = img_issai.size
        iss_plot = np.empty((x_is, y_is), dtype=np.uint32)
        iss_view=iss_plot.view(dtype=np.uint8).reshape((y_is, x_is, 4))
        iss_view[:,:,:]=np.flipud(np.asarray(img_issai))
        
        p_iss = figure(x_range=(0, x_is), y_range=(0, y_is), height=250, width=479)
        p_iss.image_rgba(image=[iss_view], x=0, y=0, dw=x_is, dh=y_is)
        p_iss.axis.visible = False
        p_iss.axis.visible = False
        p_iss.toolbar.logo = None
        p_iss.toolbar_location = None        
        

      #  create glyph for kazakhstan map
      #  p_map = figure(title = ' Kazakhstan', plot_height=600, plot_width=800, background_fill_color='black',background_fill_alpha = 0.8, toolbar_location='above')
      #  p_map.xgrid.grid_line_color=None
      #  p_map.ygrid.grid_line_color=None
      #  states=p_map.patches('xs','ys', source=geosource, fill_color='red', line_color='gray')


        # create glyph for graph plotting
        # create glyph for graph plotting
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

        #######################

        #######################

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

        #,,location=(5,30)
        legend = Legend(items=[
                                ('Exposed', [r2, r3]),
                                ('Infected', [r4, r5]),
                                ('Severe Infected', [r6, r7]),
                                ('Quarantined', [r8, r9]),
                                ('Immunized', [r10, r11]),
                                ('Dead', [r12, r13])])
        #p1.add_layout(legend, 'left')
        p1.legend.click_policy = 'hide'

        #styling
        p1.background_fill_color = "black"
        p1.background_fill_alpha = 0.8
        p1.legend.location = "top_left"
        p1.legend.background_fill_color = "cyan"
        p1.legend.background_fill_alpha = 0.5
        p1.outline_line_width = 7
        p1.outline_line_alpha = 0.9
        p1.outline_line_color = "black"

        p2.legend.click_policy = 'hide'

        #styling
        p2.background_fill_color = "black"
        p2.background_fill_alpha = 0.8
        p2.legend.location = "top_left"
        p2.legend.background_fill_color = "cyan"
        p2.legend.background_fill_alpha = 0.5
        p2.outline_line_width = 7
        p2.outline_line_alpha = 0.9
        p2.outline_line_color = "black"


       # p_map.outline_line_width = 7
       # p_map.outline_line_alpha = 0.9
       # p_map.outline_line_color = "black"
       # p_map.xaxis.visible = False
       # p_map.yaxis.visible = False

        #pAll = gridplot([[row(p1], [p_map]])
        kz_map_tag = Div(text="""<div id="svg_holder" style="float:left;"> <svg width="780" height="600" id="statesvg"></svg> <div id="tooltip"></div>   </div>""", width=960, height=600)
        kz_map_tooltip = Div(text="""<div style="float:left;" id="tooltip"></div>""", width=960, height=100)
        kz_map_row = row(kz_map_tag,kz_map_tooltip)
        pAll = row(p1, kz_map_row)
        return pAll

    #@gen.coroutine
    def update(self, change_view):
        new_nodes_all = config.new_plot_all
        counter_func = config.counter_func-2
        newx = [0]
        state_inf = [0]
        ######## change
        state_sus=[config.param_init_susceptible[config.region]]
        #### change
        state_exp = [config.param_init_exposed[config.region]]
        state_sin = [0]
        state_qua = [0]
        state_imm = [0]
        state_dea = [0]

        region_states = dict()
        tmp_state_inf = [0]
        tmp_state_sus=[config.param_init_susceptible[config.region]]
        tmp_state_exp = [config.param_init_exposed[config.region]]
        tmp_state_sin = [0]
        tmp_state_qua = [0]
        tmp_state_imm = [0]
        tmp_state_dea = [0]

        if new_nodes_all != [] and config.region != 17:
            for i in range(len(config.new_plot_all)):

                state_inf.append(new_nodes_all[i][:, config.region, 0][-1])
                state_exp.append(new_nodes_all[i][:, config.region, 1][-1])
                state_sin.append(new_nodes_all[i][:, config.region, 2][-1])
                state_qua.append(new_nodes_all[i][:, config.region, 3][-1])
                state_imm.append(new_nodes_all[i][:, config.region, 4][-1])
                state_sus.append(new_nodes_all[i][:, config.region, 5][-1])
                state_dea.append(new_nodes_all[i][:, config.region, 6][-1])

                #newx = np.arange(0,2*config.counter_func/2)
                newx = config.param_sim_len[0]*(np.arange(config.counter_func+1))
                
                # for map
                
                regions_ids = [ lregion for lregion in range(17)]
                for region in regions_ids:
                    if region in region_states:
                        #print("GOOD")
                        region_states[region]["tmp_state_inf"].append(new_nodes_all[i][:, region, 0][-1])
                        region_states[region]["tmp_state_sin"].append(new_nodes_all[i][:, region, 2][-1])
                        region_states[region]["tmp_state_exp"].append(new_nodes_all[i][:, region, 1][-1])
                        region_states[region]["tmp_state_qua"].append(new_nodes_all[i][:, region, 3][-1])
                        region_states[region]["tmp_state_imm"].append(new_nodes_all[i][:, region, 4][-1])
                        region_states[region]["tmp_state_sus"].append(new_nodes_all[i][:, region, 5][-1])
                        region_states[region]["tmp_state_dea"].append(new_nodes_all[i][:, region, 6][-1])
                    else:
                        #print("ONLY ZEROOOO")
                        tmp_data = {
                            "tmp_state_inf": [], 
                            "tmp_state_sin": [], 
                            "tmp_state_exp": [], 
                            "tmp_state_qua": [],
                            "tmp_state_imm": [],
                            "tmp_state_sus": [],
                            "tmp_state_dea": []
                            }

                        tmp_data["tmp_state_inf"].append(new_nodes_all[i][:, region, 0][-1])
                        tmp_data["tmp_state_sin"].append(new_nodes_all[i][:, region, 2][-1])
                        tmp_data["tmp_state_exp"].append(new_nodes_all[i][:, region, 1][-1])
                        tmp_data["tmp_state_qua"].append(new_nodes_all[i][:, region, 3][-1])
                        tmp_data["tmp_state_imm"].append(new_nodes_all[i][:, region, 4][-1])
                        tmp_data["tmp_state_sus"].append(new_nodes_all[i][:, region, 5][-1])
                        tmp_data["tmp_state_dea"].append(new_nodes_all[i][:, region, 6][-1])

                        region_states[region] = tmp_data
                

        elif new_nodes_all != [] and config.region == 17:
            for i in range(len(config.new_plot_all)):

                state_inf.append(sum(new_nodes_all[i][:, :, 0][-1]))
                state_exp.append(sum(new_nodes_all[i][:, :, 1][-1]))
                state_sin.append(sum(new_nodes_all[i][:, :, 2][-1]))
                state_qua.append(sum(new_nodes_all[i][:, :, 3][-1]))
                state_imm.append(sum(new_nodes_all[i][:, :, 4][-1]))
                state_sus.append(sum(new_nodes_all[i][:, :, 5][-1]))
                state_dea.append(sum(new_nodes_all[i][:, :, 6][-1]))

                #newx = np.arange(0,2*config.counter_func/2)
                newx = config.param_sim_len[0]*(np.arange(config.counter_func+1))
                
                regions_ids = [ lregion for lregion in range(17)]
                for region in regions_ids:
                    if str(region) in region_states and type(region_states[region]) is dict:
                        print("GOOOODDDDD")
                        region_states[region]["tmp_state_inf"].append(new_nodes_all[i][:, region, 0][-1])
                        region_states[region]["tmp_state_sin"].append(new_nodes_all[i][:, region, 2][-1])
                        region_states[region]["tmp_state_exp"].append(new_nodes_all[i][:, region, 1][-1])
                        region_states[region]["tmp_state_qua"].append(new_nodes_all[i][:, region, 3][-1])
                        region_states[region]["tmp_state_imm"].append(new_nodes_all[i][:, region, 4][-1])
                        region_states[region]["tmp_state_sus"].append(new_nodes_all[i][:, region, 5][-1])
                        region_states[region]["tmp_state_dea"].append(new_nodes_all[i][:, region, 6][-1])
                    else:
                        print("ONLY THIS")
                        tmp_data = {
                            "tmp_state_inf": [], 
                            "tmp_state_sin": [], 
                            "tmp_state_exp": [], 
                            "tmp_state_qua": [],
                            "tmp_state_imm": [],
                            "tmp_state_sus": [],
                            "tmp_state_dea": []
                            }

                        tmp_data["tmp_state_inf"].append(new_nodes_all[i][:, region, 0][-1])
                        tmp_data["tmp_state_sin"].append(new_nodes_all[i][:, region, 2][-1])
                        tmp_data["tmp_state_exp"].append(new_nodes_all[i][:, region, 1][-1])
                        tmp_data["tmp_state_qua"].append(new_nodes_all[i][:, region, 3][-1])
                        tmp_data["tmp_state_imm"].append(new_nodes_all[i][:, region, 4][-1])
                        tmp_data["tmp_state_sus"].append(new_nodes_all[i][:, region, 5][-1])
                        tmp_data["tmp_state_dea"].append(new_nodes_all[i][:, region, 6][-1])

                        region_states[region] = tmp_data
        #print(region_states)
        str_data = json.dumps(region_states, ensure_ascii=False)               
        new_data = dict(x=newx, sus=state_sus, exp=state_exp, inf=state_inf, sin=state_sin,
                    qua=state_qua, imm=state_imm, dea=state_dea, text=[str_data]*len(state_imm))

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
            c16=[(config.transition_matrix[16,i]) for i in range(0,17)],
                )

        
        self.source.data.update(new_data)
        #self.sourceJS.data.update(dict(text=[str_data]))
        self.sourceT.data.update(self.data1)
        self.data_tableT.update()

    def SelectRegionHandler(self, attr, old, new):
        regions = ['Almaty', 'Almaty Qalasy', 'Aqmola', 'Aqtobe', 'Atyrau', 'Batys Qazaqstan', 'Jambyl', 'Mangystau', 'Nur-Sultan', 'Pavlodar', 'Qaraqandy', 'Qostanai',
                    'Qyzylorda', 'Shygys Qazaqstan', 'Shymkent', 'Soltustik Qazaqstan', 'Turkistan']
        for i, region in enumerate(regions):
            if new == region:
                config.region = i
                break
        self.update(True)
        self.slider_update_initial_val(self, old, new)

    def save_click(self):
        nodes_num = 17
        config.param_transition_box = []
        config.param_transition_box.append(config.box1)
        config.param_transition_box.append(config.box2)
        config.param_transition_box.append(config.box3)

        #config.box_time.append(config.param_transition_box)

        tr_boxes = config.param_transition_box

        param_transition_box = np.zeros((17,3))

        for i, way in enumerate(tr_boxes): # air 0 rail 1 road 2
            for j, node in enumerate(way):
                status = int(node)
                param_transition_box[status, i] = 1

        param_transition_leakage = config.param_transition_leakage
        param_transition_scale = config.param_transition_scale

        # load transition matrix
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

        transition_railway = list(csv.reader(open(os.path.join(THIS_FOLDER, 'tr_2.csv'))))
        transition_railway = np.array(transition_railway, dtype = np.float32)

        transition_airway = list(csv.reader(open(os.path.join(THIS_FOLDER, 'tr_1.csv'))))
        transition_airway = np.array(transition_airway, dtype = np.float32)

        transition_roadway = list(csv.reader(open(os.path.join(THIS_FOLDER, 'tr_3.csv'))))
        transition_roadway = np.array(transition_roadway, dtype = np.float32)

        transition_matrix_init = (transition_railway + transition_airway + transition_roadway).astype(int)

        tr_table = [transition_airway, transition_railway, transition_roadway]

        for j, tr in enumerate(tr_table):
            for i in range(17):
                tr[i, :] = tr[i, :]*param_transition_box[i,j]
                tr[:, i] = tr[i, :]*param_transition_box[i,j]

        transition_matrix = (transition_railway + transition_airway + transition_roadway).astype(int)
        transition_matrix = 0.5*transition_matrix * (param_transition_scale )

        for i in range(nodes_num):
            for j in range(nodes_num):
                if transition_matrix[i,j] < 0.01:
                    transition_matrix[i,j] = transition_matrix_init[i,j]*param_transition_leakage # base data is for 24 days, tran_dt = 1/2

        transition_matrix = transition_matrix.astype(int)

        config.transition_matrix = transition_matrix

        self.update(False)

    def reset_click(self):
        if config.flag_sim == 0:
            config.new_plot_all = []
            config.counter_func = 0
            config.counter = 0
            config.run_iteration=False
            self.update(False)

    def run_click(self):
        if config.flag_sim == 0:
            self.save_click()
            config.run_iteration=True
            self.update(False)

    def save_file_click(self):
        if config.flag_sim == 0:
            # points*nodes*states
            info = config.header_file_csv
            params_local = np.vstack([config.param_beta_exp, config.param_qr, config.param_sir, config.param_eps_exp, config.param_eps_qua,
                    config.param_eps_sev,config.param_hosp_capacity, config.param_gamma_mor1,config.param_gamma_mor2,
                    config.param_gamma_im, config.param_init_susceptible, config.param_init_exposed])

            params_global = [config.counter_func, config.param_t_exp, config.param_t_inf, 1, 12]

            directory = 'results' + '/' +  config.param_save_file
            if not os.path.exists(directory):
                os.makedirs(directory)


            #####
            box_corr = np.zeros((17,3))
            for b in range(config.counter_func):
                curr_transition_box = config.box_time[b+1]
                tr_boxes = curr_transition_box
                param_transition_box = np.zeros((17,3))
                for i, way in enumerate(tr_boxes): # air 0 rail 1 road 2
                    for j, node in enumerate(way):
                        status = int(node)
                        param_transition_box[status, i] = 1
                box_corr = np.dstack([box_corr, param_transition_box])
                #box_corr = box_corr[:,:,]
                #, box_corr[iter,j,0], box_corr[j,1],  box_corr[j,2]
                #print(b)
                #print(box_corr)
                #print(box_corr.shape)
            #print(box_corr)
            box_corr = box_corr[:,:,1:]
                #print(box_corr[0,0,:])
            #####

            for j in range(17):
                filename =  directory + '/' + self.region_names[j] + '.csv'
                with open(filename, 'w', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',',
                            escapechar=' ', quoting=csv.QUOTE_NONE)
                    #points*nodes*states
                    spamwriter.writerow([info])
                    for iter in range(config.counter_func):
                        if config.new_plot_all:
                            one_arr = config.new_plot_all[iter] #
                            one_arr_node = one_arr[-1,j,:].astype(int)
                            one_arr_node = np.append(one_arr_node, (config.param_beta_exp[j], config.param_qr[j], config.param_sir[j],
                                config.param_hosp_capacity[j], config.param_gamma_mor1[j], config.param_gamma_mor2[j], config.param_gamma_im[j], config.param_eps_exp[j],
                                config.param_eps_qua[j], config.param_eps_sev[j], config.param_t_exp[0], config.param_t_inf[0], config.param_transition_leakage,
                                config.param_transition_scale,box_corr[j,0,iter],box_corr[j,1,iter],box_corr[j,2,iter]))
                            #print(param_transition_box)
                            #print(one_arr_node.shape)
                            #print(one_arr_node)
                            spamwriter.writerows([one_arr_node])

                    #print(np.array(config.new_plot_all[:][j][:]))
                    #spamwriter.writerow([info])
                    #spamwriter.writerows([np.array(config.new_plot_all[:][j][:])])
                    #spamwriter.writerows([[1,2,3],[4,5,6], [6,7,8]])
                    #spamwriter.writerow([arr])
                    #arr = config.new_plot_all[iter][j][:]
                    #for iter in range(config.counter_func-1):
                    #    arr = config.new_plot_all[iter][j][:]
                    #spamwriter.writerow([arr])
                    #print(config.new_plot_all[iter][j][:])
                    #print(config.new_plot_all.shape)
                        #list = [(params_local[j]) for i in range(0,17)] arr = [()] # iter * 17 * 7
                        #spamwriter.writerow([arr]) # spamwriter.writerow([a for a in arr])
                        #numpy.savetxt("FILENAME.csv", arr, delimiter=",")

            # points*nodes*states
            print('[INFO] Saving results to .csv format ..')

    def slider_update_initial_val(self, attr, old, new):
        self.init_exposed.value = config.param_init_exposed[config.region]
        self.sus_to_exp_slider.value = config.param_beta_exp[config.region]
        self.param_qr_slider.value = config.param_qr[config.region]
        self.param_sir.value = config.param_sir[config.region]
        self.param_hosp_capacity.value = config.hosp_cap_init[config.region]
        self.param_gamma_mor1.value = config.param_gamma_mor1[config.region]
        self.param_gamma_mor2.value = config.param_gamma_mor2[config.region]
        self.param_gamma_im.value = config.param_gamma_im[config.region]
        self.param_eps_exp.value = config.param_eps_exp[config.region]
        self.param_eps_qua.value = config.param_eps_qua[config.region]
        self.param_eps_sev.value = config.param_eps_sev[config.region]
        self.param_t_exp.value = config.param_t_exp[0]
        self.param_t_inf.value = config.param_t_inf[0]


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
        config.param_transition_scale=new
        self.save_click()

    def handler_param_tr_leakage(self, attr, old, new):
        config.param_transition_leakage=new
        self.save_click()

    def handler_checkbox_group1(self, new):
        config.box1 = new
        config.testing_var = config.box1
        self.save_click()

    def handler_checkbox_group2(self, new):
        config.box2 = new
        self.save_click()

    def handler_checkbox_group3(self, new):
        config.box3 = new
        self.save_click()


    def handler_param_save_file(self, attr, old, new):
        config.param_save_file= str(new)

    def layout(self):
        regions = ['Almaty', 'Almaty Qalasy', 'Aqmola', 'Aqtobe', 'Atyrau', 'Batys Qazaqstan', 'Jambyl', 'Mangystau', 'Nur-Sultan', 'Pavlodar', 'Qaraqandy', 'Qostanai',  'Qyzylorda', 'Shygys Qazaqstan', 'Shymkent', 'Soltustik Qazaqstan', 'Turkistan']

        regions_for_show = ['Almaty', 'Almaty Qalasy', 'Aqmola', 'Aqtobe', 'Atyrau', 'Batys Qazaqstan', 'Jambyl', 'Mangystau', 'Nur-Sultan', 'Pavlodar', 'Qaraqandy',
                                'Qostanai',  'Qyzylorda', 'Shygys Qazaqstan', 'Shymkent', 'Soltustik Qazaqstan', 'Turkistan']

        text_save = TextInput(value="foldername", title="")
        text_save.on_change('value', self.handler_param_save_file)

        # select region
        initial_region = 'Almaty'
        region_selection = Select(value=initial_region, title='        ', options=regions_for_show, max_width=250, max_height=20)
        region_selection.on_change('value', self.SelectRegionHandler)

        #select parameters
       #select parameters
        self.sus_to_exp_slider = Slider(start=0.0,end=0.5,step=0.005,value=config.param_beta_exp[config.region], title='Susceptible to Exposed transition constant')
        self.sus_to_exp_slider.on_change('value', self.handler_beta_exp)

        self.param_qr_slider = Slider(start=0.0,end=0.25,step=0.0025,value=config.param_qr[config.region], title='Daily Quarantine rate of the Exposed ')
        self.param_qr_slider.on_change('value', self.handler_param_qr)

        self.param_sir = Slider(start=0.0,end=0.25,step=0.0025,value=config.param_sir[config.region], title='Daily Infected to Severe Infected transition rate ')
        self.param_sir.on_change('value', self.handler_param_sir)

        self.param_eps_exp = Slider(start=0,end=1,step=0.01,value=config.param_eps_exp[config.region], title='Disease transmission rate of Exposed compared to Infected')
        self.param_eps_exp.on_change('value', self.handler_param_eps_exp)

        self.param_eps_qua = Slider(start=0,end=1,step=0.01,value=config.param_eps_qua[config.region], title='Disease transmission rate of Quarantined compared to Infected')
        self.param_eps_qua.on_change('value', self.handler_param_eps_qua)

        self.param_eps_sev = Slider(start=0,end=1,step=0.01,value=config.param_eps_sev[config.region], title='Disease transmission rate of Severe Infected compared to Infected')
        self.param_eps_sev.on_change('value', self.handler_param_eps_sev)

        self.param_hosp_capacity = Slider(start=0,end=10000,step=1,value=config.hosp_cap_init[config.region], title='Hospital Capacity')
        self.param_hosp_capacity.on_change('value', self.handler_param_hosp_capacity)

        self.param_gamma_mor1 = Slider(start=0,end=1,step=0.01,value=config.param_gamma_mor1[config.region], title='Severe Infected to Dead transition probability')
        self.param_gamma_mor1.on_change('value', self.handler_param_gamma_mor1)

        self.param_gamma_mor2 = Slider(start=0,end=1,step=0.01,value=config.param_gamma_mor2[config.region], title='Severe Infected to Dead transition probability (Hospital Cap. Exceeded)')
        self.param_gamma_mor2.on_change('value', self.handler_param_gamma_mor2)

        self.param_gamma_im = Slider(start=0,end=1,step=0.1,value=config.param_gamma_im[config.region], title='Infected to Recovery Immunized transition probability')
        self.param_gamma_im.on_change('value', self.handler_param_gamma_im)

        self.param_sim_len = Slider(start=2,end=100,step=2,value=config.loop_num, title='Length of simulation (Days)')
        self.param_sim_len.on_change('value', self.handler_param_sim_len)

        self.param_t_exp = Slider(start=1,end=20,step=1,value=config.param_t_exp[0], title='Incubation period (Days) ')
        self.param_t_exp.on_change('value', self.handler_param_t_exp)

        self.param_t_inf = Slider(start=1,end=20,step=1,value=config.param_t_inf[0], title=' Infection  period (Days) ')
        self.param_t_inf.on_change('value', self.handler_param_t_inf)


        self.init_exposed = Slider(start=0,end=100,step=1,value=config.param_init_exposed[config.region], title='Initial Exposed')
        self.init_exposed.on_change('value', self.handler_init_exposed)

        self.param_tr_scale = Slider(start=0.0,end=1,step=0.01,value=config.param_transition_scale, title='Traffic ratio')
        self.param_tr_scale.on_change('value', self.handler_param_tr_scale)

        self.param_tr_leakage = Slider(start=0.0,end=1,step=0.01,value=config.param_transition_leakage, title='Leakage ratio')
        self.param_tr_leakage.on_change('value', self.handler_param_tr_leakage)

        dumdiv = Div(text='',width=10)
        dumdiv2= Div(text='',width=10)
        dumdiv3= Div(text='',width=150)

        ######### CHANGE
        # Buttons
        reset_button = Button(label = 'Reset Button', button_type='primary')
        save_button = Button(label='Update transition matrix', button_type='primary')
        save_button_result = Button(label='Save current plot to .csv in directory results/', button_type='primary')
        run_button = Button(label='Run the simulation',button_type='primary')
        #########  CHANGE


        save_button.on_click(self.save_click)
        run_button.on_click(self.run_click)
        reset_button.on_click(self.reset_click)
        save_button_result.on_click(self.save_file_click)

        div_cb1 = Div(text = 'Airways', width = 150)
        div_cb2 = Div(text = 'Railways', width = 150)
        div_cb3 = Div(text = 'Highways', width = 150)

        checkbox_group1 = CheckboxGroup(labels=regions, active =  list(range(0, 17)))
        checkbox_group2 = CheckboxGroup(labels=regions, active= list(range(0, 17)))
        checkbox_group3 = CheckboxGroup(labels=regions, active= list(range(0, 17)))

        checkbox_group1.on_click(self.handler_checkbox_group1)
        checkbox_group2.on_click(self.handler_checkbox_group2)
        checkbox_group3.on_click(self.handler_checkbox_group3)

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
            c16=[(config.transition_matrix[16,i]) for i in range(0,17)],
                )

        self.sourceT = ColumnDataSource(self.data1)

        columns = [
                    TableColumn(field="c00", title=" ",),
                    TableColumn(field="c0", title="Almaty",),
                    TableColumn(field="c1", title="Almaty Qalasy",),
                    TableColumn(field="c2", title="Aqmola",),
                    TableColumn(field="c3", title="Aqtobe",),
                    TableColumn(field="c4", title="Atyrau",),
                    TableColumn(field="c5", title="Batys Qazaqstan",),
                    TableColumn(field="c6", title="Jambyl",),
                    TableColumn(field="c7", title="Mangystau",),
                    TableColumn(field="c8", title="Nur-Sultan",),
                    TableColumn(field="c9", title="Pavlodar",),
                    TableColumn(field="c10", title="Qaragandy",),
                    TableColumn(field="c11", title="Qostanai",),
                    TableColumn(field="c12", title="Qyzylorda",),
                    TableColumn(field="c13", title="Shygys Qazaqstan",),
                    TableColumn(field="c14", title="Shymkent",),
                    TableColumn(field="c15", title="Soltustik Qazaqstan",),
                    TableColumn(field="c16", title="Turkistan",),]

        self.data_tableT = DataTable(source=self.sourceT, columns=columns, width=1200, height=500, sortable = False)

        sliders_1 = column(self.init_exposed, self.sus_to_exp_slider, self.param_qr_slider, self.param_sir)
        sliders_2 = column(self.param_hosp_capacity, self.param_gamma_mor1, self.param_gamma_mor2, self.param_gamma_im)
        sliders_0 = column(self.param_eps_exp, self.param_eps_qua, self.param_eps_sev)

        sliders = row(sliders_1, dumdiv3, sliders_2, dumdiv3, sliders_0)
        # regions

        sliders_3 = row(self.param_t_exp, self.param_t_inf, self.param_sim_len)
        #sliders_3
        # global
        nu_logo = Div(text="""<img src='/streaming/static/nu_logo1.jpg'>""", width=650, height=100)
        issai_logo = Div(text="""<img src='/streaming/static/issai_logo_new.png'>""", width=650, height=252)
        text2 = Div(text="""<h1 style='color:black'>   issai.nu.edu.kz/episim </h1>""", width = 500, height = 100)

        text_footer_1 = Div(text="""<h3 style='color:green'> Developed by ISSAI Researchers : Askat Kuzdeuov, Daulet Baimukashev, Bauyrzhan Ibragimov, Aknur Karabay, Almas Mirzakhmetov, Mukhamet Nurpeiissov and Huseyin Atakan Varol </h3>""", width = 1500, height = 10)
        text_footer_2 = Div(text="""<h3 style='color:red'> Disclaimer : This simulator is a research tool. The simulation results will show general trends based on entered parameters and initial conditions  </h3>""", width = 1500, height = 10)
        text_footer = column(text_footer_1, text_footer_2)
        text = column(self.text1, text2)
        #header = None
        #header = row(nu_logo, text , issai_logo)

        draw_map_js = CustomJS(code=""" uStates.draw("#statesvg", sampleData, tooltipHtml); """)
        run_button.js_on_click(draw_map_js)

        ########### CHANGE ###################
        layout_t = row(save_button_result, text_save)
        buttons = row(reset_button,run_button, layout_t)

        reg1 = row(self.text2, column(self.text4rr,region_selection))

        buttons = column(buttons, reg1, self.text4)

        params =  column(sliders, self.text3, self.text4, sliders_3, self.text5, self.text4,)

        sliders_4 = column(self.param_tr_scale, self.param_tr_leakage)
        check_table = row(column(div_cb1,checkbox_group1), column(div_cb2,checkbox_group2), column(div_cb3,checkbox_group3), sliders_4)
        check_trans = row(self.data_tableT)

       
        #kz_map_tag.js_on
        ###
        dummy_div = Div(text=""" """, height=25);
        layout = column(self.pAll, buttons)
        layout = column (layout, dummy_div, params, check_table)

        layout = column (layout, check_trans, self.text4)


        

        layout = column (layout)
        layout = column (layout,self.text4) #text_footer


        self.doc.title = 'Covid Simulation'
        self.doc.add_root(layout)

        ################################# CHANGE #########################
        '''

        buttons = row(reset_button,save_button, run_button)

        params =  column(sliders, self.text3, self.text4, sliders_3, self.text5, self.text4)

        sliders_4 = column(param_tr_scale, param_tr_leakage)
        check_table = row(column(div_cb1,checkbox_group1), column(div_cb2,checkbox_group2), column(div_cb3,checkbox_group3), sliders_4)
        check_trans = row(self.data_tableT)

        layout = column(self.text1, self.pAll)
        layout = column (layout, params, check_table)
        layout = column (layout, check_trans, buttons, self.text7, self.text4)
        layout_t = column(text_save, save_button_result)
        layout = column (layout, layout_t)

        self.doc.title = 'Covid Simulation'
        self.doc.add_root(layout)
        '''
