import tkinter as tk
from tkinter import font  as tkfont # python 3
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import filedialog
from tkinter import *
global csv_name
root = tk.Tk()
root.geometry("500x200+300+300")
root.title("COVID-19 Visualizer")

def visualize():
    global csv_name
    df = pd.read_csv(csv_name)
    days = df.iloc[:,0]
    infected = df.iloc[:,2]
    isolated = df.iloc[:,9]
    sev_inf = df.iloc[:,4]
    quarantined = df.iloc[:,5]
    immunized = df.iloc[:,6]
    susceptible = df.iloc[:,7]
    dead = df.iloc[:,8]
    
    window_vis = tk.Toplevel(root)
    window_vis.geometry("1000x1000")
    window_vis.title("Visualization")
    plt = Figure()
    csv_name = csv_name.split("/")[-1]
    csv_name = csv_name.split(".")[0]
    plt.suptitle('Visualization for ' + csv_name, fontsize=16)
    plt1 = plt.add_subplot(211)
    plt1.plot(days, infected, color='tab:blue', marker='.', label='Infected')
    plt1.plot(days, isolated, color = 'brown', label='Isolated', marker='.')
    plt1.plot(days, quarantined, color = 'yellow', label='Quarantined', marker='.')
    plt1.plot(days, sev_inf, color = 'red', label='Severely Infected', marker='.')
    plt1.plot(days, immunized, color = 'gold', label='Immunized', marker='.')
    plt1.plot(days, dead, color = 'black', label='Dead', marker='.')
    plt1.legend()
    plt1.grid()
    plt2 = plt.add_subplot(212)
    plt2.plot(days, dead, color='black', marker='.', label='Dead')
    plt2.plot(days, sev_inf, color='red', marker='.', label ='Severe Infected')
    plt2.legend()
    plt2.grid()
    canvas = FigureCanvasTkAgg(plt, master=window_vis)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    toolbar=NavigationToolbar2Tk(canvas, window_vis)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def choose_file():
    global csv_name
    root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("CSV file","*.csv"),("all files","*.*")))
    csv_name = root.filename


b = tk.Button(root, text="Visualize", command=visualize)
b1 = tk.Button(root, text="Choose file", command=choose_file)
b.pack(side=tk.BOTTOM)      # pack starts packing widgets on the left 
b1.pack(side=tk.BOTTOM) 

text2 = tk.Text(root, height=10, width=50)
text2.tag_configure('bold_italics', font=('Arial', 12, 'bold', 'italic'))
text2.tag_configure('big', font=('Verdana', 20, 'bold'))
text2.tag_configure('color',
                    foreground='#476042',
                    font=('Tempus Sans ITC', 12, 'bold'))
text2.tag_bind('follow',
               '<1>',
               lambda e, t=text2: t.insert(tk.END, "Not now, maybe later!"))
text2.insert(tk.END,'\nHow to use:\n', 'big')
quote = """
         Choose the file and press Visualize
"""
text2.insert(tk.END, quote, 'color')
text2.pack()

root.mainloop()





