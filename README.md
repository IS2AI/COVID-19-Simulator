# COVID-19 Epidemic Simulator with Network Transition

## The software package contains:

### 1) In directory matlab_code/ 
Simulation of SEQISR model for single node written in Matlab software. This model is based on a previously published paper entitled “MOSES: A Matlab-based open-source stochastic epidemic simulator” (H.A. Varol) in IEEE International Conference of the Engineering in Medicine and Biology Society (EMBC), 2016 (https://ieeexplore.ieee.org/document/7591271).

### 2) In directory covid19/
Stochastic epidemic simulator to model the spread of the COVID-19 epidemic in the Republic of Kazakhstan. The software integrates the dynamic transitions between 17 regions and enables to simulate the various scenarios by adjusting the parameters and transition matrix.


## Installation guides

1) Install Anaconda3 with from https://www.anaconda.com/distribution/#download-section

2) Start Anaconda3 Terminal (Anaconda Prompt)

3) Install Bokeh visualization library using the command: **conda install bokeh**

4) Clone the repository to your working directory

5) Go to directory *covid19/* and run the following command on terminal: **bokeh serve --show streaming**

6) The web browser with visualization interface will be opened

### 3) In directory visualization_tools/

Visualization toolbox is presented for displaying the results from saved .csv files. Both Matlab and Python versions are included.

### Video tutorials

Extensive tutorials on how to use the software and analyze the results of the simulator can be found from https://www.youtube.com/channel/UCr7o_0wW4nkqx-G5b7Zopgw


More detailed information about the project can be found from the webpage (https://issai.nu.edu.kz/episim/).
