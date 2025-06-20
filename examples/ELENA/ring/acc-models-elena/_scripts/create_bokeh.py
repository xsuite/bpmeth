
#%%
import datetime
import tfs
import glob
import numpy as np
import matplotlib.pyplot as plt
import jinja2
import pandas as pnd
import httpimport 
import os
from bokeh.models import Range1d

#from plots import plots
with httpimport.remote_repo(['plots'], 'https://gitlab.cern.ch/acc-models/acc-models-utils/-/raw/master/'):
    from plots import plots

files = glob.glob('scenarios/*/*.tfs')

for f in files:
    # Load TFS table
    print(f)
    data = tfs.read(f)
    # convert to pkl
    data.to_pickle(f[:-3] + 'pkl')
    # generate plots
    plots.plot_optics(f[:-3] + 'pkl', f[:-3] + 'html', closed_orbit=False, output='file')

