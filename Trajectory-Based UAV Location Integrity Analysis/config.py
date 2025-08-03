''' Select User '''
USER='Mincheol'
# USER='Chiho'

''' Essential modules'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

''' System modules '''
import glob
import os
from os.path import join as ospj
from os.path import exists as osex
from os import mkdir #udt rqrd
from os import rmdir #udt rqrd

''' Scikit-learn modules '''
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

''' Pytorch modules'''
import torch
from torch import nn, optim
import torch.functional as F
from torch.utils.data import DataLoader

''' User handler'''
if USER=='Mincheol':
    #drive mount
    from google.colab import drive
    drive.mount('/content/drive')
    #mount google drive
    import sys
    sys.path.append('/content/drive/MyDrive')
    path_workspace=r'/content/drive/MyDrive/'
elif USER=='Chiho':
    path_workspace=r'C:\_Sync_Study\Grad_Lab\UAV\Source'
    
''' CONSTANTS '''
#udt rqrd
raw_idx=['timestamp', 'time_utc_usec', 'lat', 'lon', 'alt', 'alt_ellipsoid',
        's_variance_m_s', 'c_variance_rad', 'eph', 'epv', 'hdop', 'vdop',
        'noise_per_ms', 'jamming_indicator', 'vel_m_s', 'vel_n_m_s',
        'vel_e_m_s', 'vel_d_m_s', 'cog_rad', 'timestamp_time_relative',
        'heading', 'heading_offset', 'fix_type', 'jamming_state',
        'vel_ned_valid', 'satellites_used', 'selected'] #27 columns

''' Path locator '''
path_merged=ospj(path_workspace, 'Collected_dataset', 'Merged')
path_split=ospj(path_workspace, 'Collected_dataset', 'Split')