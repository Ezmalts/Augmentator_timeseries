import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from IPython.display import clear_output
import torch
import random
import os

class Augmentator:
    def __init__(self):
        pass
    
    def add_peak(self, ts, len_peak=4, peak_factor=2.):
        lenn = len(ts)
        ind_start = random.randint(0, lenn-len_peak-1)
        ind_end = ind_start + len_peak 
        ts_augm = ts.astype(float).copy()
        ts_augm[ind_start:ind_end] *= peak_factor
        
        return ts_augm
    
    def add_flipping(self, ts, b=None):
        if b is None:
            b = np.mean(ts)

        return np.where(2 * b - ts > 0, 2 * b - ts,  0)
    
    def add_smart_peak(self, ts, len_drop = 4, len_peak=4, peak_factor=2.):
        
        lenn = len(ts)
        ind_start = random.randint(0, lenn-len_peak-len_drop-1)
        ind_end = ind_start + len_peak
        ind_start_drop = ind_start + len_peak + 1
        ind_end_drop = ind_start + len_peak + 1 + len_drop
        
        ts_augm = ts.astype(float).copy()
        ts_augm[ind_start:ind_end] *= peak_factor
        ts_augm[ind_start_drop:ind_end_drop] = 0
        
        return ts_augm
    
    def add_exchange(self, ts, len_segm=4):
        lenn = len(ts)
        ind_start_1 = random.randint(0, lenn-len_segm-1)
        ind_end_1 = ind_start_1 + len_segm
        
        if ind_start_1 > lenn - 1 - ind_end_1:
            ind_start_2 = random.randint(0, ind_start_1-len_segm)
            ind_end_2 = ind_start_2 + len_segm
        else:
            ind_start_2 = random.randint(0, lenn - 1 - ind_end_1 - len_segm) + ind_end_1
            ind_end_2 = ind_start_2 + len_segm
        
        ts_augm = ts.astype(float).copy()
        tmp = ts_augm[ind_start_1:ind_end_1].copy()
        ts_augm[ind_start_1:ind_end_1] = ts_augm[ind_start_2:ind_end_2]
        ts_augm[ind_start_2:ind_end_2] = tmp
        
        return ts_augm
        
    
    def add_drops(self, ts, len_drop=12, num_drop=1):
        lenn = len(ts)
        ind_start = random.randint(0, lenn-len_drop-1)
        ind_end = ind_start + len_drop 
        ts_augm = ts.copy()
        ts_augm[ind_start:ind_end] = 0
        
        return ts_augm
    
    def add_norm_noise(self, ts, scale_factor=3):
        len_ts = len(ts)
        diffs = np.diff(ts, 1)
        std = np.std(diffs) / scale_factor
        
        noise = np.random.normal(loc=0.0, scale=std, size=len_ts-1)
        ts_augm = ts.copy()
        ts_augm[1:] = ts_augm[1:] + noise
        
        ts_augm = np.where(ts_augm<0,0,ts_augm)
        
        return ts_augm
        
        
    def make_ds(self, list_dfs):
        list_dfs_diff = []
        
        for df in list_dfs:
            list_dfs_diff.append(df.diff(1).dropna())
            
        tr_set = pd.concat(list_dfs_diff,axis=0)
        tr_set = tr_set.values.reshape(-1, len(tr_set.columns))
        
        return tr_set