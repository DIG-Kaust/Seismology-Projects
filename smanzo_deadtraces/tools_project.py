import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import segyio
import random
import math
import pandas as pd


import utils_project as up

######### DATA SET UTIL FUNCTIONS #########

###########################################

#### MASKS FUNCTIONS ####

"""
Function to obtain the second round of masks for the  whole dataset
Uses the function to get the masks that deals with traces that have information but are burried into other signals
"""

def get_masks_dataset_nt(data,time,rms_factor):
    
    num_data = len(data)
    nt_masks = []
    
    print(num_data)
    
    for i in range(num_data):
        masks_sets = []
        
        dataset = data[i]
        num_shots = len(data[i])
        
        for j in range(num_shots):
            
            shot = dataset[j]

            mask = up.get_dead_trace_mask_nt(shot,time,rms_factor)
            masks_sets.append(mask)
        
        nt_masks.append(masks_sets)
    
    return nt_masks

"""
Function to obtain the initial masks for the  whole dataset
Uses the function to get the masks that tries to identify zero values or constant values
"""

def get_masks_dataset_dt(data,time,amp_factor):
    
    num_data = len(data)
    dt_masks = []
    
    print(num_data)
    
    for i in range(num_data):
        masks_sets = []
        
        dataset = data[i]
        num_shots = len(data[i])
        
        for j in range(num_shots):
            
            shot = dataset[j]

            mask = up.get_dead_trace_mask_dt(shot,time,amp_factor)
            masks_sets.append(mask)
        
        dt_masks.append(masks_sets)
    
    return dt_masks

"""
Function to obtain the coherence masks for the  whole dataset
Uses the function to compare the coherence in terms of time (chunks of the data).
"""

def get_masks_dataset_coherence(data,time,amp_factor):
    
    num_data = len(data)
    coh_masks = []
    
    print(num_data)
    
    for i in range(num_data):
        masks_sets = []
        
        dataset = data[i]
        num_shots = len(data[i])
        
        for j in range(num_shots):
            
            shot = dataset[j]

            mask = up.get_dead_trace_mask_sampling(shot,time,amp_factor)
            masks_sets.append(mask)
        
        coh_masks.append(masks_sets)
    
    return coh_masks

###########################################

#### REPLACEMENT FUNCTIONS ####

def apply_replacement(datasets,masks):
    
    num_data = len(datasets)
    
    final_data = []
    
    for i in range(num_data):
        
        dataset = datasets[i]
        maskset = masks[i]
        
        num_shots = len(dataset)
        
        data = []
        
        for j in range(num_shots):
            
            shot = dataset[j]
            mask = maskset[j]
            
            tr_shot = shot.copy()

            new_traces, new_id = up.replace_traces_mask_check(tr_shot,mask)
            traces = up.traces_replacement(new_traces,new_id,tr_shot)
            
            data.append(traces)
        
        final_data.append(data)
    
    return final_data


def apply_final_replacement(datasets,masks):
    
    num_data = len(datasets)
    
    final_data = []
    
    for i in range(num_data):
        
        dataset = datasets[i]
        maskset = masks[i]
        
        num_shots = len(dataset)
        
        data = []
        
        for j in range(num_shots):
            
            shot = dataset[j]
            mask = maskset[j]
            
            traces = up.double_dead_traces(shot,mask)
            
            data.append(traces)
        
        final_data.append(data)
    
    return final_data



######### PLOTTING TOOLS #################

def plot_data(data,vmin,vmax,x,y,component):
    
    name = component + 'Test array - '

    for i in range(len(data)):

        plt.figure(figsize=(x, y))
        plt.imshow(data[i].T, cmap='gray', vmin=vmin, vmax=vmax)
        plt.axis('tight')
        plt.title(name+str(i))
        plt.show()