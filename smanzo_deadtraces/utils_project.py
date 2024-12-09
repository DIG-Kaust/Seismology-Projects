import warnings
warnings.filterwarnings('ignore')

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import segyio
import random
import math
import pandas as pd

from scipy.signal import butter, lfilter, filtfilt, freqz, medfilt, fftconvolve
from scipy.sparse import csr_matrix, vstack
from scipy.signal import filtfilt
from scipy.linalg import lstsq, solve
from scipy.sparse.linalg import LinearOperator, cg, lsqr
from scipy import misc

from segyshot import SegyShot

"""
Useful functions for the main code for dead trace identification and replacement

This script contains several functions to create. treat and analyze seismic data, to avoid 
the main Jupyter notebook becomes full of many functions.

Sofia Irene Manzo Vega - KAUST
Last modified: 25-11-2024
"""


##### REPLACEMENT UTILS #####

"""
Function to replace the damaged traces in the original data, receives the new traces
calculated by different types of interpolation
Parameters:
new_traces: np array of new traces.
new_traces_id: Index of the damaged trace to replace.
traces: original array of traces
Returns:
trace: Updated shot_gather with the dead trace replaced.
"""

def traces_replacement(new_traces,new_traces_id,traces):

    num_traces = len(new_traces)

    for i in range(num_traces):
        id = new_traces_id[i]
        traces[id] = new_traces[i]

    return traces

"""
Funtion to obtain the new interpolated traces and the index.
This manages isolated traces, meaning, there is normal traces surrounding
the identified dead trace. This ensures to have the information to do the average
between traces and interpolates them.

Parameters
traces: array of traces forming the full shot
mask: array of zeros and ones same size of the shot (# traces) where the ones represent dead traces
"""

# THIS FUNCTION MANAGES ONLY ISOLATED TRACES
# IF THE MASK IS SHOWING SEVERAL TRACES THAT FIT THE PARAMETERS, THIS METHOD WILL IGNORED THEM
# AS IT ONLY LOOKS FOR "1s" IN THE MASK SURROUNDING BY "0s"
# ANOTHER METHOD IS INTRODUCED TO MANAGE SEVERAL ANOMALOUS TRACES  

def replace_traces_mask_check(traces,mask):

    num_traces = len(traces)

    interpolated_traces = []
    interpolated_traces_id = []


    for i in range(2,num_traces-3):

        if mask[i] == 1:

            v_bef = mask[i-1]
            v_aft = mask[i+1]

            v_bef_2 = mask[i-1] + mask[i-2]
            v_aft_2 = mask[i+1] + mask[i+2]

            if v_bef_2 == 0 and v_aft_2 == 0:          
                new_trace = replace_trace_with_interpolation(traces,i)
                interpolated_traces.append(new_trace)
                interpolated_traces_id.append(i)
                #print(i)
            elif v_bef == 0 and v_aft == 0:
                new_trace = replace_trace_with_interpolation(traces,i)
                interpolated_traces.append(new_trace)
                interpolated_traces_id.append(i)
                
            elif v_bef_2 == 2 and v_aft == 2:
                pass
                #raise ValueError("No traces to interpolate.")
            
            elif v_bef_2 == 1 or v_aft_2 == 1:
                pass
                # raise ValueError("No traces to interpolate")

    if mask[1] == 1 or mask[num_traces-1] == 1:

        v_bef = mask[i-1]
        v_aft = mask[i+1]

        if v_bef == 1 and v_aft == 1:
            pass
            #raise ValueError("No traces to interpolate")


    return interpolated_traces,interpolated_traces_id

"""
Function to replace a damaged trace with the average of 2 consecutive traces before
and two consecutive tracess after the damaged trace.
traces: np array of traces, where each row is a trace.
dead_trace_id: Index of the damaged trace to replace.
Returns:
trace: Updated trace with the dead trace replaced.
"""

def replace_trace_with_average_4(traces, dead_trace_id):

    if dead_trace_id < 2 or dead_trace_id > len(traces) - 3:
        average_trace = replace_trace_with_interpolation(traces,dead_trace_id)
    traces_before = traces[dead_trace_id - 2:dead_trace_id]
    print()
    traces_after = traces[dead_trace_id + 1:dead_trace_id + 3]
    average_trace = (np.sum(traces_before, axis=0) + np.sum(traces_after, axis=0)) / 4.0

    return average_trace

"""
Function to replace a damaged trace with the average of the trace before
and the trace after the damaged trace.
Parameters:
traces: np array of traces, where each row is a trace.
dead_trace_id: Index of the damaged trace to replace.
Returns:
trace: Updated trace with the dead trace replaced.
"""
def replace_trace_with_interpolation(traces, dead_trace_id):

    if dead_trace_id <= 0 or dead_trace_id >= len(traces) - 1:
        raise ValueError("Damaged trace index must have valid neighbors.")

    trace_before = traces[dead_trace_id - 1]
    trace_after = traces[dead_trace_id + 1]
    interpolated_trace = (trace_before + trace_after) / 2.0
    
    return interpolated_trace

##############################################################################


############# IDENTIFICATION UTIL FUNCTIONS ##############################

"""
Function to identify traces with noise, amplitude or energy issues. In these function,
the masks are created with the objective of identify traces that contain information
(non zero traces, or constant traces) but the receiver malfunction cause issues that 
the trace to look noisy.

frec_mask = mask provided by the identification frequency which analyzes the frequency content
            and classifies the flattness of the frequency (three categories: flat, poor and good)

amp_mask = mask provided by the amplitude identification method, which compares the amplitude of 
            each trace in comparison of the global amplitude difference, to identify low amplitude
            traces.

energy_mask = mask provided by the energy change identification method, which compares the energy of each
                 against the mean energy. This is to identify big peaks of energy caused by traces with high peaks.

Parameters:
shot_gather: array of traces 
time: size of the traces or samples for each trace (for the frequency analysis)
amp_factor: factor to identify low amplitude traces, it compares the traces with the global
            amplitud difference to see if they are lacking amplitud in comparison of the average traces
Returns
mask: single numpy array maks with the problematic traces marked with 1 and normal traces with 0
"""

def get_dead_trace_mask_nt(shot_gather,time,amp_factor):
    
    # Frecuency analysis mask
    frec_mask = trace_identification_frecuency(shot_gather,time)
    
    # Amplitude analysis mask
    amp_mask = trace_identification_amplitude(shot_gather,amp_factor)
    
    # Energy analysis mask
    energy_mask = trace_identification_energy_change(shot_gather)
    
    
    mask = (frec_mask + amp_mask + energy_mask)
    
    
    for i in range(len(mask)):
        if mask[i] != 0:
            mask[i] = 1
    
    return mask

"""
Function to identify fully dead traces, it means zero or constant amplitude or zero energy. In these function,
the masks are created with the objective of identify traces that do not contain any information at all.
Where there no recordings lor the recordings are completly useless.

frec_mask = mask provided by the identification frequency which analyzes the content frecuency, if the trace does
not contain any complex numbers (which are expected in a signal after the Fast Fourier Transform) it is marked as dead.

amp_mask = mask provided by the zero amplitude identification method, which compares the amplitude of the traces to
see if they are constant or fully zero.

energy_mask = mask provided by the zero energy change identification method, which calculates the energy, if the trace is zero energy
it is marked as dead. Also to identify constant values by the sum of all values.

rms_mask = identify traces with a very small Root Mean Squared Error to see if they are zero or constant, spiked traces or clipped traces.

Parameters:
shot_gather: array of traces 
time: size of the traces or samples for each trace (for the frequency analysis)
rms_factor: sets a factor like a threshold to compared the RMS of all traces against the trace to identify problematic traces.
Returns
mask: single numpy array maks with the problematic traces marked with 1 and normal traces with 0
"""

def get_dead_trace_mask_dt(shot_gather,time,factor_rms):
    
    # Frecuency analysis mask
    #frec_mask = trace_identification_frecuency_zero(shot_gather,time)
    
    # Amplitude analysis mask
    #amp_mask = trace_identification_zero_amplitude(shot_gather)
    
    # Energy analysis mask
    energy_mask = trace_identification_zero_energy(shot_gather)
    
    # RMS analysis mask
    rms_mask = trace_identification_rms(shot_gather,factor_rms)
    
    mask = (energy_mask + rms_mask)
    
    
    for i in range(len(mask)):
        if mask[i] != 0:
            mask[i] = 1
    
    return mask

"""
Function to identify coherence into the traces, by diving the traces in a vertical way, or in time.
This locates traces which might be coherent in the high energy zone (reflections, arrivals, etc.)
but also have energy in the parts where all the other traces are low amplitude.
"""

def get_dead_trace_mask_sampling(traces,time_intervals,factor):
    
    # Coherence test
    new_mask = identify_anomalous_trace(traces,time_intervals,factor)
    
    for i in range(len(new_mask)):
        if new_mask[i] != 0:
            new_mask[i] = 1
    
    return new_mask


############################################################################

##### IDENTIFICATION ALGORHITMS #####

def trace_identification_frecuency(shot_gather,time):
    num_traces = len(shot_gather)
    num_samples = len(shot_gather)
    
    
    trace_fft, mag_spec_frec = frecuency_analysis_trace(shot_gather,time)
    freq = []
    freq.append(trace_fft)
    freq.append(mag_spec_frec)
    
    mag_spec_frec_mask = np.ones(num_samples)
    mag_spec_frec_zero = np.ones(num_samples)
    
    ### Frecuency flatness analysis
    for i in range(len(freq[1])):
        if freq[1][i] == 1 or freq[1][i] == 0:
            mag_spec_frec_mask[i] = 1
        else:
            mag_spec_frec_mask[i] = 0
    
    ### Frecuency complex analysis
    for i in range(len(freq[0])):
        fft_trace_sum = sum(freq[0][i])
        
        if isinstance(fft_trace_sum, complex):
            mag_spec_frec_zero[i]=0
    
    return mag_spec_frec_mask

def trace_identification_frecuency_zero(shot_gather,time):
    num_traces = len(shot_gather)
    num_samples = len(shot_gather)
    
    
    trace_fft, mag_spec_frec = frecuency_analysis_trace(shot_gather,time)
    freq = []
    freq.append(trace_fft)
    freq.append(mag_spec_frec)
    
    mag_spec_frec_mask = np.ones(num_samples)
    mag_spec_frec_zero = np.ones(num_samples)
    
    ### Frecuency flatness analysis
    for i in range(len(freq[1])):
        if freq[1][i] == 1 or freq[1][i] == 0:
            mag_spec_frec_mask[i] = 1
        else:
            mag_spec_frec_mask[i] = 0
    
    ### Frecuency complex analysis
    for i in range(len(freq[0])):
        fft_trace_sum = sum(freq[0][i])
        
        if isinstance(fft_trace_sum, complex):
            mag_spec_frec_zero[i]=0
    
    return mag_spec_frec_zero

def trace_identification_energy_change(shot_gather):
    
    mask = np.zeros(len(shot_gather))
    
    energy = trace_energy(shot_gather)
    energy_norm = energy / np.min(energy)
    
    mean = np.mean(energy_norm)
    std = np.std(energy_norm)
    
    for i in range(len(energy_norm)):
        
        if i == 0 or i == len(shot_gather)-1:
            pass
            
        else:
            next_tr = energy_norm[i+1]
            bef_tr = energy_norm[i-1]
            tr = energy_norm[i]
            
            if tr > mean:
                mask[i] = 1
    
    return mask


def trace_identification_zero_energy(shot_gather):
    
    energy_test = trace_energy(shot_gather)
    mask = np.zeros(len(shot_gather))
    
    for i in range(len(energy_test)):
        
        energy = energy_test[i]
        if energy == 0:
            mask[i] = 1
    
    return mask

def trace_identification_amplitude(arr,factor):
    
    mask = np.zeros(len(arr))
    
    amp_diff = amplitude_difference(arr)
    amp_diff_global = np.mean(amp_diff)
    
    for i in range(len(amp_diff)):
        if i == 0 or i == len(amp_diff)-1:
            pass
        
        elif (amp_diff[i]) <= factor*amp_diff_global:
            
            mask[i] = 1
    return mask

def trace_identification_zero_amplitude(arr):
    
    mask = np.zeros(len(arr))
    for i in range(len(arr)):
        
        if np.min(arr[i]) == np.max(arr[i]):
            mask[i] = 1
        
        elif np.all(arr[i]) == 0:
            mask[i] = 1

        else:
            pass
    
    return mask

def trace_identification_rms(shot_gather, factor):
    
    num_samples = len(shot_gather[0])
    rms_test = rms_trace_amplitude(shot_gather, num_samples)
    
    rms_global_average = np.mean(rms_test)
    
    mask = np.zeros(len(shot_gather))
    
    
    for i in range(len(rms_test)):
        rms_trace = rms_test[i]
        
        if rms_trace <= (factor*rms_global_average):
            mask[i] = 1
        else:
            mask[i] = 0
    
    return mask

##################################################################################

###### COMPLEX TRACES IDENTIFICATION METHODS #############

"""
Function to identify traces that are coherent in the beginning but deviate later.
Parameters:
traces : 2D array of traces (rows are traces, columns are time samples).
time_intervals: List of intervals like [(start1, end1), (start2, end2)].
factor: Threshold factor to compare deviations in variance or amplitude difference.
Returns:
mask: 1D array where 1 indicates an anomalous trace, 0 otherwise.
"""
def identify_anomalous_trace(traces, time_intervals, factor):
    num_traces = traces.shape[0]
    mask = np.zeros(num_traces)
    
    for i, trace in enumerate(traces):
        interval_metrics = []
        for start, end in time_intervals:
            segment = trace[start:end]
            amp_diff = np.max(segment) - np.min(segment)
            variance = np.var(segment)
            interval_metrics.append((amp_diff, variance))
        
        initial_amp_diff, initial_variance = interval_metrics[0]
        for amp_diff, variance in interval_metrics[1:]:
            if (amp_diff > factor * initial_amp_diff) or (variance > factor * initial_variance):
                mask[i] = 1
                break
    
    return mask

def double_dead_traces(traces,mask):
    """
    Final check to the shot gathers to see if there is any left dead traces. This function
    identifies two continuos dead traces that the original replacement function missed. It is desing to target high
    energy traces inside the region of the data whose traces have big energy in general.

    Parameters:
    mask: existing mask with 1s as index of dead traces and 0s as normal traces.
    Returns:
    traces: shot gather with the traces replaced by an average of the surrounding traces
    """
    for i in range(1, len(mask) - 2):
        # Check for pattern [0, 1, 1, 0]
        if mask[i - 1] == 0 and mask[i] == 1 and mask[i + 1] == 1 and mask[i + 2] == 0:
            average = (traces[i-1] + traces[i+2]) / 2
            average_1 = (traces[i-1] + average) / 2
            average_2 = (traces[i+2] + average) / 2
            
            traces[i] = average_1
            traces[i+1] = average_2
    return traces

####################################################################################

##### PARAMETERS UTIL FUNCTIONS #####

"""
The following functions are here to calculate the parameters which the identification functions
will use to create the masks of dead traces.
"""

# Calculate the RMS value for the array of traces
def rms_trace_amplitude(arr, n):
    rms_test = []
    for i in range(len(arr)):
        square = 0
        mean = 0.0
        root = 0.0
        for j in range(0,n):
            square += (arr[i][j]**2)
        mean = (square / (float)(n))
        root = math.sqrt(mean)
        rms_test.append(root)
    return rms_test

# Calculate the amplitude difference for flatness criterion
def amplitude_difference(arr):
    amp_dif = []
    for i in arr:
        amp_min = np.min(i)
        amp_max = np.max(i)
        difference = amp_max - amp_min
        amp_dif_abs = abs(difference)
        amp_dif.append(amp_dif_abs)
        
    return amp_dif

# Calculate the energy of each traces
def trace_energy(arr):
    tr_energy = []
    for i in arr:
        energy = 0.0
        for j in range(len(i)):
            energy += (i[j]**2)
        tr_energy.append(energy)
    return tr_energy

# Calculate the frecuency content
def frecuency_analysis_trace(arr,time):
    trace_fft = []
    mag_spec = []
    for i in arr:
        y = sp.fft(i)
        dt = np.mean(np.diff(time))
        fs = 1 / dt
        trace_fft.append(y)
        frequencies = np.linspace(0, fs/2, len(i)//2)
        mag_spectrum = np.abs(y[:len(frequencies)])
        
        variance_spectrum = np.var(mag_spectrum)
        energy_spectrum = np.sum(mag_spectrum**2)
        flatness_index = sp.stats.gmean(mag_spectrum) / np.mean(mag_spectrum)
        
        if flatness_index > 0.8:
            mag_spec.append(0)
            # flat
        elif energy_spectrum < 1e-3:
            mag_spec.append(1)
            #poor
        else:
            # normal
            mag_spec.append(2)
            
    return trace_fft,mag_spec


###############################################################################

###### UTIL FUNCTIONS ######
"""
Function to obtain the minimun and maximum vaue of a trace of data.
This takes an array with traces, so it will treat the biggest and smallest value of
amplitude for each trace of the array.
Parameters 
-traces: np array which contains the set of traces to treat

Returns
tr_min : np array that contains the minimum value of each trace from the array
tr_max : np array that contains the maximum value of each trace from the array
"""
def get_min_max_traces(traces):
    tr_min_ar = []
    tr_max_ar = []
    for i in traces:
        loc_min = np.min(i)
        loc_max = np.max(i)
        tr_min_ar.append(loc_min)
        tr_max_ar.append(loc_max)

    tr_min = np.min(tr_min_ar)
    tr_max = np.max(tr_max_ar)
    
    return tr_min, tr_max

"""
Function to obtain the mean and standard deviation values of a trace of data.
This takes an array with traces, so it will calculate the mean and standard deviation 
of the amplitude values with the Numpy function for each trace of the array.
Parameters 
-traces: np array which contains the set of traces to treat

Returns
tr_mean : np array that contains the mean of each trace from the array
tr_std : np array that contains the standard deviation of each trace from the array
"""
def get_mean_std(traces):
    tr_mean_ar = []
    tr_std_ar =  []
    for i in traces:
        mean = np.mean(i)
        std = np.std(i)
        tr_mean_ar.append(mean)
        tr_std_ar.append(std)
    tr_mean = np.mean(tr_mean_ar)
    tr_std = np.std(tr_std_ar)
    return tr_mean,tr_std