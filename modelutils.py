# Module with funtions to process and manipulate the states inthe HMM and Kalman analysis
# author : Aldo Arriola
# email  : aldo.arriolac@gmail.com / aarriola@conida.gob.pe

import numpy as np
import pandas as pd
import datetime as dt
from scipy.ndimage import binary_dilation
import os
import sys


def create_ar_features(sequences, p=2):
    X, y = [], []
    for seq in sequences:
        for t in range(p, len(seq)):
            X.append(seq[t-p:t])
            y.append(seq[t])
    return np.array(X), np.array(y)

def create_ar_features_wt(sequences, sequences_t, p=2):
    X, y , time_arr = [], [], []
    for seqt, seq in zip(sequences_t, sequences):
        for t in range(p, len(seq)):
            X.append(seq[t-p:t])
            y.append(seq[t])
            time_arr.append(seqt[t])
    return np.array(X), np.array(y), np.array(time_arr)
    
def create_ar_feat_time(sequences_t, p=2):
    X, y , time_arr = [], [], []
    for seqt in sequences_t:
        for t in range(p, len(seqt)):
            time_arr.append(seqt[t])
    return np.array(time_arr)

def find_state_ranges(states, target=1):
    mask = states==target
    dmask = np.diff(1.0*mask)
    dmask = np.pad(dmask, (1, 0), mode='constant', constant_values=0)
    ranges = []
    ind_starts = np.where(dmask==1)[0]
    ind_ends = np.where(dmask==-1)[0]
    #print(f"Found : 1s: {len(ind_starts)}   |  -1s: {len(ind_ends)}")
    nstarts = len(ind_starts)
    for k, vi in enumerate(ind_starts):
        next_vi = ind_starts[k+1] if k<nstarts-1 else len(states)
        found = np.where((vi<ind_ends) & (ind_ends<next_vi) )[0]
        if len(found):
            ranges.append([vi, ind_ends[found[0]]])
    
    return ranges

def find_state_ranges_2(states, target=1):
    # consdering diff nask gives the same +1s and -1s
    
    mask = states==target
    dmask = np.diff(1.0*mask)
    dmask = np.pad(dmask, (1, 0), mode='constant', constant_values=0)
    ranges = []
    ind_starts = np.where(dmask==1)[0]
    ind_ends = np.where(dmask==-1)[0]
    ranges = np.zeros((len(ind_starts), 2))
    ranges[:, 0] = ind_starts
    ranges[:, 1] = ind_ends
    #print(f"Found : 1s: {len(ind_starts)}   |  -1s: {len(ind_ends)}")
    
    return list(ranges.astype(int))

def hfs_from_ref(states_arr, time_arr, t_ref, delta_t=np.timedelta64(5, "m")):
    # states_arr : array of predictes states
    # time_arr   : array of time (correlated to states_arr)
    # ref_time   : datetime reference, t_ref
    # time_span  : deltatime (minutes), dt
    # so we grab the states within the [t_ref - dt : t_ref + dt] , can be monre than one
    # returns : list
    
    ws = states_arr[(t_ref - delta_t<time_arr) & (time_arr<t_ref + delta_t)]
    return np.unique(ws)

def homognize_single_state(states, hfs):
    # if multiple hfs are guiven/found for the ref time
    # equals all other to the first one
    for s in hfs:
        states[states==s] = hfs[0]
    return states

def find_state_intervals(times, states, target=1, delta_t=0, merge=False, diurnal=True, time_span=[12,23], dilate=3):
    # allows to have multiple targets as  one state.
    # as observed in the figure
    # times: time of timeserie
    # states: state array
    # target: target state
    # delta_t : displacement/offset of event times 
    # time_span: diurnal hours [hour_start, hour_end]
    # dilate 
    delta_t = np.timedelta64(delta_t, "m")
    times = pd.to_datetime(times)
    #states[(times.hour <= 12) & (times.hour > 22)]= -1
    mask = states==target
    mask = binary_dilation(mask, structure=np.array(np.ones(dilate))).astype(int)
    dmask = np.diff(1.0*mask)
    dmask = np.pad(dmask, (1, 0), mode='constant', constant_values=0)
    ranges = []
    ind_starts = np.where(dmask==1)[0]
    ind_ends = np.where(dmask==-1)[0]
    #print(f"Found : 1s: {len(ind_starts)}   |  -1s: {len(ind_ends)}")
    nstarts = len(ind_starts)
    for k, vi in enumerate(ind_starts):
        next_vi = ind_starts[k+1] if k<nstarts-1 else len(states)
        found = np.where((vi<ind_ends) & (ind_ends<next_vi) )[0]
        if len(found):
            ranges.append([times[vi], times[ind_ends[found[0]]]])
    ranges = list(map(lambda x: [x[0]+delta_t, x[1]+delta_t], ranges ))
    return ranges

def validate_match(true_intervals, pred_intervals, dfamp, amp_threshold):
    # Asumimos que true_intervals y pred_intervals están ordenados cronológicamente
    match_data = []
    
    for ref_r in true_intervals:
        # print(ref_r, len(pred_intervals))
        match = False
        
        # iterar evaluando sonbre las predicciones
        k = 0
        while k < len(pred_intervals):
            pred_r = pred_intervals[k]
            
            if pred_r[0].hour < 12 or pred_r[1].hour > 22 or dfamp.loc[pred_r[0]:pred_r[1]].mean() < amp_threshold:
                pred_intervals.pop(k)
                continue 
            
            # check if match (overlap between intervals)
            if max([ref_r[0], pred_r[0]]) < min([ref_r[1], pred_r[1]]): 
                match = True
                # drop pred
                pred_intervals.pop(k) 
                
                match_data.append([*ref_r, *pred_r,1,1])
                continue 

            # pred is before the true ref : FP
            elif pred_r[1] < ref_r[0]: 
                # Es un evento predicho que left behind with no match
                match_data.append([np.nan, np.nan, *pred_r, 0,1])
                pred_intervals.pop(k)
                continue
            
            # pred is ahead of true ref
            elif ref_r[1] < pred_r[0]:
                break 
            
            k += 1

        # no match for true ref
        if not match:
            match_data.append([*ref_r, np.nan, np.nan, 1,0])
    # remained pred as FP
    for pred_r in pred_intervals:
         # filter by time and amp
         if not (pred_r[0].hour < 12 or pred_r[1].hour > 22 or dfamp.loc[pred_r[0]:pred_r[1]].mean() < 4):
            match_data.append([np.nan, np.nan, *pred_r, 0,1])

    matchdata = pd.DataFrame(match_data, columns=["t_true_1", "t_true_2", "t_pred_1", "t_pred_2", "true", "pred"])
    return matchdata


def getmetrics(df):
    # df: well defined input
    # returns: [precision, accuracy, recall]
    y_pred = np.isnan(df["t2"])==False
    y_real = np.isnan(df["t1"])==False
    return precision_score(y_real, y_pred), accuracy_score(y_real, y_pred), recall_score(y_real, y_pred)

