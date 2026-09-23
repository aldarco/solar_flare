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

def __validate_match(true_intervals, pred_intervals, dfamp, amp_threshold):
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


def levelok(x, thr):
    return x.mean() > thr

def timeok(ti, tf, lower_tspan=4, upper_tspan=60):
    '''
    ti : time begin of event
    tf : time end of event
    lower_tspan : minimum event duration
    upper_tspan : maximum event duration
    '''
    # if lower_tspan <=(tf-ti)/np.timedelta64(1,'m') <= upper_tspan:
    #     return True
    # # elif  ti.hour >= 13 and tf.hour <= 21 and (ti.day==tf.day)
    # else: 
    #     return False
    return lower_tspan <=(tf-ti)/np.timedelta64(1,'m') <= upper_tspan



    
def validate_match(true_intervals, pred_intervals, dfamp, amp_threshold):
    '''
    Validates the detection/predicton state sequence array 
    
    '''
    # Asumimos que true_intervals y pred_intervals están ordenados cronológicamente
    match_data = []
    print("True events:\t", len(true_intervals))
    print("Predicted (raw) events:\t", len(pred_intervals))
    for ref_r in true_intervals:
        # print(ref_r, len(pred_intervals))
        match = False
        
        # iterar evaluando sonbre las predicciones
        k = 0
        while k < len(pred_intervals):
            pred_r = pred_intervals[k]
            
            if not timeok(pred_r[0], pred_r[1]) or not levelok(dfamp.loc[pred_r[0]-dt.timedelta(minutes=5):pred_r[1]-dt.timedelta(minutes=5)], amp_threshold):
                _dropped = pred_intervals.pop(k)
                # print("removed: " , _dropped)
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
         #if not (pred_r[0].hour < 12 or pred_r[1].hour > 22 or dfamp.loc[pred_r[0]:pred_r[1]].mean() < 4):
        if (pred_r[0].hour > 12 and pred_r[1].hour < 22 and dfamp.loc[pred_r[0]:pred_r[1]].mean() > amp_threshold):
            match_data.append([np.nan, np.nan, *pred_r, 0,1])
        else:
            # print(f"Discarded FP : {pred_r}:: mean={dfamp.loc[pred_r[0]-dt.timedelta(minutes=5):pred_r[1]-dt.timedelta(minutes=5)].mean() > amp_threshold}")
            # TODO ??
            pass
    matchdata = pd.DataFrame(match_data, columns=["t_true_1", "t_true_2", "t_pred_1", "t_pred_2", "true", "pred"])
    return matchdata


def getmetrics(df):
    # df: well defined input
    # returns: [precision, accuracy, recall]
    y_pred = np.isnan(df["t2"])==False
    y_real = np.isnan(df["t1"])==False
    return precision_score(y_real, y_pred), accuracy_score(y_real, y_pred), recall_score(y_real, y_pred)

# util functio to initialize params
def calculate_mean_and_cov(features, labels):
    unique_labels = np.unique(labels)
    n_states = len(unique_labels)
    n_features = features.shape[1]
    
    means = np.zeros((n_states, n_features))
    covars = np.zeros((n_states, n_features, n_features))
    
    for i, state in enumerate(unique_labels):
        state_data = features[labels == state]
        means[i] = np.mean(state_data, axis=0)
        cov = np.cov(state_data.T)
        min_covar = 1e-4  # Valor pequeño para darle "volumen" a la gaussiana
        cov = cov + np.eye(n_features) * min_covar
        
        cov = (cov + cov.T) / 2
        covars[i] = cov
        
    return means, covars


def calculate_startprob(labels):

    n_states = len(np.unique(labels))
    startprob = np.zeros(n_states)

    initial_state = int(labels[0])
    startprob[initial_state] = 1.0

    return startprob
    
def calculate_transmat(labels):

    n_states = len(np.unique(labels))
    transmat = np.zeros((n_states, n_states))

    # 1. Contar todas las transiciones en la secuencia
    for i in range(len(labels) - 1):
        from_state = int(labels[i])
        to_state = int(labels[i+1])
        transmat[from_state, to_state] += 1

    row_sums = transmat.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        transmat = np.where(row_sums > 0, transmat / row_sums, 0)
    return transmat


def gmm_init_from_labels(X, y, n_states=2, n_mix=3, min_covar=1e-4):
    D = X.shape[1]
    means  = np.zeros((n_states, n_mix, D))
    covars = np.zeros((n_states, n_mix, D))
    weights = np.zeros((n_states, n_mix))

    for s in range(n_states):
        Xs = X[y == s]
        mu = Xs.mean(0)
        var = Xs.var(0) + min_covar
        # Duplicate the single labeled mean across mixtures with small jitter
        for m in range(n_mix):
            means[s, m]  = mu + 1e-3 * np.random.randn(D)
            covars[s, m] = var
        weights[s] = np.ones(n_mix) / n_mix    # uniform mixture weights

    return means, covars, weights

def latency_metrics(p_flare, t, true_intervals, on=0.7, off=0.3):
    """
    Métricas de detección temprana.

    p_flare       : array (T,)  P(onset | pasado) por sample
    t             : DatetimeIndex de largo T
    true_intervals: lista de (t_onset, t_end) verdaderos
    on, off       : umbrales de histéresis

    Usage:
    metrics = latency_metrics(p_flare, t_test, rise_true_intervals, on=0.7, off=0.3)
    for k, v in metrics.items():
        print(f"{k:18s}  {v}")
    """
    # Estados por histéresis
    state = np.zeros(len(p_flare), dtype=int)
    cur = 0
    for i, p in enumerate(p_flare):
        if cur == 0 and p > on:
            cur = 1
        elif cur == 1 and p < off:
            cur = 0
        state[i] = cur

    # Transiciones 0 -> 1 = alarmas
    alarm_idx = np.where(np.diff(state) == 1)[0] + 1
    alarm_times = t[alarm_idx]

    # TTD por evento verdadero
    ttd = []
    for (t0, t1) in true_intervals:
        after = alarm_times[alarm_times >= t0]
        after = after[after <= t1 + pd.Timedelta(minutes=30)]
        if len(after) == 0:
            ttd.append(np.nan)
        else:
            ttd.append((after[0] - t0).total_seconds() / 60.0)

    ttd = np.array(ttd)
    valid = ~np.isnan(ttd)

    # Alarms/day (usa duración real del test)
    dur_days = (t[-1] - t[0]).total_seconds() / 86400

    # FPs / día (alarmas que no caen dentro de ningún evento real)
    in_event = np.zeros(len(alarm_times), dtype=bool)
    for i, ta in enumerate(alarm_times):
        for (t0, t1) in true_intervals:
            if t0 - pd.Timedelta(minutes=5) <= ta <= t1 + pd.Timedelta(minutes=30):
                in_event[i] = True
                break
    fp_count = (~in_event).sum()

    return {
        "n_events":       len(true_intervals),
        "n_detected":     int(valid.sum()),
        "n_alarms":       len(alarm_times),
        "ttd_min":        float(np.nanmin(ttd))  if valid.any() else np.nan,
        "ttd_median":     float(np.nanmedian(ttd)) if valid.any() else np.nan,
        "ttd_mean":       float(np.nanmean(ttd)) if valid.any() else np.nan,
        "early_rate":     float(np.mean(ttd[valid] < 0)) if valid.any() else np.nan,
        "alarms_per_day": len(alarm_times) / dur_days,
        "fps_per_day":    fp_count / dur_days,
    }



def classify_flare(peak_flux):
    if peak_flux >= 1e-4:  return "X"
    if peak_flux >= 1e-5:  return "M"
    if peak_flux >= 1e-6:  return "C"
    if peak_flux >= 1e-7:  return "B"
    return "A"

def peak_flux_in(df, t0, t1, col="GOES18", pad_min=5):
    """Pico de flujo en [t0-pad, t1+pad], robusto a desalineación de bordes."""
    t0p = t0 - pd.Timedelta(minutes=pad_min)
    t1p = t1 + pd.Timedelta(minutes=pad_min)
    seg = df.loc[t0p:t1p, col].dropna()
    return seg.max() if len(seg) > 0 else np.nan

def event_metrics_by_threshold(matchdf, true_intervals, df,
                               min_class="M", goes_col="GOES18"):
    """
    Recalcula P/R/F1 a nivel evento filtrando por intensidad mínima.

    min_class : uno de {"C", "M", "X"}
    """
    allowed = {"C": ["C","M","X"], "M": ["M","X"], "X": ["X"]}[min_class]

    # 1. Etiquetar cada evento verdadero
    true_class = {}
    for (t0, t1) in true_intervals:
        true_class[(t0, t1)] = classify_flare(peak_flux_in(df, t0, t1, goes_col))

    TP = FP = FN = 0
    for _, row in matchdf.iterrows():
        has_true = not pd.isna(row["t_true_1"])
        has_pred = not pd.isna(row["t_pred_1"])

        if has_true and has_pred:
            key = (row["t_true_1"], row["t_true_2"])
            if true_class.get(key, "A") in allowed:
                TP += 1
            else:
                FP += 1   # match a evento sub-umbral -> cuenta como falsa alarma
        elif has_true and not has_pred:
            key = (row["t_true_1"], row["t_true_2"])
            if true_class.get(key, "A") in allowed:
                FN += 1
            # evento sub-umbral perdido -> se ignora
        elif has_pred and not has_true:
            FP += 1

    P  = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    R  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = 2*P*R / (P + R)  if (P + R) > 0     else 0.0

    return {"min_class": min_class, "TP": TP, "FP": FP, "FN": FN,
            "P": P, "R": R, "F1": F1}

