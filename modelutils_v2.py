"""
Reemplazos para find_state_intervals / validate_match / event_metrics_by_threshold
de modelutils.py.

Cambios principales:
  * find_intervals_v2: sin binary_dilation (no alarga +2 min cada intervalo), merge
    explicito de huecos cortos, respeta saltos de tiempo (noches) y no pierde
    intervalos que empiezan en la muestra 0 o terminan en la ultima.
  * validate_match_1to1: cada evento verdadero y cada prediccion aparecen UNA vez
    (asignacion hungara por solape). No muta las listas de entrada. El mismo filtro
    (duracion/nivel) se aplica a todas las predicciones. No descarta predicciones
    largas por defecto (max_dur=None).
  * event_metrics_v2: precision/recall de DETECCION (cualquier clase), recall
    estratificado por clase de flare y falsas alarmas por dia.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

MIN = pd.Timedelta(minutes=1)


def find_intervals_v2(times, states, target=1, merge_gap=2, max_time_gap="2min"):
    """Intervalos [t_ini, t_fin) donde states==target.

    merge_gap : huecos de hasta `merge_gap` muestras se fusionan (sin alargar bordes).
    max_time_gap : si dos muestras consecutivas distan mas que esto (p.ej. noche
                   removida), el intervalo se corta ahi.
    """
    times = pd.DatetimeIndex(times)
    mask = np.asarray(states) == target
    n = len(mask)
    if n == 0 or not mask.any():
        return []
    step = pd.Series(times).diff().median()
    brk = np.r_[False, np.diff(times.values) > pd.Timedelta(max_time_gap).to_timedelta64()]

    change = np.r_[True, (mask[1:] != mask[:-1]) | brk[1:]]
    s_idx = np.where(change)[0]
    e_idx = np.r_[s_idx[1:], n] - 1
    keep = mask[s_idx]
    runs = [[s, e] for s, e in zip(s_idx[keep], e_idx[keep])]

    merged = [runs[0]]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        if (times[s] - times[pe]) <= (merge_gap + 1) * step:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [[times[s], times[e] + step] for s, e in merged]


def validate_match_1to1(true_intervals, pred_intervals, dfamp, amp_threshold=-3,
                        min_dur=4, max_dur=None, tol_min=0, level_shift_min=5):
    """Matching uno a uno. Devuelve el mismo esquema de columnas que validate_match
    mas `fragment` y `pred_covers_n_true`.

    fragment = prediccion sin pareja que SI solapa un evento verdadero ya detectado
               (fragmentacion/duplicado, no una falsa alarma 'pura').
    true_covered = el evento verdadero solapa ALGUNA prediccion valida (aunque esa
               prediccion haya sido asignada a otro evento del mismo cluster).
    pred_covers_n_true = cuantos eventos verdaderos solapa esa prediccion
               (>1 => una sola prediccion cubre un cluster de flares).
    """
    tol = pd.Timedelta(minutes=tol_min)
    shift = pd.Timedelta(minutes=level_shift_min)

    preds = []
    for p0, p1 in pred_intervals:
        dur = (p1 - p0) / MIN
        if dur < min_dur or (max_dur is not None and dur > max_dur):
            continue
        seg = dfamp.loc[p0 - shift:p1 - shift]
        if not (seg.mean() > amp_threshold):
            continue
        preds.append((p0, p1))

    trues = [tuple(t) for t in true_intervals]
    W = np.zeros((len(trues), len(preds)))
    for i, (t0, t1) in enumerate(trues):
        for j, (p0, p1) in enumerate(preds):
            ov = (min(t1 + tol, p1) - max(t0 - tol, p0)).total_seconds()
            if ov > 0:
                W[i, j] = 1e6 + ov          # prioriza nº de parejas, luego solape

    pairs = {}
    if W.size:
        r, c = linear_sum_assignment(W, maximize=True)
        pairs = {i: j for i, j in zip(r, c) if W[i, j] > 0}
    used_pred = set(pairs.values())
    n_cov = (W > 0).sum(axis=0) if W.size else np.zeros(len(preds), int)

    rows = []
    for i, (t0, t1) in enumerate(trues):
        if i in pairs:
            j = pairs[i]
            rows.append([t0, t1, *preds[j], 1, 1, False, int(n_cov[j]), True])
        else:
            cov = bool((W[i] > 0).any()) if W.size else False
            rows.append([t0, t1, pd.NaT, pd.NaT, 1, 0, False, 0, cov])
    for j, (p0, p1) in enumerate(preds):
        if j not in used_pred:
            rows.append([pd.NaT, pd.NaT, p0, p1, 0, 1, bool(n_cov[j] > 0), int(n_cov[j]), False])

    return pd.DataFrame(rows, columns=["t_true_1", "t_true_2", "t_pred_1", "t_pred_2",
                                       "true", "pred", "fragment", "pred_covers_n_true",
                                       "true_covered"])


def _flare_class(flux):
    if pd.isna(flux):
        return "NA"          # sin dato GOES: NO lo mezcles con la clase A
    for thr, c in [(1e-4, "X"), (1e-5, "M"), (1e-6, "C"), (1e-7, "B")]:
        if flux >= thr:
            return c
    return "A"


def event_metrics_v2(matchdf, true_intervals, df, n_days, goes_col="GOES18",
                     pad_min=5, fragments_as_fp=False):
    """Detección (todas las clases) + recall por clase + FP/día."""
    pad = pd.Timedelta(minutes=pad_min)
    cls = {}
    for t0, t1 in true_intervals:
        seg = df.loc[t0 - pad:t1 + pad, goes_col].dropna()
        cls[(t0, t1)] = _flare_class(seg.max() if len(seg) else np.nan)

    tr = matchdf[matchdf["true"] == 1]
    TP = int((tr["pred"] == 1).sum())
    FN = int((tr["pred"] == 0).sum())
    fp_rows = matchdf[(matchdf["true"] == 0) & (matchdf["pred"] == 1)]
    FP_pure = int((~fp_rows["fragment"]).sum())
    FP_frag = int(fp_rows["fragment"].sum())
    FP = FP_pure + (FP_frag if fragments_as_fp else 0)

    R_lenient = float(tr["true_covered"].mean()) if len(tr) else 0.0
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0

    by_class = {}
    for c in ["A", "B", "C", "M", "X", "NA"]:
        sub = tr[[cls.get((a, b)) == c for a, b in zip(tr["t_true_1"], tr["t_true_2"])]]
        if len(sub):
            by_class[c] = (int((sub["pred"] == 1).sum()), len(sub))

    return {"TP": TP, "FN": FN, "FP": FP, "FP_pure": FP_pure, "FP_fragments": FP_frag,
            "P": P, "R": R, "R_lenient": R_lenient, "F1": F1, "FP_per_day": FP / n_days,
            "recall_by_class": {c: f"{d}/{n}" for c, (d, n) in by_class.items()},
            "n_preds_covering_multiple_true": int((matchdf["pred_covers_n_true"] > 1).sum())}