#!/usr/bin/env python3
"""Drive-test analysis (G-NetTrack Pro export) - run1.

Reads ONLY the two data files in /home/claude/ai_stage/input/ and writes
results.json, supplementary.json and narrative.md into /home/claude/ai_stage/run1/.
Every number reported in those files is produced by this script.
"""
import json
import math
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

IN = '/home/claude/ai_stage/input/'
OUT = '/home/claude/ai_stage/run1/'
LOG = IN + 'orange_2026.10.02_12.16.18.txt'
EVT = IN + 'orange_2026.10.02_12.16.18_events.txt'

MISSING = {'', '-'}            # schema note: empty field or '-'
LOSS_CODE = -201               # schema note: Level = -201 -> loss of coverage
# Plausibility windows used only to detect unit / coding errors (3GPP reporting ranges, assumption)
RANGE = {
    'NR_SS_RSRP': (-156, -31), 'NR_SS_RSRQ': (-43, 20), 'NR_SS_SINR': (-23, 40),
    'LTE_RSRP': (-140, -44), 'LTE_RSRQ': (-34, 3), 'LTE_SINR': (-23, 40),
}
PCT_METHOD = 'linear'          # numpy default (Hyndman-Fan type 7)
GAP_CAP_S = 10                 # gap rule for time shares
JUMP_KMH = 80.0                # lower-bound implied speed above which a displacement is a 'jump'
JUMP_SPD_MAX = 10.0            # ... provided the logged Speed at both records is at most this (km/h)
RSSI_DEV_DB = 15.0             # |(LTERSSI - RSRP) - median| above this = inconsistent triplet
MERGE_S = 60.0                 # poor NR segments closer than this are merged into one window
ACC_MAX_M = 50.0               # GPS accuracy limit for spatial outputs (assumption)
R = 4                          # rounding of reported statistics


def rnd(x, n=R):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), n)


# ----------------------------------------------------------------------------- load
with open(LOG, encoding='utf-8') as f:
    raw_lines = [l for l in f.read().split('\n') if l != '']
rows = [l.split('\t') for l in raw_lines]
hdr, data = rows[0], rows[1:]
ncol = len(hdr)
core_n = hdr.index('NTech1')                       # columns before the neighbour blocks
rowlen = Counter(len(r) for r in data)
df = pd.DataFrame([r[:ncol] + [''] * (ncol - len(r)) for r in data], columns=hdr)
N = len(df)

ev = pd.read_csv(EVT, sep='\t', dtype=str, keep_default_na=False)

sup = OrderedDict()   # supplementary results
dq = []               # data quality findings (short sentences)

sup['parsing'] = {
    'header_columns': ncol, 'header_trailing_empty_column': hdr[-1] == '',
    'core_columns_before_neighbour_blocks': core_n,
    'row_field_count_distribution': {str(k): v for k, v in sorted(rowlen.items())},
    'rows_longer_than_header': int(sum(v for k, v in rowlen.items() if k > ncol)),
    'rows_shorter_than_core': int(sum(v for k, v in rowlen.items() if k < core_n)),
    'exact_duplicate_rows': int(df.duplicated().sum()),
}

# ----------------------------------------------------------------------------- timestamps
df['t'] = pd.to_datetime(df['Timestamp'], format='%Y.%m.%d_%H.%M.%S', errors='coerce')
bad_ts = int(df['t'].isna().sum())
df['sec'] = (df['t'] - df['t'].iloc[0]).dt.total_seconds()
gap = df['sec'].diff()
duration = float(df['sec'].iloc[-1] - df['sec'].iloc[0])
sup['timestamps'] = {
    'first': df['Timestamp'].iloc[0], 'last': df['Timestamp'].iloc[-1], 'duration_s': duration,
    'unparseable': bad_ts, 'backward_steps': int((gap < 0).sum()),
    'unique_timestamps': int(df['Timestamp'].nunique()),
    'records_sharing_second_with_previous': int((gap == 0).sum()),
    'gap_s': {'median': rnd(gap.median()), 'p95': rnd(np.percentile(gap.dropna(), 95)),
              'max': rnd(gap.max()), 'n_gt_cap': int((gap > GAP_CAP_S).sum()),
              'n_gt_30s': int((gap > 30).sum())},
    'mean_record_rate_per_s': rnd(N / duration),
}


def num(col):
    """numeric series with the schema's missing markers mapped to NaN"""
    s = df[col].where(~df[col].isin(MISSING), np.nan)
    return pd.to_numeric(s, errors='coerce')


is5 = df['NetworkTech'] == '5G'
is4 = df['NetworkTech'] == '4G'
rat_counts = {k: int(v) for k, v in df['NetworkTech'].value_counts().items()}
op_counts = {str(k): int(v) for k, v in df['Operator'].value_counts().items()}

lvl, qual, snr = num('Level'), num('Qual'), num('SNR')
s_rsrp, s_rsrq, s_snr = num('SecondCell_RSRP'), num('SecondCell_RSRQ'), num('SecondCell_SNR')
rssi = num('LTERSSI')

# ----------------------------------------------------------------------------- KPI construction
# NR rows: records labelled 5G. LTE rows: Level/Qual/SNR of records labelled 4G pooled with
# SecondCell_* (LTE anchor) of records labelled 5G (schema note).
kpi_src = OrderedDict()
kpi_src['NR_SS_RSRP'] = (lvl[is5], df.loc[is5, 'Level'])
kpi_src['NR_SS_RSRQ'] = (qual[is5], df.loc[is5, 'Qual'])
kpi_src['NR_SS_SINR'] = (snr[is5], df.loc[is5, 'SNR'])
kpi_src['LTE_RSRP'] = (pd.concat([lvl[is4], s_rsrp[is5]]), pd.concat([df.loc[is4, 'Level'], df.loc[is5, 'SecondCell_RSRP']]))
kpi_src['LTE_RSRQ'] = (pd.concat([qual[is4], s_rsrq[is5]]), pd.concat([df.loc[is4, 'Qual'], df.loc[is5, 'SecondCell_RSRQ']]))
kpi_src['LTE_SINR'] = (pd.concat([snr[is4], s_snr[is5]]), pd.concat([df.loc[is4, 'SNR'], df.loc[is5, 'SecondCell_SNR']]))

POOR = {'RSRP': -110.0, 'SINR': 0.0}
GOOD = {'RSRP': -85.0, 'SINR': 10.0}


def kind(name):
    return 'RSRP' if name.endswith('RSRP') else ('SINR' if name.endswith('SINR') else 'RSRQ')


def stats(name, x, raw):
    lo, hi = RANGE[name]
    raw_n = int(len(raw))
    miss = int(raw.isin(MISSING).sum())
    nonnum = int((x.isna() & ~raw.isin(MISSING)).sum())
    loss = int((x == LOSS_CODE).sum())
    oor = int(((x < lo) | (x > hi)).sum()) - loss
    v = x[(x >= lo) & (x <= hi)].astype(float)
    k = kind(name)
    out = OrderedDict(raw_n=raw_n, missing_marker_n=miss, non_numeric_n=nonnum,
                      loss_code_n=loss, out_of_range_n=oor, valid_n=int(len(v)))
    if len(v) == 0:
        out.update(mean=None, median=None, sd=None, p5=None, p95=None, min=None, max=None,
                   poor_pct=None, acceptable_pct=None, good_pct=None)
        return out
    out.update(mean=rnd(v.mean()), median=rnd(v.median()),
               sd=rnd(v.std(ddof=1)) if len(v) > 1 else None,
               p5=rnd(np.percentile(v, 5, method=PCT_METHOD)),
               p95=rnd(np.percentile(v, 95, method=PCT_METHOD)),
               min=rnd(v.min()), max=rnd(v.max()))
    if k in POOR:
        poor = int((v < POOR[k]).sum())
        good = int((v > GOOD[k]).sum())
        acc = int(len(v)) - poor - good
        out.update(poor_n=poor, acceptable_n=acc, good_n=good,
                   poor_pct=rnd(100.0 * poor / len(v)),
                   acceptable_pct=rnd(100.0 * acc / len(v)),
                   good_pct=rnd(100.0 * good / len(v)))
    else:
        out.update(poor_pct=None, acceptable_pct=None, good_pct=None)   # RSRQ: descriptive only
    return out


table2 = OrderedDict((k, stats(k, x, raw)) for k, (x, raw) in kpi_src.items())
sup['table_II'] = table2
sup['table_II_definitions'] = {
    'sd': 'sample SD (ddof=1)', 'percentiles': 'numpy.percentile method=linear (Hyndman-Fan type 7)',
    'poor': 'RSRP < -110 dBm ; SINR < 0 dB (strict)', 'good': 'RSRP > -85 dBm ; SINR > 10 dB (strict)',
    'acceptable': 'boundaries -110, -85, 0 and 10 inclusive', 'RSRQ': 'descriptive only, no category',
    'validity': 'numeric, not a missing marker (empty or -), not -201, inside 3GPP reporting range',
}

# LTE pool composition
sup['lte_pool_composition'] = {
    'from_4G_records_Level': int(lvl[is4].notna().sum()),
    'from_5G_records_SecondCell_RSRP': int(s_rsrp[is5].notna().sum()),
    '5G_records_without_anchor_values': int(s_rsrp[is5].isna().sum()),
    '4G_records_with_SecondCell_values_present_not_used': int(s_rsrp[is4].notna().sum()),
}
for nm, x4, x5 in (('LTE_RSRP', lvl[is4], s_rsrp[is5]), ('LTE_SINR', snr[is4], s_snr[is5])):
    k = kind(nm)
    sup['lte_pool_composition'][nm + '_by_source'] = {
        '4G_records': {'n': int(x4.notna().sum()), 'mean': rnd(x4.mean()), 'median': rnd(x4.median()), 'min': rnd(x4.min()),
                       'poor_pct': rnd(100.0 * (x4 < POOR[k]).sum() / x4.notna().sum())},
        '5G_anchor': {'n': int(x5.notna().sum()), 'mean': rnd(x5.mean()), 'median': rnd(x5.median()), 'min': rnd(x5.min()),
                      'poor_pct': rnd(100.0 * (x5 < POOR[k]).sum() / x5.notna().sum())},
    }

# ----------------------------------------------------------------------------- RAT-label transitions and mixed-field records
tech = df['NetworkTech']
prev_tech, next_tech = tech.shift(1), tech.shift(-1)
flip = (tech != prev_tech) & prev_tech.notna()
same_cell_prev = (df['Node'] == df['Node'].shift(1)) & (df['CellID'] == df['CellID'].shift(1))
flips_45 = int((flip & (tech == '5G')).sum())
flips_54 = int((flip & (tech == '4G')).sum())
runs = (tech != prev_tech).cumsum()
run_len = df.groupby(runs).size()
sup['rat_label_flips'] = {
    'total': int(flip.sum()), '4G_to_5G': flips_45, '5G_to_4G': flips_54,
    'flips_on_same_node_and_cell': int((flip & same_cell_prev).sum()),
    'label_runs': int(run_len.size), 'median_run_records': rnd(run_len.median()),
    'flips_per_minute': rnd(flip.sum() / (duration / 60.0)),
}

# (a) 5G-labelled records with NR SINR missing: are Level/Qual identical to LTE values?
a_mask = is5 & snr.isna()
a_eq_anchor = a_mask & (lvl == s_rsrp) & (qual == s_rsrq)
a_eq_next4g = a_mask & s_rsrp.isna() & (next_tech == '4G') & (lvl == lvl.shift(-1)) & (qual == qual.shift(-1))
a_followed_by_4g = a_mask & (next_tech == '4G')
# (b) 4G-labelled records directly before a 4G->5G flip whose Level and Qual equal the next (5G) record's NR values
b_mask = is4 & (next_tech == '5G') & (lvl == lvl.shift(-1)) & (qual == qual.shift(-1))
b_base = is4 & (next_tech == '5G')
suspect5 = a_mask.copy()
suspect4 = b_mask.copy()
sup['mixed_field_records_at_label_change'] = {
    '5G_records_with_SNR_missing': int(a_mask.sum()),
    'of_which_Level_and_Qual_equal_own_LTE_anchor': int(a_eq_anchor.sum()),
    'of_which_no_anchor_but_equal_next_4G_record': int(a_eq_next4g.sum()),
    'of_which_immediately_followed_by_4G_record': int(a_followed_by_4g.sum()),
    'timestamps_5G': df.loc[a_mask, 'Timestamp'].tolist(),
    '4G_records_directly_before_a_5G_record': int(b_base.sum()),
    'of_which_Level_and_Qual_equal_next_5G_record_NR_values': int(b_mask.sum()),
    'of_those_where_LTE_Level_differs_from_next_5G_anchor_RSRP': int((b_mask & (lvl != s_rsrp.shift(-1))).sum()),
    'timestamps_4G': df.loc[b_mask, 'Timestamp'].tolist(),
}

# Sensitivity: Table II without the mixed-field suspects
sens_src = OrderedDict()
ok5 = is5 & ~suspect5
ok4 = is4 & ~suspect4
sens_src['NR_SS_RSRP'] = (lvl[ok5], df.loc[ok5, 'Level'])
sens_src['NR_SS_RSRQ'] = (qual[ok5], df.loc[ok5, 'Qual'])
sens_src['NR_SS_SINR'] = (snr[ok5], df.loc[ok5, 'SNR'])
sens_src['LTE_RSRP'] = (pd.concat([lvl[ok4], s_rsrp[is5]]), pd.concat([df.loc[ok4, 'Level'], df.loc[is5, 'SecondCell_RSRP']]))
sens_src['LTE_RSRQ'] = (pd.concat([qual[ok4], s_rsrq[is5]]), pd.concat([df.loc[ok4, 'Qual'], df.loc[is5, 'SecondCell_RSRQ']]))
sens_src['LTE_SINR'] = (pd.concat([snr[is4], s_snr[is5]]), pd.concat([df.loc[is4, 'SNR'], df.loc[is5, 'SecondCell_SNR']]))
sup['table_II_sensitivity_excluding_mixed_field_records'] = OrderedDict(
    (k, stats(k, x, raw)) for k, (x, raw) in sens_src.items())

# ----------------------------------------------------------------------------- sample-and-hold (repeated readings)
def held(mask, cols):
    """share of records (within mask) whose value tuple equals the previous record's (same label)"""
    sub = df.loc[mask, cols]
    same = (sub == sub.shift(1)).all(axis=1)
    idx = pd.Series(np.where(mask)[0])
    consec = (idx.diff() == 1).values
    return int((same.values & consec).sum()), int(len(sub))


h5, n5 = held(is5, ['Level', 'Qual', 'SNR'])
h4, n4 = held(is4, ['Level', 'Qual', 'SNR'])
ha, na = held(is5, ['SecondCell_RSRP', 'SecondCell_RSRQ', 'SecondCell_SNR'])
sup['repeated_readings'] = {
    '5G_records_NR_triplet_identical_to_previous_record': {'n': h5, 'of': n5, 'pct': rnd(100.0 * h5 / n5)},
    '4G_records_LTE_triplet_identical_to_previous_record': {'n': h4, 'of': n4, 'pct': rnd(100.0 * h4 / n4)},
    '5G_records_anchor_triplet_identical_to_previous_record': {'n': ha, 'of': na, 'pct': rnd(100.0 * ha / na)},
}

# ----------------------------------------------------------------------------- LTE SNR floor / physical consistency
snr4 = snr[is4]
sup['lte_snr_floor_check'] = {
    '4G_records_SNR_min': rnd(snr4.min()), '4G_records_SNR_eq_0': int((snr4 == 0).sum()),
    '4G_records_SNR_lt_0': int((snr4 < 0).sum()), '4G_records_n': int(snr4.notna().sum()),
    '5G_anchor_SNR_min': rnd(s_snr[is5].min()), '5G_anchor_SNR_eq_0': int((s_snr[is5] == 0).sum()),
    '5G_anchor_SNR_lt_0': int((s_snr[is5] < 0).sum()), '5G_anchor_n': int(s_snr[is5].notna().sum()),
    'NR_SINR_eq_0': int((snr[is5] == 0).sum()), 'NR_SINR_lt_0': int((snr[is5] < 0).sum()),
}
d_rssi = (rssi - lvl)[is4]
d_med = float(d_rssi.median())
rssi_out = is4 & rssi.notna() & (((rssi - lvl) - d_med).abs() > RSSI_DEV_DB)
sup['lte_rssi_consistency'] = {
    '4G_records_with_RSSI': int((is4 & rssi.notna()).sum()),
    'RSSI_minus_RSRP_dB': {'median': rnd(d_med), 'p5': rnd(np.percentile(d_rssi.dropna(), 5)), 'p95': rnd(np.percentile(d_rssi.dropna(), 95)),
                           'min': rnd(d_rssi.min()), 'max': rnd(d_rssi.max())},
    'RSSI_not_above_RSRP_n': int((d_rssi <= 0).sum()),
    'deviation_limit_dB': RSSI_DEV_DB,
    'outlier_n': int(rssi_out.sum()),
    'outlier_rows': df.loc[rssi_out, ['Timestamp', 'Node', 'CellID', 'Level', 'Qual', 'SNR', 'LTERSSI']].to_dict('records'),
}

# ----------------------------------------------------------------------------- band / mode shares (sample shares)
def share(series, order=None):
    vc = series.value_counts(dropna=False)
    tot = int(vc.sum())
    keys = order or list(vc.index)
    return OrderedDict((str(k), {'n': int(vc.get(k, 0)), 'pct': rnd(100.0 * vc.get(k, 0) / tot)}) for k in keys)


lte_band = df['BAND'].where(~df['BAND'].isin(MISSING), 'unknown')
bw_has100 = df['BANDWIDTH'].apply(lambda s: '100' in s.split('/'))
five_g_ident_eq_anchor = int(((df['Node'] == df['SecondCell_NODE']) & (df['CellID'] == df['SecondCell_CELLID']) &
                              (df['PSC'] == df['SecondCell_PSC']) & (df['ARFCN'] == df['SecondCell_ARFCN']) &
                              (df['BAND'] == df['SecondCell_BAND']) & is5).sum())
sup['sample_shares'] = {
    'rat_label': share(df['NetworkTech'], ['4G', '5G']),
    'rat_label_unknown': {'n': int((~df['NetworkTech'].isin(['2G', '3G', '4G', '5G'])).sum())},
    'nr_mode': {'label_5G_shown_as_NSA_per_schema_note_n': int(is5.sum()),
                'NRSTATE_populated_n': int((~df['NRSTATE'].isin(MISSING)).sum()),
                'SA_or_NSA_verifiable_from_log': False},
    'lte_serving_or_anchor_band_all_records': share(lte_band),
    'lte_band_in_4G_records': share(lte_band[is4]),
    'lte_anchor_band_in_5G_records': share(lte_band[is5]),
    'arfcn_field': share(df['ARFCN']),
    'nr_band_in_5G_records': {'unknown': {'n': int(is5.sum()), 'pct': 100.0}},
    '5G_records_with_serving_identity_fields_equal_to_LTE_anchor': {'n': five_g_ident_eq_anchor, 'of': int(is5.sum())},
    'CSI_fields_populated_n': int((~df[['CSI_RSRP', 'CSI_RSRQ', 'CSI_SNR']].isin(MISSING)).any(axis=1).sum()),
    'bandwidth_list_contains_100MHz': {'5G_records': int((bw_has100 & is5).sum()), '4G_records': int((bw_has100 & is4).sum())},
    'operator_code': share(df['Operator']),
    'operator_name': share(df['Operatorname']),
    'operator_code_by_label': {t: {str(k): int(v) for k, v in df.loc[df['NetworkTech'] == t, 'Operator'].value_counts().items()} for t in ('4G', '5G')},
    'state': share(df['State']),
    'state_D_pct_by_label': {t: rnd(100.0 * (df.loc[df['NetworkTech'] == t, 'State'] == 'D').mean()) for t in ('4G', '5G')},
    'CA_true_pct_by_label': {t: rnd(100.0 * (df.loc[df['NetworkTech'] == t, 'CA'] == 'true').mean()) for t in ('4G', '5G')},
    'foreground': share(df['FOREGROUND']),
}
node_ops = df.groupby('Node')['Operator'].nunique()
sup['sample_shares']['nodes_total'] = int(node_ops.size)
sup['sample_shares']['nodes_with_more_than_one_operator_code'] = int((node_ops > 1).sum())
sup['sample_shares']['node_id_ranges_by_operator_code'] = {
    str(op): [int(g.astype(int).min()), int(g.astype(int).max()), int(g.nunique())]
    for op, g in df.groupby('Operator')['Node']}

# ----------------------------------------------------------------------------- time shares (timestamp weighting, stated gap rule)
# Rule: each record is held until the next record (zero-order hold); a hold longer than GAP_CAP_S
# is truncated to GAP_CAP_S and the remainder is counted as "unobserved". Last record: weight 0.
hold = (df['sec'].shift(-1) - df['sec']).fillna(0.0)
w = hold.clip(upper=GAP_CAP_S)
unobs = float((hold - w).sum())


def tshare(series, weights, total):
    g = weights.groupby(series).sum()
    return OrderedDict((str(k), {'s': rnd(v, 1), 'pct': rnd(100.0 * v / total)}) for k, v in g.items())


ts = OrderedDict()
ts['gap_rule'] = ('zero-order hold to next record, capped at %d s; excess = unobserved; last record weight 0' % GAP_CAP_S)
ts['total_s'] = duration
ts['observed_s'] = rnd(w.sum(), 1)
ts['unobserved_s'] = rnd(unobs, 1)
ts['unobserved_pct'] = rnd(100.0 * unobs / duration)
ts['rat_label_pct_of_total'] = tshare(df['NetworkTech'], w, duration)
ts['rat_label_pct_of_observed'] = tshare(df['NetworkTech'], w, float(w.sum()))
ts['lte_band_pct_of_observed'] = tshare(lte_band, w, float(w.sum()))
ts['operator_code_pct_of_observed'] = tshare(df['Operator'], w, float(w.sum()))
ts['uncapped_rat_label_pct_sensitivity'] = tshare(df['NetworkTech'], hold, duration)
sup['time_shares'] = ts

# record density foreground vs background
fg = df['FOREGROUND'] == 'true'
first_bg = df.index[~fg][0]
sup['record_density'] = {
    'first_background_record': df['Timestamp'].iloc[first_bg],
    'records_before_first_background': int(first_bg),
    'seconds_before_first_background': float(df['sec'].iloc[first_bg]),
    'records_from_first_background': int(N - first_bg),
    'seconds_from_first_background': float(duration - df['sec'].iloc[first_bg]),
    'pct_records_before': rnd(100.0 * first_bg / N),
    'pct_time_before': rnd(100.0 * df['sec'].iloc[first_bg] / duration),
    'hold_s_foreground_true': rnd(hold[fg].sum(), 1), 'hold_s_foreground_false': rnd(hold[~fg].sum(), 1),
    'median_gap_s_foreground_true': rnd(hold[fg].median()), 'median_gap_s_foreground_false': rnd(hold[~fg].median()),
}

# ----------------------------------------------------------------------------- GPS validation (spatial outputs only)
lon, lat, spd, acc = num('Longitude'), num('Latitude'), num('Speed'), num('Accuracy')


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


gps_bad_basic = lon.isna() | lat.isna() | (lat.abs() > 90) | (lon.abs() > 180) | ((lat == 0) & (lon == 0)) | (acc > ACC_MAX_M)
dist = hav(lat.shift(1), lon.shift(1), lat, lon)
dt = df['sec'].diff()
# timestamps have 1 s resolution, so the true elapsed time is below dt + 1 s: lower bound of the implied speed
imp_kmh = dist / (dt + 1.0) * 3.6
pos_id = ((lon != lon.shift(1)) | (lat != lat.shift(1))).cumsum()
jumps = df.index[(imp_kmh > JUMP_KMH) & (spd <= JUMP_SPD_MAX) & (spd.shift(1) <= JUMP_SPD_MAX)].tolist()
stale = pd.Series(False, index=df.index)
jump_rows = []
for j in jumps:
    run = df.index[pos_id == pos_id[j - 1]]
    # the first record of a frozen run is the genuine fix; later repeats cannot be located
    stale.loc[run[1:]] = True
    jump_rows.append({'from': df['Timestamp'][j - 1], 'to': df['Timestamp'][j], 'gap_s': float(dt[j]),
                      'distance_m': rnd(dist[j], 0), 'implied_kmh': rnd(imp_kmh[j], 0),
                      'logged_speed_kmh_before_after': [float(spd[j - 1]), float(spd[j])],
                      'frozen_run_start': df['Timestamp'][run[0]], 'frozen_run_records': int(len(run)),
                      'frozen_position': [float(lon[j - 1]), float(lat[j - 1])],
                      'foreground_false_in_run': int((~fg[run]).sum())})
runsz = df.groupby(pos_id).agg(n=('sec', 'size'), t0=('sec', 'min'), t1=('sec', 'max'))
gps_valid = ~(gps_bad_basic | stale)
track_m = float(np.nansum(dist))
sup['gps'] = {
    'records': N, 'non_numeric_or_out_of_range_or_0_0': int((lon.isna() | lat.isna() | (lat.abs() > 90) | (lon.abs() > 180) | ((lat == 0) & (lon == 0))).sum()),
    'accuracy_m_min_median_max': [rnd(acc.min()), rnd(acc.median()), rnd(acc.max())],
    'accuracy_gt_limit_n': int((acc > ACC_MAX_M).sum()), 'accuracy_limit_m': ACC_MAX_M,
    'bbox_lon': [float(lon.min()), float(lon.max())], 'bbox_lat': [float(lat.min()), float(lat.max())],
    'distinct_positions': int(pos_id.nunique()),
    'frozen_position_runs_ge_5_records': int((runsz['n'] >= 5).sum()),
    'longest_frozen_run': {'records': int(runsz['n'].max()),
                           'seconds': float((runsz['t1'] - runsz['t0'])[runsz['n'].idxmax()])},
    'jump_rule': 'distance/(gap+1 s) > %.0f km/h while logged Speed <= %.0f km/h at both records; all but the first record of the frozen run before the jump are stale' % (JUMP_KMH, JUMP_SPD_MAX),
    'implied_speed_limit_kmh': JUMP_KMH, 'max_logged_speed_kmh': float(spd.max()),
    'jumps_over_limit': jump_rows,
    'stale_position_records_excluded_from_spatial_outputs': int(stale.sum()),
    'valid_for_spatial_outputs_n': int(gps_valid.sum()),
    'speed_eq_0_records': int((spd == 0).sum()),
    'speed_eq_0_and_foreground_false': int(((spd == 0) & ~fg).sum()),
    'sum_of_fix_to_fix_distance_km': rnd(track_m / 1000.0, 3),
    'straight_line_first_to_last_km': rnd(hav(lat.iloc[0], lon.iloc[0], lat.iloc[-1], lon.iloc[-1]) / 1000.0, 3),
}

# ----------------------------------------------------------------------------- active tests / passive traffic
test_cols = ['Test_Status', 'PINGAVG', 'PINGMIN', 'PINGMAX', 'PINGSTDEV', 'PINGLOSS',
             'TESTDOWNLINK', 'TESTUPLINK', 'TESTDOWNLINKMAX', 'TESTUPLINKMAX']
active = {
    'download_n': int((~df['TESTDOWNLINK'].isin(MISSING)).sum()),
    'upload_n': int((~df['TESTUPLINK'].isin(MISSING)).sum()),
    'ping_n': int((~df['PINGAVG'].isin(MISSING)).sum()),
}
dl, ul = num('DL_bitrate'), num('UL_bitrate')
sup['active_tests'] = dict(active, populated_cells_in_any_test_column=int((~df[test_cols].isin(MISSING)).sum().sum()),
                           throughput_stats='N/A (no active-test interval)', rtt_stats='N/A (no active-test interval)')
sup['passive_bitrate_not_throughput'] = {
    'DL_kbps_gt0_n': int((dl > 0).sum()), 'UL_kbps_gt0_n': int((ul > 0).sum()),
    'DL_kbps_max': rnd(dl.max()), 'DL_max_at': df['Timestamp'][dl.idxmax()],
    'UL_kbps_max': rnd(ul.max()), 'UL_max_at': df['Timestamp'][ul.idxmax()],
    'DL_kbps_median_all': rnd(dl.median()), 'state_D_n': int((df['State'] == 'D').sum()),
}
sup['other_fields'] = {
    'CQI_missing_n': int(df['CQI'].isin(MISSING).sum()), 'CQI_eq_6_n': int((df['CQI'] == '6').sum()),
    'TA_values': {str(k): int(v) for k, v in df['TA'].value_counts().items()},
    'LTERSSI_missing_n': int(df['LTERSSI'].isin(MISSING).sum()),
    'Level_eq_minus201_n': int((lvl == LOSS_CODE).sum()),
    'always_empty_columns_of_interest': [c for c in ['Cellname', 'NRSTATE', 'CSI_RSRP', 'CSI_RSRQ', 'CSI_SNR', 'IMEI', 'IMSI', 'INFO'] + test_cols
                                         if df[c].isin(MISSING).all()],
    'constant_columns': {c: df[c].iloc[0] for c in ['Distance', 'Bearing', 'Site_BEARING', 'AVGSPEED', 'BATTERY', 'NetworkTypeNum', 'DataConnection_Info', 'DEVICE', 'VERSION_NAME']
                         if df[c].nunique() == 1},
}

# ----------------------------------------------------------------------------- mobility events
ev_counts = OrderedDict((str(k), int(v)) for k, v in ev['EventType'].value_counts().items())
log_ev = df[~df['EVENT'].isin(MISSING)]
log_mob = log_ev[log_ev['EVENT'] != 'LOG START']
cross = (len(log_mob) == len(ev)) and bool(
    (log_mob['Timestamp'].values == ev['Timestamp'].values).all()
    and (log_mob['EVENT'].values == ev['EventType'].values).all()
    and (log_mob['EVENTDETAILS'].values == ev['EventDetails'].values).all())


def parse_side(s):
    mcc, mnc, tac, node, cell, arfcn, pci = s.split('-')
    return dict(plmn=mcc + mnc, tac=tac, node=node, cell=cell, arfcn=arfcn, pci=pci)


E = []
for i, r in ev.iterrows():
    a, b = r['EventDetails'].split(':')
    A, B = parse_side(a), parse_side(b)
    E.append(dict(i=i, ts=r['Timestamp'], type=r['EventType'], src=a, dst=b, A=A, B=B,
                  t=pd.to_datetime(r['Timestamp'], format='%Y.%m.%d_%H.%M.%S')))
same_cell = [e for e in E if (e['A']['plmn'], e['A']['node'], e['A']['cell']) == (e['B']['plmn'], e['B']['node'], e['B']['cell'])]
same_node = [e for e in E if e['A']['node'] == e['B']['node']]
plmn_x = [e for e in E if e['A']['plmn'] != e['B']['plmn']]
tac_x = [e for e in E if e['A']['tac'] != e['B']['tac']]
freq_x = [e for e in E if e['A']['arfcn'] != e['B']['arfcn']]
chain_break = [E[k]['ts'] for k in range(1, len(E)) if E[k]['src'] != E[k - 1]['dst']]


def cellkey(d):
    return (d['plmn'], d['node'], d['cell'])


rev = []            # event k+1 returns to the cell that event k left
for k in range(len(E) - 1):
    if cellkey(E[k]['A']) == cellkey(E[k + 1]['B']) and cellkey(E[k]['B']) == cellkey(E[k + 1]['A']) \
            and cellkey(E[k]['A']) != cellkey(E[k]['B']):
        rev.append((E[k + 1]['t'] - E[k]['t']).total_seconds())
rev = np.array(rev)
# event type RAT codes vs the RAT label of the adjacent main-log records
mm_src = mm_dst = 0
for idx, r in log_mob.iterrows():
    code = r['EVENT'].rsplit('_', 1)[1]            # e.g. 4G5G
    s, d_ = code[:2], code[2:]
    if df['NetworkTech'][idx - 1] != s:
        mm_src += 1
    if df['NetworkTech'][idx] != d_:
        mm_dst += 1
irat_n = sum(v for k, v in ev_counts.items() if k.startswith('IRAT'))
irat_same_label = int(sum(1 for idx, r in log_mob.iterrows()
                          if r['EVENT'].startswith('IRAT') and df['NetworkTech'][idx - 1] == df['NetworkTech'][idx]))
ev_on_label_flip = int(flip[log_mob.index].sum())
ev_spd0 = int((num('Speed')[log_mob.index] == 0).sum())
ev_bg = int((log_mob['FOREGROUND'] == 'false').sum())
ev_frozen = int(stale[log_mob.index].sum())
ho = sum(v for k, v in ev_counts.items() if 'HANDOVER' in k)
resel = sum(v for k, v in ev_counts.items() if 'RESELECTION' in k)
sup['mobility'] = {
    'events_total': int(len(ev)), 'events_by_type': ev_counts,
    'handover_type_events': int(ho), 'reselection_type_events': int(resel), 'irat_type_events': int(irat_n),
    'main_log_EVENT_non_empty': int(len(log_ev)), 'main_log_LOG_START': int((log_ev['EVENT'] == 'LOG START').sum()),
    'events_file_identical_to_main_log_EVENT_column': cross,
    'events_per_minute': rnd(len(ev) / (duration / 60.0)),
    'source_target_same_node_and_cell_only_ARFCN_or_PCI_differs': {'n': len(same_cell), 'events': [[e['ts'], e['type'], e['src'] + ':' + e['dst']] for e in same_cell]},
    'intra_node_events': len(same_node), 'inter_node_events': len(E) - len(same_node),
    'operator_code_changes_across_event': len(plmn_x), 'tac_changes': len(tac_x), 'arfcn_changes': len(freq_x),
    'event_chain_breaks_source_not_equal_previous_target': chain_break,
    'immediate_reversals_A_B_then_B_A': {'n': int(len(rev)), 'within_10s': int((rev <= 10).sum()), 'within_30s': int((rev <= 30).sum()),
                                         'median_return_s': rnd(np.median(rev)) if len(rev) else None},
    'event_type_source_RAT_differs_from_label_of_previous_record': int(mm_src),
    'event_type_target_RAT_differs_from_label_of_event_record': int(mm_dst),
    'rat_label_flips_in_main_log': int(flip.sum()),
    'irat_type_events_with_same_label_before_and_at_event': irat_same_label,
    'events_coinciding_with_a_label_flip': ev_on_label_flip,
    'events_logged_with_Speed_0': ev_spd0, 'events_logged_with_FOREGROUND_false': ev_bg,
    'events_logged_on_stale_position_records': ev_frozen,
    'success_rate': 'N/A (not observable from the handset)',
}

# ----------------------------------------------------------------------------- 5G -> 4G label changes and NR level before them
fb = []
for i in df.index[flip & (tech == '4G')]:
    j = i - 1
    while j >= 0 and tech[j] == '5G' and pd.isna(snr[j]):
        j -= 1
    if j >= 0 and tech[j] == '5G':
        fb.append(dict(ts=df['Timestamp'][i], nr_rsrp=float(lvl[j]), nr_sinr=float(snr[j]), state=df['State'][i],
                       cell_changed=bool((df['Node'][i] != df['Node'][i - 1]) or (df['CellID'][i] != df['CellID'][i - 1]))))
fbd = pd.DataFrame(fb)
sup['label_5G_to_4G_changes'] = {
    'n': flips_54, 'with_preceding_valid_NR_sample': int(len(fbd)),
    'last_NR_RSRP_not_poor_ge_-110_n': int((fbd['nr_rsrp'] >= -110).sum()),
    'last_NR_RSRP_poor_lt_-110_n': int((fbd['nr_rsrp'] < -110).sum()),
    'last_NR_SINR_poor_lt_0_n': int((fbd['nr_sinr'] < 0).sum()),
    'last_NR_RSRP_not_poor_and_SINR_not_poor_n': int(((fbd['nr_rsrp'] >= -110) & (fbd['nr_sinr'] >= 0)).sum()),
    'last_NR_RSRP_median': rnd(fbd['nr_rsrp'].median()), 'last_NR_SINR_median': rnd(fbd['nr_sinr'].median()),
    'first_4G_record_state_I_n': int((fbd['state'] == 'I').sum()),
    'with_cell_change_n': int(fbd['cell_changed'].sum()),
}

# ----------------------------------------------------------------------------- poor segments (consecutive 5G records with poor NR RSRP or SINR)
nr_poor = is5 & ((lvl < -110) | (snr < 0))
seg_id = (nr_poor != nr_poor.shift(1)).cumsum()
segs = []
for _, g in df[nr_poor].groupby(seg_id[nr_poor]):
    if len(g) >= 3:
        ii = g.index
        segs.append({'from': g['Timestamp'].iloc[0], 'to': g['Timestamp'].iloc[-1], 'records': int(len(g)),
                     'nr_rsrp_min': float(lvl[ii].min()), 'nr_sinr_min': float(snr[ii].min()),
                     'anchor_rsrp_range': [float(s_rsrp[ii].min()), float(s_rsrp[ii].max())],
                     'anchor_snr_min': float(s_snr[ii].min()),
                     'lon_range': [float(lon[ii].min()), float(lon[ii].max())],
                     'lat_range': [float(lat[ii].min()), float(lat[ii].max())],
                     'gps_valid_records': int(gps_valid[ii].sum()),
                     'cells': sorted(set((g['Operator'] + '-' + g['Node'] + '-' + g['CellID']).tolist())),
                     'state_D_records': int((g['State'] == 'D').sum())})
sup['nr_poor_segments_ge_3_consecutive_records'] = segs
# merge poor segments that are closer than MERGE_S into windows and describe each window
def to_sec(ts):
    return float((pd.to_datetime(ts, format='%Y.%m.%d_%H.%M.%S') - df['t'].iloc[0]).total_seconds())


wins = []
for sg in segs:
    if wins and to_sec(sg['from']) - to_sec(wins[-1][1]) <= MERGE_S:
        wins[-1][1] = sg['to']
    else:
        wins.append([sg['from'], sg['to']])
windows = []
for w0, w1_ in wins:
    win = (df['Timestamp'] >= w0) & (df['Timestamp'] <= w1_)
    lte_r = pd.concat([lvl[win & is4], s_rsrp[win & is5]])
    lte_s = pd.concat([snr[win & is4], s_snr[win & is5]])
    windows.append({
        'from': w0, 'to': w1_, 'records': int(win.sum()), '5G_records': int((win & is5).sum()),
        'nr_rsrp_median': rnd(lvl[win & is5].median()), 'nr_rsrp_min': rnd(lvl[win & is5].min()),
        'nr_rsrp_poor_pct': rnd(100.0 * (lvl[win & is5] < -110).mean()),
        'nr_sinr_median': rnd(snr[win & is5].median()),
        'nr_sinr_poor_pct': rnd(100.0 * (snr[win & is5] < 0).sum() / snr[win & is5].notna().sum()),
        'lte_rsrp_median_pooled': rnd(lte_r.median()), 'lte_rsrp_min_pooled': rnd(lte_r.min()),
        'lte_sinr_median_pooled': rnd(lte_s.median()), 'lte_sinr_poor_pct_pooled': rnd(100.0 * (lte_s < 0).sum() / lte_s.notna().sum()),
        'events_in_window': int(((ev['Timestamp'] >= w0) & (ev['Timestamp'] <= w1_)).sum()),
        'operator_codes': sorted(df.loc[win, 'Operator'].unique().tolist()),
        'cells': sorted(set((df.loc[win, 'Operator'] + '-' + df.loc[win, 'Node'] + '-' + df.loc[win, 'CellID']).tolist())),
        'lon_range': [float(lon[win].min()), float(lon[win].max())], 'lat_range': [float(lat[win].min()), float(lat[win].max())],
        'gps_valid_records': int(gps_valid[win].sum()), 'state_D_records': int((df.loc[win, 'State'] == 'D').sum()),
        'DL_kbps_max': rnd(dl[win].max()), 'speed_kmh_range': [float(spd[win].min()), float(spd[win].max())],
    })
sup['nr_poor_windows_segments_merged_within_60s'] = windows

# ----------------------------------------------------------------------------- inter-KPI relationships (same-record pairs)
def corr(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 3:
        return None
    return {'n': int(m.sum()), 'pearson': rnd(a[m].corr(b[m], method='pearson'), 3),
            'spearman': rnd(a[m].corr(b[m], method='spearman'), 3)}


l_rsrp = pd.concat([lvl[is4], s_rsrp[is5]]).sort_index()
l_rsrq = pd.concat([qual[is4], s_rsrq[is5]]).sort_index()
l_snr = pd.concat([snr[is4], s_snr[is5]]).sort_index()
sup['relationships'] = {
    'NR_RSRP_vs_NR_SINR': corr(lvl[is5], snr[is5]),
    'NR_RSRP_vs_NR_RSRQ': corr(lvl[is5], qual[is5]),
    'NR_RSRQ_vs_NR_SINR': corr(qual[is5], snr[is5]),
    'LTE_RSRP_vs_LTE_SINR_pooled': corr(l_rsrp, l_snr),
    'LTE_RSRP_vs_LTE_RSRQ_pooled': corr(l_rsrp, l_rsrq),
    'LTE_RSRQ_vs_LTE_SINR_pooled': corr(l_rsrq, l_snr),
    'NR_RSRP_vs_anchor_LTE_RSRP_same_record': corr(lvl[is5], s_rsrp[is5]),
    'NR_SINR_vs_anchor_LTE_SNR_same_record': corr(snr[is5], s_snr[is5]),
    'NR_RSRP_vs_passive_DL_kbps_state_D_only': corr(lvl[is5 & (df['State'] == 'D')], dl[is5 & (df['State'] == 'D')]),
    'note': 'records are sample-and-hold and unevenly spaced; coefficients are descriptive, no significance test',
}
both = is5 & s_rsrp.notna()
sup['relationships']['NR_minus_anchor_RSRP_dB'] = {'n': int(both.sum()), 'mean': rnd((lvl - s_rsrp)[both].mean()), 'median': rnd((lvl - s_rsrp)[both].median())}
sup['relationships']['NR_poor_RSRP_and_poor_SINR_overlap'] = {
    'nr_rsrp_poor_n': int((is5 & (lvl < -110)).sum()),
    'of_which_sinr_poor_n': int((is5 & (lvl < -110) & (snr < 0)).sum()),
    'of_which_sinr_le_0_n': int((is5 & (lvl < -110) & (snr <= 0)).sum()),
    'nr_rsrp_not_poor_with_sinr_poor_n': int((is5 & (lvl >= -110) & (snr < 0)).sum()),
}

# ----------------------------------------------------------------------------- findings + anomaly candidates
T = table2
mf = sup['mixed_field_records_at_label_change']
g = sup['gps']
mob = sup['mobility']
sens = sup['table_II_sensitivity_excluding_mixed_field_records']
fl = sup['lte_snr_floor_check']
rc = sup['lte_rssi_consistency']
dens = sup['record_density']
rep = sup['repeated_readings']
rel = sup['relationships']
fbk = sup['label_5G_to_4G_changes']
shr = sup['sample_shares']
n_mixed = mf['5G_records_with_SNR_missing'] + mf['of_which_Level_and_Qual_equal_next_5G_record_NR_values']


def hms(ts):
    return ts.split('_')[1].replace('.', ':')


dq = [
    "%d records from %s to %s (%.0f s); all timestamps parse and none goes backwards; %d records share a second with the previous record and gaps reach %.0f s, so sampling is not uniform." % (
        N, df['Timestamp'].iloc[0], df['Timestamp'].iloc[-1], duration, sup['timestamps']['records_sharing_second_with_previous'], gap.max()),
    "The header has %d named columns; data rows carry %d-%d fields (fewer than the header, never more); all %d core fields are aligned in every row; no exact duplicate rows." % (
        ncol - 1, min(rowlen), max(rowlen), core_n),
    "No Level = -201 loss-of-coverage code and no out-of-range value in any Table II field; Level and Qual are numeric in all %d records." % N,
    "NR SS-SINR is '-' in %d of %d records labelled 5G (valid N %d); NR SS-RSRP and SS-RSRQ are populated in all %d." % (
        T['NR_SS_SINR']['missing_marker_n'], T['NR_SS_SINR']['raw_n'], T['NR_SS_SINR']['valid_n'], T['NR_SS_RSRP']['valid_n']),
    "LTE pool = %d values from 4G-labelled records + %d anchor values from 5G-labelled records = %d; %d 5G-labelled record has empty SecondCell fields; %d 4G-labelled record has SecondCell values populated (not used)." % (
        sup['lte_pool_composition']['from_4G_records_Level'], sup['lte_pool_composition']['from_5G_records_SecondCell_RSRP'], T['LTE_RSRP']['valid_n'],
        sup['lte_pool_composition']['5G_records_without_anchor_values'], sup['lte_pool_composition']['4G_records_with_SecondCell_values_present_not_used']),
    "In all %d 5G-labelled records with missing NR SINR, Level and Qual equal LTE values (%d equal the record's own LTE anchor, %d equals the next 4G record) and each is immediately followed by a 4G record; they are kept in NR RSRP/RSRQ; excluding them moves the NR RSRP mean from %.2f to %.2f dBm." % (
        mf['5G_records_with_SNR_missing'], mf['of_which_Level_and_Qual_equal_own_LTE_anchor'], mf['of_which_no_anchor_but_equal_next_4G_record'],
        T['NR_SS_RSRP']['mean'], sens['NR_SS_RSRP']['mean']),
    "%d of the %d 4G-labelled records directly preceding a 5G record carry Level and Qual identical to that 5G record's NR values (all %d differ from the LTE anchor RSRP reported next); they are kept in the LTE pool; excluding them moves the LTE RSRP mean from %.2f to %.2f dBm." % (
        mf['of_which_Level_and_Qual_equal_next_5G_record_NR_values'], mf['4G_records_directly_before_a_5G_record'],
        mf['of_those_where_LTE_Level_differs_from_next_5G_anchor_RSRP'], T['LTE_RSRP']['mean'], sens['LTE_RSRP']['mean']),
    "Readings are sample-and-hold: the Level/Qual/SNR triplet equals the previous record's in %.1f%% of 5G-labelled and %.1f%% of 4G-labelled records, so records are not independent samples." % (
        rep['5G_records_NR_triplet_identical_to_previous_record']['pct'], rep['4G_records_LTE_triplet_identical_to_previous_record']['pct']),
    "LTE SNR in 4G-labelled records is never below 0 dB (min %.1f; %d of %d values exactly 0.0) while the LTE anchor SNR in 5G-labelled records is negative in %d of %d; all %d poor LTE SINR samples come from anchor values, so a floor at 0 dB in 4G-labelled records cannot be excluded and the LTE SINR poor share may be understated." % (
        fl['4G_records_SNR_min'], fl['4G_records_SNR_eq_0'], fl['4G_records_n'], fl['5G_anchor_SNR_lt_0'], fl['5G_anchor_n'], T['LTE_SINR']['poor_n']),
    "In 5G-labelled records Node, CellID, PSC, ARFCN and BAND equal the LTE anchor fields in %d of %d records; no NR ARFCN, NR PCI or NR band is logged; CSI_* and NRSTATE are empty in all records." % (
        five_g_ident_eq_anchor, int(is5.sum())),
    "Operatorname is 'orange' in all records, but the serving-cell operator code is 22610 in %d and 22601 in %d records; each of the %d nodes maps to exactly one code." % (
        op_counts.get('22610', 0), op_counts.get('22601', 0), shr['nodes_total']),
    "No active test was run: Test_Status, PING* and TESTDOWNLINK/TESTUPLINK are empty in all %d records, so throughput and RTT are N/A; DL_bitrate/UL_bitrate are passive traffic (max %.0f / %.0f kbps)." % (
        N, dl.max(), ul.max()),
    "GPS: all %d fixes are numeric and in range, accuracy %.0f-%.0f m; %d records lie on frozen positions that end in a jump (%d jumps: distance/(gap+1 s) > %.0f km/h with logged Speed <= %.0f km/h) and are excluded from spatial outputs only (valid for spatial use: %d)." % (
        N, acc.min(), acc.max(), g['stale_position_records_excluded_from_spatial_outputs'], len(jump_rows), JUMP_KMH, JUMP_SPD_MAX, g['valid_for_spatial_outputs_n']),
    "FOREGROUND first turns false at %s; %.1f%% of the records fall in the first %.1f%% of the time (%.0f s), so sample shares over-represent the start of the drive." % (
        dens['first_background_record'], dens['pct_records_before'], dens['pct_time_before'], dens['seconds_before_first_background']),
    "The events file (%d rows) is identical to the main-log EVENT/EVENTDETAILS columns apart from the LOG START marker; %d events have the same node and cell on both sides (only ARFCN or PCI differs)." % (
        mob['events_total'], mob['source_target_same_node_and_cell_only_ARFCN_or_PCI_differs']['n']),
    "LTERSSI is not above RSRP in %d of %d 4G-labelled records (median difference %.0f dB), so it does not behave as a wideband RSSI and is not used; %d records deviate by more than %.0f dB from that median." % (
        rc['RSSI_not_above_RSRP_n'], rc['4G_records_with_RSSI'], rc['RSSI_minus_RSRP_dB']['median'], rc['outlier_n'], RSSI_DEV_DB),
    "CQI is '-' in %d records and equals 6 in %d; TA is 0 or empty in all records; neither is used." % (
        sup['other_fields']['CQI_missing_n'], sup['other_fields']['CQI_eq_6_n']),
    "Timestamps are handset local time with no time-zone field; BANDWIDTH lists carriers without band labels (a 100 MHz entry appears in %d 5G-labelled and %d 4G-labelled records), so the NR band cannot be resolved." % (
        shr['bandwidth_list_contains_100MHz']['5G_records'], shr['bandwidth_list_contains_100MHz']['4G_records']),
]

anom = []
for wd in windows:
    anom.append({
        'when': '%s - %s' % (wd['from'], wd['to']),
        'what': 'Observation: cluster of consecutive 5G-labelled records with poor NR SS-RSRP or SS-SINR; %d mobility events inside the interval. Hypothesis: cell-edge or cell-overlap area; load and interference cannot be tested from this log.' % wd['events_in_window'],
        'evidence': 'NR SS-RSRP median %.1f dBm, min %.0f dBm, %.1f%% poor; NR SS-SINR median %.1f dB, %.1f%% poor; pooled LTE RSRP median %.1f dBm (min %.0f), LTE SINR median %.1f dB; %d records (%d labelled 5G); lon %.6f-%.6f, lat %.6f-%.6f (%d GPS-valid); Speed %.0f-%.0f km/h; State D in %d records, passive DL max %.0f kbps; cells %s' % (
            wd['nr_rsrp_median'], wd['nr_rsrp_min'], wd['nr_rsrp_poor_pct'], wd['nr_sinr_median'], wd['nr_sinr_poor_pct'], wd['lte_rsrp_median_pooled'],
            wd['lte_rsrp_min_pooled'], wd['lte_sinr_median_pooled'], wd['records'], wd['5G_records'], wd['lon_range'][0], wd['lon_range'][1],
            wd['lat_range'][0], wd['lat_range'][1], wd['gps_valid_records'], wd['speed_kmh_range'][0], wd['speed_kmh_range'][1],
            wd['state_D_records'], wd['DL_kbps_max'], ', '.join(wd['cells']))})
anom.append({
    'when': 'whole log (%s - %s)' % (df['Timestamp'].iloc[0], df['Timestamp'].iloc[-1]),
    'what': 'Observation: the 4G/5G label flips %d times, every time on an unchanged node and cell, and 5G->4G changes mostly follow a non-poor NR sample. Hypothesis: the NR leg is added and released with traffic demand or inactivity; the fallback alone is not evidence of a coverage failure.' % int(flip.sum()),
    'evidence': 'NetworkTech flips: %d 4G->5G, %d 5G->4G (%d on same Node+CellID); last valid NR sample before a 5G->4G change: SS-RSRP >= -110 dBm in %d of %d (median %.1f dBm), RSRP >= -110 and SINR >= 0 in %d; first 4G record in State I in %d of %d; State D share %.1f%% of 5G-labelled vs %.1f%% of 4G-labelled records' % (
        flips_45, flips_54, sup['rat_label_flips']['flips_on_same_node_and_cell'], fbk['last_NR_RSRP_not_poor_ge_-110_n'], fbk['with_preceding_valid_NR_sample'],
        fbk['last_NR_RSRP_median'], fbk['last_NR_RSRP_not_poor_and_SINR_not_poor_n'], fbk['first_4G_record_state_I_n'], fbk['n'],
        shr['state_D_pct_by_label']['5G'], shr['state_D_pct_by_label']['4G'])})
i_ex = int(np.where(b_mask)[0][0])
anom.append({
    'when': 'records adjacent to label changes: ' + ', '.join(sorted(mf['timestamps_5G'] + mf['timestamps_4G'])),
    'what': 'Observation: mixed-RAT values next to a label change. Hypothesis: Level/Qual, SNR and the RAT label are refreshed asynchronously (stale readings); such values bias the NR and LTE pools slightly.',
    'evidence': '%d 5G-labelled records with SNR "-" whose Level/Qual equal LTE values; %d 4G-labelled records whose Level/Qual equal the NR values of the next 5G record; example: %s 4G record Level %s dBm, %s 4G record Level %s dBm, next 5G record NR Level %s dBm with LTE anchor %s dBm' % (
        mf['5G_records_with_SNR_missing'], mf['of_which_Level_and_Qual_equal_next_5G_record_NR_values'],
        df['Timestamp'][i_ex - 1], df['Level'][i_ex - 1], df['Timestamp'][i_ex], df['Level'][i_ex], df['Level'][i_ex + 1], df['SecondCell_RSRP'][i_ex + 1])})
if jump_rows:
    j0 = max(jump_rows, key=lambda r: r['frozen_run_records'])
    anom.append({
        'when': '%s - %s (and %d further frozen runs)' % (j0['frozen_run_start'], j0['from'], len(jump_rows) - 1),
        'what': 'Observation: position frozen with Speed 0 and FOREGROUND=false, then a jump. Hypothesis: location updates stalled while the application was in the background, so position and Speed are stale and "stationary" cannot be assumed.',
        'evidence': '%d records at lon %.6f, lat %.6f (%d with FOREGROUND=false); next fix at %s is %.0f m away after %.0f s (>= %.0f km/h, logged Speed %.0f km/h); other jumps: %s' % (
            j0['frozen_run_records'], j0['frozen_position'][0], j0['frozen_position'][1], j0['foreground_false_in_run'], j0['to'], j0['distance_m'], j0['gap_s'],
            j0['implied_kmh'], j0['logged_speed_kmh_before_after'][1],
            '; '.join('%s %.0f m in %.0f s' % (r['to'], r['distance_m'], r['gap_s']) for r in jump_rows if r is not j0))})
anom.append({
    'when': '%s - %s' % (E[0]['ts'], E[-1]['ts']),
    'what': 'Observation: frequent cell changes with immediate returns to the previous cell, many between cells of the two operator codes; whether the handset was moving cannot be verified for most of them (stale GPS). Hypothesis: overlapping cells or layers of similar level; no root cause is asserted.',
    'evidence': '%d events in %.0f s (%.2f per minute): %d reselection-type, %d handover-type; %d immediate A->B, B->A reversals (%d within 10 s, %d within 30 s); %d events change operator code, %d change ARFCN, %d stay on the same node; %d logged with Speed 0, %d with FOREGROUND=false, %d on stale-position records' % (
        mob['events_total'], duration, mob['events_per_minute'], mob['reselection_type_events'], mob['handover_type_events'],
        mob['immediate_reversals_A_B_then_B_A']['n'], mob['immediate_reversals_A_B_then_B_A']['within_10s'], mob['immediate_reversals_A_B_then_B_A']['within_30s'],
        mob['operator_code_changes_across_event'], mob['arfcn_changes'], mob['intra_node_events'], mob['events_logged_with_Speed_0'],
        mob['events_logged_with_FOREGROUND_false'], mob['events_logged_on_stale_position_records'])})
anom.append({
    'when': ', '.join(e[0] for e in mob['source_target_same_node_and_cell_only_ARFCN_or_PCI_differs']['events']),
    'what': 'Observation: events whose source and target are the same node and cell (only ARFCN or PCI differs), each directly after an event into that cell. Hypothesis: one cell change logged as two events because identity fields update one after another; the event total may be inflated by these.',
    'evidence': '; '.join(e[1] + ' ' + e[2] for e in mob['source_target_same_node_and_cell_only_ARFCN_or_PCI_differs']['events'])})
anom.append({
    'when': 'events file, all %d rows' % mob['events_total'],
    'what': 'Observation: IRAT-type events are not RAT changes of the record label, and "5G" event types refer to LTE anchor cells; the types describe LTE (anchor) cell changes plus the label state, not NR cell changes.',
    'evidence': '%d of %d IRAT-type events lie between two records with the same NetworkTech label; only %d event coincides with one of the %d label flips; the source RAT in EventType differs from the previous record label in %d events; identity fields equal the LTE anchor fields in %d of %d 5G-labelled records' % (
        mob['irat_type_events_with_same_label_before_and_at_event'], mob['irat_type_events'], mob['events_coinciding_with_a_label_flip'],
        mob['rat_label_flips_in_main_log'], mob['event_type_source_RAT_differs_from_label_of_previous_record'], five_g_ident_eq_anchor, int(is5.sum()))})
if rc['outlier_rows']:
    anom.append({
        'when': ', '.join(sorted(set(x['Timestamp'] for x in rc['outlier_rows']))),
        'what': 'Observation: LTE RSRP (Level) and LTERSSI disagree by more than %.0f dB from their usual difference in the same record. Hypothesis: one of the two values is stale or belongs to the other RAT (not refreshed after a cell or label change).' % RSSI_DEV_DB,
        'evidence': 'typical LTERSSI - RSRP: median %.0f dB (P5 %.0f, P95 %.0f); outliers: %s' % (
            rc['RSSI_minus_RSRP_dB']['median'], rc['RSSI_minus_RSRP_dB']['p5'], rc['RSSI_minus_RSRP_dB']['p95'],
            '; '.join(list(OrderedDict.fromkeys('%s cell %s-%s Level %s dBm, Qual %s dB, SNR %s dB, LTERSSI %s dBm' % (
                x['Timestamp'], x['Node'], x['CellID'], x['Level'], x['Qual'], x['SNR'], x['LTERSSI']) for x in rc['outlier_rows'])))
            + '; %d of the %d outlier records are 4G-labelled mixed-field records' % (int((rssi_out & b_mask).sum()), int(rssi_out.sum())))})
anom.append({
    'when': 'whole log, same-record pairs',
    'what': 'Observation (aligned inter-KPI relationships): SINR and RSRQ rise with RSRP on both layers and NR level follows the LTE anchor level; passive DL bitrate shows no relation with NR level. Coefficients are descriptive only (held, unevenly spaced samples); passive bitrate depends on traffic demand and is not a capacity or throughput measurement.',
    'evidence': 'Spearman: NR RSRP-SINR %.3f (n=%d), NR RSRQ-SINR %.3f, LTE RSRP-SINR %.3f (n=%d), NR RSRP vs anchor LTE RSRP %.3f (n=%d; NR minus LTE median %.1f dB), NR RSRP vs passive DL kbps in State D %.3f (n=%d); NR RSRP poor in %d records, of which SINR < 0 dB in %d' % (
        rel['NR_RSRP_vs_NR_SINR']['spearman'], rel['NR_RSRP_vs_NR_SINR']['n'], rel['NR_RSRQ_vs_NR_SINR']['spearman'],
        rel['LTE_RSRP_vs_LTE_SINR_pooled']['spearman'], rel['LTE_RSRP_vs_LTE_SINR_pooled']['n'],
        rel['NR_RSRP_vs_anchor_LTE_RSRP_same_record']['spearman'], rel['NR_RSRP_vs_anchor_LTE_RSRP_same_record']['n'], rel['NR_minus_anchor_RSRP_dB']['median'],
        rel['NR_RSRP_vs_passive_DL_kbps_state_D_only']['spearman'], rel['NR_RSRP_vs_passive_DL_kbps_state_D_only']['n'],
        rel['NR_poor_RSRP_and_poor_SINR_overlap']['nr_rsrp_poor_n'], rel['NR_poor_RSRP_and_poor_SINR_overlap']['of_which_sinr_poor_n'])})

# ----------------------------------------------------------------------------- results.json (contract layout)
def contract(d):
    return OrderedDict((k, d[k]) for k in ('valid_n', 'mean', 'median', 'sd', 'p5', 'p95', 'poor_pct'))


results = OrderedDict()
results['records'] = N
results['duration_s'] = duration
results['rat_label_records'] = {'4G': rat_counts.get('4G', 0), '5G': rat_counts.get('5G', 0)}
results['serving_cell_operator_code_records'] = op_counts
results['kpi'] = OrderedDict((k, contract(v)) for k, v in table2.items())
results['active_tests'] = active
results['mobility'] = {'events_total': int(len(ev)), 'events_by_type': ev_counts}
results['data_quality_findings'] = dq
results['anomaly_candidates'] = anom
results['model_self_report'] = 'claude-opus-5-5 (Opus 5.5)'

with open(OUT + 'results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=1, ensure_ascii=False)
with open(OUT + 'supplementary.json', 'w', encoding='utf-8') as f:
    json.dump(sup, f, indent=1, ensure_ascii=False, default=str)

# ----------------------------------------------------------------------------- narrative.md (all numbers inserted from the computed objects)
def row(name, unit, cats=True):
    d = T[name]
    s_ = '- %s, N=%d: %.2f/%.1f/%.2f/%.1f/%.1f %s' % (name, d['valid_n'], d['mean'], d['median'], d['sd'], d['p5'], d['p95'], unit)
    if cats:
        s_ += '; %.2f/%.2f/%.2f.' % (d['poor_pct'], d['acceptable_pct'], d['good_pct'])
    else:
        s_ += '; descriptive.'
    return s_


w0 = windows[0]
jmax = max(jump_rows, key=lambda r: r['frozen_run_records'])
tsr = sup['time_shares']
lb = shr['lte_serving_or_anchor_band_all_records']
P = []
P.append('# Narrative')
P.append('**Scope, validation.** %d records, 2026-10-02 %s-%s (%.0f s, handset local time, zone not logged), longitude %.4f-%.4f, latitude %.4f-%.4f. Timestamps parse, never go backwards; gaps reach %.0f s. No -201 code or out-of-range value. Excluded: missing markers only (NR SINR "-" in %d of %d 5G-labelled records; one 5G record lacks anchor values). No active test was run: throughput and RTT are N/A; logged bitrates are passive traffic.' % (
    N, hms(df['Timestamp'].iloc[0]), hms(df['Timestamp'].iloc[-1]), duration, lon.min(), lon.max(), lat.min(), lat.max(), gap.max(),
    T['NR_SS_SINR']['missing_marker_n'], T['NR_SS_SINR']['raw_n']))
P.append('**Table II** (mean/median/sample SD/P5/P95, linear-interpolation percentiles; then poor/acceptable/good %, strict thresholds):\n' + '\n'.join([
    row('NR_SS_RSRP', 'dBm'), row('NR_SS_RSRQ', 'dB', False), row('NR_SS_SINR', 'dB'),
    row('LTE_RSRP', 'dBm'), row('LTE_RSRQ', 'dB', False), row('LTE_SINR', 'dB')]))
P.append('**Shares.** Records: 4G %.2f%%, 5G %.2f%% (NSA per schema note; NRSTATE empty, SA not verifiable), unknown 0. LTE serving/anchor band: L7 %.2f%%, L20 %.2f%%, L3 %.2f%%. NR band unknown for all 5G records (no NR ARFCN, PCI or band logged). Time shares (record held until the next, capped at %d s, remainder unobserved): 4G %.1f%%, 5G %.1f%%, unobserved %.1f%%.' % (
    shr['rat_label']['4G']['pct'], shr['rat_label']['5G']['pct'], lb['L7']['pct'], lb['L20']['pct'], lb['L3']['pct'], GAP_CAP_S,
    tsr['rat_label_pct_of_total']['4G']['pct'], tsr['rat_label_pct_of_total']['5G']['pct'], tsr['unobserved_pct']))
P.append('**Mobility.** %d events (%d reselection-type, %d handover-type; types in results.json); success rate not observable.' % (
    mob['events_total'], mob['reselection_type_events'], mob['handover_type_events']))
P.append('**Observations.** (1) %s-%s, longitude %.4f-%.4f, latitude %.4f-%.4f: NR RSRP median %.0f dBm (%.1f%% poor), NR SINR median %.0f dB, LTE RSRP median %.0f dBm, %d events. (2) The label flips %d times, always on an unchanged cell; %d of %d 5G-to-4G changes follow a non-poor NR RSRP, %d start idle. (3) Positions freeze in background: %d records at one point, %s-%s, then %.0f m in %.0f s; %d frozen-position records (jumps above %.0f km/h against logged speed up to %.0f km/h) are excluded from spatial outputs only. (4) %d immediate cell reversals; %d events change operator code (22610, 22601). (5) Spearman RSRP-SINR: NR %.2f, LTE %.2f.' % (
    hms(w0['from']), hms(w0['to']), w0['lon_range'][0], w0['lon_range'][1], w0['lat_range'][0], w0['lat_range'][1],
    w0['nr_rsrp_median'], w0['nr_rsrp_poor_pct'], w0['nr_sinr_median'], w0['lte_rsrp_median_pooled'], w0['events_in_window'],
    int(flip.sum()), fbk['last_NR_RSRP_not_poor_ge_-110_n'], fbk['n'], fbk['first_4G_record_state_I_n'],
    jmax['frozen_run_records'], hms(jmax['frozen_run_start']), hms(jmax['from']), jmax['distance_m'], jmax['gap_s'],
    g['stale_position_records_excluded_from_spatial_outputs'], JUMP_KMH, JUMP_SPD_MAX,
    mob['immediate_reversals_A_B_then_B_A']['n'], mob['operator_code_changes_across_event'],
    rel['NR_RSRP_vs_NR_SINR']['spearman'], rel['LTE_RSRP_vs_LTE_SINR_pooled']['spearman']))
P.append('**Hypotheses, unverified.** (1) cell-edge or overlap area; (2) NR leg released on inactivity, not a coverage failure; (3) stale location in background; (4) shared RAN with overlapping layers. Load, interference and server limits cannot be tested.')
P.append('**Limitations.** Readings are held (%.0f%% of 5G records repeat the previous triplet). %d records beside label changes carry mixed-RAT values; kept (without them the NR RSRP mean is %.2f dBm). LTE SNR in 4G-labelled records never falls below 0 dB, so the LTE SINR poor share may be understated. %.1f%% of records fall in the first %.1f%% of the time. All outputs need human verification.' % (
    rep['5G_records_NR_triplet_identical_to_previous_record']['pct'], n_mixed, sens['NR_SS_RSRP']['mean'],
    dens['pct_records_before'], dens['pct_time_before']))
narr = '\n\n'.join(P) + '\n'
with open(OUT + 'narrative.md', 'w', encoding='utf-8') as f:
    f.write(narr)
sup['narrative_word_count_whitespace_split'] = len(narr.split())
with open(OUT + 'supplementary.json', 'w', encoding='utf-8') as f:
    json.dump(sup, f, indent=1, ensure_ascii=False, default=str)

print(json.dumps(sup, indent=1, ensure_ascii=False, default=str))
