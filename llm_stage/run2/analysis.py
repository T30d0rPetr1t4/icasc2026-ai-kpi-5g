#!/usr/bin/env python3
"""Drive-test analysis (run2). Reads ONLY the two data files in ../input and writes
results.json, supplement.json and narrative.md next to this script.
Every number reported in the outputs is produced by this script."""
import json, os, re
import numpy as np
import pandas as pd

IN = '/home/claude/ai_stage/input'
OUT = '/home/claude/ai_stage/run2'
LOG = os.path.join(IN, 'orange_2026.10.02_12.16.18.txt')
EVT = os.path.join(IN, 'orange_2026.10.02_12.16.18_events.txt')

# ----------------------------------------------------------------- choices
PCT_METHOD = 'linear'          # numpy default, Hyndman-Fan type 7
GAP_S = 10.0                   # forward-hold; an interval > GAP_S is 'unobserved'
FROZEN_S = 10.0                # identical coordinates held >= FROZEN_S -> stale fix
ACC_MAX_M = 50.0               # GPS accuracy limit for spatial use
MISSING = {'', '-'}
# plausibility ranges (analyst choice, wide reporting ranges) - used only to trap unit errors
RANGE = {'RSRP': (-156.0, -31.0), 'RSRQ': (-43.0, 20.0), 'SINR': (-23.0, 40.0)}
R = lambda x, n=3: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)

# ----------------------------------------------------------------- load
with open(LOG, encoding='utf-8', errors='replace') as fh:
    raw = [ln.rstrip('\n').rstrip('\r').split('\t') for ln in fh]
hdr = raw[0]
NB = hdr.index('NTech1')                      # 90 base columns, neighbours after
body = [r for r in raw[1:] if any(c.strip() for c in r)]
field_counts = pd.Series([len(r) for r in body]).value_counts().sort_index().to_dict()
over_header = int(sum(len(r) > len(hdr) for r in body))
short_rows = int(sum(len(r) < NB for r in body))
df = pd.DataFrame([r[:NB] + [''] * (NB - len(r)) for r in body], columns=hdr[:NB])
N = len(df)

def num(s):
    """string series -> float, with missing markers mapped to NaN"""
    s = s.astype(str).str.strip()
    return pd.to_numeric(s.where(~s.isin(MISSING)), errors='coerce')

def nonnumeric(s):
    s = s.astype(str).str.strip()
    return int((~s.isin(MISSING) & pd.to_numeric(s, errors='coerce').isna()).sum())

# timestamps
ts_ok = df.Timestamp.str.fullmatch(r'\d{4}\.\d\d\.\d\d_\d\d\.\d\d\.\d\d')
df['ts'] = pd.to_datetime(df.Timestamp, format='%Y.%m.%d_%H.%M.%S', errors='coerce')
dt_next = (df.ts.shift(-1) - df.ts).dt.total_seconds()      # interval to next record
dt_prev = df.ts.diff().dt.total_seconds()
duration_s = float((df.ts.iloc[-1] - df.ts.iloc[0]).total_seconds())

is5 = df.NetworkTech == '5G'
is4 = df.NetworkTech == '4G'
for c in ['Level', 'Qual', 'SNR', 'SecondCell_RSRP', 'SecondCell_RSRQ', 'SecondCell_SNR', 'LTERSSI',
          'Speed', 'Accuracy', 'Longitude', 'Latitude', 'DL_bitrate', 'UL_bitrate', 'CQI']:
    df['n_' + c] = num(df[c])

df['n_lteRSRP'] = df.n_Level.where(is4, df.n_SecondCell_RSRP)      # LTE serving (4G label) or anchor (5G label)
df['n_lteRSRQ'] = df.n_Qual.where(is4, df.n_SecondCell_RSRQ)
df['n_lteSINR'] = df.n_SNR.where(is4, df.n_SecondCell_SNR)

# ----------------------------------------------------------------- KPI series (schema-literal)
def kpi_series():
    s = {}
    s['NR_SS_RSRP'] = (df.loc[is5, 'Level'], 'RSRP')
    s['NR_SS_RSRQ'] = (df.loc[is5, 'Qual'], 'RSRQ')
    s['NR_SS_SINR'] = (df.loc[is5, 'SNR'], 'SINR')
    s['LTE_RSRP'] = (pd.concat([df.loc[is4, 'Level'], df.loc[is5, 'SecondCell_RSRP']]), 'RSRP')
    s['LTE_RSRQ'] = (pd.concat([df.loc[is4, 'Qual'], df.loc[is5, 'SecondCell_RSRQ']]), 'RSRQ')
    s['LTE_SINR'] = (pd.concat([df.loc[is4, 'SNR'], df.loc[is5, 'SecondCell_SNR']]), 'SINR')
    return s

def validate(strs, kind):
    st = strs.astype(str).str.strip()
    v = num(st)
    miss = st.isin(MISSING)
    nonnum = ~miss & v.isna()
    code201 = v == -201
    lo, hi = RANGE[kind]
    oor = v.notna() & ~code201 & ((v < lo) | (v > hi))
    valid = v[v.notna() & ~code201 & ~oor]
    return valid, dict(raw_n=int(len(st)), valid_n=int(len(valid)), excl_missing=int(miss.sum()),
                       excl_non_numeric=int(nonnum.sum()), excl_code_minus201=int(code201.sum()),
                       excl_out_of_range=int(oor.sum()))

def stats(v, kind):
    v = v.astype(float)
    o = dict(valid_n=int(len(v)), mean=R(v.mean()), median=R(v.median()),
             sd=R(v.std(ddof=1)) if len(v) > 1 else None,
             p5=R(np.percentile(v, 5, method=PCT_METHOD)), p95=R(np.percentile(v, 95, method=PCT_METHOD)),
             min=R(v.min()), max=R(v.max()))
    if kind == 'RSRP':
        good, poor = (v > -85).sum(), (v < -110).sum()
    elif kind == 'SINR':
        good, poor = (v > 10).sum(), (v < 0).sum()
    else:
        o.update(poor_pct=None, good_pct=None, acceptable_pct=None, poor_n=None)
        return o
    n = len(v)
    o.update(poor_pct=R(100.0 * poor / n), good_pct=R(100.0 * good / n),
             acceptable_pct=R(100.0 * (n - good - poor) / n), poor_n=int(poor))
    return o

kpi, counts, valid_vals = {}, {}, {}
for name, (strs, kind) in kpi_series().items():
    v, c = validate(strs, kind)
    valid_vals[name] = v
    counts[name] = c
    kpi[name] = stats(v, kind)

# ----------------------------------------------------------------- label-transition artefacts (sensitivity only)
nxt = df.shift(-1)
# (a) 5G-labelled record with NR SINR missing: is RSRP/RSRQ a copy of the LTE anchor?
nr_sinr_missing = is5 & df.SNR.str.strip().isin(MISSING)
copy_anchor = nr_sinr_missing & (df.Level == df.SecondCell_RSRP) & (df.Qual == df.SecondCell_RSRQ)
copy_next4g = nr_sinr_missing & ~copy_anchor & (nxt.NetworkTech == '4G') & (df.Level == nxt.Level) & (df.Qual == nxt.Qual)
nr_art = copy_anchor | copy_next4g
# (b) 4G-labelled record directly followed by a 5G-labelled record with identical Level and Qual,
#     where that pair differs from the LTE anchor pair of the following 5G record
lte_art = is4 & (nxt.NetworkTech == '5G') & (df.Level == nxt.Level) & (df.Qual == nxt.Qual) & \
          ~((nxt.Level == nxt.SecondCell_RSRP) & (nxt.Qual == nxt.SecondCell_RSRQ))
lte_art_big = lte_art & ((num(df.Level) - num(nxt.SecondCell_RSRP)).abs() >= 3)
sens = {}
def sens_stats(name, strs, kind):
    v, c = validate(strs, kind)
    sens[name] = stats(v, kind)
sens_stats('NR_SS_RSRP', df.loc[is5 & ~nr_art, 'Level'], 'RSRP')
sens_stats('NR_SS_RSRQ', df.loc[is5 & ~nr_art, 'Qual'], 'RSRQ')
sens_stats('NR_SS_SINR', df.loc[is5 & ~nr_art, 'SNR'], 'SINR')
sens_stats('LTE_RSRP', pd.concat([df.loc[is4 & ~lte_art, 'Level'], df.loc[is5, 'SecondCell_RSRP']]), 'RSRP')
sens_stats('LTE_RSRQ', pd.concat([df.loc[is4 & ~lte_art, 'Qual'], df.loc[is5, 'SecondCell_RSRQ']]), 'RSRQ')
sens_stats('LTE_SINR', pd.concat([df.loc[is4, 'SNR'], df.loc[is5, 'SecondCell_SNR']]), 'SINR')

# ----------------------------------------------------------------- GPS validation
lat, lon = df.n_Latitude, df.n_Longitude
def hav(a1, o1, a2, o2):
    p1, p2 = np.radians(a1), np.radians(a2)
    x = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(o2 - o1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(x))
gps_missing = lat.isna() | lon.isna()
gps_range = ~gps_missing & (((lat == 0) & (lon == 0)) | (lat.abs() > 90) | (lon.abs() > 180))
gps_acc = df.n_Accuracy > ACC_MAX_M
run_id = ((lat != lat.shift()) | (lon != lon.shift())).cumsum()
run_start = df.groupby(run_id).ts.transform('min')
held_s = (df.ts - run_start).dt.total_seconds()
first_of_run = run_id != run_id.shift()
gps_frozen = ~first_of_run & (held_s >= FROZEN_S)            # stale repeat of an old fix
gps_invalid = gps_missing | gps_range | gps_acc | gps_frozen
df['gps_ok'] = ~gps_invalid
# frozen runs and the jump that ends them
frozen_runs = []
for rid, g in df.groupby(run_id):
    dur = (g.ts.iloc[-1] - g.ts.iloc[0]).total_seconds()
    if dur < FROZEN_S:
        continue
    j = g.index[-1] + 1
    rec = dict(start=g.Timestamp.iloc[0], end=g.Timestamp.iloc[-1], records=int(len(g)), held_s=dur,
               lon=float(g.n_Longitude.iloc[0]), lat=float(g.n_Latitude.iloc[0]),
               logged_speed_kmh_min=int(g.n_Speed.min()), logged_speed_kmh_max=int(g.n_Speed.max()),
               foreground_false_records=int((g.FOREGROUND == 'false').sum()))
    if j < N:
        d = float(hav(g.n_Latitude.iloc[-1], g.n_Longitude.iloc[-1], lat[j], lon[j]))
        gap = float((df.ts[j] - g.ts.iloc[-1]).total_seconds())
        rec.update(next_fix=df.Timestamp[j], jump_m=R(d, 1), gap_to_next_fix_s=gap,
                   implied_kmh_over_gap=R(d / gap * 3.6, 1) if gap > 0 else None,
                   implied_kmh_over_whole_run=R(d / (gap + dur) * 3.6, 1))
    frozen_runs.append(rec)
max_logged_speed = int(df.n_Speed.max())
frozen_impossible = [r for r in frozen_runs if r.get('implied_kmh_over_gap') and r['implied_kmh_over_gap'] > max_logged_speed]
frozen_time_s = float(sum(r['held_s'] for r in frozen_runs))
# speed-unit check on well-sampled moving stretches (consecutive valid fixes, gap 1..3 s, position changed)
step_d = pd.Series(hav(lat.shift(), lon.shift(), lat, lon), index=df.index)
okstep = df.gps_ok & df.gps_ok.shift(fill_value=False) & (dt_prev >= 1) & (dt_prev <= 3) & (step_d > 0)
gps_kmh = float(step_d[okstep].sum() / dt_prev[okstep].sum() * 3.6)
log_kmh = float((df.n_Speed[okstep] * dt_prev[okstep]).sum() / dt_prev[okstep].sum())
vg = df[df.gps_ok]
track_m = float(hav(vg.n_Latitude.shift(), vg.n_Longitude.shift(), vg.n_Latitude, vg.n_Longitude).dropna().sum())
spatial = dict(gps_raw_n=N, gps_valid_n=int(df.gps_ok.sum()), excl_missing=int(gps_missing.sum()),
               excl_out_of_range_or_zero=int(gps_range.sum()), excl_accuracy_gt_50m=int(gps_acc.sum()),
               excl_frozen_fix=int(gps_frozen.sum()), accuracy_m_max=float(df.n_Accuracy.max()),
               bbox_valid=dict(lon_min=float(vg.n_Longitude.min()), lon_max=float(vg.n_Longitude.max()),
                               lat_min=float(vg.n_Latitude.min()), lat_max=float(vg.n_Latitude.max())),
               track_length_m_between_valid_fixes_lower_bound=R(track_m, 0),
               speed_check=dict(gps_derived_kmh=R(gps_kmh, 1), logged_speed_kmh=R(log_kmh, 1),
                                steps=int(okstep.sum())),
               max_logged_speed_kmh=max_logged_speed,
               frozen_runs=frozen_runs, frozen_runs_n=len(frozen_runs), frozen_time_s=frozen_time_s,
               frozen_runs_ending_faster_than_max_logged_speed=len(frozen_impossible))

# ----------------------------------------------------------------- shares
def share(series, base=None):
    vc = series.value_counts(dropna=False)
    n = int(vc.sum()) if base is None else base
    return {str(k): dict(n=int(v), pct=R(100.0 * v / n, 2)) for k, v in vc.items()}

rat_n = df.NetworkTech.value_counts().to_dict()
band = df.BAND.where(~df.BAND.str.strip().isin(MISSING), 'unknown')
bw_has100 = df.BANDWIDTH.str.split('/').apply(lambda x: '100' in x)
# time weighting: forward-hold, intervals > GAP_S are unobserved
w = dt_next.fillna(0.0)
obs = w.where(w <= GAP_S, 0.0)
gap_time = float(w.where(w > GAP_S, 0.0).sum())
obs_time = float(obs.sum())
def tshare(series):
    t = obs.groupby(series).sum()
    return {str(k): dict(seconds=float(v), pct_of_observed=R(100.0 * v / obs_time, 2)) for k, v in t.items()}
def tshare_ffill(series):
    t = w.groupby(series).sum()
    return {str(k): dict(seconds=float(v), pct=R(100.0 * v / w.sum(), 2)) for k, v in t.items()}
shares = dict(
    rat_label_samples=share(df.NetworkTech),
    lte_serving_or_anchor_band_samples_all=share(band),
    lte_serving_band_samples_4G_label=share(band[is4]),
    lte_anchor_band_samples_5G_label=share(band[is5]),
    nr_band_samples_5G_label={'unknown': dict(n=int(is5.sum()), pct=100.0)},
    nr_fields_present=dict(NRSTATE=int((~df.NRSTATE.str.strip().isin(MISSING)).sum()),
                           CSI_RSRP=int((~df.CSI_RSRP.str.strip().isin(MISSING)).sum()),
                           CSI_RSRQ=int((~df.CSI_RSRQ.str.strip().isin(MISSING)).sum()),
                           CSI_SNR=int((~df.CSI_SNR.str.strip().isin(MISSING)).sum())),
    arfcn_values=share(df.ARFCN), network_type_num=share(df.NetworkTypeNum),
    bandwidth_list_contains_100MHz=dict(label_5G=int((bw_has100 & is5).sum()), label_4G=int((bw_has100 & is4).sum()),
                                        label_5G_without=int((~bw_has100 & is5).sum())),
    ca_flag_samples=share(df.CA), state_samples=share(df.State), foreground_samples=share(df.FOREGROUND),
    operator_code_samples=share(df.Operator),
    operator_code_by_rat_label=pd.crosstab(df.Operator, df.NetworkTech).to_dict(),
    time_weighting=dict(rule='forward-hold to next record; intervals > %g s counted as unobserved' % GAP_S,
                        total_s=duration_s, observed_s=obs_time, unobserved_gap_s=gap_time,
                        unobserved_pct=R(100.0 * gap_time / duration_s, 2), gaps_n=int((w > GAP_S).sum()),
                        max_gap_s=float(w.max()),
                        rat_label=tshare(df.NetworkTech), lte_band=tshare(band), operator_code=tshare(df.Operator),
                        rat_label_ffill_all_intervals=tshare_ffill(df.NetworkTech)))
# 5G-labelled records: do serving-cell identifiers equal the LTE anchor?
has_anchor = is5 & ~df.SecondCell_NODE.str.strip().isin(MISSING)
ident_same = has_anchor & (df.Node == df.SecondCell_NODE) & (df.CellID == df.SecondCell_CELLID) & \
             (df.PSC == df.SecondCell_PSC) & (df.ARFCN == df.SecondCell_ARFCN) & (df.BAND == df.SecondCell_BAND)
anchor_in_4g = is4 & ~df.SecondCell_RSRP.str.strip().isin(MISSING)

# ----------------------------------------------------------------- active tests / passive traffic
def present(c):
    return int((~df[c].astype(str).str.strip().isin(MISSING)).sum())
active = dict(download_n=present('TESTDOWNLINK'), upload_n=present('TESTUPLINK'), ping_n=present('PINGAVG'))
active_other = {c: present(c) for c in ['Test_Status', 'PINGMIN', 'PINGMAX', 'PINGSTDEV', 'PINGLOSS',
                                        'TESTDOWNLINKMAX', 'TESTUPLINKMAX']}
passive = dict(dl_gt0_records=int((df.n_DL_bitrate > 0).sum()), ul_gt0_records=int((df.n_UL_bitrate > 0).sum()),
               dl_max_kbps=float(df.n_DL_bitrate.max()), ul_max_kbps=float(df.n_UL_bitrate.max()),
               note='passive interface bitrate, not an active throughput test; no statistics reported as throughput')

# ----------------------------------------------------------------- mobility
with open(EVT, encoding='utf-8', errors='replace') as fh:
    eraw = [ln.rstrip('\n').rstrip('\r').split('\t') for ln in fh if ln.strip()]
ev = pd.DataFrame([r[:len(eraw[0])] + [''] * (len(eraw[0]) - len(r)) for r in eraw[1:]], columns=eraw[0])
ev['ts'] = pd.to_datetime(ev.Timestamp, format='%Y.%m.%d_%H.%M.%S', errors='coerce')
ev_by_type = ev.EventType.value_counts().to_dict()
log_ev = df[(df.EVENT.str.strip() != '') & (df.EVENT != 'LOG START')]
ev_match = bool(len(log_ev) == len(ev) and (log_ev.Timestamp.values == ev.Timestamp.values).all()
                and (log_ev.EVENT.values == ev.EventType.values).all()
                and (log_ev.EVENTDETAILS.values == ev.EventDetails.values).all())
def parse(side):
    p = side.split('-')
    return dict(plmn=p[0] + p[1], tac=p[2], node=p[3], cell=p[4], arfcn=p[5], pci=p[6])
src = ev.EventDetails.str.split(':').str[0].apply(parse).apply(pd.Series)
tgt = ev.EventDetails.str.split(':').str[1].apply(parse).apply(pd.Series)
same_cell_pci_only = (src.plmn == tgt.plmn) & (src.node == tgt.node) & (src.cell == tgt.cell) & \
                     (src.arfcn == tgt.arfcn) & (src.pci != tgt.pci)
newnode_oldpci = (src.node != tgt.node) & (src.pci == tgt.pci)
cross_plmn = src.plmn != tgt.plmn
lte_arfcns = set(df.ARFCN.unique())
ev_all_lte_arfcn = bool(src.arfcn.isin(lte_arfcns).all() and tgt.arfcn.isin(lte_arfcns).all())
inter_freq = src.arfcn != tgt.arfcn
# ping-pong: event A->B followed by B->A (cell = plmn-node-cell-arfcn) within 60 s
key = lambda d: d.plmn + '-' + d.node + '-' + d.cell + '-' + d.arfcn
sk, tk = src.apply(key, axis=1), tgt.apply(key, axis=1)
pp = 0
for i in range(len(ev) - 1):
    if sk[i] == tk[i + 1] and tk[i] == sk[i + 1] and (ev.ts[i + 1] - ev.ts[i]).total_seconds() <= 60:
        pp += 1
ev_gps = ev.merge(df[['Timestamp', 'EVENTDETAILS', 'gps_ok', 'FOREGROUND', 'Speed']].drop_duplicates(['Timestamp', 'EVENTDETAILS']),
                  left_on=['Timestamp', 'EventDetails'], right_on=['Timestamp', 'EVENTDETAILS'], how='left')
half = df.ts.iloc[0] + pd.Timedelta(seconds=duration_s / 2)
mobility_extra = dict(events_file_rows=int(len(ev)), main_log_event_rows_excl_LOG_START=int(len(log_ev)),
                      events_file_matches_main_log=ev_match, cross_plmn_events=int(cross_plmn.sum()),
                      inter_frequency_events=int(inter_freq.sum()),
                      same_cell_pci_only_change_events=int(same_cell_pci_only.sum()),
                      new_node_with_old_pci_events=int(newnode_oldpci.sum()),
                      all_event_arfcns_are_lte_earfcns_of_log=ev_all_lte_arfcn,
                      ping_pong_pairs_within_60s=int(pp),
                      events_with_invalid_gps=int((ev_gps.gps_ok == False).sum()),
                      events_app_background=int((ev_gps.FOREGROUND == 'false').sum()),
                      events_logged_speed_0=int((ev_gps.Speed == '0').sum()),
                      events_first_half=int((ev.ts < half).sum()), events_second_half=int((ev.ts >= half).sum()),
                      events_per_min=R(len(ev) / (duration_s / 60.0), 2),
                      distinct_cells_main_log=int((df.Operator + '-' + df.Node + '-' + df.CellID).nunique()),
                      distinct_nodes_main_log=int(df.Node.nunique()),
                      success_rate='N/A (not observable from the handset)')

# ----------------------------------------------------------------- relationships / staleness
def corr(a, b):
    m = a.notna() & b.notna()
    return dict(n=int(m.sum()), pearson=R(a[m].corr(b[m])), spearman=R(a[m].rank().corr(b[m].rank())))
A = df[is5]
rel = dict(
    NR_RSRP_vs_NR_SINR=corr(A.n_Level, A.n_SNR), NR_RSRP_vs_NR_RSRQ=corr(A.n_Level, A.n_Qual),
    NR_RSRQ_vs_NR_SINR=corr(A.n_Qual, A.n_SNR),
    NR_RSRP_vs_anchor_RSRP_same_record=corr(A.n_Level, A.n_SecondCell_RSRP),
    NR_SINR_vs_anchor_SINR_same_record=corr(A.n_SNR, A.n_SecondCell_SNR),
    anchor_RSRP_vs_anchor_SINR=corr(A.n_SecondCell_RSRP, A.n_SecondCell_SNR),
    LTE4G_RSRP_vs_SINR=corr(df[is4].n_Level, df[is4].n_SNR),
    LTE4G_RSRQ_vs_SINR=corr(df[is4].n_Qual, df[is4].n_SNR))
lte_rsrp_all = pd.concat([df.loc[is4, 'n_Level'], df.loc[is5, 'n_SecondCell_RSRP']])
lte_sinr_all = pd.concat([df.loc[is4, 'n_SNR'], df.loc[is5, 'n_SecondCell_SNR']])
lte_rsrq_all = pd.concat([df.loc[is4, 'n_Qual'], df.loc[is5, 'n_SecondCell_RSRQ']])
rel['LTE_pooled_RSRP_vs_SINR'] = corr(lte_rsrp_all, lte_sinr_all)
rel['LTE_pooled_RSRQ_vs_SINR'] = corr(lte_rsrq_all, lte_sinr_all)
nr_both = A.n_Level.notna() & A.n_SNR.notna()
nr_poor_rsrp = A.n_Level < -110
rel['NR_SINR_poor_pct_when_RSRP_poor'] = dict(n=int((nr_both & nr_poor_rsrp).sum()),
    pct=R(100.0 * (A.n_SNR[nr_both & nr_poor_rsrp] < 0).mean(), 2))
rel['NR_SINR_poor_pct_when_RSRP_not_poor'] = dict(n=int((nr_both & ~nr_poor_rsrp).sum()),
    pct=R(100.0 * (A.n_SNR[nr_both & ~nr_poor_rsrp] < 0).mean(), 2))
lb = lte_rsrp_all.notna() & lte_sinr_all.notna()
rel['LTE_SINR_poor_with_RSRP_not_poor'] = dict(n=int((lb & (lte_sinr_all < 0) & (lte_rsrp_all >= -110)).sum()),
                                              of_poor_SINR_n=int((lb & (lte_sinr_all < 0)).sum()))
rel['NR_SINR_exactly_0'] = dict(n=int((A.n_SNR == 0).sum()), pct_of_valid=R(100.0 * (A.n_SNR == 0).sum() / A.n_SNR.notna().sum(), 2))
rel['LTE_SINR_exactly_0'] = dict(n=int((lte_sinr_all == 0).sum()), pct_of_valid=R(100.0 * (lte_sinr_all == 0).sum() / lte_sinr_all.notna().sum(), 2))
# repeated (unchanged) measurement triplets between consecutive records of the same label
def repeat_share(mask, cols):
    sub = df[cols].astype(str).agg('/'.join, axis=1)
    same = mask & mask.shift(fill_value=False) & (sub == sub.shift())
    pairs = int((mask & mask.shift(fill_value=False)).sum())
    return dict(consecutive_pairs=pairs, identical=int(same.sum()), pct=R(100.0 * same.sum() / pairs, 2))
stale = dict(NR_triplet_5G_label=repeat_share(is5, ['Level', 'Qual', 'SNR']),
             LTE_triplet_4G_label=repeat_share(is4, ['Level', 'Qual', 'SNR']),
             anchor_triplet_5G_label=repeat_share(is5, ['SecondCell_RSRP', 'SecondCell_RSRQ', 'SecondCell_SNR']))
trip5 = df.loc[is5, ['Level', 'Qual', 'SNR']].astype(str).agg('/'.join, axis=1)
stale['NR_distinct_value_runs_5G_label'] = int((trip5 != trip5.shift()).sum())
# first record after a cell-change event keeps the previous record's values?
evrow = df.EVENT.str.strip().ne('') & (df.EVENT != 'LOG START')
carry = evrow & (df.Level == df.Level.shift()) & (df.Qual == df.Qual.shift()) & (df.SNR == df.SNR.shift())
stale['event_records_with_values_identical_to_previous_record'] = dict(n=int(carry.sum()), of=int(evrow.sum()))
# LTE RSSI consistency (4G label): RSRP is a per-resource-element power and cannot reach the wideband RSSI
r4 = df[is4 & df.n_LTERSSI.notna() & df.n_Level.notna()]
rssi = dict(lte_rssi_valid_n=int(len(r4)), rsrp_ge_rssi_n=int((r4.n_Level >= r4.n_LTERSSI).sum()),
            rsrp_ge_rssi_pct=R(100.0 * (r4.n_Level >= r4.n_LTERSSI).mean(), 2),
            rssi_minus_rsrp_median_db=R((r4.n_LTERSSI - r4.n_Level).median()),
            rssi_minus_rsrp_max_db=R((r4.n_LTERSSI - r4.n_Level).max()),
            rssi_minus_rsrp_min_db=R((r4.n_LTERSSI - r4.n_Level).min()))

# sampling density: records before the application first went to background
i_bg = df.index[df.FOREGROUND == 'false'][0]
t_bg = df.ts[i_bg]
sampling = dict(first_background_record=df.Timestamp[i_bg], records_before=int((df.ts < t_bg).sum()),
                records_before_pct=R(100.0 * (df.ts < t_bg).sum() / N, 2),
                seconds_before=float((t_bg - df.ts.iloc[0]).total_seconds()),
                seconds_before_pct=R(100.0 * (t_bg - df.ts.iloc[0]).total_seconds() / duration_s, 2),
                records_per_5min_bin={str(k): int(v) for k, v in ((df.ts - df.ts.iloc[0]).dt.total_seconds() // 300).astype(int).value_counts().sort_index().items()},
                foreground_false_records_with_event=int(((df.FOREGROUND == 'false') & df.EVENT.str.strip().ne('')).sum()),
                foreground_false_records=int((df.FOREGROUND == 'false').sum()))
nr_snr_hist = {str(int(k)): int(v) for k, v in df.loc[is5, 'n_SNR'].dropna().value_counts().sort_index().items()}
ca_by_label = pd.crosstab(df.CA, df.NetworkTech).to_dict()
_z = (df.loc[is5, 'n_SNR'] == 0)
nr_sinr_zero_runs = int((_z & ~_z.shift(fill_value=False)).sum())     # runs of consecutive 5G-labelled records at 0.0 dB

# ----------------------------------------------------------------- poor-sample clusters with time/place
def clusters(mask, valcol, label, join_s=10.0):
    out, idx = [], list(df.index[mask])
    if not idx:
        return out
    grp = [idx[0]]
    def close(g):
        s = df.loc[g]
        gv = s[s.gps_ok]
        out.append(dict(kpi=label, start=s.Timestamp.iloc[0], end=s.Timestamp.iloc[-1], samples=int(len(s)),
                        min=float(s[valcol].min()), max=float(s[valcol].max()),
                        nodes=sorted(set(s.Operator + '/' + s.Node + '-' + s.CellID)),
                        gps_valid_samples=int(len(gv)),
                        start_lon_lat=[float(gv.n_Longitude.iloc[0]), float(gv.n_Latitude.iloc[0])] if len(gv) else None,
                        end_lon_lat=[float(gv.n_Longitude.iloc[-1]), float(gv.n_Latitude.iloc[-1])] if len(gv) else None))
    for i in idx[1:]:
        if (df.ts[i] - df.ts[grp[-1]]).total_seconds() <= join_s:
            grp.append(i)
        else:
            close(grp); grp = [i]
    close(grp)
    return out
cl_nr_rsrp = clusters(is5 & (df.n_Level < -110), 'n_Level', 'NR_SS_RSRP<-110', join_s=30.0)
cl_nr_sinr = clusters(is5 & (df.n_SNR < 0), 'n_SNR', 'NR_SS_SINR<0', join_s=30.0)
cl_lte_rsrp = clusters(df.n_lteRSRP < -110, 'n_lteRSRP', 'LTE_RSRP<-110', join_s=30.0)
cl_lte_sinr = clusters(df.n_lteSINR < 0, 'n_lteSINR', 'LTE_SINR<0', join_s=30.0)

# ----------------------------------------------------------------- assemble anomaly candidates (values pulled from data)
def rowtxt(i):
    r = df.loc[i]
    return '%s %s Level/Qual/SNR=%s/%s/%s anchor=%s/%s/%s LTERSSI=%s cell=%s/%s-%s PCI%s EARFCN%s' % (
        r.Timestamp[11:], r.NetworkTech, r.Level, r.Qual, r.SNR, r.SecondCell_RSRP or '', r.SecondCell_RSRQ or '',
        r.SecondCell_SNR or '', r.LTERSSI, r.Operator, r.Node, r.CellID, r.PSC, r.ARFCN)
T = lambda t: pd.to_datetime(t, format='%Y.%m.%d_%H.%M.%S')
def win(t0, t1):
    return df[(df.ts >= T(t0)) & (df.ts <= T(t1))]
def ev_win(t0, t1):
    return ev[(ev.ts >= T(t0)) & (ev.ts <= T(t1))]
fr0 = max(frozen_runs, key=lambda r: r['held_s'])
fastest = max([r for r in frozen_runs if r.get('implied_kmh_over_gap')], key=lambda r: r['implied_kmh_over_gap'])
i_min_lte = df.n_lteRSRP.idxmin()
i_min_asnr = df.n_lteSINR.idxmin()
op_changes = int((df.Operator != df.Operator.shift()).sum() - 1)
jump = df[is4 & ((df.n_Level - df.n_LTERSSI) >= 15)]         # 4G-label RSRP far above LTE RSSI
ev_bg = ev_gps[ev_gps.FOREGROUND == 'false']
nr_poor_total = int((is5 & (df.n_Level < -110)).sum())

anomalies = []
for c in sorted(cl_nr_rsrp, key=lambda c: -c['samples']):
    s = win(c['start'], c['end']); s5 = s[s.NetworkTech == '5G']; e = ev_win(c['start'], c['end'])
    anomalies.append(dict(
        when='%s to %s' % (c['start'], c['end']) if c['start'] != c['end'] else c['start'],
        what='Poor NR SS-RSRP (< -110 dBm) in %d of the %d 5G-labelled records of this interval (%d of all %d poor NR SS-RSRP samples); observation only, cause not determined' % (
            c['samples'], len(s5), c['samples'], nr_poor_total),
        evidence='NR Level min/median %.0f/%.0f dBm; NR SNR min/median %.0f/%.1f dB, <0 in %d; LTE anchor RSRP %.0f..%.0f dBm, anchor SNR %.0f..%.0f dB (<0 in %d); cells %s; events in interval %d%s; valid GPS %d/%d poor samples, lon/lat %s -> %s' % (
            s5.n_Level.min(), s5.n_Level.median(), s5.n_SNR.min(), s5.n_SNR.median(), int((s5.n_SNR < 0).sum()),
            s5.n_SecondCell_RSRP.min(), s5.n_SecondCell_RSRP.max(), s5.n_SecondCell_SNR.min(), s5.n_SecondCell_SNR.max(),
            int((s5.n_SecondCell_SNR < 0).sum()), ','.join(c['nodes']), len(e),
            (' (' + ', '.join('%s x%d' % (k, v) for k, v in e.EventType.value_counts().items()) + ')') if len(e) else '',
            c['gps_valid_samples'], c['samples'], c['start_lon_lat'], c['end_lon_lat'])))
anomalies.append(dict(
    when='%s to %s' % (fr0['start'], fr0['end']),
    what='Position frozen for %.0f s (%d records, logged Speed %d-%d km/h), then a %.0f m jump in %.0f s: the fix was stale (the handset moved without position updates)' % (
        fr0['held_s'], fr0['records'], fr0['logged_speed_kmh_min'], fr0['logged_speed_kmh_max'], fr0['jump_m'], fr0['gap_to_next_fix_s']),
    evidence='Longitude/Latitude constant %.6f/%.6f; next fix %s implies %.1f km/h over the gap (max logged Speed %d km/h); FOREGROUND=false in %d of %d records; %d mobility events in the interval cannot be located' % (
        fr0['lon'], fr0['lat'], fr0['next_fix'], fr0['implied_kmh_over_gap'], max_logged_speed,
        fr0['foreground_false_records'], fr0['records'], len(ev_win(fr0['start'], fr0['end'])))))
if fastest is not fr0:
    anomalies.append(dict(
        when='%s to %s' % (fastest['start'], fastest['next_fix']),
        what='Second stale-position interval ending in an impossible jump',
        evidence='coordinates constant for %.0f s (%d records), then %.0f m in %.0f s = %.1f km/h while logged Speed is %d-%d km/h; in total %d of %d frozen runs end faster than the maximum logged Speed' % (
            fastest['held_s'], fastest['records'], fastest['jump_m'], fastest['gap_to_next_fix_s'], fastest['implied_kmh_over_gap'],
            fastest['logged_speed_kmh_min'], fastest['logged_speed_kmh_max'], len(frozen_impossible), len(frozen_runs))))
anomalies.append(dict(
    when=df.Timestamp[i_min_lte],
    what='Lowest LTE RSRP of the log, in consecutive idle reselection records whose LTE RSSI contradicts the RSRP (poor but valid by the schema; kept)',
    evidence='; '.join(rowtxt(i) for i in range(max(i_min_lte - 1, 0), i_min_lte + 1)) + '; GPS valid: %s; FOREGROUND=%s' % (bool(df.gps_ok[i_min_lte]), df.FOREGROUND[i_min_lte])))
if len(jump):
    anomalies.append(dict(
        when='%s to %s' % (jump.Timestamp.iloc[0], jump.Timestamp.iloc[-1]),
        what='4G-labelled records whose RSRP exceeds the LTE RSSI by >= 15 dB after an abrupt step (inconsistent pair; candidate stale or mis-assigned value)',
        evidence='%d records; previous record: %s; then %s' % (len(jump), rowtxt(jump.index[0] - 1), rowtxt(jump.index[0]))))
anomalies.append(dict(
    when=df.Timestamp[i_min_asnr],
    what='Lowest LTE SINR of the log (anchor) at non-poor anchor RSRP; %d of %d poor LTE SINR samples have RSRP >= -110 dBm (observation; interference, load or stale readings are hypotheses only)' % (
        rel['LTE_SINR_poor_with_RSRP_not_poor']['n'], rel['LTE_SINR_poor_with_RSRP_not_poor']['of_poor_SINR_n']),
    evidence=rowtxt(i_min_asnr) + '; anchor SNR in the next distinct reading: %s dB' % df.SecondCell_SNR[i_min_asnr + 2]))
anomalies.append(dict(
    when='whole log, at 4G<->5G label switches',
    what='Label-transition artefacts: 5G-labelled records with NR SINR missing whose Level/Qual equal LTE values, and 4G-labelled records whose Level/Qual equal the NR values of the next 5G record',
    evidence='5G label, NR SINR missing: %d records at %s (%d copy the anchor pair of the same record, %d copies the next 4G record); 4G label carrying the next record NR Level/Qual: %d records (%d differ from the anchor RSRP by >= 3 dB) at %s' % (
        int(nr_sinr_missing.sum()), ', '.join(df.Timestamp[nr_sinr_missing].str[11:]), int(copy_anchor.sum()), int(copy_next4g.sum()),
        int(lte_art.sum()), int(lte_art_big.sum()), ', '.join(df.Timestamp[lte_art].str[11:]))))
anomalies.append(dict(
    when='whole log',
    what='Serving-cell operator code alternates between two PLMN codes under one Operatorname (observation; network sharing or roaming is a hypothesis the files cannot confirm)',
    evidence='Operatorname=orange in %d/%d records; Operator=%s; %d code changes between consecutive records; %d of %d mobility events cross the two codes; both codes occur under both RAT labels (%s)' % (
        int((df.Operatorname == 'orange').sum()), N, ', '.join('%s x%d' % (k, v) for k, v in df.Operator.value_counts().items()),
        op_changes, int(cross_plmn.sum()), len(ev), json.dumps(shares['operator_code_by_rat_label']).replace('"', ''))))
anomalies.append(dict(
    when='%s to %s' % (ev_bg.Timestamp.iloc[0], ev_bg.Timestamp.iloc[-1]),
    what='Frequent cell changes while the application was in background, with back-and-forth reversals; logged Speed 0 in this period is contradicted by the position jumps, so neither a stationary ping-pong nor normal mobility can be concluded',
    evidence='%d of %d events have FOREGROUND=false (%d with logged Speed 0, %d without valid GPS); whole log: %d A->B then B->A reversals within 60 s, %d events change only the PCI of an unchanged node-cell and %d show a new node with the old PCI (identity-update artefacts); %d of %d background records carry an event' % (
        len(ev_bg), len(ev), int((ev_bg.Speed == '0').sum()), int((ev_bg.gps_ok == False).sum()), pp,
        int(same_cell_pci_only.sum()), int(newnode_oldpci.sum()), sampling['foreground_false_records_with_event'], sampling['foreground_false_records'])))
anomalies.append(dict(
    when='whole log, 5G-labelled records',
    what='NR SS-SINR equals exactly 0.0 dB more often than any other value (candidate floor/placeholder or repeated stale value; counted as acceptable, not poor, because the criterion is < 0)',
    evidence='SNR=0.0 in %d of %d valid NR SINR samples (%.1f%%), in %d separate runs of consecutive 5G-labelled records, versus %d samples at -1 dB and %d at 1 dB' % (
        rel['NR_SINR_exactly_0']['n'], kpi['NR_SS_SINR']['valid_n'], rel['NR_SINR_exactly_0']['pct_of_valid'],
        nr_sinr_zero_runs, nr_snr_hist.get('-1', 0), nr_snr_hist.get('1', 0))))

# ----------------------------------------------------------------- data-quality findings
dq = [
    'Main log: %d records, %d base columns used; rows carry %d-%d tab fields; %d rows exceed the %d-column header; rows parsed by position, neighbour fields not analysed.' % (N, NB, min(field_counts), max(field_counts), over_header, len(hdr)),
    'Timestamps: %d/%d match YYYY.MM.DD_HH.MM.SS, non-decreasing=%s, %d distinct seconds; %d records share a second with another record; sampling is irregular (median interval %.0f s, max %.0f s, %d intervals > %g s covering %.0f s = %.1f%% of the duration).' % (
        int(ts_ok.sum()), N, bool(df.ts.is_monotonic_increasing), int(df.Timestamp.nunique()), int(df.Timestamp.duplicated(keep=False).sum()),
        dt_prev.median(), dt_prev.max(), int((w > GAP_S).sum()), GAP_S, gap_time, 100.0 * gap_time / duration_s),
    'Units/ranges: valid values lie inside plausible reporting ranges - NR SS-RSRP %.0f..%.0f dBm, SS-RSRQ %.0f..%.0f dB, SS-SINR %.0f..%.0f dB; LTE RSRP %.0f..%.0f dBm, RSRQ %.0f..%.0f dB, SINR %.0f..%.0f dB; no Level=-201 code; non-numeric values other than "-"/empty: %d.' % (
        kpi['NR_SS_RSRP']['min'], kpi['NR_SS_RSRP']['max'], kpi['NR_SS_RSRQ']['min'], kpi['NR_SS_RSRQ']['max'], kpi['NR_SS_SINR']['min'], kpi['NR_SS_SINR']['max'],
        kpi['LTE_RSRP']['min'], kpi['LTE_RSRP']['max'], kpi['LTE_RSRQ']['min'], kpi['LTE_RSRQ']['max'], kpi['LTE_SINR']['min'], kpi['LTE_SINR']['max'],
        sum(nonnumeric(df[c]) for c in ['Level', 'Qual', 'SNR', 'SecondCell_RSRP', 'SecondCell_RSRQ', 'SecondCell_SNR'])),
    'Missing values: NR SS-SINR "-" in %d of %d 5G-labelled records; LTE anchor fields empty in %d 5G-labelled record(s); no other KPI value missing.' % (
        counts['NR_SS_SINR']['excl_missing'], int(is5.sum()), int((is5 & ~has_anchor).sum())),
    'Suspected label-transition artefacts kept in the main statistics (schema-literal): %d 5G-labelled records whose Level/Qual duplicate LTE values (NR SINR missing) and %d 4G-labelled records whose Level/Qual duplicate the next NR values; excluding them changes the NR SS-RSRP mean from %.3f to %.3f dBm and the LTE RSRP mean from %.3f to %.3f dBm.' % (
        int(nr_art.sum()), int(lte_art.sum()), kpi['NR_SS_RSRP']['mean'], sens['NR_SS_RSRP']['mean'], kpi['LTE_RSRP']['mean'], sens['LTE_RSRP']['mean']),
    'No NR identifiers are logged: ARFCN (%s) and BAND (%s) are LTE values in all records, also in 5G-labelled ones, where Node/CellID/PSC/ARFCN/BAND equal the SecondCell (anchor) fields in %d of %d records with anchor data; NR band is unknown for all %d 5G-labelled records.' % (
        ', '.join(sorted(df.ARFCN.unique())), ', '.join(sorted(df.BAND.unique())), int(ident_same.sum()), int(has_anchor.sum()), int(is5.sum())),
    'NRSTATE, CSI_RSRP, CSI_RSRQ and CSI_SNR are empty in all %d records; NetworkTypeNum=%s in all records; SA cannot be inferred and the 5G label is treated as NSA only because the schema note says so.' % (N, '/'.join(sorted(df.NetworkTypeNum.unique()))),
    'No active test was run: Test_Status, PINGAVG/MIN/MAX, TESTDOWNLINK and TESTUPLINK are empty in all records, so throughput and RTT are N/A; DL_bitrate/UL_bitrate (non-zero in %d/%d records) are passive traffic counters.' % (passive['dl_gt0_records'], passive['ul_gt0_records']),
    'GPS: coordinates numeric and in range in %d/%d records, Accuracy <= %.0f m; %d records repeat an identical fix for >= %g s (%d frozen runs, %.0f s) and are excluded from spatial outputs only; %d of these runs end with a jump faster than the maximum logged Speed (%d km/h), i.e. the position was stale.' % (
        int((~gps_missing & ~gps_range).sum()), N, df.n_Accuracy.max(), int(gps_frozen.sum()), FROZEN_S, len(frozen_runs), frozen_time_s, len(frozen_impossible), max_logged_speed),
    'Speed unit check: on %d well-sampled steps GPS-derived speed is %.1f km/h against %.1f km/h logged, consistent with km/h.' % (int(okstep.sum()), gps_kmh, log_kmh),
    'Operator code is %s although Operatorname is "orange" in every record.' % ' and '.join('%s in %d records' % (k, v) for k, v in df.Operator.value_counts().items()),
    'Readings are refreshed more slowly than records are written: %.1f%% of consecutive 5G-labelled record pairs and %.1f%% of consecutive 4G-labelled pairs repeat the identical Level/Qual/SNR triplet, so samples are not independent.' % (
        stale['NR_triplet_5G_label']['pct'], stale['LTE_triplet_4G_label']['pct']),
    'LTERSSI is not consistent with RSRP: RSRP >= LTERSSI in %d of %d 4G-labelled records (%.1f%%); LTERSSI is not used in any statistic.' % (rssi['rsrp_ge_rssi_n'], rssi['lte_rssi_valid_n'], rssi['rsrp_ge_rssi_pct']),
    'Events file: %d events, identical to the EVENT/EVENTDETAILS column of the main log=%s; every source and target ARFCN is an LTE EARFCN, so IRAT/5G event names reflect the application label, not NR cell changes; %d events change only the PCI of an unchanged node-cell.' % (
        len(ev), ev_match, int(same_cell_pci_only.sum())),
    'Sampling is uneven: %d records (%.1f%%) were written in the first %.0f s (%.1f%% of the duration) before the application went to background at %s; afterwards %d of %d background records coincide with a cell-change event, so sample statistics over-represent the first part of the route and cell-change instants.' % (
        sampling['records_before'], sampling['records_before_pct'], sampling['seconds_before'], sampling['seconds_before_pct'], sampling['first_background_record'][11:],
        sampling['foreground_false_records_with_event'], sampling['foreground_false_records']),
    'Altitude equals Height in %d/%d records; BATTERY constant (%s); TA empty in %d records; IMEI/IMSI/MSISDN/Cellname empty.' % (
        int((df.Altitude == df.Height).sum()), N, '/'.join(sorted(df.BATTERY.unique())), int(df.TA.str.strip().isin(MISSING).sum())),
]

# ----------------------------------------------------------------- results.json (contract layout)
keep = ['valid_n', 'mean', 'median', 'sd', 'p5', 'p95', 'poor_pct']
results = {
    'records': int(N),
    'duration_s': duration_s,
    'rat_label_records': {'4G': int(rat_n.get('4G', 0)), '5G': int(rat_n.get('5G', 0))},
    'serving_cell_operator_code_records': {str(k): int(v) for k, v in df.Operator.value_counts().sort_index().items()},
    'kpi': {k: {f: kpi[k][f] for f in keep} for k in ['NR_SS_RSRP', 'NR_SS_RSRQ', 'NR_SS_SINR', 'LTE_RSRP', 'LTE_RSRQ', 'LTE_SINR']},
    'active_tests': active,
    'mobility': {'events_total': int(len(ev)), 'events_by_type': {str(k): int(v) for k, v in ev_by_type.items()}},
    'data_quality_findings': dq,
    'anomaly_candidates': anomalies,
    'model_self_report': 'Opus 5.5 (model ID: claude-opus-5-5)',
}
with open(os.path.join(OUT, 'results.json'), 'w', encoding='utf-8') as fh:
    json.dump(results, fh, indent=1, ensure_ascii=False)

supplement = dict(
    choices=dict(percentile_method='linear interpolation (numpy default, Hyndman-Fan type 7)', sd='sample, ddof=1',
                 poor_rule='RSRP < -110 dBm; SINR < 0 dB (strict); good RSRP > -85, SINR > 10; acceptable = remainder (bounds inclusive)',
                 gap_rule='forward-hold; intervals > %g s unobserved' % GAP_S,
                 gps_rule='invalid for spatial outputs if missing, 0/0 or out of range, Accuracy > %g m, or identical coordinates repeated for >= %g s (first record of a frozen run kept)' % (ACC_MAX_M, FROZEN_S),
                 plausibility_ranges=RANGE, rounding='3 decimals'),
    table_II=kpi, table_II_counts_and_exclusions=counts,
    table_II_sensitivity_excluding_label_transition_artefacts=sens,
    artefacts=dict(nr_5G_label_sinr_missing=df.Timestamp[nr_sinr_missing].tolist(), nr_copy_of_anchor=int(copy_anchor.sum()),
                   nr_copy_of_next_4G=int(copy_next4g.sum()), lte_4G_label_carrying_next_NR_pair=df.Timestamp[lte_art].tolist(),
                   lte_4G_label_records_with_anchor_fields_filled=df.Timestamp[anchor_in_4g].tolist()),
    shares=shares, active_tests=dict(active, other_fields_present=active_other, throughput='N/A', rtt='N/A'), passive_traffic=passive,
    mobility=dict(events_total=int(len(ev)), events_by_type=ev_by_type, **mobility_extra),
    spatial=spatial, relationships=rel, staleness=stale, lte_rssi_check=rssi,
    poor_clusters=dict(rule='poor samples joined when <= 30 s apart', NR_SS_RSRP=cl_nr_rsrp, NR_SS_SINR=cl_nr_sinr, LTE_RSRP=cl_lte_rsrp, LTE_SINR=cl_lte_sinr),
    sampling=sampling, nr_sinr_histogram=nr_snr_hist, ca_flag_by_rat_label=ca_by_label,
    file_structure=dict(header_columns=len(hdr), base_columns=NB, field_count_histogram={str(k): int(v) for k, v in field_counts.items()},
                        rows_longer_than_header=over_header, rows_shorter_than_base=short_rows))
with open(os.path.join(OUT, 'supplement.json'), 'w', encoding='utf-8') as fh:
    json.dump(supplement, fh, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


# ----------------------------------------------------------------- narrative.md (all figures taken from the objects above)
def line(k, unit, label):
    c, x = counts[k], kpi[k]
    poor = 'poor N/A' if x['poor_pct'] is None else 'poor %.2f%%' % x['poor_pct']
    return '- %s: %d/%d; %.3f, %.1f, %.3f, %.1f, %.1f %s; %s' % (label, c['valid_n'], c['raw_n'], x['mean'], x['median'], x['sd'], x['p5'], x['p95'], unit, poor)
c1 = cl_nr_rsrp[0] if cl_nr_rsrp[0]['samples'] >= cl_nr_rsrp[1]['samples'] else cl_nr_rsrp[1]
srt = sorted(cl_nr_rsrp, key=lambda c: -c['samples'])
c1, c2 = srt[0], srt[1]
w1 = win(c1['start'], c1['end']); w15 = w1[w1.NetworkTech == '5G']
hm = lambda t: t[11:].replace('.', ':')
ll = lambda p: '%.4f, %.4f' % (p[0], p[1])
tw = shares['time_weighting']
bs = shares['lte_serving_or_anchor_band_samples_all']
narrative = """**Scope, validation.** {N} records, {t0}-{t1} ({dur:.0f} s). Timestamps well-formed, non-decreasing; units plausible (dBm, dB, km/h); no -201 code. Missing: NR SS-SINR "-" in {m1} 5G-labelled records, anchor fields empty in {m2}; nothing else excluded.

**Table II** (valid/raw N; mean, median, sample SD, P5, P95 by linear interpolation; poor share):
{l1}
{l2}
{l3}
{l4}
{l5}
{l6}

Good/acceptable: NR RSRP {g1:.1f}/{a1:.1f}%, NR SINR {g2:.1f}/{a2:.1f}%, LTE RSRP {g3:.1f}/{a3:.1f}%, LTE SINR {g4:.1f}/{a4:.1f}%.

**Mode, band.** Sample shares: 5G label {s5:.1f}%, 4G {s4:.1f}%; LTE serving/anchor band L7 {b7:.1f}%, L20 {b20:.1f}%, L3 {b3:.1f}%; NR band unknown 100% (only LTE EARFCNs logged; n77/n78 unresolved). NRSTATE/CSI empty: SA not inferable; "NSA" rests on the schema note. Time shares (forward-hold; intervals >10 s unobserved): 5G {t5:.1f}%, 4G {t4:.1f}% of {obs:.0f} observed s; {unk:.1f}% of time unknown.

**Mobility.** {ne} events: {evlist}. Success rate N/A. {xp} cross operator codes 22610/22601; "IRAT/5G" names reflect labels, every event ARFCN being LTE. LTE fallback is not treated as coverage failure.

**Throughput/RTT.** N/A: no active test; DL/UL bitrate is passive.

**Observations.** {p1} of {pt} poor NR RSRP samples lie in {c1s}-{c1e} ({c1a} to {c1b}), with negative anchor SINR in {an} records and {e1} events; a second cluster at {c2s}-{c2e} ({c2a}). NR SINR is poor in {r1:.1f}% of poor-RSRP samples against {r2:.1f}% otherwise; {lq} of {lqt} poor LTE SINR samples have non-poor RSRP. GPS froze {f0}-{f1}, then jumped {fj:.0f} m in {fg:.0f} s; {gx} stale-fix records excluded from spatial outputs only.

**Hypotheses (unverified).** Cell-edge conditions, interference or load; network sharing behind the two operator codes; background logging restrictions. Stale readings likely: {st:.1f}% of consecutive 5G-labelled records repeat identical values.

**Assumptions, limitations.** Schema-literal mapping; {a_nr} + {a_lte} suspected label-transition artefacts retained (without them NR RSRP mean is {sm:.3f} dBm). {rb:.1f}% of records fall in the first {sb:.1f}% of the time; samples are not independent; one drive, one handset; no load, traffic-demand or server data. All figures require human verification.
""".format(
    N=N, t0=hm(df.Timestamp.iloc[0]), t1=hm(df.Timestamp.iloc[-1]), dur=duration_s,
    m1=counts['NR_SS_SINR']['excl_missing'], m2=counts['LTE_RSRP']['excl_missing'],
    l1=line('NR_SS_RSRP', 'dBm', 'NR SS-RSRP'), l2=line('NR_SS_RSRQ', 'dB', 'NR SS-RSRQ'), l3=line('NR_SS_SINR', 'dB', 'NR SS-SINR'),
    l4=line('LTE_RSRP', 'dBm', 'LTE RSRP (4G+anchor)'), l5=line('LTE_RSRQ', 'dB', 'LTE RSRQ'), l6=line('LTE_SINR', 'dB', 'LTE SINR'),
    g1=kpi['NR_SS_RSRP']['good_pct'], a1=kpi['NR_SS_RSRP']['acceptable_pct'], g2=kpi['NR_SS_SINR']['good_pct'], a2=kpi['NR_SS_SINR']['acceptable_pct'],
    g3=kpi['LTE_RSRP']['good_pct'], a3=kpi['LTE_RSRP']['acceptable_pct'], g4=kpi['LTE_SINR']['good_pct'], a4=kpi['LTE_SINR']['acceptable_pct'],
    s5=shares['rat_label_samples']['5G']['pct'], s4=shares['rat_label_samples']['4G']['pct'],
    b7=bs['L7']['pct'], b20=bs['L20']['pct'], b3=bs['L3']['pct'],
    t5=tw['rat_label']['5G']['pct_of_observed'], t4=tw['rat_label']['4G']['pct_of_observed'], obs=tw['observed_s'], unk=tw['unobserved_pct'],
    ne=len(ev), evlist=', '.join('%s %d' % (k, v) for k, v in ev_by_type.items()), xp=int(cross_plmn.sum()),
    p1=c1['samples'], pt=nr_poor_total, c1s=hm(c1['start']), c1e=hm(c1['end']), c1a=ll(c1['start_lon_lat']), c1b=ll(c1['end_lon_lat']),
    an=int((w15.n_SecondCell_SNR < 0).sum()), e1=len(ev_win(c1['start'], c1['end'])),
    c2s=hm(c2['start']), c2e=hm(c2['end']), c2a=ll(c2['start_lon_lat']),
    r1=rel['NR_SINR_poor_pct_when_RSRP_poor']['pct'], r2=rel['NR_SINR_poor_pct_when_RSRP_not_poor']['pct'], rho=rel['NR_RSRP_vs_NR_SINR']['spearman'],
    lq=rel['LTE_SINR_poor_with_RSRP_not_poor']['n'], lqt=rel['LTE_SINR_poor_with_RSRP_not_poor']['of_poor_SINR_n'],
    f0=hm(fr0['start']), f1=hm(fr0['end']), fj=fr0['jump_m'], fg=fr0['gap_to_next_fix_s'], gx=int(gps_frozen.sum()),
    st=stale['NR_triplet_5G_label']['pct'], a_nr=int(nr_art.sum()), a_lte=int(lte_art.sum()), sm=sens['NR_SS_RSRP']['mean'],
    rb=sampling['records_before_pct'], sb=sampling['seconds_before_pct'])
narrative_words = len(narrative.split())
assert narrative_words <= 400, narrative_words
with open(os.path.join(OUT, 'narrative.md'), 'w', encoding='utf-8') as fh:
    fh.write(narrative)

if __name__ == '__main__':
    print(json.dumps(results, indent=1, ensure_ascii=False))
    print('=' * 100)
    print(json.dumps({k: v for k, v in supplement.items() if k not in ('poor_clusters',)}, indent=1, ensure_ascii=False,
                     default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print('=' * 100)
    print(narrative); print('narrative words:', narrative_words)
    for k, v in supplement['poor_clusters'].items():
        print(k)
        for c in (v if isinstance(v, list) else []):
            print('   ', c)
