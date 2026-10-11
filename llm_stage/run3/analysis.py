#!/usr/bin/env python3
"""Drive-test analysis (G-NetTrack Pro export, operator name 'orange', 2026-10-02).

Reads only the two data files in /home/claude/ai_stage/input/ and writes
results.json and tables.md into the run folder. Every number quoted in
narrative.md is printed by this script (see the NARRATIVE NUMBERS block) or
written to tables.md.

Analyst choices (also stated in narrative.md / tables.md):
  * Field mapping strictly as in schema_note.md:
      NR_SS_*  = Level/Qual/SNR of records labelled 5G;
      LTE_*    = Level/Qual/SNR of records labelled 4G pooled with
                 SecondCell_RSRP/RSRQ/SNR of records labelled 5G.
  * Missing markers: empty string, '-', and Level == -201 (loss of coverage).
  * Plausibility ranges (anything outside is counted as invalid, none found):
      RSRP -156..-31 dBm, RSRQ -43..20 dB, SINR -30..40 dB.
  * Percentiles: linear interpolation between order statistics
    (Hyndman-Fan type 7 = numpy/pandas default). SD: sample SD (ddof = 1).
  * Table I bins: RSRP good > -85, acceptable -110..-85 inclusive, poor < -110;
    SINR good > 10, acceptable 0..10 inclusive, poor < 0.
  * Time shares: a record's state is held until the next record, credited for
    at most GAP_CAP_S seconds; the remainder of longer gaps is 'unobserved'.
  * GPS: a fix is 'frozen' when it belongs to a run of identical coordinates
    lasting >= 10 s that ends with a jump >= 100 m at an implied speed
    > 80 km/h (max logged Speed is 71 km/h); all records of the run after its
    first are flagged. Flagged records stay in KPI statistics and are excluded
    only from location statements.
"""
import csv
import itertools
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

IN = '/home/claude/ai_stage/input'
OUT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(IN, 'orange_2026.10.02_12.16.18.txt')
EVT = os.path.join(IN, 'orange_2026.10.02_12.16.18_events.txt')
GAP_CAP_S = 10
MODEL = 'claude-opus-5-5'

# ----------------------------------------------------------------- load
with open(LOG, encoding='utf-8', errors='replace', newline='') as fh:
    rows = list(csv.reader(fh, delimiter='\t', quoting=csv.QUOTE_NONE))
hdr, data = rows[0], rows[1:]
ncol = len(hdr)
row_len = Counter(len(r) for r in data)
longer_than_header = sum(1 for r in data if len(r) > ncol)
df = pd.DataFrame([r[:ncol] + [''] * (ncol - len(r)) for r in data], columns=hdr)
ev = pd.read_csv(EVT, sep='\t', dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)

N = len(df)
df['t'] = pd.to_datetime(df['Timestamp'], format='%Y.%m.%d_%H.%M.%S', errors='coerce')
ts_bad = int(df['t'].isna().sum())
ts_monotonic = bool(df['t'].is_monotonic_increasing)
sec = (df['t'] - df['t'].iloc[0]).dt.total_seconds().values
duration_s = float(sec[-1] - sec[0])
dt_next = np.r_[np.diff(sec), 0.0]            # time until the next record
dup_ts_records = int((np.diff(sec) == 0).sum())
exact_dup_rows = len(data) - len(set(tuple(r) for r in data))


def num(s):
    """numeric conversion with the documented missing markers"""
    s = s.astype(str).str.strip()
    return pd.to_numeric(s.where(~s.isin(['', '-'])), errors='coerce')


is5 = (df['NetworkTech'] == '5G').values
is4 = (df['NetworkTech'] == '4G').values
rat_counts = {k: int(v) for k, v in df['NetworkTech'].value_counts().items()}
op_counts = {str(k): int(v) for k, v in df['Operator'].value_counts().items()}

# ----------------------------------------------------------------- KPI
RANGES = {'RSRP': (-156, -31), 'RSRQ': (-43, 20), 'SINR': (-30, 40)}


def series_for(kind, which):
    col = {'RSRP': 'Level', 'RSRQ': 'Qual', 'SINR': 'SNR'}[kind]
    col2 = {'RSRP': 'SecondCell_RSRP', 'RSRQ': 'SecondCell_RSRQ', 'SINR': 'SecondCell_SNR'}[kind]
    if which == 'NR':
        raw = df.loc[is5, col]
    else:  # LTE: 4G-labelled serving values pooled with anchor values of 5G-labelled records
        raw = pd.concat([df.loc[is4, col], df.loc[is5, col2]])
    return raw


def kpi_stats(kind, which, mask_extra=None):
    raw = series_for(kind, which)
    if mask_extra is not None:
        raw = raw[~raw.index.isin(mask_extra)]
    v = num(raw)
    n_raw = int(len(raw))
    n_missing = int(v.isna().sum())
    lo, hi = RANGES[kind]
    code201 = int((v == -201).sum())
    out_of_range = int(((v < lo) | (v > hi)).sum())
    v = v[(v >= lo) & (v <= hi)]
    x = v.values.astype(float)
    res = dict(raw_n=n_raw, missing_n=n_missing, code_minus201_n=code201,
               out_of_range_n=out_of_range, valid_n=int(len(x)),
               mean=float(np.mean(x)), median=float(np.median(x)),
               sd=float(np.std(x, ddof=1)),
               p5=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)),
               min=float(x.min()), max=float(x.max()))
    if kind == 'RSRP':
        res.update(good_pct=100 * float((x > -85).mean()),
                   acceptable_pct=100 * float(((x >= -110) & (x <= -85)).mean()),
                   poor_n=int((x < -110).sum()), poor_pct=100 * float((x < -110).mean()))
    elif kind == 'SINR':
        res.update(good_pct=100 * float((x > 10).mean()),
                   acceptable_pct=100 * float(((x >= 0) & (x <= 10)).mean()),
                   poor_n=int((x < 0).sum()), poor_pct=100 * float((x < 0).mean()))
    else:
        res.update(good_pct=None, acceptable_pct=None, poor_n=None, poor_pct=None)
    return res


KPIS = [('NR_SS_RSRP', 'RSRP', 'NR'), ('NR_SS_RSRQ', 'RSRQ', 'NR'), ('NR_SS_SINR', 'SINR', 'NR'),
        ('LTE_RSRP', 'RSRP', 'LTE'), ('LTE_RSRQ', 'RSRQ', 'LTE'), ('LTE_SINR', 'SINR', 'LTE')]
full = {name: kpi_stats(kind, which) for name, kind, which in KPIS}

R = 6  # decimals in results.json
kpi_json = {}
for name, _, _ in KPIS:
    s = full[name]
    kpi_json[name] = {'valid_n': s['valid_n'], 'mean': round(s['mean'], R), 'median': round(s['median'], R),
                      'sd': round(s['sd'], R), 'p5': round(s['p5'], R), 'p95': round(s['p95'], R),
                      'poor_pct': None if s['poor_pct'] is None else round(s['poor_pct'], R)}

# ----------------------------------------------------------------- transition-record checks
lvl, qual, snr = df['Level'], df['Qual'], df['SNR']
a_rsrp, a_rsrq = df['SecondCell_RSRP'], df['SecondCell_RSRQ']
nr_sinr_missing = df.index[is5 & snr.isin(['', '-']).values]
nr_dup_anchor_all = df.index[is5 & (lvl == a_rsrp).values & (qual == a_rsrq).values]
nr_missing_and_dup = [i for i in nr_sinr_missing if i in set(nr_dup_anchor_all)]
nr_missing_eq_next4g = [i for i in nr_sinr_missing if i + 1 < N and is4[i + 1]
                        and lvl[i] == lvl[i + 1] and qual[i] == qual[i + 1]]
nr_suspect = sorted(set(nr_missing_and_dup) | set(nr_missing_eq_next4g))
t45 = [i for i in range(N - 1) if is4[i] and is5[i + 1]]
t54 = [i for i in range(N - 1) if is5[i] and is4[i + 1]]
lte_pre5g_equal_nr = [i for i in t45 if lvl[i] == lvl[i + 1] and qual[i] == qual[i + 1]]
lab_runs = [(k, len(list(g))) for k, g in itertools.groupby(df['NetworkTech'])]
anchor_missing_in_5g = int((is5 & (a_rsrp == '').values).sum())
anchor_present_in_4g = df.index[is4 & (a_rsrp != '').values].tolist()

# sensitivity: drop the suspect transition records
sens = {}
for name, kind, which in KPIS:
    drop = nr_suspect if which == 'NR' else lte_pre5g_equal_nr
    if which == 'NR' and kind == 'SINR':
        drop = []
    sens[name] = kpi_stats(kind, which, mask_extra=drop)

# ----------------------------------------------------------------- stale readings
def stale_share(mask, cols):
    key = df.loc[mask, cols].agg('|'.join, axis=1)
    idx = key.index
    pairs = [(i, i - 1) for i in idx if (i - 1) in idx]
    same = sum(1 for i, j in pairs if key[i] == key[j])
    return same, len(pairs), 100.0 * same / len(pairs), int((key != key.shift()).sum())


stale_nr = stale_share(is5, ['Level', 'Qual', 'SNR'])
stale_lte = stale_share(is4, ['Level', 'Qual', 'SNR'])
stale_anchor = stale_share(is5, ['SecondCell_RSRP', 'SecondCell_RSRQ', 'SecondCell_SNR'])

# ----------------------------------------------------------------- GPS
lon, lat = num(df['Longitude']).values, num(df['Latitude']).values
gps_format_bad = int((np.isnan(lon) | np.isnan(lat) | (lon == 0) | (lat == 0) |
                      (np.abs(lat) > 90) | (np.abs(lon) > 180)).sum())


def hav(lo1, la1, lo2, la2):
    p1, p2 = np.radians(la1), np.radians(la2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))


same_pos = np.r_[False, (lon[1:] == lon[:-1]) & (lat[1:] == lat[:-1])]
run_id = np.cumsum(~same_pos)
gps_frozen = np.zeros(N, bool)
frozen_runs = []
for rid in np.unique(run_id):
    ii = np.where(run_id == rid)[0]
    a, b = ii[0], ii[-1]
    if b + 1 >= N:
        continue
    dist = float(hav(lon[b], lat[b], lon[b + 1], lat[b + 1]))
    dtc = sec[b + 1] - sec[b]
    run_s = sec[b] - sec[a]
    v = dist / dtc * 3.6 if dtc > 0 else np.inf
    if run_s >= 10 and dist >= 100 and v > 80:
        gps_frozen[a + 1:b + 1] = True
        frozen_runs.append(dict(first=df['Timestamp'][a], last=df['Timestamp'][b], records=int(len(ii)),
                                held_s=float(run_s), exit_jump_m=round(dist, 1), exit_dt_s=float(dtc),
                                implied_kmh=round(float(v), 1)))
gps_frozen_n = int(gps_frozen.sum())
speed = num(df['Speed']).values
# unit check of Speed: single-fix steps of >= 1 s
step_d = hav(lon[:-1], lat[:-1], lon[1:], lat[1:])
step_dt = np.diff(sec)
ok = (step_dt >= 1) & (step_dt <= 3) & (~same_pos[1:]) & (~gps_frozen[1:]) & (~gps_frozen[:-1])
speed_gps = step_d[ok] / step_dt[ok] * 3.6
speed_med_diff = float(np.median(speed_gps - speed[1:][ok]))
path_m = float(step_d.sum())
acc = num(df['Accuracy']).values

# ----------------------------------------------------------------- shares
def sample_share(col):
    vc = df[col].replace({'': 'unknown', '-': 'unknown'}).value_counts()
    return {str(k): (int(v), 100.0 * v / N) for k, v in vc.items()}


w = np.minimum(dt_next, GAP_CAP_S)
observed_s = float(w.sum())
unobserved_s = duration_s - observed_s
gaps_over_cap = int((dt_next > GAP_CAP_S).sum())


def time_share(col):
    s = pd.Series(w).groupby(df[col].replace({'': 'unknown', '-': 'unknown'}).values).sum()
    return {str(k): (float(v), 100.0 * v / observed_s, 100.0 * v / duration_s) for k, v in s.items()}


band_s, band_t = sample_share('BAND'), time_share('BAND')
rat_s, rat_t = sample_share('NetworkTech'), time_share('NetworkTech')
op_s, op_t = sample_share('Operator'), time_share('Operator')
state_s = sample_share('State')
bw_has100 = df['BANDWIDTH'].str.split('/').map(lambda v: '100' in v).values
bw_has15 = df['BANDWIDTH'].str.split('/').map(lambda v: '15' in v).values
band_by_rat = pd.crosstab(df['NetworkTech'], df['BAND'])
op_by_rat = pd.crosstab(df['NetworkTech'], df['Operator'])
nr_band_known = 0   # no NR-ARFCN / NR band field is populated in the export
nrstate_filled = int((df['NRSTATE'] != '').sum())
csi_filled = int(((df['CSI_RSRP'] != '') | (df['CSI_RSRQ'] != '') | (df['CSI_SNR'] != '')).sum())
nettypenum = {str(k): int(v) for k, v in df['NetworkTypeNum'].value_counts().items()}
anchor_arfcn_equal = int((is5 & (df['ARFCN'] == df['SecondCell_ARFCN']).values).sum())

# ----------------------------------------------------------------- active tests / passive traffic
test_cols = ['Test_Status', 'PINGAVG', 'PINGMIN', 'PINGMAX', 'PINGSTDEV', 'PINGLOSS',
             'TESTDOWNLINK', 'TESTUPLINK', 'TESTDOWNLINKMAX', 'TESTUPLINKMAX']
test_filled = {c: int(num(df[c]).notna().sum()) if c != 'Test_Status' else int((df[c] != '').sum())
               for c in test_cols}
download_n, upload_n, ping_n = test_filled['TESTDOWNLINK'], test_filled['TESTUPLINK'], test_filled['PINGAVG']
dl, ul = num(df['DL_bitrate']).values, num(df['UL_bitrate']).values

# ----------------------------------------------------------------- mobility
ev_total = int(len(ev))
ev_by_type = {str(k): int(v) for k, v in ev['EventType'].value_counts().items()}
log_ev = df[(df['EVENT'] != '') & (df['EVENT'] != 'LOG START')]
ev_match_log = bool(len(log_ev) == ev_total
                    and (log_ev['Timestamp'].values == ev['Timestamp'].values).all()
                    and (log_ev['EVENT'].values == ev['EventType'].values).all()
                    and (log_ev['EVENTDETAILS'].values == ev['EventDetails'].values).all())
sd = ev['EventDetails'].str.split(':', expand=True)
S = sd[0].str.split('-', expand=True)
T = sd[1].str.split('-', expand=True)
S.columns = T.columns = ['mcc', 'mnc', 'tac', 'node', 'cell', 'arfcn', 'pci']
ev_plmn_change = (S['mnc'] != T['mnc']).values
ev_same_cell = ((S['mnc'] == T['mnc']) & (S['node'] == T['node']) & (S['cell'] == T['cell'])).values
ev_irat = ev['EventType'].str.startswith('IRAT').values
ev_5g5g = ev['EventType'].str.endswith('5G5G').values
ev_4g4g = ev['EventType'].str.endswith('4G4G').values
ev_t = pd.to_datetime(ev['Timestamp'], format='%Y.%m.%d_%H.%M.%S')
ck = lambda d: d['mnc'] + '-' + d['node'] + '-' + d['cell']
sk, tk = ck(S).values, ck(T).values
pingpong = [(ev['Timestamp'][i - 1], ev['Timestamp'][i], float((ev_t[i] - ev_t[i - 1]).total_seconds()))
            for i in range(1, ev_total) if tk[i] == sk[i - 1] and sk[i] == tk[i - 1] and sk[i] != tk[i]]
pingpong_le10 = sum(1 for p in pingpong if p[2] <= 10)
ev_frozen_gps = int(gps_frozen[log_ev.index].sum())
ev_rate_per_min = ev_total / (duration_s / 60.0)
# identity consistency of (PLMN, node, cell) in the main log
ident = df.groupby(['Operator', 'Node', 'CellID']).agg(arfcn=('ARFCN', lambda x: sorted(set(x))),
                                                       pci=('PSC', lambda x: sorted(set(x))))
ident_multi = ident[(ident['arfcn'].map(len) > 1) | (ident['pci'].map(len) > 1)]
n_cells = int(len(ident))
n_nodes = int(df[['Operator', 'Node']].drop_duplicates().shape[0])

# ----------------------------------------------------------------- relationships
nr_rsrp, nr_sinr = num(lvl.where(is5)), num(snr.where(is5))
an_rsrp, an_sinr = num(a_rsrp.where(is5)), num(df['SecondCell_SNR'].where(is5))


def corr(x, y):
    m = x.notna() & y.notna()
    return (float(np.corrcoef(x[m], y[m])[0, 1]), float(x[m].rank().corr(y[m].rank())), int(m.sum()))


c_nr = corr(nr_rsrp, nr_sinr)
c_an = corr(an_rsrp, an_sinr)
c_nr_an = corr(nr_rsrp, an_rsrp)
m = nr_rsrp.notna() & an_rsrp.notna()
nr_minus_anchor = float((nr_rsrp[m] - an_rsrp[m]).mean())
m2 = nr_rsrp.notna() & nr_sinr.notna()
poor_rsrp_mask = (nr_rsrp < -110) & m2
poor_both = int(((nr_sinr < 0) & poor_rsrp_mask).sum())
poor_rsrp_n_paired = int(poor_rsrp_mask.sum())
# poor NR samples by serving-cell operator code
nr_poor_by_op = pd.crosstab(df.loc[is5, 'Operator'], (nr_rsrp[is5] < -110))
# LTE RSSI plausibility (4G records)
rssi = num(df['LTERSSI'].where(is4))
lte_rsrp4 = num(lvl.where(is4))
rssi_valid = int(rssi.notna().sum())
rssi_le_rsrp = int((lte_rsrp4 >= rssi).sum())
rssi_all_odd = bool(((rssi.dropna() % 2) != 0).all())

# ----------------------------------------------------------------- anomaly windows
def pos(i):
    if gps_frozen[i]:
        return 'position frozen (not usable)'
    return f'{lat[i]:.5f}N {lon[i]:.5f}E'


def window(t0, t1):
    return df.index[(df['Timestamp'] >= t0) & (df['Timestamp'] <= t1)]


def wstats(t0, t1):
    ii = window(t0, t1)
    r, s = nr_rsrp[ii].dropna(), nr_sinr[ii].dropna()
    e = log_ev[(log_ev['Timestamp'] >= t0) & (log_ev['Timestamp'] <= t1)]
    e_plmn = int(ev_plmn_change[[list(log_ev.index).index(k) for k in e.index]].sum()) if len(e) else 0
    return dict(records=int(len(ii)), nr_n=int(len(r)), nr_rsrp_min=float(r.min()), nr_rsrp_median=float(r.median()),
                nr_rsrp_poor=int((r < -110).sum()), nr_sinr_min=float(s.min()), nr_sinr_median=float(s.median()),
                nr_sinr_poor=int((s < 0).sum()), anchor_rsrp_min=float(an_rsrp[ii].min()),
                anchor_sinr_min=float(an_sinr[ii].min()), events=int(len(e)), events_plmn_change=e_plmn,
                start_pos=pos(ii[0]), end_pos=pos(ii[-1]), dl_max_kbps=float(dl[ii].max()), ul_max_kbps=float(ul[ii].max()))


# runs of consecutive 5G-labelled records with poor NR RSRP or poor NR SINR
poor_any = ((nr_rsrp < -110) | (nr_sinr < 0)).fillna(False).values & is5
clusters, i = [], 0
while i < N:
    if poor_any[i]:
        j = i
        while j + 1 < N and (poor_any[j + 1] or (is5[j + 1] and sec[j + 1] - sec[j] <= 2 and j + 2 < N and poor_any[j + 2])):
            j += 1
        clusters.append((i, j, int(poor_any[i:j + 1].sum())))
        i = j + 1
    else:
        i += 1
clusters = sorted(clusters, key=lambda c: -c[2])

W1 = ('2026.10.02_12.19.13', '2026.10.02_12.20.29')
W2 = ('2026.10.02_12.34.03', '2026.10.02_12.34.18')
w1, w2 = wstats(*W1), wstats(*W2)
w1_idx, w2_idx = window(*W1), window(*W2)
w1_speed = (speed[w1_idx[0]], speed[w1_idx[-1]])
w2_4g_records = int(is4[w2_idx].sum())
gap_open_bg = int(((dt_next > GAP_CAP_S) & (df['FOREGROUND'] == 'false').values).sum())
ident_when = []
for k, v in ident_multi.iterrows():
    sub = df[(df['Operator'] == k[0]) & (df['Node'] == k[1]) & (df['CellID'] == k[2])]
    col = 'PSC' if len(v['pci']) > 1 else 'ARFCN'
    minority = sub[col].value_counts().index[-1]
    ident_when.append(f"{k[0]}-{k[1]}-{k[2]} {col}={minority}: " + ', '.join(sub.loc[sub[col] == minority, 'Timestamp'].str[11:].unique()))
i_minanchor = int(an_sinr.idxmin())
i_last = N - 1
rssi_odd_rows = df.index[is4 & (num(df['LTERSSI']) >= -60).values & (num(lvl) <= -110).values].tolist()

anomalies = [
    {'when': f'{W1[0]} to {W1[1]}',
     'what': 'Sustained weak NR SS-RSRP/SS-SINR while the LTE anchor is also weak and the serving cell alternates '
             'between operator codes 22610 and 22601 (observation; cause not established).',
     'evidence': f"{w1['nr_n']} NR samples: SS-RSRP median {w1['nr_rsrp_median']:.0f} dBm, min {w1['nr_rsrp_min']:.0f}; "
                 f"{w1['nr_rsrp_poor']} below -110 dBm; SS-SINR median {w1['nr_sinr_median']:.0f} dB, min {w1['nr_sinr_min']:.0f}; "
                 f"anchor RSRP min {w1['anchor_rsrp_min']:.0f} dBm, anchor SNR min {w1['anchor_sinr_min']:.0f} dB "
                 f"(at {df['Timestamp'][i_minanchor]}); {w1['events']} mobility events, {w1['events_plmn_change']} between operator codes; "
                 f"from {w1['start_pos']} to {w1['end_pos']}; Speed {w1_speed[0]:.0f} -> {w1_speed[1]:.0f} km/h."},
    {'when': f'{W2[0]} to {W2[1]}',
     'what': 'NR quality drops after the serving cell changes to 22601 node 947902 cell 7; label then falls back to 4G '
             f'for {w2_4g_records} records and returns to 5G (observation).',
     'evidence': f"{w2['nr_n']} NR samples: SS-RSRP min {w2['nr_rsrp_min']:.0f} dBm ({w2['nr_rsrp_poor']} below -110), "
                 f"SS-SINR min {w2['nr_sinr_min']:.0f} dB ({w2['nr_sinr_poor']} below 0); anchor RSRP min {w2['anchor_rsrp_min']:.0f} dBm; "
                 f"passive UL_bitrate up to {w2['ul_max_kbps']:.0f} kbps; from {w2['start_pos']} to {w2['end_pos']}."},
    {'when': ', '.join(df['Timestamp'][k] for k in nr_sinr_missing),
     'what': 'Records labelled 5G with missing SNR whose Level/Qual duplicate LTE values (label/field race at the '
             '5G-to-4G label change); their Level/Qual are probably LTE, not NR (hypothesis).',
     'evidence': f"{len(nr_sinr_missing)} records with SNR='-'; in {len(nr_missing_and_dup)} Level==SecondCell_RSRP and Qual==SecondCell_RSRQ; "
                 f"in the remaining {len(nr_sinr_missing) - len(nr_missing_and_dup)} SecondCell is empty and Level/Qual equal the next 4G record; "
                 f"no other 5G record has both equalities ({len(nr_dup_anchor_all)} in total)."},
    {'when': ', '.join(df['Timestamp'][k] for k in lte_pre5g_equal_nr),
     'what': 'Last 4G-labelled record before a 5G label already carries the Level/Qual of the following 5G record '
             '(possible NR values under a 4G label, or stale LTE values under a 5G label; undecidable from the log).',
     'evidence': f"{len(lte_pre5g_equal_nr)} of {len(t45)} 4G-to-5G label changes; one 4G record ({', '.join(df['Timestamp'][k] for k in anchor_present_in_4g)}) "
                 f"even has SecondCell_* filled."},
    {'when': '; '.join(f"{r['first']} to {r['last']}" for r in frozen_runs),
     'what': 'GPS position frozen while cell changes continue; fix resumes with a jump that implies an impossible speed.',
     'evidence': '; '.join(f"held {r['held_s']:.0f} s then {r['exit_jump_m']:.0f} m in {r['exit_dt_s']:.0f} s = {r['implied_kmh']:.0f} km/h"
                           for r in frozen_runs) + f"; {gps_frozen_n} records flagged; FOREGROUND=false in "
                 f"{int((df['FOREGROUND'][gps_frozen] == 'false').sum())} of them."},
    {'when': f"{df['Timestamp'].iloc[0]} to {df['Timestamp'].iloc[-1]}",
     'what': "Serving-cell operator code alternates between 22610 and 22601 although Operatorname is always 'orange'.",
     'evidence': f"22610 in {op_counts.get('22610', 0)} records, 22601 in {op_counts.get('22601', 0)}; "
                 f"{int(ev_plmn_change.sum())} of {ev_total} mobility events change the operator code; "
                 f"{len(pingpong)} immediate A-B-A returns, {pingpong_le10} within 10 s."},
    {'when': ', '.join(df['Timestamp'][k] for k in rssi_odd_rows),
     'what': 'Very low LTE RSRP (including the log minimum) while LTERSSI is at its log maximum and SNR is positive: internally inconsistent '
             'reading, candidate stale or mis-reported field (position frozen at the time).',
     'evidence': '; '.join(f"Level {lvl[k]} dBm, LTERSSI {df['LTERSSI'][k]} dBm, SNR {snr[k]} dB, node {df['Node'][k]} cell {df['CellID'][k]}"
                           for k in rssi_odd_rows)},
    {'when': '; '.join(ident_when),
     'what': 'Cell identity fields change asynchronously: same (operator code, node, cell) appears with two EARFCNs or two PCIs.',
     'evidence': '; '.join(f"{k[0]}-{k[1]}-{k[2]}: ARFCN {v['arfcn']}, PCI {v['pci']}" for k, v in ident_multi.iterrows())
                 + f"; {int(ev_same_cell.sum())} events have identical source and target cell identity."},
    {'when': df['Timestamp'][i_last],
     'what': 'Lowest NR SS-RSRP of the log in the final record while the LTE anchor is good and traffic is flowing.',
     'evidence': f"Level {lvl[i_last]} dBm, Qual {qual[i_last]} dB, SNR {snr[i_last]} dB; anchor RSRP {a_rsrp[i_last]} dBm, "
                 f"anchor SNR {df['SecondCell_SNR'][i_last]} dB; DL_bitrate {df['DL_bitrate'][i_last]} kbps; {pos(i_last)}."},
]

dq = [
    f"{N} data records, header has {ncol} columns; rows carry {min(row_len)} to {max(row_len)} fields, none longer than the header.",
    f"Timestamps: all {N} parse as YYYY.MM.DD_HH.MM.SS, non-decreasing, 1 s resolution; {dup_ts_records} records share the timestamp of the previous record; no exact duplicate rows.",
    f"Sampling is irregular: {gaps_over_cap} inter-record gaps exceed {GAP_CAP_S} s (longest {dt_next.max():.0f} s), {gap_open_bg} of them opened by a record with FOREGROUND=false ({int((df['FOREGROUND'] == 'false').sum())} such records).",
    f"Level and Qual are numeric in all {N} records; no -201 loss-of-coverage code and no out-of-range RSRP/RSRQ/SINR value.",
    f"SNR is '-' in {int(snr.isin(['', '-']).sum())} records, all labelled 5G (NR_SS_SINR valid N {full['NR_SS_SINR']['valid_n']} of {full['NR_SS_SINR']['raw_n']}).",
    f"SecondCell_* (LTE anchor) is empty in {anchor_missing_in_5g} record labelled 5G (LTE KPIs valid N {full['LTE_RSRP']['valid_n']} of {full['LTE_RSRP']['raw_n']}) and filled in {len(anchor_present_in_4g)} record labelled 4G (not pooled, to avoid double counting).",
    f"The {len(nr_sinr_missing)} 5G records with missing SNR duplicate LTE values in Level/Qual; kept in NR_SS_RSRP/RSRQ per schema; excluding them moves the NR_SS_RSRP mean from {full['NR_SS_RSRP']['mean']:.2f} to {sens['NR_SS_RSRP']['mean']:.2f} dBm and poor share from {full['NR_SS_RSRP']['poor_pct']:.2f} to {sens['NR_SS_RSRP']['poor_pct']:.2f} %.",
    f"{len(lte_pre5g_equal_nr)} of {len(t45)} 4G-to-5G label changes show the last 4G record with the Level/Qual of the next 5G record; excluding them moves the LTE_RSRP mean from {full['LTE_RSRP']['mean']:.2f} to {sens['LTE_RSRP']['mean']:.2f} dBm.",
    f"Readings are refreshed more slowly than records are written: {stale_nr[2]:.1f} % of consecutive 5G records repeat the same Level/Qual/SNR triple ({stale_lte[2]:.1f} % for 4G records, {stale_anchor[2]:.1f} % for anchor values); samples are not independent.",
    f"GPS: all {N} coordinates are well-formed (Accuracy {np.nanmin(acc):.0f} to {np.nanmax(acc):.0f} m), but {gps_frozen_n} records lie in {len(frozen_runs)} frozen-position runs and are excluded from location statements only.",
    f"Speed is consistent with km/h (median difference to GPS-derived speed {speed_med_diff:.1f} km/h over {int(ok.sum())} steps).",
    f"No active test was run: Test_Status, PING* and TESTDOWNLINK/TESTUPLINK are empty in all {N} records; throughput and RTT are N/A. DL_bitrate/UL_bitrate are passive counters (non-zero in {int((dl > 0).sum())}/{int((ul > 0).sum())} records).",
    f"CSI_RSRP/CSI_RSRQ/CSI_SNR and NRSTATE are empty in all records; no NR-ARFCN or NR band field is populated: ARFCN/BAND hold the LTE carrier (equal to SecondCell_ARFCN in {anchor_arfcn_equal} of {rat_counts.get('5G', 0)} 5G records).",
    f"LTERSSI is quantised in 2 dB steps (all {rssi_valid} values odd) and is not above RSRP in {rssi_le_rsrp} of {rssi_valid} 4G records, which is not consistent with a wideband RSSI; not used.",
    f"Distance=-1, Bearing=1000, Site_BEARING=1000, AVGSPEED=0, TA=0 or empty, Cellname/IMEI/IMSI empty: placeholders, no site database; not used.",
    f"NetworkTypeNum is {', '.join(nettypenum)} in all {N} records, including those labelled 5G; the field is not described in the schema note and is not interpreted here.",
    f"Events file ({ev_total} events) is identical in timestamp, type and details to the EVENT column of the main log (plus one LOG START).",
]

results = {
    'records': N,
    'duration_s': duration_s,
    'rat_label_records': {'4G': rat_counts.get('4G', 0), '5G': rat_counts.get('5G', 0)},
    'serving_cell_operator_code_records': op_counts,
    'kpi': kpi_json,
    'active_tests': {'download_n': download_n, 'upload_n': upload_n, 'ping_n': ping_n},
    'mobility': {'events_total': ev_total, 'events_by_type': ev_by_type},
    'data_quality_findings': dq,
    'anomaly_candidates': anomalies,
    'model_self_report': MODEL,
}
with open(os.path.join(OUT, 'results.json'), 'w', encoding='utf-8') as fh:
    json.dump(results, fh, indent=1, ensure_ascii=False)

# ----------------------------------------------------------------- tables.md (supplement)
L = []
L.append('# Supplementary tables (generated by analysis.py)\n')
L.append('## Table II - radio KPI statistics\n')
L.append('| KPI | unit | raw N | missing | invalid | valid N | mean | median | SD | P5 | P95 | min | max | good % | acceptable % | poor % |')
L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
f2 = lambda v: 'N/A' if v is None else f'{v:.2f}'
for name, kind, which in KPIS:
    s = full[name]
    unit = 'dBm' if kind == 'RSRP' else 'dB'
    L.append(f"| {name} | {unit} | {s['raw_n']} | {s['missing_n']} | {s['out_of_range_n']} | {s['valid_n']} | {s['mean']:.2f} | "
             f"{s['median']:.1f} | {s['sd']:.2f} | {s['p5']:.2f} | {s['p95']:.2f} | {s['min']:.0f} | {s['max']:.0f} | "
             f"{f2(s['good_pct'])} | {f2(s['acceptable_pct'])} | {f2(s['poor_pct'])} |")
L.append('\nPercentiles: linear interpolation (type 7). SD: sample (n-1). RSRQ: descriptive only. '
         'Throughput and RTT: N/A (no active test in the log).\n')
L.append('## Sensitivity - suspect label-transition records removed\n')
L.append('| KPI | valid N | mean | median | SD | P5 | P95 | poor % |')
L.append('|---|---|---|---|---|---|---|---|')
for name, kind, which in KPIS:
    s = sens[name]
    L.append(f"| {name} | {s['valid_n']} | {s['mean']:.2f} | {s['median']:.1f} | {s['sd']:.2f} | {s['p5']:.2f} | {s['p95']:.2f} | {f2(s['poor_pct'])} |")
L.append(f"\nNR rows: without the {len(nr_suspect)} records labelled 5G that have no SNR and duplicate LTE values. "
         f"LTE rows: without the {len(lte_pre5g_equal_nr)} records labelled 4G that equal the following 5G record.\n")
L.append('## Sample and time shares\n')
L.append(f'Time weighting: record state held until the next record, at most {GAP_CAP_S} s; observed {observed_s:.0f} s of '
         f'{duration_s:.0f} s ({100 * observed_s / duration_s:.1f} %), unobserved {unobserved_s:.0f} s in {gaps_over_cap} gaps.\n')
L.append('| category | value | records | sample % | time s | % of observed time |')
L.append('|---|---|---|---|---|---|')
for label, s_, t_ in [('RAT label', rat_s, rat_t), ('LTE band (serving or anchor carrier)', band_s, band_t),
                      ('serving-cell operator code', op_s, op_t)]:
    for k in s_:
        L.append(f'| {label} | {k} | {s_[k][0]} | {s_[k][1]:.2f} | {t_[k][0]:.0f} | {t_[k][1]:.2f} |')
L.append(f"| NR band | unknown | {rat_counts.get('5G', 0)} | 100.00 of 5G records | - | - |")
L.append(f"| mode | 5G NSA (application label, per schema note) | {rat_counts.get('5G', 0)} | {rat_s['5G'][1]:.2f} | {rat_t['5G'][0]:.0f} | {rat_t['5G'][1]:.2f} |")
L.append(f"| mode | LTE only | {rat_counts.get('4G', 0)} | {rat_s['4G'][1]:.2f} | {rat_t['4G'][0]:.0f} | {rat_t['4G'][1]:.2f} |")
L.append(f"| mode | 5G SA | 0 | 0.00 | 0 | 0.00 |")
L.append(f"| NRSTATE | unknown (empty) | {N - nrstate_filled} | 100.00 | - | - |")
for k in state_s:
    L.append(f'| State | {k} | {state_s[k][0]} | {state_s[k][1]:.2f} | - | - |')
L.append(f"\nBANDWIDTH list contains a 100 MHz element in {int(bw_has100.sum())} records ({int((bw_has100 & is5).sum())} labelled 5G, "
         f"{int((bw_has100 & is4).sum())} labelled 4G) and a 15 MHz element in {int(bw_has15.sum())}; "
         f"the carrier it belongs to is not identified in the export.\n")
L.append('LTE band by RAT label (records):\n')
L.append('| label | ' + ' | '.join(band_by_rat.columns) + ' |')
L.append('|---|' + '---|' * len(band_by_rat.columns))
for k, r in band_by_rat.iterrows():
    L.append(f'| {k} | ' + ' | '.join(str(int(v)) for v in r.values) + ' |')
L.append('\nOperator code by RAT label (records):\n')
L.append('| label | ' + ' | '.join(op_by_rat.columns) + ' |')
L.append('|---|' + '---|' * len(op_by_rat.columns))
for k, r in op_by_rat.iterrows():
    L.append(f'| {k} | ' + ' | '.join(str(int(v)) for v in r.values) + ' |')
L.append('\n## Mobility events (application classification)\n')
L.append('| EventType | count |')
L.append('|---|---|')
for k, v in ev_by_type.items():
    L.append(f'| {k} | {v} |')
L.append(f'| total | {ev_total} |')
L.append(f"\n{int(ev_4g4g.sum())} events are 4G4G, {int(ev_5g5g.sum())} are 5G5G and {int(ev_irat.sum())} are labelled IRAT; "
         f"{int(ev_plmn_change.sum())} change the operator code; {len(pingpong)} are immediate returns to the previous cell "
         f"({pingpong_le10} within 10 s); {int(ev_same_cell.sum())} have the same cell identity on both sides; "
         f"{ev_frozen_gps} were logged with a frozen position. {n_cells} distinct (operator code, node, cell) identities on {n_nodes} nodes. "
         f"Success rate: not observable.\n")
L.append('## Frozen-GPS runs\n')
L.append('| first record | last record | records | held s | exit jump m | exit dt s | implied km/h |')
L.append('|---|---|---|---|---|---|---|')
for r in frozen_runs:
    L.append(f"| {r['first']} | {r['last']} | {r['records']} | {r['held_s']:.0f} | {r['exit_jump_m']:.0f} | {r['exit_dt_s']:.0f} | {r['implied_kmh']:.0f} |")
L.append(f"\n{gps_frozen_n} records flagged (all but the first of each run); {N - gps_frozen_n} records keep a usable position. "
         f"Sum of inter-fix distances {path_m / 1000:.2f} km.\n")
L.append('## Inter-KPI relationships (descriptive, samples are autocorrelated)\n')
L.append('| pair | n | Pearson r | Spearman rho |')
L.append('|---|---|---|---|')
L.append(f'| NR SS-RSRP vs NR SS-SINR | {c_nr[2]} | {c_nr[0]:.2f} | {c_nr[1]:.2f} |')
L.append(f'| LTE anchor RSRP vs anchor SNR (5G records) | {c_an[2]} | {c_an[0]:.2f} | {c_an[1]:.2f} |')
L.append(f'| NR SS-RSRP vs LTE anchor RSRP | {c_nr_an[2]} | {c_nr_an[0]:.2f} | {c_nr_an[1]:.2f} |')
L.append(f"\nMean (NR SS-RSRP minus anchor RSRP) = {nr_minus_anchor:.2f} dB. Of {poor_rsrp_n_paired} NR samples with SS-RSRP < -110 dBm, "
         f"{poor_both} also have SS-SINR < 0 dB.\n")
L.append('## Largest runs of poor NR samples (SS-RSRP < -110 dBm or SS-SINR < 0 dB)\n')
L.append('| from | to | poor records | start position | end position |')
L.append('|---|---|---|---|---|')
for a, b, k in clusters[:6]:
    L.append(f"| {df['Timestamp'][a]} | {df['Timestamp'][b]} | {k} | {pos(a)} | {pos(b)} |")
with open(os.path.join(OUT, 'tables.md'), 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(L) + '\n')

# ----------------------------------------------------------------- console dump
print('=== NARRATIVE NUMBERS ===')
print('records', N, 'duration_s', duration_s, 'first', df['Timestamp'].iloc[0], 'last', df['Timestamp'].iloc[-1])
print('row lengths', dict(row_len), 'header', ncol, 'longer than header', longer_than_header)
print('timestamps bad', ts_bad, 'monotonic', ts_monotonic, 'dup-ts records', dup_ts_records, 'exact dup rows', exact_dup_rows)
print('rat', rat_counts, 'op', op_counts, 'nettypenum', nettypenum)
for name, _, _ in KPIS:
    s = full[name]
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
print('--- sensitivity')
for name, _, _ in KPIS:
    s = sens[name]
    print(name, s['valid_n'], round(s['mean'], 3), round(s['median'], 3), round(s['sd'], 3), round(s['p5'], 3), round(s['p95'], 3),
          None if s['poor_pct'] is None else round(s['poor_pct'], 3))
print('NR SNR missing idx', list(nr_sinr_missing), 'dup anchor', nr_missing_and_dup, 'eq next 4G', nr_missing_eq_next4g, 'all dup', len(nr_dup_anchor_all))
print('4G->5G', len(t45), 'equal next NR', len(lte_pre5g_equal_nr), '5G->4G', len(t54), 'label runs', len(lab_runs))
print('stale NR (same,pairs,pct,distinct)', stale_nr, 'LTE', stale_lte, 'anchor', stale_anchor)
print('gps format bad', gps_format_bad, 'frozen', gps_frozen_n, 'runs', frozen_runs)
print('speed med diff', speed_med_diff, 'steps', int(ok.sum()), 'path km', path_m / 1000, 'speed max', np.nanmax(speed))
print('observed_s', observed_s, 'unobserved_s', unobserved_s, 'gaps>cap', gaps_over_cap, 'max gap', dt_next.max())
print('rat sample', rat_s, 'time', rat_t)
print('band sample', band_s, 'time', band_t)
print('op sample', op_s, 'time', op_t)
print('state', state_s)
print('bw has 100', int(bw_has100.sum()), 'in 5G', int((bw_has100 & is5).sum()), 'in 4G', int((bw_has100 & is4).sum()), 'has 15', int(bw_has15.sum()))
print('tests', test_filled, 'DL>0', int((dl > 0).sum()), 'UL>0', int((ul > 0).sum()), 'DL max', np.nanmax(dl), 'UL max', np.nanmax(ul))
print('events', ev_total, ev_by_type, 'match log', ev_match_log)
print('ev plmn change', int(ev_plmn_change.sum()), 'irat', int(ev_irat.sum()), '5g5g', int(ev_5g5g.sum()), '4g4g', int(ev_4g4g.sum()),
      'same cell', int(ev_same_cell.sum()), 'pingpong', len(pingpong), '<=10s', pingpong_le10, 'frozen gps', ev_frozen_gps,
      'per min', round(ev_rate_per_min, 2), 'cells', n_cells, 'nodes', n_nodes)
print('corr NR', c_nr, 'anchor', c_an, 'NR-anchor', c_nr_an, 'mean diff', nr_minus_anchor, 'poor both', poor_both, 'of', poor_rsrp_n_paired)
print('NR poor by op\n', nr_poor_by_op)
print('rssi', rssi_valid, rssi_le_rsrp, rssi_all_odd, 'rssi-odd rows', rssi_odd_rows)
print('W1', w1, w1_speed, 'W2 4G records', w2_4g_records, 'gaps opened in background', gap_open_bg)
print('ident_when', ident_when)
print('W2', w2)
print('clusters', [(df['Timestamp'][a], df['Timestamp'][b], k, pos(a), pos(b)) for a, b, k in clusters[:6]])
print('ident multi\n', ident_multi)
print('last', i_last, lvl[i_last], qual[i_last], snr[i_last], a_rsrp[i_last], pos(i_last))
