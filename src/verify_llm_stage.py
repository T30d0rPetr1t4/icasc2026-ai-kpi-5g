#!/usr/bin/env python3
"""
Compares the three LLM-stage runs (llm_stage/run*/results.json) with the reference JSON and
re-computes, independently of the LLM's own code, the additional claims made in the narratives.

Usage (from the repository root):
    python src/verify_llm_stage.py <main_log.txt> <events.txt> results/<main_log>_results.json
Writes results/llm_stage_verification.json.
"""
import json, sys, glob, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # repository root
sys.path.insert(0, HERE)
from analyze_gnettrack import load, prepare, haversine

LOG = sys.argv[1]; EV = sys.argv[2]; REF = sys.argv[3]
ref = json.load(open(REF, encoding="utf-8"))
kmap = {"NR_SS_RSRP": "NR_SS_RSRP_dBm", "NR_SS_RSRQ": "NR_SS_RSRQ_dB", "NR_SS_SINR": "NR_SS_SINR_dB",
        "LTE_RSRP": "LTE_RSRP_dBm", "LTE_RSRQ": "LTE_RSRQ_dB", "LTE_SINR": "LTE_SINR_dB_all"}
out = {"runs": {}}
for rj in sorted(glob.glob(os.path.join(ROOT, "llm_stage", "run*", "results.json"))):
    run = os.path.basename(os.path.dirname(rj)); r = json.load(open(rj, encoding="utf-8"))
    n = ok = 0; bad = []
    for k, rk in kmap.items():
        for f in ["valid_n", "mean", "median", "sd", "p5", "p95", "poor_pct"]:
            a, b = r["kpi"][k].get(f), ref["kpi"][rk].get(f)
            if a is None and b is None:
                continue
            n += 1
            tol = 0.0051                                         # LLM runs report 3-6 decimals
            if a is not None and b is not None and abs(float(a) - float(b)) <= tol: ok += 1
            else: bad.append((k, f, a, b))
    cn = cok = 0
    checks = [(r["records"], ref["audit"]["records"]), (r["duration_s"], ref["audit"]["duration_s_first_to_last"]),
              (r["rat_label_records"]["4G"], ref["shares"]["rat_label"]["records_n"]["4G"]),
              (r["rat_label_records"]["5G"], ref["shares"]["rat_label"]["records_n"]["5G"]),
              (r["serving_cell_operator_code_records"].get("22610"), ref["shares"]["serving_plmn"]["records_n"]["22610"]),
              (r["serving_cell_operator_code_records"].get("22601"), ref["shares"]["serving_plmn"]["records_n"]["22601"]),
              (r["active_tests"]["download_n"], 0), (r["active_tests"]["upload_n"], 0), (r["active_tests"]["ping_n"], 0),
              (r["mobility"]["events_total"], ref["mobility"]["events_total"])]
    for t, c in ref["mobility"]["events_by_type"].items():
        checks.append((r["mobility"]["events_by_type"].get(t), c))
    for a, b in checks:
        cn += 1; cok += int(a == b)
    t = [int(x) for x in open(os.path.join(os.path.dirname(rj), "timing.txt")).read().split()]
    out["runs"][run] = {"table2_values_compared": n, "table2_values_equal": ok, "table2_mismatch": bad,
                        "counts_compared": cn, "counts_equal": cok, "wall_clock_s": t[-1] - t[0],
                        "data_quality_findings_n": len(r["data_quality_findings"]),
                        "anomaly_candidates_n": len(r["anomaly_candidates"]),
                        "model_self_report": r["model_self_report"],
                        "narrative_words": len(open(os.path.join(os.path.dirname(rj), "narrative.md"), encoding="utf-8").read().split())}

# ------------------------------------------------------------------ independent re-computation of narrative claims
df = load(LOG); d = prepare(df)
ev = pd.read_csv(EV, sep="\t", dtype=str, index_col=False, keep_default_na=False)
is5, is4 = d["NetworkTech"] == "5G", d["NetworkTech"] == "4G"
c = {}
# (1) label-transition artefacts
miss = is5 & (d["SNR"] == "-")
eq_anchor = miss & (d["Level_n"] == d["SecondCell_RSRP_n"]) & (d["Qual_n"] == d["SecondCell_RSRQ_n"])
nxt = d.shift(-1)
followed_by_4g = miss & (nxt["NetworkTech"] == "4G")
c["5G_records_SNR_missing"] = int(miss.sum())
c["...whose_Level_Qual_equal_own_LTE_anchor"] = int(eq_anchor.sum())
c["...immediately_followed_by_4G_record"] = int(followed_by_4g.sum())
pre5 = is4 & (nxt["NetworkTech"] == "5G")
same_as_next = pre5 & (d["Level_n"] == nxt["Level_n"]) & (d["Qual_n"] == nxt["Qual_n"])
c["4G_records_directly_before_a_5G_record"] = int(pre5.sum())
c["...carrying_Level_Qual_of_that_5G_record"] = int(same_as_next.sum())
nr_clean = d["NR_RSRP"].where(~miss)
c["NR_RSRP_without_the_7"] = {"n": int(nr_clean.notna().sum()), "mean": round(float(nr_clean.mean()), 4),
                              "median": float(nr_clean.median()),
                              "poor_pct": round(100 * float((nr_clean < -110).sum()) / int(nr_clean.notna().sum()), 4)}
lte_clean = d["LTE_RSRP"].where(~same_as_next)
c["LTE_RSRP_without_the_12"] = {"n": int(lte_clean.notna().sum()), "mean": round(float(lte_clean.mean()), 4)}
# (2) held readings
trip = d["Level"] + "|" + d["Qual"] + "|" + d["SNR"]
prev_same = trip == trip.shift()
c["triplet_equals_previous_record_pct"] = {"5G": round(100 * float(prev_same[is5].mean()), 1),
                                           "4G": round(100 * float(prev_same[is4].mean()), 1)}
both5 = is5 & (d["NetworkTech"].shift() == "5G"); both4 = is4 & (d["NetworkTech"].shift() == "4G")
c["triplet_repeat_within_consecutive_same_label_pairs_pct"] = {
    "5G": round(100 * float(prev_same[both5].mean()), 1), "4G": round(100 * float(prev_same[both4].mean()), 1)}
# (3) uneven record density
t0 = d["t"].iloc[0]; el = (d["t"] - t0).dt.total_seconds()
first_bg = d.index[d["FOREGROUND"] == "false"][0]
c["first_background_record"] = d.loc[first_bg, "Timestamp"]
c["records_before_first_background_record"] = int(first_bg)
c["...as_pct_of_records"] = round(100 * first_bg / len(d), 1)
c["...elapsed_s"] = float(el[first_bg]); c["...as_pct_of_duration"] = round(100 * float(el[first_bg]) / float(el.iloc[-1]), 1)
c["records_sharing_timestamp_with_previous"] = int((d["t"].diff().dt.total_seconds() == 0).sum())
# (4) label flips
flip = (d["NetworkTech"] != d["NetworkTech"].shift()); flip.iloc[0] = False
cell = d["Operator"] + "-" + d["Node"] + "-" + d["CellID"]
c["rat_label_flips"] = int(flip.sum())
c["...on_unchanged_node_and_cell"] = int((flip & (cell == cell.shift())).sum())
f54 = flip & is4
c["5G_to_4G_flips"] = int(f54.sum())
last_nr = d["NR_RSRP"].where(~miss).ffill().shift()
c["...preceded_by_valid_NR_RSRP_ge_-110"] = int((f54 & (last_nr >= -110)).sum())
c["...first_4G_record_idle"] = int((f54 & (d["State"] == "I")).sum())
# (5) poor NR cluster
w = d[(d["t"] >= "2026-10-02 12:19:17") & (d["t"] <= "2026-10-02 12:20:38")]
c["poor_NR_RSRP_total"] = int((d["NR_RSRP"] < -110).sum())
c["poor_NR_RSRP_in_12:19:17-12:20:38"] = int((w["NR_RSRP"] < -110).sum())
c["5G_records_in_that_interval"] = int((w["NetworkTech"] == "5G").sum())
w2 = d[(d["t"] >= "2026-10-02 12:19:17") & (d["t"] <= "2026-10-02 12:20:29")]
c["interval_12:19:17-12:20:29"] = {"records": int(len(w2)), "5G_records": int((w2["NetworkTech"] == "5G").sum()),
                                   "NR_RSRP_median": float(w2["NR_RSRP"].median()),
                                   "NR_RSRP_poor_pct": round(100 * float((w2["NR_RSRP"] < -110).sum()) / int(w2["NR_RSRP"].notna().sum()), 1),
                                   "NR_SINR_median": float(w2["NR_SINR"].median()),
                                   "LTE_RSRP_median": float(w2["LTE_RSRP"].median())}
et = pd.to_datetime(ev["Timestamp"], format="%Y.%m.%d_%H.%M.%S")
c["interval_12:19:17-12:20:29"]["events"] = int(((et >= "2026-10-02 12:19:17") & (et <= "2026-10-02 12:20:29")).sum())
w3 = d[(d["t"] >= "2026-10-02 12:34:05") & (d["t"] <= "2026-10-02 12:34:10")]
c["interval_12:34:05-12:34:10"] = {"5G_records": int((w3["NetworkTech"] == "5G").sum()),
                                   "NR_RSRP_poor": int((w3["NR_RSRP"] < -110).sum()),
                                   "NR_RSRP_median": float(w3["NR_RSRP"].median()), "NR_SINR_median": float(w3["NR_SINR"].median())}
# (6) co-occurrence
both = d[["NR_RSRP", "NR_SINR"]].dropna()
pr = both["NR_RSRP"] < -110
c["NR_SINR_poor_pct_when_RSRP_poor"] = round(100 * float((both.loc[pr, "NR_SINR"] < 0).mean()), 1)
c["NR_SINR_poor_pct_when_RSRP_not_poor"] = round(100 * float((both.loc[~pr, "NR_SINR"] < 0).mean()), 1)
c["NR_RSRP_poor_and_SINR_poor_n"] = int((pr & (both["NR_SINR"] < 0)).sum())
l = d[["LTE_RSRP", "LTE_SINR"]].dropna()
c["poor_LTE_SINR_n"] = int((l["LTE_SINR"] < 0).sum())
c["...with_RSRP_ge_-110"] = int(((l["LTE_SINR"] < 0) & (l["LTE_RSRP"] >= -110)).sum())
c["pearson_NR_RSRP_SINR"] = round(float(both.corr().iloc[0, 1]), 2)
c["spearman_NR_RSRP_SINR"] = round(float(both.corr(method="spearman").iloc[0, 1]), 2)
c["spearman_NR_RSRQ_SINR"] = round(float(d[["NR_RSRQ", "NR_SINR"]].dropna().corr(method="spearman").iloc[0, 1]), 2)
c["median_NR_minus_LTE_anchor_RSRP_dB"] = float((d["NR_RSRP"] - d["LTE_RSRP"])[is5].median())
# (7) time shares under the three gap rules used
dt = d["dt_next"]
def share(w_):
    s = w_.groupby(d["NetworkTech"]).sum(); return {k: round(100 * float(v) / float(w_.sum()), 1) for k, v in s.items()}, round(float(w_.sum()), 0)
c["time_share_hold_capped_10s"] = share(dt.clip(upper=10))
c["time_share_intervals_gt10s_dropped"] = share(dt.where(dt <= 10))
c["time_share_intervals_gt30s_dropped"] = share(dt.where(dt <= 30))
c["time_share_no_rule"] = share(dt)
c["intervals_gt10s"] = {"n": int((dt > 10).sum()), "sum_s": float(dt[dt > 10].sum()),
                        "pct_of_duration": round(100 * float(dt[dt > 10].sum()) / float(dt.sum()), 1)}
# (8) events
src = ev["EventDetails"].str.split(":").str[0].str.split("-").str[:5].str.join("-")
dst = ev["EventDetails"].str.split(":").str[1].str.split("-").str[:5].str.join("-")
rev = [(src[i - 1] == dst[i]) and (dst[i - 1] == src[i]) and (src[i] != dst[i]) for i in range(1, len(ev))]
gap = et.diff().dt.total_seconds().iloc[1:].values
c["immediate_reversals"] = {"all": int(sum(rev)), "within_10s": int(sum(r and g <= 10 for r, g in zip(rev, gap))),
                            "within_30s": int(sum(r and g <= 30 for r, g in zip(rev, gap)))}
c["events_4G4G"] = int(ev["EventType"].str.endswith("4G4G").sum()); c["events_5G5G"] = int(ev["EventType"].str.endswith("5G5G").sum())
c["events_IRAT"] = int(ev["EventType"].str.startswith("IRAT").sum())
main_ev = d[(d["EVENT"] != "") & (d["EVENT"] != "LOG START")]
c["events_file_equals_main_log_EVENT_column"] = bool(len(main_ev) == len(ev) and (main_ev["EVENT"].values == ev["EventType"].values).all()
                                                     and (main_ev["EVENTDETAILS"].values == ev["EventDetails"].values).all())
c["events_with_FOREGROUND_false"] = int((main_ev["FOREGROUND"] == "false").sum())
c["events_with_logged_speed_0"] = int((main_ev["Speed_n"] == 0).sum())
c["events_on_stale_position_records_reference_rule"] = int(main_ev["pos_stale"].sum())
# NR identifiers
c["5G_records_identity_equals_anchor"] = int((is5 & (d["Node"] == d["SecondCell_NODE"]) & (d["CellID"] == d["SecondCell_CELLID"])).sum())
c["SNR_exact_zero_in_valid_NR_SINR"] = int((d["NR_SINR"] == 0).sum())
# additional re-computations for claims made by single runs
ie = main_ev.index
prev_lab = d["NetworkTech"].shift()
irat = main_ev["EVENT"].str.startswith("IRAT")
c["IRAT_events_between_records_of_same_label"] = int((irat & (d.loc[ie, "NetworkTech"] == prev_lab[ie])).sum())
l4 = d[is4]
c["4G_records_LTERSSI_not_above_RSRP"] = int((l4["LTERSSI_n"] <= l4["Level_n"]).sum())
c["CQI_missing_marker"] = int((d["CQI"] == "-").sum()); c["CQI_equal_6"] = int((d["CQI"] == "6").sum())
c["events_same_node"] = int((ev["EventDetails"].str.split(":").str[0].str.split("-").str[:4].str.join("-") ==
                             ev["EventDetails"].str.split(":").str[1].str.split("-").str[:4].str.join("-")).sum())
c["events_change_arfcn"] = int((ev["EventDetails"].str.split(":").str[0].str.split("-").str[5] !=
                                ev["EventDetails"].str.split(":").str[1].str.split("-").str[5]).sum())
bw100 = d["BANDWIDTH"].str.split("/").apply(lambda xs: "100" in xs)
c["bw100_by_label"] = {"5G": int((bw100 & is5).sum()), "4G": int((bw100 & is4).sum())}
run216 = d[d["run"] == d.loc[d["Timestamp"] == "2026.10.02_12.20.29", "run"].iloc[0]]
nxt_fix = d.loc[run216.index[-1] + 1]
c["frozen_run_12:20:29"] = {"records": int(len(run216)), "last": run216["Timestamp"].iloc[-1],
    "background": int((run216["FOREGROUND"] == "false").sum()), "next_fix": nxt_fix["Timestamp"],
    "jump_m": round(float(haversine(run216["Latitude_n"].iloc[0], run216["Longitude_n"].iloc[0], nxt_fix["Latitude_n"], nxt_fix["Longitude_n"])), 0),
    "gap_s": float((nxt_fix["t"] - run216["t"].iloc[-1]).total_seconds())}
c["state_D_share_pct"] = {"5G": round(100 * float((d["State"][is5] == "D").mean()), 1), "4G": round(100 * float((d["State"][is4] == "D").mean()), 1)}
out["independent_recomputation"] = c

# ---------------------------------------------------------------- explicit claim table (claimed value typed from the run outputs)
ts10, ts10d = c["time_share_hold_capped_10s"], c["time_share_intervals_gt10s_dropped"]
obs = ts10[1] / 1895.0
claims = [
 ("5G-labelled records with missing SNR", "1,2,3", 7, c["5G_records_SNR_missing"]),
 ("... of which Level/Qual equal the record's own LTE anchor", "1,2,3", 6, c["...whose_Level_Qual_equal_own_LTE_anchor"]),
 ("... each immediately followed by a 4G record", "1", 7, c["...immediately_followed_by_4G_record"]),
 ("4G records directly preceding a 5G record", "1,3", 25, c["4G_records_directly_before_a_5G_record"]),
 ("... carrying that 5G record's Level/Qual", "1,2,3", 12, c["...carrying_Level_Qual_of_that_5G_record"]),
 ("NR SS-RSRP mean without the 7 (dBm)", "1,2,3", -99.44, c["NR_RSRP_without_the_7"]["mean"]),
 ("NR SS-RSRP poor share without the 7 (%)", "3", 14.49, c["NR_RSRP_without_the_7"]["poor_pct"]),
 ("LTE RSRP mean without the 12 (dBm)", "1,2,3", -92.75, c["LTE_RSRP_without_the_12"]["mean"]),
 ("triplet equals previous record, 5G (%)", "1", 38.9, c["triplet_equals_previous_record_pct"]["5G"]),
 ("triplet equals previous record, 4G (%)", "1", 32.5, c["triplet_equals_previous_record_pct"]["4G"]),
 ("triplet repeats in consecutive 5G pairs (%)", "2,3", 44.1, c["triplet_repeat_within_consecutive_same_label_pairs_pct"]["5G"]),
 ("triplet repeats in consecutive 4G pairs (%)", "2,3", 37.0, c["triplet_repeat_within_consecutive_same_label_pairs_pct"]["4G"]),
 ("first background record", "1,2", "2026.10.02_12.20.49", c["first_background_record"]),
 ("records before it", "2", 254, c["records_before_first_background_record"]),
 ("... share of records (%)", "1,2", 59.5, c["...as_pct_of_records"]),
 ("... elapsed time (s)", "1,2", 271.0, c["...elapsed_s"]),
 ("... share of duration (%)", "1,2", 14.3, c["...as_pct_of_duration"]),
 ("records sharing a timestamp with the previous one", "1,3", 74, c["records_sharing_timestamp_with_previous"]),
 ("RAT label flips", "1", 50, c["rat_label_flips"]),
 ("... on unchanged node and cell", "1", 50, c["...on_unchanged_node_and_cell"]),
 ("5G-to-4G flips", "1", 25, c["5G_to_4G_flips"]),
 ("... preceded by NR SS-RSRP >= -110 dBm", "1", 20, c["...preceded_by_valid_NR_RSRP_ge_-110"]),
 ("... first 4G record idle", "1", 24, c["...first_4G_record_idle"]),
 ("poor NR SS-RSRP samples in 12:19:17-12:20:38", "2", 22, c["poor_NR_RSRP_in_12:19:17-12:20:38"]),
 ("5G records in that interval", "2", 40, c["5G_records_in_that_interval"]),
 ("12:19:17-12:20:29 records", "1", 58, c["interval_12:19:17-12:20:29"]["records"]),
 ("12:19:17-12:20:29 5G records", "1", 39, c["interval_12:19:17-12:20:29"]["5G_records"]),
 ("12:19:17-12:20:29 NR SS-RSRP median (dBm)", "1", -112.0, c["interval_12:19:17-12:20:29"]["NR_RSRP_median"]),
 ("12:19:17-12:20:29 NR SS-RSRP poor (%)", "1", 53.8, c["interval_12:19:17-12:20:29"]["NR_RSRP_poor_pct"]),
 ("12:19:17-12:20:29 NR SS-SINR median (dB)", "1", 0.0, c["interval_12:19:17-12:20:29"]["NR_SINR_median"]),
 ("12:19:17-12:20:29 LTE RSRP median (dBm)", "1", -97.0, c["interval_12:19:17-12:20:29"]["LTE_RSRP_median"]),
 ("12:19:17-12:20:29 events", "1,2", 8, c["interval_12:19:17-12:20:29"]["events"]),
 ("12:34:05-12:34:10 5G records", "1,2", 6, c["interval_12:34:05-12:34:10"]["5G_records"]),
 ("12:34:05-12:34:10 poor NR SS-RSRP", "1,2", 6, c["interval_12:34:05-12:34:10"]["NR_RSRP_poor"]),
 ("12:34:05-12:34:10 NR SS-RSRP median (dBm)", "1,2", -113.0, c["interval_12:34:05-12:34:10"]["NR_RSRP_median"]),
 ("12:34:05-12:34:10 NR SS-SINR median (dB)", "1,2", -4.0, c["interval_12:34:05-12:34:10"]["NR_SINR_median"]),
 ("NR SINR poor when RSRP poor (%)", "2", 48.4, c["NR_SINR_poor_pct_when_RSRP_poor"]),
 ("NR SINR poor when RSRP not poor (%)", "2", 6.0, c["NR_SINR_poor_pct_when_RSRP_not_poor"]),
 ("NR RSRP poor and SINR poor (records)", "1", 15, c["NR_RSRP_poor_and_SINR_poor_n"]),
 ("poor LTE SINR samples", "1,2", 57, c["poor_LTE_SINR_n"]),
 ("... with RSRP >= -110 dBm", "2", 55, c["...with_RSRP_ge_-110"]),
 ("Spearman NR RSRP-SINR", "1", 0.72, c["spearman_NR_RSRP_SINR"]),
 ("Pearson NR RSRP-SINR", "3", 0.70, c["pearson_NR_RSRP_SINR"]),
 ("Spearman NR RSRQ-SINR", "1", 0.94, c["spearman_NR_RSRQ_SINR"]),
 ("Spearman LTE RSRP-SINR", "1", 0.52, ref["kpi"]["spearman"]["LTE_RSRP_vs_LTE_SINR_all"]),
 ("Spearman NR RSRP vs anchor RSRP", "1", 0.61, ref["kpi"]["spearman"]["NR_RSRP_vs_LTEanchor_RSRP"]),
 ("median NR minus anchor RSRP (dB)", "1", -6.0, c["median_NR_minus_LTE_anchor_RSRP_dB"]),
 ("time share 5G, hold capped at 10 s (% of observed)", "3", 35.0, ts10[0]["5G"]),
 ("observed time, hold capped at 10 s (s)", "3", 958.0, ts10[1]),
 ("time share 5G, hold capped at 10 s (% of total)", "1", 17.7, round(ts10[0]["5G"] * obs, 1)),
 ("time share 4G, hold capped at 10 s (% of total)", "1", 32.9, round(ts10[0]["4G"] * obs, 1)),
 ("unobserved time, hold capped at 10 s (%)", "1", 49.4, round(100 * (1 - obs), 1)),
 ("time share 5G, intervals > 10 s dropped (%)", "2", 50.2, ts10d[0]["5G"]),
 ("observed time, intervals > 10 s dropped (s)", "2", 528.0, ts10d[1]),
 ("intervals > 10 s", "2,3", 43, c["intervals_gt10s"]["n"]),
 ("... share of duration (%)", "2", 72.1, c["intervals_gt10s"]["pct_of_duration"]),
 ("immediate reversals (any delay)", "1,2,3", 28, c["immediate_reversals"]["all"]),
 ("... within 10 s", "1,3", 18, c["immediate_reversals"]["within_10s"]),
 ("... within 30 s", "1", 26, c["immediate_reversals"]["within_30s"]),
 ("events crossing the two operator codes", "1,2,3", 36, ref["mobility"]["events_cross_plmn_n"]),
 ("events 4G4G / 5G5G / IRAT", "3", [50, 11, 18], [c["events_4G4G"], c["events_5G5G"], c["events_IRAT"]]),
 ("reselection-type / handover-type events", "1", [61, 18], [ref["mobility"]["events_reselection_n"], ref["mobility"]["events_handover_n"]]),
 ("events with identical source and target cell identity", "1,3", 3, ref["mobility"]["events_same_cell_identity_n"]),
 ("events file identical to main-log EVENT column", "1,2,3", True, c["events_file_equals_main_log_EVENT_column"]),
 ("events logged with FOREGROUND=false", "1,2", 60, c["events_with_FOREGROUND_false"]),
 ("events logged with Speed 0", "1,2", 52, c["events_with_logged_speed_0"]),
 ("IRAT-type events between records of the same label", "1", 17, c["IRAT_events_between_records_of_same_label"]),
 ("events staying on the same node", "1", 33, c["events_same_node"]),
 ("events changing ARFCN", "1", 18, c["events_change_arfcn"]),
 ("5G records whose identity equals the anchor fields", "1,2,3", 220, c["5G_records_identity_equals_anchor"]),
 ("NR SINR exactly 0.0 dB", "2", 32, c["SNR_exact_zero_in_valid_NR_SINR"]),
 ("LTE-only SNR exactly 0.0 dB", "1", 31, ref["kpi"]["LTE_only4G_SINR_exact_zero_n"]),
 ("LTE-only SNR below 0 dB", "1,2", 0, ref["kpi"]["LTE_only4G_SINR_negative_n"]),
 ("4G records with LTERSSI not above RSRP", "1,2,3", 135, c["4G_records_LTERSSI_not_above_RSRP"]),
 ("CQI '-' / CQI = 6", "1", [137, 73], [c["CQI_missing_marker"], c["CQI_equal_6"]]),
 ("100 MHz bandwidth entry in 5G / 4G records", "1", [142, 88], [c["bw100_by_label"]["5G"], c["bw100_by_label"]["4G"]]),
 ("frozen run from 12:20:29: records", "1,2,3", 21, c["frozen_run_12:20:29"]["records"]),
 ("... of which background", "1,2", 18, c["frozen_run_12:20:29"]["background"]),
 ("... jump to next fix (m)", "1,2,3", 3418.0, c["frozen_run_12:20:29"]["jump_m"]),
 ("... after (s)", "1,2,3", 60.0, c["frozen_run_12:20:29"]["gap_s"]),
 ("State D share, 5G / 4G records (%)", "1", [50.2, 21.8], [c["state_D_share_pct"]["5G"], c["state_D_share_pct"]["4G"]]),
 ("passive DL / UL bitrate non-zero records", "2,3", [114, 112], [ref["kpi"]["passive_DL_bitrate_nonzero_n"], ref["kpi"]["passive_UL_bitrate_nonzero_n"]]),
 ("band shares L7 / L20 / L3 (%)", "1,2,3", [66.0, 17.6, 16.4], [ref["shares"]["lte_band_label"]["records_pct"][k] for k in ("L7", "L20", "L3")]),
 ("NR RSRP good / acceptable (%)", "1,2", [3.6, 82.4], [ref["kpi"]["NR_SS_RSRP_dBm"]["good_pct"], ref["kpi"]["NR_SS_RSRP_dBm"]["acceptable_pct"]]),
 ("NR SINR good / acceptable (%)", "1,2", [22.4, 65.4], [ref["kpi"]["NR_SS_SINR_dB"]["good_pct"], ref["kpi"]["NR_SS_SINR_dB"]["acceptable_pct"]]),
 ("LTE RSRP good / acceptable (%)", "1,2", [21.6, 76.8], [ref["kpi"]["LTE_RSRP_dBm"]["good_pct"], ref["kpi"]["LTE_RSRP_dBm"]["acceptable_pct"]]),
 ("LTE SINR good / acceptable (%)", "1,2", [20.7, 66.0], [ref["kpi"]["LTE_SINR_dB_all"]["good_pct"], ref["kpi"]["LTE_SINR_dB_all"]["acceptable_pct"]]),
]
w4 = d[(d["t"] >= "2026-10-02 12:19:13") & (d["t"] <= "2026-10-02 12:20:29")]
claims += [("12:19:13-12:20:29 NR samples", "3", 44, int(w4["NR_RSRP"].notna().sum())),
           ("12:19:13-12:20:29 poor NR SS-RSRP", "3", 21, int((w4["NR_RSRP"] < -110).sum())),
           ("12:19:13-12:20:29 NR SS-RSRP median (dBm)", "3", -110.0, float(w4["NR_RSRP"].median())),
           ("NetworkTypeNum = 13 in all records", "2,3", 427, int((d["NetworkTypeNum"] == "13").sum()))]
def same(a, b):
    """claimed value a (as printed by the LLM) vs recomputed b, compared at the precision of the claim"""
    if isinstance(a, list): return all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, (bool, str)): return a == b
    nd = len(str(a).split(".")[1]) if isinstance(a, float) and "." in str(a) else 0
    return abs(float(a) - round(float(b), nd)) < 1e-9
table = [{"claim": k, "runs": r_, "claimed": a, "recomputed": b, "equal": bool(same(a, b))} for k, r_, a, b in claims]
out["narrative_claims_checked"] = len(table)
out["narrative_claims_reproduced"] = sum(t["equal"] for t in table)
out["narrative_claims_not_reproduced"] = [t for t in table if not t["equal"]]
out["narrative_claim_table"] = table
out["stale_position_records"] = {"run1": 63, "run2": 64, "run3": 63, "reference_rule": ref["audit"]["stale_position_records"]}
print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
json.dump(out, open(os.path.join(ROOT, "results", "llm_stage_verification.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
