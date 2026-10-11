#!/usr/bin/env python3
"""
Reference analysis (Stages 1-3 and the screening part of Stage 4) for G-NetTrack Pro logs.

Usage (from the repository root):
    python src/analyze_gnettrack.py <main_log.txt> [<events.txt>] --out results [--figures] [--ref LAT LON LABEL]
    python src/analyze_gnettrack.py <supplementary_log.txt> --out results/supplementary --audit-only

Everything reported in the paper's tables comes from the JSON written by this script.
The functions plot_route_map() and plot_timeseries() are also used by notebooks/figures.ipynb.
Only pandas / numpy / scikit-learn / matplotlib are needed.
"""
import argparse, csv, json, os, sys
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- Table I (study-defined bins)
RSRP_GOOD, RSRP_POOR = -85.0, -110.0          # dBm : good > -85 ; poor < -110
SINR_GOOD, SINR_POOR = 10.0, 0.0              # dB  : good > 10  ; poor < 0
# Plausibility ranges used for validation (outside = invalid, not "poor")
VALID = {
    "NR_RSRP": (-140, -44), "NR_RSRQ": (-43, 20), "NR_SINR": (-23, 40),
    "LTE_RSRP": (-140, -44), "LTE_RSRQ": (-34, 3), "LTE_SINR": (-23, 40),
}
JUMP_M = 100.0        # consecutive distinct fixes farther apart than this = position discontinuity
STALE_S = 10.0        # identical coordinates for >= this long, followed by a discontinuity = stale position
GAP_S = 30.0          # inter-record intervals longer than this are not used for time weighting
PINGPONG_S = 30.0     # A->B->A within this time = ping-pong candidate
REF_POINT = None      # optional (lat, lon, label) orientation marker for the map; set with --ref
R_EARTH = 6371000.0


def load(path):
    """Tolerant TSV reader: G-NetTrack rows may carry more neighbour fields than the header."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    hdr = rows[0]
    while hdr and hdr[-1] == "":
        hdr = hdr[:-1]
    n = len(hdr)
    body = [r for r in rows[1:] if any(x != "" for x in r)]
    overlong = sum(1 for r in body if len([x for x in r[n:] if x != ""]) > 0)
    df = pd.DataFrame([(r + [""] * n)[:n] for r in body], columns=hdr)
    df.attrs["n_header_fields"] = n
    df.attrs["overlong_rows"] = overlong
    return df


def num(s):
    return pd.to_numeric(s, errors="coerce")


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def in_range(s, key):
    lo, hi = VALID[key]
    return s.where((s >= lo) & (s <= hi))


def stats(s, poor_lt=None, good_gt=None):
    """valid N, mean, median, sample SD, P5, P95 (+ category shares where a criterion exists)."""
    v = s.dropna()
    out = {"valid_n": int(v.size)}
    if v.size == 0:
        return out
    out.update(
        mean=round(float(v.mean()), 4), median=round(float(v.median()), 4),
        sd=round(float(v.std(ddof=1)), 4) if v.size > 1 else None,
        p5=round(float(np.percentile(v, 5)), 4), p95=round(float(np.percentile(v, 95)), 4),
        min=float(v.min()), max=float(v.max()),
    )
    if poor_lt is not None:
        out["poor_pct"] = round(100.0 * float((v < poor_lt).sum()) / v.size, 4)
        out["poor_n"] = int((v < poor_lt).sum())
    if good_gt is not None:
        out["good_pct"] = round(100.0 * float((v > good_gt).sum()) / v.size, 4)
        out["good_n"] = int((v > good_gt).sum())
    if poor_lt is not None and good_gt is not None:
        acc = int(((v >= poor_lt) & (v <= good_gt)).sum())
        out["acceptable_n"] = acc
        out["acceptable_pct"] = round(100.0 * acc / v.size, 4)
    return out


def prepare(df):
    """Stage 1: typing, validity, RAT-aware KPI columns, position quality."""
    d = df.copy()
    d["t"] = pd.to_datetime(d["Timestamp"], format="%Y.%m.%d_%H.%M.%S", errors="coerce")
    for c in ["Longitude", "Latitude", "Speed", "Level", "Qual", "SNR", "LTERSSI", "Accuracy",
              "SecondCell_RSRP", "SecondCell_RSRQ", "SecondCell_SNR", "DL_bitrate", "UL_bitrate",
              "PINGAVG", "TESTDOWNLINK", "TESTUPLINK"]:
        d[c + "_n"] = num(d[c]) if c in d else np.nan
    is5, is4 = d["NetworkTech"] == "5G", d["NetworkTech"] == "4G"
    # NR measurements exist only in records labelled 5G (Level/Qual/SNR = SS-RSRP/SS-RSRQ/SS-SINR)
    d["NR_RSRP"] = in_range(d["Level_n"].where(is5), "NR_RSRP")
    d["NR_RSRQ"] = in_range(d["Qual_n"].where(is5), "NR_RSRQ")
    d["NR_SINR"] = in_range(d["SNR_n"].where(is5), "NR_SINR")
    # LTE measurements: serving cell in 4G records, anchor (SecondCell_*) in 5G NSA records
    d["LTE_RSRP"] = in_range(d["Level_n"].where(is4).fillna(d["SecondCell_RSRP_n"].where(is5)), "LTE_RSRP")
    d["LTE_RSRQ"] = in_range(d["Qual_n"].where(is4).fillna(d["SecondCell_RSRQ_n"].where(is5)), "LTE_RSRQ")
    d["LTE_SINR"] = in_range(d["SNR_n"].where(is4).fillna(d["SecondCell_SNR_n"].where(is5)), "LTE_SINR")
    d["LTE_SINR_only4G"] = in_range(d["SNR_n"].where(is4), "LTE_SINR")
    d["LTE_SINR_anchor"] = in_range(d["SecondCell_SNR_n"].where(is5), "LTE_SINR")
    # time step to next record
    d["dt_next"] = (d["t"].shift(-1) - d["t"]).dt.total_seconds()
    # position runs / discontinuities
    pos = d["Longitude"] + "," + d["Latitude"]
    d["run"] = (pos != pos.shift()).cumsum()
    g = d.groupby("run").agg(t0=("t", "first"), t1=("t", "last"), n=("t", "size"),
                             lat=("Latitude_n", "first"), lon=("Longitude_n", "first"))
    g["dur"] = (g["t1"] - g["t0"]).dt.total_seconds()
    g["step_to_next_m"] = haversine(g["lat"], g["lon"], g["lat"].shift(-1), g["lon"].shift(-1))
    g["stale"] = (g["dur"] >= STALE_S) & (g["step_to_next_m"] > JUMP_M)
    d["pos_stale"] = d["run"].map(g["stale"]).fillna(False).astype(bool)
    d.attrs["runs"] = g
    return d


def transition_and_hold_checks(d):
    """Checks added after the LLM stage reported these artefacts (Section VI of the paper)."""
    is5, is4 = d["NetworkTech"] == "5G", d["NetworkTech"] == "4G"
    nxt = d.shift(-1)
    miss5 = is5 & (d["SNR"] == "-")
    eq_anchor = miss5 & (d["Level_n"] == d["SecondCell_RSRP_n"]) & (d["Qual_n"] == d["SecondCell_RSRQ_n"])
    pre5 = is4 & (nxt["NetworkTech"] == "5G")
    carries = pre5 & (d["Level_n"] == nxt["Level_n"]) & (d["Qual_n"] == nxt["Qual_n"])
    trip = d["Level"] + "|" + d["Qual"] + "|" + d["SNR"]
    held = trip == trip.shift()
    return {
        "label_transition_5G_records_without_SNR": int(miss5.sum()),
        "label_transition_5G_records_without_SNR_equal_to_anchor": int(eq_anchor.sum()),
        "label_transition_4G_records_before_5G": int(pre5.sum()),
        "label_transition_4G_records_before_5G_carrying_NR_values": int(carries.sum()),
        "held_triplet_pct": {"5G": round(100.0 * float(held[is5].mean()), 1) if is5.any() else None,
                             "4G": round(100.0 * float(held[is4].mean()), 1) if is4.any() else None},
    }


def audit(df, d):
    """Stage 1 report: what is in the file and what must not be trusted."""
    g = d.attrs["runs"]
    t = d["t"]
    dt = t.diff().dt.total_seconds()
    lvl = d["Level_n"]
    a = {
        "records": int(len(d)),
        "header_fields": int(df.attrs["n_header_fields"]),
        "rows_longer_than_header": int(df.attrs["overlong_rows"]),
        "first_timestamp": str(d["Timestamp"].iloc[0]), "last_timestamp": str(d["Timestamp"].iloc[-1]),
        "timestamp_backward_steps": int((dt < 0).sum()),
        "largest_backward_step_s": float(dt.min()) if (dt < 0).any() else 0.0,
        "duration_s_first_to_last": float((t.iloc[-1] - t.iloc[0]).total_seconds()),
        "dt_median_s": float(dt.median()), "dt_mean_s": round(float(dt.mean()), 2), "dt_max_s": float(dt.max()),
        "dt_gt_gap_rule_n": int((dt > GAP_S).sum()),
        "device": sorted(d["DEVICE"].unique().tolist()) if "DEVICE" in d else [],
        "app_version": sorted(d["VERSION_NAME"].unique().tolist()) if "VERSION_NAME" in d else [],
        "operator_names": d["Operatorname"].value_counts().to_dict(),
        "operator_codes": d["Operator"].value_counts().to_dict(),
        "rat_labels": d["NetworkTech"].value_counts().to_dict(),
        "band_labels": d["BAND"].value_counts().to_dict(),
        "arfcn": d["ARFCN"].value_counts().to_dict(),
        "level_out_of_range_n": int(((lvl < -140) | (lvl > -44)).sum()),
        "level_out_of_range_values": sorted(set(lvl[(lvl < -140) | (lvl > -44)].dropna().astype(int).tolist())),
        "operator_code_invalid_n": int((~d["Operator"].str.fullmatch(r"\d{5,6}")).sum()),
        "snr_missing_marker_n": int((d["SNR"] == "-").sum()),
        "snr_blank_n": int((d["SNR"] == "").sum()),
        "qual_missing_n": int(d["Qual_n"].isna().sum()),
        "active_test_rows": d["Test_Status"].replace("", np.nan).dropna().value_counts().to_dict() if "Test_Status" in d else {},
        "ping_rows_n": int(d["PINGAVG_n"].notna().sum()),
        "dl_test_rows_n": int(d["TESTDOWNLINK_n"].notna().sum()),
        "csi_rsrp_populated_n": int((d["CSI_RSRP"] != "").sum()) if "CSI_RSRP" in d else 0,
        "android_network_type_codes": d["NetworkTypeNum"].value_counts().to_dict() if "NetworkTypeNum" in d else {},
        "nrstate_populated_n": int((d["NRSTATE"] != "").sum()) if "NRSTATE" in d else 0,
        "app_not_foreground_n": int((d["FOREGROUND"] == "false").sum()) if "FOREGROUND" in d else None,
        "distinct_positions": int(g.shape[0]),
        "position_discontinuities_gt100m": int((g["step_to_next_m"] > JUMP_M).sum()),
        "largest_discontinuity_m": round(float(g["step_to_next_m"].max()), 0),
        "stale_position_records": int(d["pos_stale"].sum()),
        "stale_position_seconds": float(g.loc[g["stale"], "dur"].sum()),
        "longest_stale_run_s": float(g.loc[g["stale"], "dur"].max()) if g["stale"].any() else 0.0,
        "path_m_tracked_steps_le100m": round(float(g.loc[g["step_to_next_m"] <= JUMP_M, "step_to_next_m"].sum()), 0),
        "path_m_all_steps_chained": round(float(g["step_to_next_m"].sum()), 0),
        **transition_and_hold_checks(d),
        "bbox": {"lat_min": float(d["Latitude_n"].min()), "lat_max": float(d["Latitude_n"].max()),
                 "lon_min": float(d["Longitude_n"].min()), "lon_max": float(d["Longitude_n"].max())},
        "speed_kmh": {"median": float(d["Speed_n"].median()), "max": float(d["Speed_n"].max())},
    }
    return a


def shares(d):
    out = {}
    n = len(d)
    w = d["dt_next"].where(d["dt_next"] <= GAP_S)          # time weights under the gap rule
    def both(col):
        rec = (d[col].value_counts() / n * 100).round(1).to_dict()
        tw = (w.groupby(d[col]).sum() / w.sum() * 100).round(1).to_dict()
        w_all = d["dt_next"]
        tw_all = (w_all.groupby(d[col]).sum() / w_all.sum() * 100).round(1).to_dict()
        return {"records_pct": rec, "time_weighted_pct": tw, "time_weighted_no_gap_rule_pct": tw_all,
                "records_n": d[col].value_counts().to_dict()}
    out["rat_label"] = both("NetworkTech")
    out["serving_plmn"] = both("Operator")
    out["lte_band_label"] = both("BAND")
    out["time_weight_total_s"] = float(w.sum())
    out["time_weight_excluded_s"] = float(d["dt_next"].where(d["dt_next"] > GAP_S).sum())
    out["plmn_by_rat_n"] = pd.crosstab(d["Operator"], d["NetworkTech"]).to_dict()
    bw100 = d["BANDWIDTH"].str.split("/").apply(lambda xs: "100" in xs)
    out["bandwidth_list_contains_100MHz_n"] = int(bw100.sum())
    out["bandwidth_list_contains_100MHz_pct"] = round(100.0 * bw100.mean(), 1)
    out["bandwidth_100MHz_by_rat_n"] = bw100.groupby(d["NetworkTech"]).sum().astype(int).to_dict()
    out["ca_flag_true_n"] = int((d["CA"] == "true").sum())
    out["state_n"] = d["State"].value_counts().to_dict()
    out["distinct_cells"] = int(d.groupby(["Operator", "Node", "CellID"]).ngroups)
    out["distinct_nodes"] = int(d.groupby(["Operator", "Node"]).ngroups)
    out["distinct_nodes_by_plmn"] = d.groupby("Operator")["Node"].nunique().to_dict()
    out["distinct_pci"] = int(d["PSC"].nunique())
    return out


def kpis(d):
    k = {
        "NR_SS_RSRP_dBm": stats(d["NR_RSRP"], RSRP_POOR, RSRP_GOOD),
        "NR_SS_RSRQ_dB": stats(d["NR_RSRQ"]),
        "NR_SS_SINR_dB": stats(d["NR_SINR"], SINR_POOR, SINR_GOOD),
        "LTE_RSRP_dBm": stats(d["LTE_RSRP"], RSRP_POOR, RSRP_GOOD),
        "LTE_RSRQ_dB": stats(d["LTE_RSRQ"]),
        "LTE_SINR_dB_all": stats(d["LTE_SINR"], SINR_POOR, SINR_GOOD),
        "LTE_SINR_dB_only4G_records": stats(d["LTE_SINR_only4G"], SINR_POOR, SINR_GOOD),
        "LTE_SINR_dB_anchor_in_NSA": stats(d["LTE_SINR_anchor"], SINR_POOR, SINR_GOOD),
        "LTE_RSRP_dBm_only4G_records": stats(d["LTE_RSRP"].where(d["NetworkTech"] == "4G"), RSRP_POOR, RSRP_GOOD),
        "LTE_RSRP_dBm_anchor_in_NSA": stats(d["LTE_RSRP"].where(d["NetworkTech"] == "5G"), RSRP_POOR, RSRP_GOOD),
        "passive_DL_bitrate_kbps": stats(d["DL_bitrate_n"]),
        "passive_UL_bitrate_kbps": stats(d["UL_bitrate_n"]),
    }
    k["passive_DL_bitrate_nonzero_n"] = int((d["DL_bitrate_n"] > 0).sum())
    k["passive_UL_bitrate_nonzero_n"] = int((d["UL_bitrate_n"] > 0).sum())
    # exact-zero share: a possible placeholder / censoring indicator
    k["NR_SINR_exact_zero_n"] = int((d["NR_SINR"] == 0).sum())
    k["LTE_only4G_SINR_exact_zero_n"] = int((d["LTE_SINR_only4G"] == 0).sum())
    k["LTE_only4G_SINR_negative_n"] = int((d["LTE_SINR_only4G"] < 0).sum())
    # correlations (Spearman; consecutive samples are autocorrelated, so no p-values are reported)
    k["spearman"] = {
        "NR_RSRP_vs_NR_SINR": round(float(d[["NR_RSRP", "NR_SINR"]].dropna().corr(method="spearman").iloc[0, 1]), 2),
        "NR_RSRP_vs_LTEanchor_RSRP": round(float(d.loc[d["NetworkTech"] == "5G", ["NR_RSRP", "LTE_RSRP"]].dropna().corr(method="spearman").iloc[0, 1]), 2),
        "LTE_RSRP_vs_LTE_SINR_all": round(float(d[["LTE_RSRP", "LTE_SINR"]].dropna().corr(method="spearman").iloc[0, 1]), 2),
    }
    return k


def mobility(d, ev):
    m = {}
    cell = d["Operator"] + "-" + d["Node"] + "-" + d["CellID"]
    chg = cell != cell.shift()
    chg.iloc[0] = False
    m["serving_cell_changes_in_main_log"] = int(chg.sum())
    m["plmn_changes_in_main_log"] = int(((d["Operator"] != d["Operator"].shift()) & (d.index > 0)).sum())
    m["rat_label_changes_in_main_log"] = int(((d["NetworkTech"] != d["NetworkTech"].shift()) & (d.index > 0)).sum())
    # dwell per serving-cell visit
    visit = chg.cumsum()
    first = d.groupby(visit)["t"].first()
    dw = (first.shift(-1) - first).dt.total_seconds().dropna()      # last visit is open-ended -> dropped
    m["cell_visits"] = int(first.size)
    m["dwell_s"] = {"n": int(dw.size), "median": float(dw.median()), "p5": round(float(np.percentile(dw, 5)), 1),
                    "p95": round(float(np.percentile(dw, 95)), 1), "max": float(dw.max()),
                    "le_5s_n": int((dw <= 5).sum()), "le_5s_pct": round(100.0 * float((dw <= 5).mean()), 1)}
    if ev is not None:
        e = ev.copy()
        e["t"] = pd.to_datetime(e["Timestamp"], format="%Y.%m.%d_%H.%M.%S", errors="coerce")
        m["events_total"] = int(len(e))
        m["events_by_type"] = e["EventType"].value_counts().to_dict()
        m["events_handover_n"] = int(e["EventType"].str.contains("HANDOVER").sum())
        m["events_reselection_n"] = int(e["EventType"].str.contains("RESELECTION").sum())
        m["events_irat_n"] = int(e["EventType"].str.startswith("IRAT").sum())
        src = e["EventDetails"].str.split(":").str[0]
        dst = e["EventDetails"].str.split(":").str[1]
        cid = lambda s: s.str.split("-").str[:5].str.join("-")          # MCC-MNC-TAC-node-cell
        e["src"], e["dst"] = cid(src), cid(dst)
        e["src_plmn"] = src.str.split("-").str[:2].str.join("-")
        e["dst_plmn"] = dst.str.split("-").str[:2].str.join("-")
        m["events_cross_plmn_n"] = int((e["src_plmn"] != e["dst_plmn"]).sum())
        m["events_cross_plmn_pct"] = round(100.0 * m["events_cross_plmn_n"] / len(e), 1)
        m["events_same_cell_identity_n"] = int((e["src"] == e["dst"]).sum())
        # ping-pong: A->B immediately followed by B->A within PINGPONG_S
        pp = []
        for i in range(1, len(e)):
            a, b = e.iloc[i - 1], e.iloc[i]
            if a["src"] == b["dst"] and a["dst"] == b["src"] and a["src"] != a["dst"] \
                    and (b["t"] - a["t"]).total_seconds() <= PINGPONG_S:
                pp.append({"t_out": a["Timestamp"], "t_back": b["Timestamp"], "a": a["src"], "b": a["dst"],
                           "return_s": (b["t"] - a["t"]).total_seconds(),
                           "cross_plmn": bool(a["src_plmn"] != a["dst_plmn"])})
        m["pingpong_pairs_n"] = len(pp)
        m["pingpong_cross_plmn_n"] = int(sum(p["cross_plmn"] for p in pp))
        m["pingpong_pairs"] = pp
        m["events_per_minute"] = round(len(e) / ((d["t"].iloc[-1] - d["t"].iloc[0]).total_seconds() / 60.0), 2)
        ie = e["t"].diff().dt.total_seconds().dropna()
        m["inter_event_s"] = {"median": float(ie.median()), "min": float(ie.min()), "max": float(ie.max())}
    return m


def screening(d):
    """Stage 4a: evidence-linked anomaly candidates (rules + Isolation Forest). No labels -> no accuracy claim."""
    from sklearn.ensemble import IsolationForest
    s = {}
    nsa = d[d["NetworkTech"] == "5G"]
    s["rule_NR_poor_RSRP_n"] = int((nsa["NR_RSRP"] < RSRP_POOR).sum())
    s["rule_NR_poor_SINR_n"] = int((nsa["NR_SINR"] < SINR_POOR).sum())
    r1 = (d["NR_RSRP"] > RSRP_GOOD) & (d["NR_SINR"] < SINR_POOR)               # strong signal, poor quality
    r1b = (d["NR_RSRP"] >= RSRP_POOR) & (d["NR_SINR"] < SINR_POOR)             # not-poor signal, poor quality
    r2 = (d["NR_RSRP"] < RSRP_POOR) | (d["LTE_RSRP"] < RSRP_POOR)              # poor coverage (either layer)
    r3 = (d["LTE_SINR_anchor"] < SINR_POOR)                                    # poor LTE anchor quality in NSA
    r4 = (d["NR_RSRP"] - d["LTE_RSRP"] <= -15) & (d["NetworkTech"] == "5G")    # NR >= 15 dB below its anchor
    s["rule_goodRSRP_poorSINR_NR_n"] = int(r1.sum())
    s["rule_notpoorRSRP_poorSINR_NR_n"] = int(r1b.sum())
    s["rule_poor_coverage_any_layer_n"] = int(r2.sum())
    s["rule_poor_LTE_anchor_SINR_n"] = int(r3.sum())
    s["rule_NR_15dB_below_anchor_n"] = int(r4.sum())
    rule_any = r1b | r2 | r3
    s["rule_any_n"] = int(rule_any.sum())
    # Isolation Forest on the NSA records with all six radio features present
    feats = ["NR_RSRP", "NR_RSRQ", "NR_SINR", "LTE_RSRP", "LTE_RSRQ", "LTE_SINR"]
    X = nsa[feats].dropna()
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=0).fit(X)
    flag = pd.Series(iso.predict(X) == -1, index=X.index)
    score = pd.Series(-iso.score_samples(X), index=X.index)
    s["iforest"] = {"features": feats, "n_estimators": 200, "contamination": 0.05, "random_state": 0,
                    "records_used": int(len(X)), "flagged_n": int(flag.sum()),
                    "flagged_also_rule_n": int((flag & rule_any.reindex(flag.index)).sum())}
    top = score[flag].sort_values(ascending=False)
    s["iforest"]["flagged_records"] = [
        {"Timestamp": d.loc[i, "Timestamp"], "score": round(float(score[i]), 3),
         **{f: float(d.loc[i, f]) for f in feats},
         "cell": f'{d.loc[i, "Operator"]}-{d.loc[i, "Node"]}-{d.loc[i, "CellID"]}',
         "rule_flag": bool(rule_any[i])} for i in top.index]
    # seed sensitivity of the flagged set
    sets = []
    for seed in range(10):
        m = IsolationForest(n_estimators=200, contamination=0.05, random_state=seed).fit(X)
        sets.append(set(X.index[m.predict(X) == -1]))
    core = set.intersection(*sets)
    union = set.union(*sets)
    s["iforest"]["seed_sensitivity_10_seeds"] = {"flagged_in_all": len(core), "flagged_in_any": len(union)}
    # worst contiguous NR segment (30-s rolling median of SS-RSRP)
    x = d.set_index("t")["NR_RSRP"].dropna()
    roll = x.rolling("30s").median()
    cnt = x.rolling("30s").count()
    roll = roll[cnt >= 5]
    if len(roll):
        s["worst_30s_NR_RSRP_median"] = {"value_dBm": float(roll.min()), "window_end": str(roll.idxmin())}
    d["_rule_any"], d["_if_flag"] = rule_any, flag.reindex(d.index).fillna(False)
    return s


C_NR, C_LTE = "#1f6fb4", "#e08a1e"
C_GOOD, C_ACC, C_POOR = "#2e9e5b", "#e3b505", "#d1352b"


def set_style():
    """Matplotlib settings used for the paper figures (IEEE column width, 7 pt text)."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "savefig.dpi": 600})


def plot_route_map(d):
    """Fig. 3: route in local km coordinates, RAT label per record, poor RSRP, frozen positions."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    set_style()
    ok = d[~d["pos_stale"]]
    stale = d[d["pos_stale"]]
    lat0 = np.radians(d["Latitude_n"].mean())
    kx, ky = 111.320 * np.cos(lat0), 110.574                      # km per degree
    x0, y0 = d["Longitude_n"].min(), d["Latitude_n"].min()
    X = lambda lon: (lon - x0) * kx
    Y = lambda lat: (lat - y0) * ky
    g = d.attrs["runs"]

    def base(ax):
        # dashed connectors across position discontinuities
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            if a["step_to_next_m"] > JUMP_M:
                ax.plot([X(a["lon"]), X(b["lon"])], [Y(a["lat"]), Y(b["lat"])], ls=(0, (3, 2)), lw=0.6,
                        color="#9a9a9a", zorder=1)
        st = stale.drop_duplicates("run")
        ax.scatter(X(st["Longitude_n"]), Y(st["Latitude_n"]), marker="x", s=22, lw=0.9, color="#444444", zorder=4)
        ax.set_aspect("equal")
        ax.set_xlim(-0.6, X(d["Longitude_n"].max()) + 0.6)
        ax.set_ylim(-1.0, Y(d["Latitude_n"].max()) + 0.4)
        ax.grid(True, lw=0.3, color="#dddddd")
        ax.set_axisbelow(True)
        ax.tick_params(length=2, width=0.5)
        s_, e_ = d.iloc[0], d.iloc[-1]
        ax.annotate(f'start {s_["t"]:%H:%M}', (X(s_["Longitude_n"]), Y(s_["Latitude_n"])), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=6.5)
        ax.annotate(f'end {e_["t"]:%H:%M}', (X(e_["Longitude_n"]), Y(e_["Latitude_n"])), xytext=(2, 6),
                    textcoords="offset points", ha="left", va="bottom", fontsize=6.5)
        if REF_POINT is not None:
            ax.plot(X(REF_POINT[1]), Y(REF_POINT[0]), marker="+", ms=6, mew=0.9, color="k", zorder=5)
            ax.annotate(REF_POINT[2], (X(REF_POINT[1]), Y(REF_POINT[0])), xytext=(0, -5),
                        textcoords="offset points", ha="center", va="top", fontsize=6)
        # north arrow
        xa, ya = X(d["Longitude_n"].max()) + 0.25, Y(d["Latitude_n"].max()) - 1.3
        ax.annotate("", xy=(xa, ya + 0.9), xytext=(xa, ya), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="k"))
        ax.text(xa, ya + 0.95, "N", ha="center", va="bottom", fontsize=7)

    fig, ax = plt.subplots(1, 1, figsize=(3.25, 2.42))
    base(ax)
    for rat, c in [("4G", C_LTE), ("5G", C_NR)]:
        q = ok[ok["NetworkTech"] == rat]
        ax.scatter(X(q["Longitude_n"]), Y(q["Latitude_n"]), s=7, color=c, lw=0, zorder=3)
    v = ok["NR_RSRP"].fillna(ok["LTE_RSRP"])
    pq = ok[v < RSRP_POOR]
    ax.scatter(X(pq["Longitude_n"]), Y(pq["Latitude_n"]), s=26, facecolors="none", edgecolors=C_POOR, lw=0.7, zorder=6)
    ax.set_ylabel("north (km)")
    ax.set_xlabel("east (km)")
    h = [Line2D([], [], marker="o", ls="", ms=3.2, color=C_NR, label="5G NSA record"),
         Line2D([], [], marker="o", ls="", ms=3.2, color=C_LTE, label="LTE record"),
         Line2D([], [], marker="o", ls="", ms=4.6, mfc="none", mec=C_POOR, mew=0.7, label="RSRP < −110 dBm"),
         Line2D([], [], marker="x", ls="", ms=4.5, mew=0.9, color="#444444", label="frozen position"),
         Line2D([], [], ls=(0, (3, 2)), lw=0.6, color="#9a9a9a", label="gap between fixes")]
    ax.legend(handles=h, loc="lower right", fontsize=5.8, frameon=True, framealpha=0.9, borderpad=0.3,
              handletextpad=0.3, labelspacing=0.2, edgecolor="#cccccc")
    fig.tight_layout(pad=0.3)
    return fig


def plot_timeseries(d, ev=None):
    """Fig. 4: RSRP and SINR against time since log start, with PLMN / RAT strips and event ticks."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    set_style()
    fig, axs = plt.subplots(3, 1, figsize=(3.25, 3.0), sharex=True,
                            gridspec_kw={"height_ratios": [3, 2.4, 0.9]})
    tm = (d["t"] - d["t"].iloc[0]).dt.total_seconds() / 60.0
    ax = axs[0]
    ax.axhspan(-160, RSRP_POOR, color=C_POOR, alpha=0.07, lw=0)
    ax.plot(tm, d["LTE_RSRP"], ".", ms=2.2, color=C_LTE, label="LTE RSRP (serving / anchor)")
    ax.plot(tm, d["NR_RSRP"], ".", ms=2.2, color=C_NR, label="NR SS-RSRP")
    ax.axhline(RSRP_POOR, lw=0.5, color=C_POOR); ax.axhline(RSRP_GOOD, lw=0.5, color=C_GOOD)
    ax.set_ylim(-125, -58); ax.set_ylabel("RSRP (dBm)")
    ax.legend(loc="upper right", fontsize=5.8, ncol=2, frameon=False, handletextpad=0.1, columnspacing=0.8,
              borderpad=0.1, markerscale=2)
    ax = axs[1]
    ax.axhspan(-30, SINR_POOR, color=C_POOR, alpha=0.07, lw=0)
    ax.plot(tm, d["LTE_SINR"], ".", ms=2.2, color=C_LTE)
    ax.plot(tm, d["NR_SINR"], ".", ms=2.2, color=C_NR)
    ax.axhline(SINR_POOR, lw=0.5, color=C_POOR); ax.axhline(SINR_GOOD, lw=0.5, color=C_GOOD)
    ax.set_ylim(-21, 27); ax.set_ylabel("SINR (dB)")
    ax = axs[2]
    # strips: serving-cell PLMN (top) and RAT label (bottom); width = time to next record (capped)
    w = (d["dt_next"].fillna(1).clip(upper=GAP_S) / 60.0).values
    plmns = sorted(d["Operator"].unique())
    pc = {plmns[0]: "#7b3294", plmns[-1]: "#5aae61"}
    ax.bar(tm, 0.8, width=w, bottom=1.1, align="edge", color=[pc[p] for p in d["Operator"]], lw=0)
    ax.bar(tm, 0.8, width=w, bottom=0.1, align="edge",
           color=[C_NR if r == "5G" else C_LTE for r in d["NetworkTech"]], lw=0)
    if ev is not None:
        te = (pd.to_datetime(ev["Timestamp"], format="%Y.%m.%d_%H.%M.%S") - d["t"].iloc[0]).dt.total_seconds() / 60
        ax.plot(te, np.full(len(te), 2.25), "|", ms=3.5, mew=0.5, color="k")
    ax.set_ylim(0, 2.55); ax.set_yticks([0.5, 1.5, 2.25])
    ax.set_yticklabels(["RAT", "PLMN", "events"], fontsize=6)
    hs = [Patch(color=pc[p], label=f"{p[:3]}-{p[3:]}") for p in sorted(pc, reverse=True)] + \
         [Patch(color=C_NR, label="5G NSA"), Patch(color=C_LTE, label="LTE")]
    fig.legend(handles=hs, loc="lower center", ncol=4, fontsize=6, frameon=False, handlelength=1.0,
               handletextpad=0.3, columnspacing=1.0, bbox_to_anchor=(0.55, -0.005))
    ax.set_xlabel("time since log start (min)")
    for a in axs:
        a.tick_params(length=2, width=0.5); a.grid(True, axis="x", lw=0.3, color="#dddddd"); a.set_axisbelow(True)
        a.set_xlim(0, tm.max() + 0.2)
    fig.tight_layout(pad=0.3, h_pad=0.4, rect=(0, 0.045, 1, 1))
    fig.plmn_strip_colours = dict(pc)
    return fig


def figures(d, ev, out):
    """Writes fig_route_map.png (Fig. 3) and fig_timeseries.png (Fig. 4) to `out`."""
    import matplotlib.pyplot as plt
    fig = plot_route_map(d)
    fig.savefig(os.path.join(out, "fig_route_map.png"))
    plt.close(fig)
    fig = plot_timeseries(d, ev)
    pc = fig.plmn_strip_colours
    fig.savefig(os.path.join(out, "fig_timeseries.png"))
    plt.close(fig)
    return {"plmn_strip_colours": {k: v for k, v in pc.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("events", nargs="?")
    ap.add_argument("--out", default="out")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--ref", nargs=3, metavar=("LAT", "LON", "LABEL"), help="orientation marker on the map")
    a = ap.parse_args()
    global REF_POINT
    if a.ref:
        REF_POINT = (float(a.ref[0]), float(a.ref[1]), a.ref[2])
    os.makedirs(a.out, exist_ok=True)
    df = load(a.log)
    d = prepare(df)
    res = {"file": os.path.basename(a.log), "audit": audit(df, d)}
    ev = None
    if not a.audit_only:
        if a.events and os.path.exists(a.events):
            ev = pd.read_csv(a.events, sep="\t", dtype=str, index_col=False, keep_default_na=False)
        res["shares"] = shares(d)
        res["kpi"] = kpis(d)
        res["mobility"] = mobility(d, ev)
        res["screening"] = screening(d)
        res["parameters"] = {"RSRP_GOOD": RSRP_GOOD, "RSRP_POOR": RSRP_POOR, "SINR_GOOD": SINR_GOOD,
                             "SINR_POOR": SINR_POOR, "JUMP_M": JUMP_M, "STALE_S": STALE_S, "GAP_S": GAP_S,
                             "PINGPONG_S": PINGPONG_S, "VALID": VALID}
        if a.figures:
            import matplotlib
            matplotlib.use("Agg")
            res["figures"] = figures(d, ev, a.out)
    name = os.path.splitext(os.path.basename(a.log))[0]
    with open(os.path.join(a.out, f"{name}_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
