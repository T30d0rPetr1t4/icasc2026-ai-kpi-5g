# Output contract (so that the run can be compared automatically with an independent calculation)

Write these files into your run folder:

1. `timing.txt` : the output of `date +%s` taken as your very first action, and again as your very last action (two lines).
2. `analysis.py` : the code you executed to obtain every number you report.
3. `results.json` with exactly this layout (use null where a value is not defined or not available):

{
 "records": int,
 "duration_s": number,                       // last timestamp minus first timestamp
 "rat_label_records": {"4G": int, "5G": int},
 "serving_cell_operator_code_records": {"<code>": int, ...},
 "kpi": {
   "NR_SS_RSRP": {"valid_n": int, "mean": x, "median": x, "sd": x, "p5": x, "p95": x, "poor_pct": x},
   "NR_SS_RSRQ": {... "poor_pct": null},
   "NR_SS_SINR": {...},
   "LTE_RSRP":   {...},
   "LTE_RSRQ":   {... "poor_pct": null},
   "LTE_SINR":   {...}
 },
 "active_tests": {"download_n": int, "upload_n": int, "ping_n": int},
 "mobility": {"events_total": int, "events_by_type": {"<EventType>": int, ...}},
 "data_quality_findings": ["short sentence", ...],
 "anomaly_candidates": [{"when": "timestamp or interval", "what": "observation", "evidence": "fields/values"}],
 "model_self_report": "the model name/identifier you are running as, exactly as stated in your own system context, or 'unknown'"
}

4. `narrative.md` : the narrative of at most 400 words requested in the prompt.
