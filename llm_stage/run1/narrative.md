# Narrative

**Scope, validation.** 427 records, 2026-10-02 12:16:18-12:47:53 (1895 s, handset local time, zone not logged), longitude 21.2023-21.3201, latitude 45.7582-45.8073. Timestamps parse, never go backwards; gaps reach 93 s. No -201 code or out-of-range value. Excluded: missing markers only (NR SINR "-" in 7 of 221 5G-labelled records; one 5G record lacks anchor values). No active test was run: throughput and RTT are N/A; logged bitrates are passive traffic.

**Table II** (mean/median/sample SD/P5/P95, linear-interpolation percentiles; then poor/acceptable/good %, strict thresholds):
- NR_SS_RSRP, N=221: -99.07/-98.0/9.57/-114.0/-85.0 dBm; 14.03/82.35/3.62.
- NR_SS_RSRQ, N=221: -12.42/-12.0/1.79/-16.0/-10.0 dB; descriptive.
- NR_SS_SINR, N=214: 5.28/4.5/6.15/-3.0/17.0 dB; 12.15/65.42/22.43.
- LTE_RSRP, N=426: -92.85/-95.0/10.98/-109.0/-71.0 dBm; 1.64/76.76/21.60.
- LTE_RSRQ, N=426: -12.10/-12.0/3.11/-17.0/-7.0 dB; descriptive.
- LTE_SINR, N=426: 4.98/4.0/6.84/-6.0/17.0 dB; 13.38/65.96/20.66.

**Shares.** Records: 4G 48.24%, 5G 51.76% (NSA per schema note; NRSTATE empty, SA not verifiable), unknown 0. LTE serving/anchor band: L7 66.04%, L20 17.56%, L3 16.39%. NR band unknown for all 5G records (no NR ARFCN, PCI or band logged). Time shares (record held until the next, capped at 10 s, remainder unobserved): 4G 32.9%, 5G 17.7%, unobserved 49.4%.

**Mobility.** 79 events (61 reselection-type, 18 handover-type; types in results.json); success rate not observable.

**Observations.** (1) 12:19:17-12:20:29, longitude 21.3067-21.3116, latitude 45.7773-45.7827: NR RSRP median -112 dBm (53.8% poor), NR SINR median 0 dB, LTE RSRP median -97 dBm, 8 events. (2) The label flips 50 times, always on an unchanged cell; 20 of 25 5G-to-4G changes follow a non-poor NR RSRP, 24 start idle. (3) Positions freeze in background: 21 records at one point, 12:20:29-12:27:07, then 3418 m in 60 s; 63 frozen-position records (jumps above 80 km/h against logged speed up to 10 km/h) are excluded from spatial outputs only. (4) 28 immediate cell reversals; 36 events change operator code (22610, 22601). (5) Spearman RSRP-SINR: NR 0.72, LTE 0.52.

**Hypotheses, unverified.** (1) cell-edge or overlap area; (2) NR leg released on inactivity, not a coverage failure; (3) stale location in background; (4) shared RAN with overlapping layers. Load, interference and server limits cannot be tested.

**Limitations.** Readings are held (39% of 5G records repeat the previous triplet). 19 records beside label changes carry mixed-RAT values; kept (without them the NR RSRP mean is -99.44 dBm). LTE SNR in 4G-labelled records never falls below 0 dB, so the LTE SINR poor share may be understated. 59.5% of records fall in the first 14.3% of the time. All outputs need human verification.
