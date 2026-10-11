**Scope, validation.** 427 records, 12:16:18-12:47:53 (1895 s). Timestamps well-formed, non-decreasing; units plausible (dBm, dB, km/h); no -201 code. Missing: NR SS-SINR "-" in 7 5G-labelled records, anchor fields empty in 1; nothing else excluded.

**Table II** (valid/raw N; mean, median, sample SD, P5, P95 by linear interpolation; poor share):
- NR SS-RSRP: 221/221; -99.068, -98.0, 9.567, -114.0, -85.0 dBm; poor 14.03%
- NR SS-RSRQ: 221/221; -12.421, -12.0, 1.791, -16.0, -10.0 dB; poor N/A
- NR SS-SINR: 214/221; 5.276, 4.5, 6.148, -3.0, 17.0 dB; poor 12.15%
- LTE RSRP (4G+anchor): 426/427; -92.845, -95.0, 10.975, -109.0, -71.0 dBm; poor 1.64%
- LTE RSRQ: 426/427; -12.096, -12.0, 3.113, -17.0, -7.0 dB; poor N/A
- LTE SINR: 426/427; 4.977, 4.0, 6.837, -6.0, 17.0 dB; poor 13.38%

Good/acceptable: NR RSRP 3.6/82.4%, NR SINR 22.4/65.4%, LTE RSRP 21.6/76.8%, LTE SINR 20.7/66.0%.

**Mode, band.** Sample shares: 5G label 51.8%, 4G 48.2%; LTE serving/anchor band L7 66.0%, L20 17.6%, L3 16.4%; NR band unknown 100% (only LTE EARFCNs logged; n77/n78 unresolved). NRSTATE/CSI empty: SA not inferable; "NSA" rests on the schema note. Time shares (forward-hold; intervals >10 s unobserved): 5G 50.2%, 4G 49.8% of 528 observed s; 72.1% of time unknown.

**Mobility.** 79 events: CELL_RESELECTION_4G4G 49, HANDOVER_DATA_5G5G 9, IRAT_CELL_RESELECTION_5G4G 8, IRAT_HANDOVER_DATA_4G5G 7, IRAT_CELL_RESELECTION_4G5G 2, CELL_RESELECTION_5G5G 2, HANDOVER_DATA_4G4G 1, IRAT_HANDOVER_DATA_5G4G 1. Success rate N/A. 36 cross operator codes 22610/22601; "IRAT/5G" names reflect labels, every event ARFCN being LTE. LTE fallback is not treated as coverage failure.

**Throughput/RTT.** N/A: no active test; DL/UL bitrate is passive.

**Observations.** 22 of 31 poor NR RSRP samples lie in 12:19:17-12:20:38 (21.3116, 45.7827 to 21.3067, 45.7773), with negative anchor SINR in 16 records and 8 events; a second cluster at 12:34:05-12:34:10 (21.2451, 45.7619). NR SINR is poor in 48.4% of poor-RSRP samples against 6.0% otherwise; 55 of 57 poor LTE SINR samples have non-poor RSRP. GPS froze 12:20:29-12:27:07, then jumped 3418 m in 60 s; 64 stale-fix records excluded from spatial outputs only.

**Hypotheses (unverified).** Cell-edge conditions, interference or load; network sharing behind the two operator codes; background logging restrictions. Stale readings likely: 44.1% of consecutive 5G-labelled records repeat identical values.

**Assumptions, limitations.** Schema-literal mapping; 7 + 12 suspected label-transition artefacts retained (without them NR RSRP mean is -99.439 dBm). 59.5% of records fall in the first 14.3% of the time; samples are not independent; one drive, one handset; no load, traffic-demand or server data. All figures require human verification.
