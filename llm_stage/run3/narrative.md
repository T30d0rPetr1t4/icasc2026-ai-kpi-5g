# Narrative for human verification

**Scope.** 427 records, 2026-10-02 12:16:18–12:47:53 (1895 s), operator name "orange". Labels: 5G 221 (51.8%), 4G 206 (48.2%). Time shares use a 10 s hold cap: only 958 s are observed, of which 5G 35.0%, 4G 65.0%.

**Validation.** Timestamps parse and never decrease; 74 records repeat the previous second. No −201 code, no out-of-range value. Raw/valid N: NR SS-RSRP and SS-RSRQ 221/221; NR SS-SINR 221/214 (seven "-"); LTE KPIs 427/426 (one empty anchor). GPS is well-formed but frozen in six runs (63 records; e.g. 12:20:29–12:27:07, then 3.4 km in 60 s); kept in KPI statistics, excluded from locations only.

**Table II** (mean, median, SD, P5, P95; poor share):
- NR SS-RSRP: −99.07, −98, 9.57, −114, −85 dBm; 14.03%
- NR SS-RSRQ: −12.42, −12, 1.79, −16, −10 dB; N/A
- NR SS-SINR: 5.28, 4.5, 6.15, −3, 17 dB; 12.15%
- LTE RSRP: −92.85, −95, 10.98, −109, −71 dBm; 1.64%
- LTE RSRQ: −12.10, −12, 3.11, −17, −7 dB; N/A
- LTE SINR: 4.98, 4, 6.84, −6, 17 dB; 13.38%

Throughput, RTT: N/A (no active test logged).

**Bands and mode.** LTE carrier band, sample shares: L7 66.0%, L20 17.6%, L3 16.4%. NR band: unknown for all 221 5G records (no NR-ARFCN logged). Mode: NSA as labelled by the application; SA not evidenced; NRSTATE and CSI fields are empty.

**Mobility.** 79 events: 50 4G4G, 11 5G5G, 18 IRAT-labelled. 36 change the serving operator code (22610 in 224 records, 22601 in 203); 28 are immediate returns. Success rate: not observable.

**Observations.**
1. 12:19:13–12:20:29, 45.78329N 21.31208E to 45.77728N 21.30673E: 21 of the 31 poor NR RSRP samples, anchor also weak (RSRP −114 dBm, SNR −19 dB), nine events.
2. 12:34:03–12:34:18, from 45.76207N 21.24551E: NR SS-SINR down to −6 dB.
3. NR SS-RSRP and SS-SINR correlate (r = 0.70, n = 214).

**Hypotheses, unverified.** The two operator codes may reflect a shared RAN, and the weak stretch a border area between cells. Load, traffic demand and server limits cannot be assessed from passive counters. LTE fallback alone is not read as a coverage failure.

**Assumptions and limitations.** Schema field mapping applied literally; type-7 percentiles; sample SD. The seven 5G records without SNR duplicate LTE values in Level/Qual (excluding them: NR RSRP mean −99.44 dBm), and 12 4G records equal the following 5G record. 44% of consecutive 5G records repeat identical readings, so samples are not independent. One drive, one device.
