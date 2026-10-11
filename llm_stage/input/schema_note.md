# Schema note for the supplied G-NetTrack Pro export

Source of the field meanings: G-NetTrack Pro manual (Gyokov Solutions), https://gyokovsolutions.com/manual-g-nettrack/

## Files
- `orange_2026.10.02_12.16.18.txt` : main log, tab-separated, one header row. A data row may carry more
  trailing neighbour-cell fields than the header has columns.
- `orange_2026.10.02_12.16.18_events.txt` : mobility events as classified by the application.

## Record generation
A record is written on a change of position (distance interval), on a change of cell, or on a timer when
neither changes. The log is therefore NOT sampled at a fixed rate.

## Main-log fields (only those needed)
- `Timestamp` : handset local time, `YYYY.MM.DD_HH.MM.SS`.
- `Longitude`, `Latitude` (decimal degrees), `Speed` (km/h), `Accuracy` (m), `FOREGROUND` (true/false: application in foreground).
- `Operatorname` : name of the registered operator. `Operator` : MCC+MNC reported for the serving cell.
- `NetworkTech` : RAT label of the record (2G/3G/4G/5G).
- `Level`, `Qual`, `SNR` :
    - in records labelled 4G: LTE RSRP (dBm), LTE RSRQ (dB), LTE SNR (dB) of the serving cell;
    - in records labelled 5G (the application showed "5G NSA"): NR SS-RSRP (dBm), SS-RSRQ (dB), SS-SINR (dB).
- `SecondCell_RSRP`, `SecondCell_RSRQ`, `SecondCell_SNR`, `SecondCell_NODE`, `SecondCell_CELLID` :
  serving 4G (LTE anchor) cell values in 5G NSA mode; empty otherwise.
- `Node`, `CellID`, `PSC` (PCI), `ARFCN`, `BAND`, `BANDWIDTH` (MHz, "/"-separated list of the carriers the handset reports), `CA`.
- `LTERSSI` : LTE RSSI (dBm), 4G only. `CQI`. `State` : I idle, D data, V voice.
- `DL_bitrate`, `UL_bitrate` : instantaneous interface bitrate (kbps). This is passive traffic, not an active throughput test.
- Active tests, if any were run: `Test_Status`, `PINGAVG`/`PINGMIN`/`PINGMAX` (ms), `TESTDOWNLINK`, `TESTUPLINK`.
- `CSI_RSRP`, `CSI_RSRQ`, `CSI_SNR`, `NRSTATE` : NR CSI-based values / NR state, when the handset exposes them.
- Missing value markers: empty field or `-`. The manual documents `Level = -201` for loss of coverage.

## Events file
`EventType` is the application's own classification. `EventDetails` is `source:target`, each written as
`MCC-MNC-TAC-node-cell-ARFCN-PCI`.

## Table I criteria (study-defined engineering bins, not 3GPP limits)
- RSRP (NR SS-RSRP or LTE RSRP), dBm: good > -85 ; acceptable -110 to -85 ; poor < -110.
- SINR (NR SS-SINR or LTE SNR), dB: good > 10 ; acceptable 0 to 10 ; poor < 0.
- RSRQ: descriptive only (no poor threshold). Throughput and RTT: only from active tests; distribution only.
- Mobility: report counts; success rate is not observable from the handset.

## Table II rows to report
`NR_SS_RSRP`, `NR_SS_RSRQ`, `NR_SS_SINR` : from records labelled 5G.
`LTE_RSRP`, `LTE_RSRQ`, `LTE_SINR` : LTE serving-cell values of the records labelled 4G pooled with the
LTE anchor values of the records labelled 5G.
