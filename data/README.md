# Data

Drive-test logs exported by G-NetTrack Pro (Gyokov Solutions), tab-separated text with one header row. Field
meanings: [G-NetTrack Pro manual](https://gyokovsolutions.com/manual-g-nettrack/) and
[`llm_stage/input/schema_note.md`](../llm_stage/input/schema_note.md).

## Clock times

Clock times (the `Timestamp` column and the time part of the file names) are anonymised by a constant offset, the
same for every file. Dates, the order of records, intervals between records, coordinates and every other field are
as exported.

## Primary log — `main/`

| File | Content |
|---|---|
| `orange_2026.10.02_12.16.18.txt` | 427 records, 1895 s, 270 header fields |
| `orange_2026.10.02_12.16.18_events.txt` | 79 mobility events (cell reselections and handovers as labelled by the application) |

Orange Romania subscription, Timișoara, 2 October 2026. Samsung Galaxy S22 (SM-S901B, Android
build BP2A.250605.031.A3), G-NetTrack Pro 33.8. No active download, upload or ping test was run. About 4.5 minutes after
the start the application left the foreground; from then on records are sparse and positions are missing or frozen
on much of the route (see the audit in `results/` and Section VI of the paper).

## Supplementary logs — `supplementary/`

Recorded in June and July 2026 while travelling, with G-NetTrack Pro 33.3. Used only to test whether Stage 1 copes
with other handsets, networks and technologies (Table III); no performance statement is made about these networks.

| Folder | Log(s) | Handset | Records |
|---|---|---|---|
| `Logs5G_RO-HU_BorderCrossing/` | `orange_2026.06.09_09.03.11` | Nokia X30 5G | 2365 |
| `Logs_5G_XR30_Dublin/` | `eir_2026.06.12_16.27.29` | Nokia X30 5G | 108 |
| `Logs_LTE_S22_Barca/` | `Orange_2026.07.12_14.48.32` | Samsung Galaxy S22 | 59 |
| `Logs_LTE_Gdansk/` | `Orange_2026.07.26_13.47.11`, `Orange_2026.07.27_11.07.59` | Samsung Galaxy S22 | 95, 152 |

Besides the main log and `_events.txt`, some folders contain the application's `_datatest.txt` (active test
results), `_cellinfo.txt` (cells seen) and `_stats.txt` (call statistics).

The RO–HU and Dublin logs contain one-hour backward clock steps (Table III); the constant offset leaves them as they
were.

## What the logs contain

- GPS coordinates, speed and altitude of the handset, as recorded.
- Serving and neighbour cell identities, channel numbers, band labels and signal levels as exposed by Android.
- The `IMEI`, `IMSI` and `MSISDN` columns are empty in every file. The `IP` column holds the private (10.x.x.x)
  address assigned by the operator's network.
