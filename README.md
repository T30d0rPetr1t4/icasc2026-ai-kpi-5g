# AI-assisted KPI evaluation of a live 5G network — data, code and figures

Supporting material for the paper

> T.-G. Furdui and T. Petrița, “AI-Assisted KPI Evaluation of a Live 5G Network: A Drive-Test Study on Romanian
> Mobile Operators,” in *Proc. International Conference on Advanced Scientific Computing
> ([ICASC 2026](https://www.itim-cj.ro/icasc/))*, Cluj-Napoca, Romania, Oct. 21–23, 2026, to appear.

How to cite it is under [Citation](#citation).

The paper proposes a framework for evaluating user-equipment-side KPIs from smartphone drive-test logs
(G-NetTrack Pro): data-quality validation, deterministic statistics computed by code, route mapping, and an
interpretation stage in which a general-purpose large language model (LLM) drafts an evidence-linked reading that is
checked against the code. This repository holds everything needed to reproduce the numbers and figures of the
proof-of-concept demonstration.

Clock times in the published logs are anonymised; durations, intervals and all statistics are unchanged
(see [`data/README.md`](data/README.md)).

## Contents

```
data/
  main/                      primary log (Orange Romania, Timișoara, Samsung Galaxy S22) + events file
  supplementary/             five logs used only to test Stage 1 (Table III)
src/
  analyze_gnettrack.py       reference analysis: Stages 1–3 and the screening part of Stage 4
  verify_llm_stage.py        compares the three LLM runs with the reference and re-computes 91 narrative figures
notebooks/
  figures.ipynb              reads the logs from data/ and draws every figure (executed, with outputs)
results/
  orange_..._results.json    audit, shares, KPI statistics (Table II), mobility, screening
  supplementary/*.json       Stage 1 audit of the supplementary logs (Table III)
  llm_stage_verification.json
  figures/                   PNG figures written by the notebook (600 dpi)
llm_stage/
  input/                     prompt (Fig. 2 of the paper), schema note, output contract
  run1/ run2/ run3/          the three independent LLM runs: results, narrative, code written by the model
```

## Quick start

```bash
pip install -r requirements.txt
jupyter lab notebooks/figures.ipynb          # or: jupyter notebook notebooks/figures.ipynb
```

The notebook reads the logs from `DATA_DIR` (default `../data`, set in its first code cell), so another folder of
G-NetTrack Pro logs with the same layout (`main/` with a log and its `_events.txt`, optional `supplementary/`)
can be analysed by changing that one line. To re-execute it from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/figures.ipynb
```

| Notebook output | Paper |
|---|---|
| `results/figures/fig_route_map.png` | Fig. 3 |
| `results/figures/fig_timeseries.png` | Fig. 4 |
| Table II and Table III cells | Tables II and III |
| `fig_ecdf.png`, `fig_shares.png`, `fig_screening.png` | supporting views, not in the paper |

Fig. 1 of the paper (a screenshot of the application) and Fig. 2 (the prompt, here `llm_stage/input/prompt.txt`)
are not generated from data.

## Reproducing the result files

From the repository root:

```bash
# primary log -> results/orange_2026.10.02_12.16.18_results.json
# (--figures also writes Fig. 3 and Fig. 4 to results/, identical to the notebook's files; git ignores them)
python src/analyze_gnettrack.py data/main/orange_2026.10.02_12.16.18.txt \
    data/main/orange_2026.10.02_12.16.18_events.txt --out results --figures --ref 45.7581 21.2289 "Piața Unirii"

# supplementary logs (Stage 1 only) -> results/supplementary/
python src/analyze_gnettrack.py data/supplementary/Logs_LTE_Gdansk/Orange_2026.07.26_13.47.11.txt \
    --out results/supplementary --audit-only               # repeat for each supplementary log

# LLM stage check -> results/llm_stage_verification.json
python src/verify_llm_stage.py data/main/orange_2026.10.02_12.16.18.txt \
    data/main/orange_2026.10.02_12.16.18_events.txt results/orange_2026.10.02_12.16.18_results.json
```

Tested with Python 3.13, pandas 3.0, NumPy 2.5, scikit-learn 1.9.1 and matplotlib 3.11. The Isolation Forest
result depends on the scikit-learn version.

## Definitions used by the reference script

- Plausibility ranges (outside = invalid, not poor): RSRP −140…−44 dBm; NR RSRQ −43…20 dB; LTE RSRQ −34…3 dB;
  SINR −23…40 dB.
- Table I bins: RSRP good > −85 dBm, poor < −110 dBm; SINR good > 10 dB, poor < 0 dB.
- RAT handling: in 5G-labelled (NSA) records the level/quality/SNR fields carry NR SS values and the `SecondCell_*`
  group carries the LTE anchor; in 4G-labelled records the main fields are the LTE serving cell. The two are never
  pooled.
- Stale position: a run of records with identical coordinates lasting ≥ 10 s and followed by a jump > 100 m
  (excluded from maps only).
- Time weighting: each inter-record interval is credited to the record that starts it; with the gap rule,
  intervals > 30 s are left out.
- Dwell: time between successive changes of serving-cell identity (operator code + node + cell).
- Return to the previous cell: event A→B followed by B→A within 30 s (cell = MCC-MNC-TAC-node-cell).
- Isolation Forest: scikit-learn, 200 trees, contamination 0.05, `random_state=0`, on the 5G-labelled records with
  all six radio values; used only to rank candidates (seed sensitivity is reported over seeds 0–9).

## LLM stage

`llm_stage/input/` holds what each run received besides the two log files: the fixed prompt of Fig. 2, a one-page
schema note (field meanings from the G-NetTrack Pro manual, Table I criteria, definition of the Table II rows) and
the requested JSON layout. Each `run*/` folder holds the model's `results.json`, `narrative.md`, the Python code it
wrote and executed (`analysis.py`, which keeps the absolute paths of the sandbox it ran in), one file of
supplementary tables and `timing.txt` (Unix time of its first and last action).

Model: Claude Opus 5.5 (Anthropic, identifier `claude-opus-5-5`), 5 October 2026, run as an agent with file access
and a Python tool, without fine-tuning or retrieval; sampling settings were not controlled. Each run received only
the files in `input/` and the two log files; the reference results were not accessible. Run 3 reported one deviation:
it re-read a truncated printout of its own command output from a temporary file outside the input folder.

File names and clock times inside `llm_stage/` are anonymised in the same way as the logs; nothing else in those
files was changed. All three runs
reproduce the 40 Table II values of the reference, and the 91 further figures quoted in their outputs are
re-computed by `src/verify_llm_stage.py` (summary at the end of the notebook).

As stated in the paper's acknowledgment, the same AI system was also used to write the analysis scripts.

## Data

See [`data/README.md`](data/README.md) for the files, handsets and what the logs contain.

## Citation

If you use the data, code or figures, please cite the paper. The reference below will be updated with the pages and
the DOI once the proceedings are published.

T.-G. Furdui and T. Petrița, “AI-Assisted KPI Evaluation of a Live 5G Network: A Drive-Test Study on Romanian Mobile
Operators,” in *Proc. International Conference on Advanced Scientific Computing (ICASC 2026)*, Cluj-Napoca, Romania,
Oct. 21–23, 2026, to appear.

```bibtex
@inproceedings{furdui2026icasc,
  author    = {Furdui, Tudor-Gabriel and Petri{\c{t}}a, Teodor},
  title     = {{AI}-Assisted {KPI} Evaluation of a Live {5G} Network: A Drive-Test Study on {Romanian} Mobile Operators},
  booktitle = {Proceedings of the International Conference on Advanced Scientific Computing (ICASC 2026)},
  address   = {Cluj-Napoca, Romania},
  month     = oct,
  year      = {2026},
  note      = {To appear},
  url       = {https://www.itim-cj.ro/icasc/}
}
```

GitHub also reads [`CITATION.cff`](CITATION.cff) and shows the reference under “Cite this repository”.

## License

- **Code** (`src/`, `notebooks/`, the `analysis.py` files in `llm_stage/`): [MIT](LICENSE).
- **Data, results, figures and documentation** (`data/`, `results/`, the other files in `llm_stage/`, the README
  files): [Creative Commons Attribution 4.0 International](LICENSE-CC-BY-4.0.txt) (CC BY 4.0). When you reuse them,
  cite the paper (see [`CITATION.cff`](CITATION.cff)).

Copyright © 2026 Tudor-Gabriel Furdui and Teodor Petrița.
