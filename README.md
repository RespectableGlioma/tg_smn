# TG-SMN Experiments

This repository packages the TG-SMN experimental code as importable Python modules.

The intended workflow is:

1. Use the **root notebook** `TG_SMN_Run.ipynb` as the top-level interface.
2. The notebook calls library functions to:
   - build environments (WikiText-2 permuted-vocab; multi-domain continual LM)
   - run baselines and TG-SMN variants
   - run hyperparameter / expert-count / seed sweeps
   - load and visualize results

## Quickstart (Colab)

1. Upload / clone this repo.
2. In Colab:

```python
!pip -q install -e .
```

3. Open `TG_SMN_Run.ipynb` (repo root) and run it top-to-bottom.

Notes:
- If you open the notebook from Google Drive or GitHub, Colab's working directory is often `/content`.
  The first cell in the notebook will search for the repo root and `cd` into it automatically.

## Notes on Hugging Face `datasets>=4.0`

As of `datasets>=4.0.0`, loading *script-based* datasets is no longer supported (you may see
`RuntimeError: Dataset scripts are no longer supported, but found <name>.py`).

This repo avoids script-based loaders in the multi-domain environment by using parquet-only
sources where needed (e.g. PTB). If you add new domains, prefer datasets that are available
as standard parquet/arrow exports.

## Outputs

Each run writes to an output directory you specify (Drive recommended in Colab).

A run directory contains:

- `config.json`
- `metrics.csv` (step-level controller + TIUR proxy signals)
- `eval.csv` (task-level evaluation)
- `summary.json`
- `ckpt_task*.pt`

The sweep runner additionally writes a `grid_results.csv` at the sweep root.

## Progress tracking

Long sweeps can take a while. `run_grid(...)` displays a progress bar and also writes:

- `grid_progress.json` (snapshot: total/done/skipped/failed, elapsed seconds, current job)
- `grid_results.jsonl` (append-only stream of results so partial progress isn't lost on runtime reset)

These do **not** include time-to-completion estimates by default; they report counts + elapsed time only.

## Design

- `tg_smn/envs/*`: environment builders
- `tg_smn/models/*`: dense and hierarchical-sparse Transformer LM
- `tg_smn/controllers/*`: fixed + learned RNN controllers
- `tg_smn/train/*`: training loops and logging
- `tg_smn/sweep.py`: grid sweeps + skip-existing
- `tg_smn/analysis.py`: loading + plotting helpers

