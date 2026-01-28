# Google Colab Workflow

This document explains how to run your TG-SMN experiments on Google Colab without git/Drive sync conflicts.

## The Problem (Old Approach)

Previously, the notebook was:
1. Stored in Google Drive
2. Running git commands from within that same Drive directory
3. Creating conflicts as Colab tried to sync the notebook while git modified it

## The Solution (New Approach)

**Separate code from outputs:**

- **Repository**: Cloned to Colab's local storage (`/content/tg_smn`) - fast, no sync issues
- **Outputs**: Saved to Google Drive (`/content/drive/MyDrive/tg_smn_outputs/`) - persistent and accessible
- **Notebook**: Can live anywhere (Drive or local) - doesn't matter since it doesn't live inside the repo

## Architecture

```
/content/
├── tg_smn/                           # Git repo (LOCAL, fast)
│   ├── world_models/
│   ├── pyproject.toml
│   └── ...
└── drive/MyDrive/
    ├── tg_smn_outputs/               # Outputs (DRIVE, persistent)
    │   ├── othello/
    │   │   ├── ckpt_final.pt
    │   │   └── rollout_*.png
    │   └── 2048/
    │       ├── ckpt_final.pt
    │       └── rollout_*.png
    └── Run_StochMuZeroHarness.ipynb  # Notebook (DRIVE, persistent)
```

## Benefits

1. **No more git conflicts**: Repo lives in local Colab storage, separate from Drive
2. **Fast operations**: Git operations run on local SSD, not over Drive sync
3. **Persistent outputs**: Checkpoints and visualizations saved to Drive survive disconnects
4. **Clean workflow**: Fresh clone each session ensures you're always on latest code

## Usage

1. Upload `Run_StochMuZeroHarness_Benchmarks.ipynb` to Google Colab
2. Run all cells
3. Outputs appear in `Google Drive/tg_smn_outputs/`

### Important: Update the Git URL

In the notebook, replace `YOUR_USERNAME` with your actual GitHub username:

```bash
git clone -b \"$BRANCH\" https://github.com/YOUR_USERNAME/tg_smn.git \"$REPO_DIR\"
```

Or use the full HTTPS/SSH URL if your repo is elsewhere.

## What Happens Each Session

1. Mount Google Drive
2. Clone repo to `/content/tg_smn` (or pull if exists)
3. Install package with `pip install -e .`
4. Run training/eval scripts
5. Save outputs to Drive at `/content/drive/MyDrive/tg_smn_outputs/`

## Accessing Outputs

Your outputs are always available at:
```
Google Drive > tg_smn_outputs/
```

Even after Colab disconnects, all checkpoints and visualizations remain in Drive.

## Customization

Edit these variables in the notebook's setup cell:

```python
# Repo cloned to LOCAL Colab storage
REPO_DIR = '/content/tg_smn'

# Outputs saved to Drive (persistent)
OUTROOT = '/content/drive/MyDrive/tg_smn_outputs'

# Branch to use
BRANCH = 'stoch-muzero-harness'
```

## Troubleshooting

**Q: Notebook outputs are huge and slow to save**
- A: This is normal. Colab saves execution outputs in the notebook. The important data (checkpoints, images) is in Drive.

**Q: Can I run multiple experiments?**
- A: Yes! Change `OUTROOT` to different directories like:
  - `/content/drive/MyDrive/tg_smn_outputs_experiment1/`
  - `/content/drive/MyDrive/tg_smn_outputs_experiment2/`

**Q: What if I update the code on GitHub?**
- A: Just restart the Colab runtime. It will pull latest code on next run.

**Q: Do I need to push the notebook to Git?**
- A: No! Keep notebooks out of git (already in `.gitignore`). Store them in Drive or download locally.
