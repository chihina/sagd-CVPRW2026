# End-to-End Shared Attention Estimation via Group Detection with Feedback Refinement

**CVPR 2026 Workshop on Eye and Gaze in Computer Vision (GAZE 2026)**

Chihiro Nakatani, Norimichi Ukita (Toyota Technological Institute), Jean-Marc Odobez (Idiap Research Institute / EPFL)

In collaboration with Prof. Jean-Marc Odobez (Idiap Research Institute / EPFL)

[[Project Page]](https://chihina.github.io/sagd-CVPRW2026-page/) [[arXiv]](https://arxiv.org/abs/2604.01714) [[Code]](https://github.com/chihina/sagd-CVPRW2026)

---

## Overview

This is the official implementation of our CVPRW 2026 paper. We propose a unified end-to-end approach for shared attention estimation via group detection with feedback refinement. Our method consists of three stages: individual attention estimation, initial shared attention generation through group detection, and refinement via feedback from initial attention predictions.

Experiments on three datasets (VideoCoAtt, VideoAttentionTarget, ChildPlay) demonstrate superior performance in both group detection and shared attention localization over competing baselines.

This codebase supports multiple datasets (e.g., GazeFollow, VideoCoAtt, ChildPlay, VideoAttentionTarget, UCO-LAEO, and Combined Social) through a shared experiment interface built on Hydra + PyTorch Lightning.

## Repository Structure

- `main.py`: primary entry point (Hydra-based)
- `src/conf/train.yaml`: default configuration loaded by `main.py`
- `src/experiments.py`: task and dataset dispatch (`train`, `val`, `test`, `predict`)

## Requirements

- Linux (recommended)
- Python 3.12+
- NVIDIA GPU + CUDA (recommended for training)

> You can still run on CPU by overriding `train.device=cpu`, but training will be much slower.

## Setup

### 1) Clone and enter the repository

```bash
git clone <your-repo-url>
cd mtgs
```

### 2) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3) Install dependencies

Use one of the following options:

```bash
pip install -r requirements.txt
```

## Configure Paths Before Running

`main.py` loads `src/conf/train.yaml` by default.

Important: the default config includes machine-specific absolute paths (for datasets and pretrained weights). You must provide valid paths for your environment.

You can either:

- edit `src/conf/train.yaml`, or
- override fields directly from the command line (recommended for reproducibility).

## Anonymous/Public Config Workflow

A shareable template is provided as `src/conf/main.yaml`.
`main.yaml` is intentionally anonymized and contains `/path/to/...` placeholders by design.

Use it like this:

```bash
cp src/conf/main.yaml src/conf/train.yaml
```

Then update placeholders (e.g., `/path/to/...`) in your local `src/conf/train.yaml` and run:

```bash
python main.py
```

## Run `main.py`

### Minimal training command (with command-line overrides)

```bash
python main.py \
	experiment.task=train \
	experiment.dataset=gazefollow \
	experiment.cuda=0 \
	data.gf.root=/path/to/gazefollow_extended \
	model.sharingan.gaze_weights=/path/to/gaze360_resnet18.pt \
	wandb.log=False
```

### Combined Social example

```bash
python main.py \
	experiment.task=train \
	experiment.dataset=combined_social \
	experiment.cuda=0 \
	data.gf.root=/path/to/gazefollow_extended \
	data.coatt.root=/path/to/VideoCoAtt_Dataset \
	data.vat.root=/path/to/videoattentiontarget \
	data.laeo.root=/path/to/ucolaeodb \
	data.childplay.root=/path/to/ChildPlay-gaze \
	model.sharingan.gaze_weights=/path/to/gaze360_resnet18.pt \
	wandb.log=False
```

### CPU-only fallback

```bash
python main.py train.device=cpu wandb.log=False
```

## Weights and Checkpoints

- If resuming training, set:
	- `train.resume=True`
	- `train.resume_from=/path/to/checkpoint.ckpt`

Example:

```bash
python main.py train.resume=True train.resume_from=/path/to/last.ckpt
```

## Logging (Weights & Biases)

By default, training config may enable W&B logging (`wandb.log=True`).

If you do not want W&B:

```bash
python main.py wandb.log=False
```

If you use W&B, authenticate first:

```bash
wandb login
```

## Outputs

- Checkpoints are saved under:
	- `checkpoints/<dataset>/<experiment_id>/`
- A resolved config snapshot is saved with each run.
- Lightning logs are written to `lightning_logs/`.

## Citation

```
@inproceedings{DBLP:conf/cvpr/NakataniKU26,
  author       = {Chihiro Nakatani and
                  Norimichi Ukita and
                  Jean-Marc Odobez},
  title        = {End-to-End Shared Attention Estimation via Group Detection with Feedback Refinement},
  booktitle    = {CVPRW},
  year         = {2026},
}
```

## Acknowledgments

This codebase is based on https://github.com/idiap/MTGS.
Thank you for authors contributions.
