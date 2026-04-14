# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multitask Diffusion-Transformer (DiT) Policy for robot manipulation. A PyTorch-based model that uses transformer-based diffusion/flow-matching to predict robot action trajectories, conditioned on vision (CLIP/DINOv3), text (CLIP), and proprioceptive state observations. Built around the LeRobot dataset format.

## Commands

### Setup
```bash
uv python install && uv sync          # Install Python and dependencies
uv sync --extra dev                    # Include dev tools (ruff, pre-commit)
uv sync --extra modal                  # Include Modal cloud training
pre-commit install                     # Set up pre-commit hooks
```

### Training
```bash
uv run -m multitask_dit_policy.train --dataset_path=/path/to/dataset --batch_size=16 --train_steps=2000
uv run -m multitask_dit_policy.train --help   # Show all config options
```

### Inference
```bash
uv run -m multitask_dit_policy.examples.inference --checkpoint_dir=path/to/checkpoint --dataset_path=/path/to/dataset
```

### Modal (cloud) training
```bash
uv run -m multitask_dit_policy.train_modal --dataset_path=modal/path --gpu_type=B200 --batch_size=320
```

### Linting & Formatting
```bash
ruff format .          # Format code
ruff check --fix .     # Lint with autofix
pre-commit run --all-files   # Run all pre-commit hooks
```

## Architecture

All source code lives under `src/multitask_dit_policy/`.

### Model (`model/`)
- **`model.py`** — `MultiTaskDiTPolicy(nn.Module)`: Top-level policy. Composes observation encoder + diffusion transformer + objective. Handles save/load via safetensors + draccus JSON config. Key methods: `forward()` (training loss), `select_action()` (inference with action queue caching).
- **`observation_encoder.py`** — Encodes vision (CLIP or DINOv3 via timm), text (CLIP via HuggingFace transformers), and robot state into a conditioning vector for the transformer.
- **`transformer.py`** — `DiffusionTransformer`: DiT-style transformer for noise/velocity prediction on 1D action trajectories. Supports RoPE. Uses AdaLN (adaptive layer norm) conditioned on diffusion timestep.
- **`objectives.py`** — `DiffusionObjective` (DDPM/DDIM) and `FlowMatchingObjective` (Euler/RK4 ODE). Both implement `compute_loss()` and `conditional_sample()`.

### Configuration (`utils/configuration.py`)
All config is dataclass-based using **draccus** (not argparse/hydra). `MultiTaskDiTConfig` is the policy config; `TrainConfig` wraps it with training params. Vision/text encoder configs use `draccus.ChoiceRegistry` for polymorphic dispatch (e.g., `--policy.observation_encoder.vision.type=clip`).

### Training (`train.py`, `train_modal.py`)
- `train.py` — Local training loop with LeRobotDataset, Adam optimizer, AMP support, wandb logging (if `WANDB_API_KEY` set).
- `train_modal.py` — Extends TrainConfig with Modal-specific GPU/compute params for cloud deployment.

### Utils (`utils/`)
- `utils.py` — Batch normalization (min-max / mean-std), device movement, dataset stats saving.

## Key Conventions

- **Python 3.12**, managed with **uv** (not pip/conda)
- **ruff** for linting and formatting: 120 char line length, double quotes, spaces
- Config passed via CLI using draccus `--key=value` syntax (dot-separated for nested: `--policy.transformer.num_layers=8`)
- Datasets must be local LeRobotDataset format (no HF Hub pulling)
- Model checkpoints contain: `model.safetensors`, `config.json`, `dataset_stats.json`
- Environment vars: `HUGGINGFACE_TOKEN` (required for dataset), `WANDB_API_KEY` (optional)
- CI checks for version bump on PRs (`.github/workflows/version-bump-check.yml`)
