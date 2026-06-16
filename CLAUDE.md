# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fast-WAM: World Action Model on Wan2.2-TI2V-5B backbone, trained/evaluated on LIBERO and RoboTwin. Standalone package — no separate diffsynth/wan checkout required.

## Environment

- Conda env: `fastwam` (python 3.10, torch 2.7.1+cu128).
- Install: `pip install -e .` from repo root.
- LIBERO eval also requires the official LIBERO repo + `mujoco==3.3.2` in the same env.
- RoboTwin eval requires the upstream RoboTwin install + asset download. The repo's `third_party/RoboTwin` only contains the eval scaffolding; the policy plugin is wired by symlinking `experiments/robotwin/fastwam_policy` into `third_party/RoboTwin/policy/fastwam_policy`.
- `DIFFSYNTH_MODEL_BASE_PATH` controls where Wan checkpoints get downloaded/cached (default `./checkpoints`).

## Common commands

Preprocess (must run before any training, once per arch):
```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```

Precompute T5 text-embedding cache (per task config — must run before training):
```bash
python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
```

Training (DeepSpeed ZeRO-1 single-node, args after `<nproc>` are Hydra overrides):
```bash
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4
```
Multi-node: set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`. The script syncs a shared `RUN_ID` across nodes via a `torch.distributed.TCPStore` and writes outputs to `runs/{task}/{RUN_ID}`. There is also `train_zero2.sh`.

LIBERO evaluation (parallel manager, dispatches one Hydra job per (task_suite, task_id) across GPUs):
```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=8
```
Single LIBERO task (what the manager dispatches under the hood): `python experiments/libero/eval_libero_single.py task=... EVALUATION.task_suite_name=libero_spatial EVALUATION.task_id=0 ckpt=...`.

RoboTwin evaluation:
```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 ckpt=... EVALUATION.dataset_stats_path=... MULTIRUN.num_gpus=8
```

There is no test suite or linter in `pyproject.toml`. The closest "smoke test" is running the manager scripts above with a small `MULTIRUN.num_gpus`.

## Architecture

### Hydra config layering
All entrypoints (`scripts/train.py`, `experiments/*/eval_*.py`, the managers) load Hydra configs from `configs/`. Three-axis composition:
- `configs/data/*.yaml` — dataset + processor (image shapes, action/proprio dims, normalization, transforms, text-embed cache dir).
- `configs/model/*.yaml` — Wan2.2 video DiT + ActionDiT hyperparams. Three variants: `fastwam.yaml` (uncond, action-only output), `fastwam_idm.yaml` (inverse-dynamics), `fastwam_joint.yaml` (joint video+action). Each binds a `_target_` that points at a `runtime.create_*` factory.
- `configs/task/*.yaml` — wraps a (data, model) pair plus training hyperparams (`batch_size`, `learning_rate`, `num_epochs`, etc.). The task-name → `(data, model)` mapping lives in this layer.

`configs/train.yaml` is the training entrypoint base; `configs/sim_libero.yaml` and `configs/sim_robotwin.yaml` are eval bases that `defaults: - train` then override task and add an `EVALUATION` block. Eval configs reference `${eval_num_inference_steps}` etc. inherited from train.

### Code layout (`src/fastwam/`)
- `runtime.py` — factory functions (`create_fastwam`, `create_fastwam_idm`, `create_fastwam_joint`, `create_wan22_model`) referenced by `_target_` in model configs, plus `run_training` (the function `scripts/train.py` calls).
- `trainer.py` — `Wan22Trainer`. Wraps `accelerate.Accelerator`, builds optimizer + LR schedule, handles checkpointing, eval, wandb logging.
- `models/wan22/` — model code:
  - `wan22.py` — `Wan22Core` (text encoder + VAE + video DiT, loaded from Wan-AI HF weights).
  - `wan_video_dit.py`, `wan_video_vae.py`, `wan_video_text_encoder.py` — Wan2.2 components.
  - `action_dit.py` — action-prediction DiT trunk; pretrained weights are produced offline by `scripts/preprocess_action_dit_backbone.py` (interpolated from Wan22 DiT) and loaded via `action_dit_pretrained_path` in the model config.
  - `mot.py` — multi-modal-of-thought / mixed-attn block shared by the three FastWAM variants.
  - `fastwam.py`, `fastwam_idm.py`, `fastwam_joint.py` — top-level model classes assembled in the runtime factories.
- `datasets/lerobot/` — LeRobot-format dataset (`base_lerobot_dataset.py`, `robot_video_dataset.py`) plus `processors/fastwam_processor.py` and `transforms/`. Handles multi-camera concat (e.g. 2-cam → horizontal 224x448), action/state normalization (`min/max`, optional stepwise), proprio/action delta-masking, and reads precomputed T5 embeddings from `text_embedding_cache_dir`.
- `utils/config_resolvers.py` — registers OmegaConf resolvers used inside the YAMLs.

### MOT model variants
- `fastwam` (uncond): action-only output head; `video_dit_config.action_conditioned: false`. The "no test-time imagination" config from the paper.
- `fastwam_idm`: inverse-dynamics — action conditioned on past + future video.
- `fastwam_joint`: joint video+action denoising.
The `mot_checkpoint_mixed_attn` flag in model config gates gradient checkpointing on the mixed-attention layers; the per-task config sets it `false` for LIBERO at batch_size=16 (faster) and the default `true` keeps memory bounded for RoboTwin.

### Training launcher
`scripts/train_zero1.sh` is the canonical entrypoint. It:
1. Parses `task=<name>` (or `--config-name task/<name>`) from the Hydra overrides to derive `TASK_BASENAME`.
2. Generates or syncs a shared `RUN_ID` (timestamp). Multi-node sync uses `MASTER_ADDR:MASTER_PORT+11`; defaults timeout 180s.
3. Runs `accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml scripts/train.py output_dir=./runs/{task}/{run_id} wandb.name={task} <overrides>`.
DeepSpeed JSON lives at `scripts/ds_configs/ds_zero1_config.json` (zero stage 1, no offload). `mixed_precision` is controlled from the training Hydra config, NOT the accelerate yaml (which sets it to null).

### Eval managers
`run_libero_manager.py` / `run_robotwin_manager.py` build a task list (LIBERO uses `libero.benchmark.get_benchmark_dict()`; RoboTwin uses task names from the policy YAML), dispatch up to `MULTIRUN.num_gpus * MULTIRUN.max_tasks_per_gpu` worker processes, each invoking the corresponding `eval_*_single.py` with the parent's Hydra overrides minus a blocked set (`task`, `ckpt`, `gpu_id`, the `EVALUATION.task_*` keys, anything under `MULTIRUN.` or `hydra.`). Pass non-blocked overrides through normally.

### Dataset stats
On the first training run for a new dataset config, `pretrained_norm_stats` in `configs/data/*.yaml` must be `null`; the trainer writes `runs/{task}/{run_id}/dataset_stats.json`. Set `pretrained_norm_stats` to that path for subsequent runs to skip the (slow) initial pass. Released checkpoints ship with their own `*_dataset_stats.json` — pass via `EVALUATION.dataset_stats_path=...`.

## Output layout

- `runs/{task}/{run_id}/` — training: checkpoints, `dataset_stats.json`, logs, optional wandb run dir.
- `evaluate_results/{libero,robotwin}/{task}/{timestamp}/` — eval rollouts + summary.
- `data/text_embeds_cache/` — precomputed T5 embeddings, one subdir per task as set by `data.train.text_embedding_cache_dir`.
- `checkpoints/` — Wan2.2 base, ActionDiT preprocessed weights, optional released FastWAM checkpoints.

## Gotchas

- The `task=` Hydra override is special-cased by `train_zero1.sh` — pass it without quoting: `task=libero_uncond_2cam224_1e-4`, not `--config-name=task/...`.
- `MULTIRUN.num_gpus` defaults to 8 in both `sim_libero.yaml` and `sim_robotwin.yaml`. Override on smaller hosts.
- RoboTwin eval has `EVALUATION.skip_get_obs_within_replan=true` for speed — saved videos look low-FPS by design. Set false to rerender.
- LIBERO evaluation uses `unseen` instruction split by default (matching Motus); `EVALUATION.instruction_type=seen` switches to the easier split.
- Some model configs reference fields under `data.train.processor` (e.g. `action_output_dim`); Hydra interpolation requires `data:` to be set, so don't load a model config standalone.
