"""Helpers for distilling FastWAM (teacher) world-features into a fast VLA student.

This module is the public, *training-only* surface used by an external distillation
precompute (e.g. the starVLA `tools/fastwam_precompute_targets.py`) to:

1. Load a frozen FastWAM teacher from a Hydra config + checkpoint (`load_teacher`).
2. Build the exact frame-0 input the teacher expects from a pair of camera images
   (`build_frame0_image`) — per-cam resize, horizontal concat, normalize to [-1, 1].
3. Format the instruction prompt the teacher was trained on (`format_distill_prompt`).

The heavy lifting (frame-0 video-DiT hidden extraction) lives on the model itself:
``FastWAM.extract_world_features`` (REPA / Channel-1 target) and
``FastWAM.extract_future_latents`` (Channel-2 target).

Nothing here runs at student inference time.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch

# Must match `fastwam.datasets.lerobot.robot_video_dataset.DEFAULT_PROMPT` so the
# teacher sees the same instruction text distribution it was trained on.
DEFAULT_DISTILL_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)

# Dataset image contract (configs/data/libero_2cam.yaml): per-cam 224, hcat -> 224x448,
# left = agentview (`image`), right = wrist (`wrist_image`), normalized to [-1, 1].
CAMERA_RESIZE = 224
NUM_CAMERAS = 2


def format_distill_prompt(task: str) -> str:
    """Return the teacher-side instruction string for a raw LIBERO task description."""
    return DEFAULT_DISTILL_PROMPT.format(task=task)


def _to_chw_float(image: Any) -> torch.Tensor:
    """Coerce a single image (PIL.Image, HxWxC uint8 ndarray, or CHW tensor) to CHW float in [0, 1]."""
    if isinstance(image, torch.Tensor):
        t = image.float()
        if t.ndim == 3 and t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3):
            t = t.permute(2, 0, 1)  # HWC -> CHW
        if t.max() > 1.5:
            t = t / 255.0
        return t
    # numpy ndarray or PIL.Image
    try:
        import numpy as np

        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = arr[..., None].repeat(3, axis=-1)
            t = torch.from_numpy(arr.copy()).float()
            if t.ndim == 3 and t.shape[-1] in (1, 3):
                t = t.permute(2, 0, 1)
            if t.max() > 1.5:
                t = t / 255.0
            return t
    except ImportError:  # pragma: no cover
        pass
    # Assume PIL.Image
    from torchvision.transforms.functional import pil_to_tensor

    return pil_to_tensor(image.convert("RGB")).float() / 255.0


def build_frame0_image(
    images: Sequence[Any],
    size: int = CAMERA_RESIZE,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the teacher frame-0 input from per-camera images.

    Mirrors the dataset preprocessing: each camera is resized to ``size x size``,
    concatenated horizontally (left = agentview, right = wrist) into ``size x (N*size)``,
    and normalized to [-1, 1] via ``Normalize(0.5, 0.5)``.

    Args:
        images: Sequence of ``NUM_CAMERAS`` images (PIL.Image, HWC uint8 ndarray, or
            CHW tensor), camera-ordered ``[agentview, wrist]``.
        size: Per-camera square resize (default 224).
        device, dtype: Output placement.

    Returns:
        Tensor of shape ``[1, 3, size, NUM_CAMERAS * size]`` in [-1, 1].
    """
    from torchvision.transforms.functional import resize

    if len(images) != NUM_CAMERAS:
        raise ValueError(
            f"Expected {NUM_CAMERAS} camera images (agentview, wrist), got {len(images)}."
        )
    cams = []
    for img in images:
        chw = _to_chw_float(img)
        if chw.shape[0] != 3:
            raise ValueError(f"Each camera image must have 3 channels, got shape {tuple(chw.shape)}.")
        chw = resize(chw, [size, size], antialias=True)
        cams.append(chw)
    hcat = torch.cat(cams, dim=2)  # [3, size, N*size]
    hcat = (hcat - 0.5) / 0.5  # Normalize(mean=0.5, std=0.5) -> [-1, 1]
    return hcat.unsqueeze(0).to(device=device, dtype=dtype)


def load_teacher(
    config_dir: str,
    config_name: str = "sim_libero",
    overrides: Optional[Sequence[str]] = None,
    ckpt: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    return_cfg: bool = False,
):
    """Load a frozen FastWAM teacher from a Hydra config + checkpoint.

    Composes a *full* config (model configs interpolate ``data.*`` fields, so a
    ``task=`` override is required), instantiates the model, loads the checkpoint via
    ``model.load_checkpoint`` (same path as eval), and freezes it for inference.

    Args:
        config_dir: Absolute path to the FastWAM ``configs/`` directory.
        config_name: Base config to compose (default ``sim_libero``; ``train`` also works).
        overrides: Hydra overrides, e.g.
            ``["task=libero_uncond_2cam224_1e-4", "model.load_text_encoder=true"]``.
            Must set ``task`` so ``cfg.data`` is populated (model config interpolates it).
        ckpt: Path to the trained FastWAM checkpoint.
        device, dtype: Target placement / compute dtype.
        return_cfg: If True, also return the composed config.

    Returns:
        The eval-mode ``FastWAM`` model (with ``requires_grad_(False)``), or
        ``(model, cfg)`` if ``return_cfg``.
    """
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    overrides = list(overrides or [])
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides)

    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    if ckpt is not None:
        model.load_checkpoint(str(ckpt))
    model.eval()
    model.requires_grad_(False)
    return (model, cfg) if return_cfg else model
