# Distilling FastWAM (teacher) → fast starVLA QwenGR00T student on LIBERO

Concrete, execution-ready. Two repos, both already present:
`/Users/termanteus/workspace/uwm/labs/FastWAM` (teacher) and
`/Users/termanteus/workspace/uwm/labs/starVLA` (student).

## 0. Thesis

FastWAM's edge = a *training objective* (future-video prediction + video↔action
co-attention on a Wan-pretrained trunk), not multi-frame test-time input. At inference
it's already a single-frame VLA carrying world-shaped weights. Transplant that into a
fast `QwenGR00T` flow-matching student via **two training-only channels**, both dropped
at inference:
- **Channel 1 (REPA):** align student vision-token features to FastWAM frame-0
  video-DiT hidden states (`Z_teacher`). Zero inference cost.
- **Channel 2 (future head):** small flow-matching head predicts future video latents
  (Wan VAE space), supervised by real future frames (`Fut_real`). Replicates the
  world-model objective. Dropped at inference.

Goal: match/approach FastWAM LIBERO success at a large speedup (student = Qwen-VL +
4 tiny flow steps vs teacher's 1×5B prefill + 10×1.1B steps).

---

## 0.5 System figure

```
══════════════════════════════════════════════════════════════════════════════════════
 PHASE A — OFFLINE TEACHER EXTRACTION   (conda env: fastwam · frozen FastWAM · run once)
══════════════════════════════════════════════════════════════════════════════════════
  LIBERO frame (agentview+wrist)
        │  FastWAM preprocessing (2-cam → 224×448, [-1,1])
        ▼
   Wan VAE.encode ──► z0  [48,14,28]
        │                                  T5 text cache [128,4096] ─┐
        ▼                                  proprio[8]→proprio_enc ───┤ (+1 ctx token)
   video_expert.pre_dit (t=0)  ◄──────────────────────────────────────┘
        │
   MoT video blocks 0..L   (frame-0 self-attn only; mask blocks →future)
        │
        ├──► TAP @ layer L ─────────►  Z_teacher  [7,14,3072]      ─┐
        │                                                          ├─►  write cache/
   (also) Wan VAE.encode(future frames +4..+32) ─► Fut_real [8,48,14,28] ─┘
                                                                key = (suite,task,ep,frame)
══════════════════════════════════════════════════════════════════════════════════════
 PHASE B — STUDENT DISTILLATION TRAINING        (conda env: starVLA · 8×GPU · accelerate)
══════════════════════════════════════════════════════════════════════════════════════
  cur frame (agentview,wrist 224×224) + instruction + state[7]
        │
   Qwen3-VL-4B  (output_hidden_states=True)
        │   ├─ image tokens (per cam, grid from image_grid_thw)
        │   │        │ tap @~0.5 depth ─► RepaProjector(d→3072)
        │   │        │                         │
        │   │        │            ╔════════════▼═══════════╗   Z_teacher (cache, resize→grid)
        │   │        │            ║  L_REPA = 1-cos        ║◄──────── stop-grad
        │   │        │            ╚════════════════════════╝
        │   └─ last_hidden [B,seq,2048] ─────────────┬───────────────┐
        │                                            ▼               ▼
        │                              ┌─ Future flow head      GR00T action head
        │                              │   (small DiT, 48-ch)   (flow-matching, chunk 8)
        │                              ▼                              │
        │                    ╔══════════════════╗            ╔════════▼═════════╗
        │                    ║ L_future (FM vel)║            ║  L_flow_action   ║
        │                    ╚════════▲═════════╝            ╚══════════════════╝
        │                             │ Fut_real (cache)                │
        │                                                               │
   TOTAL  L = L_flow_action + 0.5·L_REPA + 0.5·L_future  [+0.5·L_futureREPA, gated]
══════════════════════════════════════════════════════════════════════════════════════
 PHASE C — INFERENCE   (conda env: libero · student only · fast · single frame)
══════════════════════════════════════════════════════════════════════════════════════
  frame+instruction+state ─► Qwen-VL ─► GR00T action head (4 flow steps) ─► action[8,7]
  NO 5B video DiT.  NO REPA/future heads.  replan every 8 steps.
══════════════════════════════════════════════════════════════════════════════════════
```

---

## 1. Teacher facts (verified — FastWAM uncond)

- Inference (`fastwam.py:905-1048`): VAE-encode current frame → video DiT once at t=0 →
  `prefill_video_cache` (`mot.py:257-341`) → 1.1B ActionDiT × `num_inference_steps`
  (eval default 10) attending frame-0 K/V. No future at test time.
- Training (`fastwam.py:448-568`): `loss = lambda_video*loss_video + lambda_action*loss_action`,
  `lambda_video` default **1.0** (`runtime.py:156`), not zeroed by uncond cfg
  (`configs/model/fastwam.yaml:57-58`). Denoises 9 future latent frames (`first_frame_causal`,
  `wan_video_dit.py:501-505`); whole MoT incl. 5B DiT finetuned (`trainer.py:82-95`).
- **Frame-0 = clean conditioning, zero recon loss** (`fastwam.py:467-468, 534-537`); masked
  off the future but read by future+action tokens → its representation is shaped purely to
  be good conditioning ⇒ `Z_teacher` carries task/dynamics/action structure, not appearance.
  Frame-0 hidden states are **invariant to whether future tokens are present** ⇒ single-frame
  extraction == training-time features (faithful cache).
- I/O: 2-cam concat **224×448**; T5 [128,4096]; proprio **[8]** → `Linear(8→4096)` appended as
  one context token (`fastwam.py:59,219-235`); action **[32,7]**. Frame-0 latent grid:
  VAE/16 → 14×28, patch[1,2,2]/2 → **7×14 = 98 tokens, dim 3072**. The 448 width = 2 cams →
  teacher grid is `[7×7 agentview | 7×7 wrist]`.

## 2. Student facts (verified — starVLA `QwenGR00T`)

- Framework `starVLA/model/framework/VLM4A/QwenGR00T.py` (flow-matching, last-layer
  cross-attn DiT). Backbone **Qwen3-VL-4B-Instruct**, `vl_hidden_dim=2048`.
- Action head `starVLA/model/modules/action_model/GR00T_ActionHeader.py:312-363`
  (velocity-MSE flow matching). `action_dim=7`, `action_horizon=8`, `state_dim=7`,
  `num_inference_timesteps=4`, `num_target_vision_tokens=32` (learnable future tokens).
- `forward` (`QwenGR00T.py:170-218`) runs Qwen with `output_hidden_states=True` →
  `qwenvl_outputs.hidden_states` = tuple(37); returns `{"action_loss": ...}`. **This is
  where aux losses are added.** Mid-depth tap: `hidden_states[1 + num_layers//2]`.
- Images: 2 cams (`obs: ["image_0"]` + auto wrist), each resized **224×224**, encoded
  **separately** by Qwen-VL (NOT concatenated). Per-cam patch grid recoverable from the
  processor's `image_grid_thw` (post patch-merge, e.g. ~8×8 or ~16×16 tokens/cam).
- Train: `starVLA/training/train_starvla.py` + YAML
  `examples/LIBERO/train_files/starvla_cotrain_libero.yaml` (OmegaConf + CLI dotlist).
  Loss routed via `base_framework.compute_loss` (tag `vla`). `_train_step`
  (`train_starvla.py:~300`) does `output = model.forward(batch); loss = output["action_loss"]`.
- Eval: `examples/LIBERO/eval_files/eval_libero.py` + `model2libero_interface.py`; replan
  every `action_horizon` (8) steps; 4 flow steps (or ddim 10). Action mode `delta_qpos`,
  norm stats in `dataset_statistics.json`. **Student keeps its OWN action normalization** —
  we do NOT do action distillation, so no need to import FastWAM action stats.

---

## 3. The critical integration: cache-key alignment + data contract

Teacher and student read **different LeRobot copies** of LIBERO (FastWAM's
`configs/data/libero_2cam.yaml` dirs vs starVLA `playground/Datasets/LIBERO`, gr00t format).
Global/internal indices will **not** match. Fix by keying every cache entry on canonical
LIBERO identity:

```
KEY = (task_suite, task_id, episode_id, frame_id)     # e.g. "libero_spatial/3/000012/000045"
```

Both sides can produce this:
- **Teacher (Phase A):** iterate FastWAM's LeRobot; FastWAM's dataset exposes
  `episode_index` + frame offset within episode (`base_lerobot_dataset.py`), and the
  task suite/id come from the dataset dir + `meta/episodes.jsonl`/`tasks.jsonl`.
- **Student (Phase B):** starVLA's `gr00t_lerobot/datasets.py:1357-1404`
  `all_steps[index] = (trajectory_id, base_index)` → `(episode_id, frame_id)`; suite/task
  from the dataset path / mix.

**Action item M1:** write a tiny `canonical_key(sample)` helper in *each* repo and assert,
on a 100-sample overlap, that decoding the same KEY yields the *same RGB frame* (pixel MSE
≈ 0 after identical resize). This is the single highest-risk step — do it first.

**Image-content contract (for REPA correspondence):** REPA aligns *features*, so per-encoder
pixel normalization may differ (teacher [-1,1]; Qwen-VL its own mean/std). What MUST match
is the **underlying crop/content** per camera: same source frame, same agentview/wrist, same
resize target (224). Both already use 224 — keep crops identical (no extra random crop on the
distill path; use val transforms).

---

## 4. Channel 1 — REPA (concrete)

- **Teacher target:** `Z_teacher[KEY]` = frame-0 hidden at tap layer L (start L=18/30),
  reshaped `[7,14,3072]` → split into per-cam `[7,7,3072]` halves (cols 0:7 agentview,
  7:14 wrist).
- **Student source:** Qwen image tokens at LLM layer `1+18` (≈0.5 depth), sliced per camera
  via `image_grid_thw`, reshaped `[B, hs, ws, 2048]`.
- **Projector:** `RepaProjector: 2048 → 2048 → 3072` (SiLU). Project student up; **stop-grad
  teacher**.
- **Loss:** resize teacher per-cam map `[7,7]→[hs,ws]` (bilinear), per-patch negative cosine,
  averaged over both cams + batch. `λ_repa = 0.5`, constant, from step 0.
- **v2 option:** distill the action-expert K/V the policy actually reads (`mot.py:412-419`)
  instead of hidden `x`; or multi-layer. Gate behind M5.

## 5. Channel 2 — future-latent head (concrete)

- **Target:** `Fut_real[KEY]` = Wan-VAE latents of real future frames at teacher strides
  (+4,+8,…,+32 → 8 latents `[8,48,14,28]`), **cached offline in Phase A** (avoids loading
  Wan VAE in the student env).
- **Head:** small flow-matching DiT (4–6 layers, d=2048→internal 768, out 48-ch patches),
  cross-attends `last_hidden` (`vl_embs`), conditioned on flow timestep. Frame-0 not
  predicted (it's the given current frame). Reuse starVLA's
  `flow_matching_head/cross_attention_dit.py` as the block.
- **Loss:** velocity flow-matching to match teacher's objective:
  `L_future = E_{t,ε} || head(z_t,t,vl_embs) - (ε - z0) ||²` over the 8 future frames; same
  shift-5.0 continuous scheduler family. `λ_fut = 0.5` with ~5% warmup; optional stop-grad
  into vision encoder for first ~2k steps.
- **Rejected:** `Fut_teacher` (teacher *denoised* latents) as the target — it's
  `real + teacher_error`; real frames dominate.
- **future-REPA (kept, gated, §M5):** align the future-head hidden states to the teacher's
  *future-frame* hidden states (analog of `Z_teacher`). Needs the teacher's full multi-frame
  forward (~10× extraction + much larger cache) → enable only after M3 if budget allows.
  Term `λ_futrepa` (~0.5), off by default.

## 6. Combined loss

```
L = L_flow_action  +  0.5·L_REPA  +  0.5·L_future  [ + 0.5·L_futureREPA (gated) ]
```
Monitor grad-norms; if `L_future` grad > 2× action-loss grad, halve `λ_fut`. Channel flags
in config so each is independently toggleable (M2/M3 ablations) and all vanish at inference.

---

## 7. Concrete code artifacts

### 7.1 Teacher: hidden-state tap (FastWAM repo)

`src/fastwam/models/wan22/mot.py` — add (non-breaking; mirrors `prefill_video_cache`):
```python
@torch.no_grad()
def extract_video_hidden(self, video_tokens, video_freqs, video_t_mod,
                         video_context_payload, video_attention_mask, tap_layer):
    expert = self.mixtures["video"]; x = video_tokens
    for layer_idx in range(tap_layer + 1):
        block = expert.blocks[layer_idx]
        (q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, _) = \
            self._build_expert_attention_io(expert=expert, block=block, x=x,
                                            freqs=video_freqs, t_mod=video_t_mod)
        mixed = self._mixed_attention(q_cat=q, k_cat=k, v_cat=v,
                                      attention_mask=video_attention_mask)
        x = self._apply_post_with_optional_checkpoint(
            block=block, residual_x=residual_x, gate_msa=gate_msa, shift_mlp=shift_mlp,
            scale_mlp=scale_mlp, gate_mlp=gate_mlp, use_gradient_checkpointing=False,
            mixed_slice=mixed, context_payload=video_context_payload)
    return x   # [B, Sv, 3072] at layer tap_layer
```

`src/fastwam/models/wan22/fastwam.py` — add (mirrors `infer_action:960-1022`):
```python
@torch.no_grad()
def extract_world_features(self, input_image, context, context_mask,
                           proprio=None, tap_layer=18, tiled=False):
    self.eval()
    if input_image.ndim == 3: input_image = input_image.unsqueeze(0)
    input_image = input_image.to(self.device, self.torch_dtype)
    z0 = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
    fuse = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
    context = context.to(self.device, self.torch_dtype)
    context_mask = context_mask.to(self.device, torch.bool)
    if proprio is not None:
        context, context_mask = self._append_proprio_to_context(
            context, context_mask, proprio.to(self.device, self.torch_dtype))
    t0 = torch.zeros((z0.shape[0],), dtype=z0.dtype, device=self.device)
    vp = self.video_expert.pre_dit(x=z0, timestep=t0, context=context,
            context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=fuse)
    Sv = vp["tokens"].shape[1]
    vmask = self.video_expert._build_video_attention_mask(   # video-only square mask
                video_seq_len=Sv,
                video_tokens_per_frame=int(vp["meta"]["tokens_per_frame"]),
                device=vp["tokens"].device)
    x = self.mot.extract_video_hidden(
            video_tokens=vp["tokens"], video_freqs=vp["freqs"], video_t_mod=vp["t_mod"],
            video_context_payload={"context": vp["context"], "mask": vp["context_mask"]},
            video_attention_mask=vmask, tap_layer=tap_layer)
    tpf = int(vp["meta"]["tokens_per_frame"])              # =98 for 224×448
    h, w = 7, 14                                           # confirm from vp["meta"]
    return x[:, :tpf].reshape(x.shape[0], h, w, x.shape[-1])   # [B,7,14,3072]
```
*(Confirm the exact mask-builder name/signature and `meta` grid keys when implementing;
fall back to building the `first_frame_causal` mask inline if the helper isn't public.)*

### 7.2 Teacher: precompute script (FastWAM repo)

`scripts/precompute_distill_cache.py` (model on `precompute_text_embeds.py` infra:
`_init_distributed`, `_atomic_torch_save`, Hydra `@hydra.main(config_name="train")`,
`torchrun` sharding). Per **unique (suite,task,ep,frame)** frame-0:
```
for KEY, sample in iterate_frame0(dataset, dedupe_by=KEY, shard=rank::world):
    img   = sample.frame0_image          # [1,3,224,448] in [-1,1]  (FastWAM preprocessing)
    ctx,m = load_text_cache(sample.prompt)            # reuse existing T5 cache
    pro   = sample.proprio0               # [1,8] normalized
    Z = model.extract_world_features(img, ctx, m, pro, tap_layer=L)   # [1,7,14,3072]
    Fut = model.vae.encode(future_frames(sample, strides=[4,8,..,32]))# [1,8,48,14,28]
    _atomic_torch_save({"Z": Z.bf16().cpu(), "Fut": Fut.half().cpu()},
                       cache_dir / f"{KEY}.pt")
```
Run with `--device cuda --dtype bfloat16`; dedupe so each episode timestep is encoded once.

### 7.3 Student: REPA module (starVLA repo)

`starVLA/model/modules/distill/repa.py`:
```python
class RepaProjector(nn.Module):
    def __init__(self, d_in=2048, d_out=3072, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, d_out))
    def forward(self, x): return self.net(x)

def repa_loss(student_tok, z_teacher, grid_hw, proj):   # per camera
    B = student_tok.shape[0]; hs, ws = grid_hw
    s = proj(student_tok).reshape(B, hs, ws, -1).permute(0, 3, 1, 2)   # [B,3072,hs,ws]
    t = F.interpolate(z_teacher.permute(0, 3, 1, 2), size=(hs, ws),
                      mode="bilinear", align_corners=False).detach()    # [B,3072,hs,ws]
    s = s.flatten(2).transpose(1, 2); t = t.flatten(2).transpose(1, 2)
    return (1.0 - F.cosine_similarity(s, t, dim=-1)).mean()
```

### 7.4 Student: integrate into `QwenGR00T.forward` (starVLA repo)

`starVLA/model/framework/VLM4A/QwenGR00T.py:170-218` — after the existing Qwen call:
```python
qo = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)
last_hidden = qo.hidden_states[-1]
action_loss = self.action_model(last_hidden_rep, actions_rep, state_rep)   # unchanged

losses = {"action_loss": action_loss, "flow_loss": action_loss.detach()}
if self.cfg.distill.repa:
    mid = qo.hidden_states[1 + len(self.qwen_layers)//2]
    tok_a, grid_a = slice_image_tokens(mid, qwen_inputs["image_grid_thw"], cam=0)
    tok_w, grid_w = slice_image_tokens(mid, qwen_inputs["image_grid_thw"], cam=1)
    Z = batch["Z_teacher"]                       # [B,7,14,3072] from cache loader
    lr = 0.5*(repa_loss(tok_a, Z[..., :7, :], grid_a, self.repa_proj)
              + repa_loss(tok_w, Z[..., 7:, :], grid_w, self.repa_proj))
    losses["action_loss"] = losses["action_loss"] + self.cfg.distill.lambda_repa*lr
    losses["repa_loss"] = lr.detach()
if self.cfg.distill.future:
    lf = self.future_head.flow_loss(last_hidden, batch["Fut_real"])   # cached targets
    losses["action_loss"] = losses["action_loss"] + self.cfg.distill.lambda_fut*lf
    losses["future_loss"] = lf.detach()
return losses
```
`slice_image_tokens` uses `image_grid_thw` (post merge-size) to find each camera's token
span and `(h,w)`. Build `repa_proj` / `future_head` in `__init__` gated by `cfg.distill`.

### 7.5 Student: cache loader + dataloader (starVLA repo)

`gr00t_lerobot/datasets.py` `_pack_sample`: compute `KEY` from `(trajectory_id, base_index)`
+ suite/task, then `blob = torch.load(cache_dir/f"{KEY}.pt")`; add
`sample["Z_teacher"] = blob["Z"]`, `sample["Fut_real"] = blob["Fut"]`. Collate as tensors.

### 7.6 Config additions (starVLA YAML)

`examples/LIBERO/train_files/starvla_distill_libero.yaml` (extends the cotrain yaml):
```yaml
distill:
  repa: true
  future: true
  future_repa: false
  lambda_repa: 0.5
  lambda_fut: 0.5
  lambda_futrepa: 0.5
  teacher_tap_layer: 18
  student_tap_layer: 18
  cache_dir: ./playground/Datasets/LIBERO_distill_cache
  future_strides: [4, 8, 12, 16, 20, 24, 28, 32]
```

---

## 8. Caching sizing (Phase A output)

- `Z_teacher`: `7·14·3072·2 B ≈ 0.59 MB/frame` (bf16, 1 tap layer).
- `Fut_real`: `8·48·14·28·2 B ≈ 3.0 MB/frame` (fp16).
- LIBERO ≈ 300k unique frames ⇒ ~175 GB (Z) + ~900 GB (Fut). **Mitigate:** cache `Fut`
  only at a frame **stride** (every 4th anchor → ~225 GB) or for a task-suite subset first.
  `Z` at every frame is fine. Keep one tap layer for v1.

---

## 9. Remote GPU runbook

Assume repos synced to the GPU host. Three conda envs: **fastwam** (teacher/preprocess),
**starVLA** (train), **libero** (eval). `cd` paths are absolute.

**0) One-time setup**
```bash
# teacher env (already used by FastWAM)
conda activate fastwam   # torch 2.7.1+cu128; pip install -e /…/FastWAM
# student env
conda create -n starVLA python=3.10 -y && conda activate starVLA
cd /…/starVLA && pip install -r requirements.txt && pip install flash-attn --no-build-isolation && pip install -e .
bash examples/LIBERO/data_preparation.sh          # downloads starVLA LIBERO
# eval env
conda create -n libero python=3.10 -y && conda activate libero && pip install libero mujoco==3.2.3
```

**1) M0 baselines (sanity before any distill)**
```bash
# student baseline (no distill)
conda activate starVLA && cd /…/starVLA
bash examples/LIBERO/train_files/run_libero_train.sh   # or accelerate launch … train_starvla.py
# teacher eval reproduces on 1 suite
conda activate fastwam && cd /…/FastWAM
python experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

**2) M1 key-alignment check** — run the `canonical_key` overlap assert (pixel-MSE≈0) across
both dataloaders before extraction.

**3) Phase A — precompute teacher cache** (fastwam env, multi-GPU)
```bash
conda activate fastwam && cd /…/FastWAM
torchrun --standalone --nproc_per_node=8 scripts/precompute_distill_cache.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  +distill.tap_layer=18 +distill.future_strides=[4,8,12,16,20,24,28,32] \
  +distill.cache_dir=/…/starVLA/playground/Datasets/LIBERO_distill_cache
```

**4) Phase B — distillation training** (starVLA env, 8×GPU)
```bash
conda activate starVLA && cd /…/starVLA
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/LIBERO/train_files/starvla_distill_libero.yaml \
  --framework.name QwenGR00T --run_root_dir ./playground/Checkpoints --run_id libero_distill_v1 \
  --distill.repa true --distill.future true
```

**5) Phase C — eval** (libero env)
```bash
conda activate libero && cd /…/starVLA
python examples/LIBERO/eval_files/eval_libero.py --task_suite_name libero_spatial \
  --num_trials_per_task 10 --pretrained_path ./playground/Checkpoints/libero_distill_v1
# report success/suite + per-step latency & flow-step count vs teacher
```

---

## 10. Milestones (go/no-go)

1. **M0** baselines reproduce (student ±2% of published; teacher 1-suite OK).
2. **M1** key alignment verified (pixel-MSE≈0 on overlap). *Hard gate — blocks everything.*
3. **M2** Channel 1 alone: +REPA neutral-or-better on `libero_spatial`; PCA overlay of
   `Z_teacher` is object-aligned; student cosine rises.
4. **M3** Channel 2 alone: future head trains (L_future↓, action loss intact), ≥ baseline.
5. **M4** both, full LIBERO: success ≥ FastWAM−3% AND ≥2× faster (count heavy passes + wall-clock).
6. **M5** (gated) future-REPA / K/V target / multi-layer — only if it beats M4 enough to
   justify the extra extraction + cache.

## 11. Risks

- **Key misalignment** (different LeRobot copies) → M1 hard gate; anchor on canonical LIBERO id.
- **Qwen image-token slicing** wrong → verify `image_grid_thw`→grid mapping on one batch
  (overlay) before trusting REPA.
- **Grid mismatch per cam** (teacher 7×7 vs student hs×ws) → bilinear-resize teacher; never
  downsample student.
- **Future head into 48-ch Wan latents hard to learn** → warmup λ_fut, stop-grad-to-encoder
  early; fallback to feature-space future match.
- **Aux losses corrupt action head** → grad-norm caps, channel-toggle ablations (M2/M3).
- **Disk blowup (Fut_real)** → stride/subset (§8).
- **Speed must actually improve** → it does by construction (no 5B prefill, 4 vs 10 steps);
  verify by counting, not assuming.

## 12. File reference index

- Teacher: `fastwam.py:905-1048` (infer), `448-568` (train loss), `467-468/534-537`
  (frame-0 clean+excluded), `59,219-235` (proprio); `mot.py:257-341` (prefill),
  `412-419` (action reads frame-0 K/V), `447-556` (joint); `wan_video_dit.py:501-505`
  (mask), `:509` (pre_dit); `runtime.py:156-157`; `configs/{model/fastwam,data/libero_2cam}.yaml`;
  `scripts/precompute_text_embeds.py` (infra to mirror); eval `experiments/libero/*`.
- Student: `QwenGR00T.py:170-218` (forward/hook), `GR00T_ActionHeader.py:312-363` (flow loss),
  `:365-415` (predict), `:257-264` (state enc); `train_starvla.py:~288-330` (loop);
  `base_framework.py:145-181` (compute_loss); `gr00t_lerobot/datasets.py:1357-1404`
  (indexing/_pack_sample); `flow_matching_head/cross_attention_dit.py` (future-head block);
  cfg `examples/LIBERO/train_files/starvla_cotrain_libero.yaml`; eval
  `examples/LIBERO/eval_files/{eval_libero,model2libero_interface}.py`.
```
