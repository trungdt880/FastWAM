# Distilling FastWAM (teacher) → fast starVLA QwenGR00T student on LIBERO

Concrete, execution-ready. Two repos, both already present:
`/Users/termanteus/workspace/uwm/labs/FastWAM` (teacher) and
`/Users/termanteus/workspace/uwm/labs/starVLA` (student).

> **Revision note (v2):** Verified every code claim against both repos. Several signatures, mask-builder names, `meta` keys, the future-frame definition, the student forward signature, and the cache key derivation were wrong or underspecified in v1 and are corrected below. The two biggest *real* risks — (a) recovering a per-camera spatial grid from Qwen3-VL hidden states, and (b) deriving a canonical cache key on the starVLA side — are now called out as hard gates with concrete verification steps, and Channel 1 is sequenced to ship before Channel 2. See **Changelog vs v1** at the end.

## 0. Thesis

FastWAM's edge = a *training objective* (future-video prediction + video↔action
co-attention on a Wan-pretrained trunk), not multi-frame test-time input. At inference
it is already a single-frame VLA carrying world-shaped weights (verified: `infer_action`
encodes one frame, prefills the video branch once, then runs only the action expert —
`fastwam.py:905-1048`). Transplant that into a fast `QwenGR00T` flow-matching student via
**two training-only channels**, both dropped at inference:
- **Channel 1 (REPA):** align student vision-token features to FastWAM frame-0
  video-DiT hidden states (`Z_teacher`). Zero inference cost. **Ship this first.**
- **Channel 2 (future head):** small flow-matching head predicts the teacher's 8 future
  latent frames (Wan VAE space), supervised by *real* future frames (`Fut_real`).
  Replicates the world-model objective. Dropped at inference. **Add only after C1 works.**

Goal: match/approach FastWAM LIBERO success at a large speedup (student = 1 Qwen-VL
prefill + 4 tiny flow steps vs teacher's 1 Wan-VAE encode + 1×5B video-DiT prefill +
10×1.1B action steps).

---

## 0.5 System figure

```
══════════════════════════════════════════════════════════════════════════════════════
 PHASE A — OFFLINE TEACHER EXTRACTION   (conda env: fastwam · frozen FastWAM · run once)
══════════════════════════════════════════════════════════════════════════════════════
  LIBERO clip (agentview+wrist, 33 px frames)   FastWAM preprocessing:
        │   per-cam resize 224×224, hcat → 224×448, Normalize(0.5,0.5) → [-1,1]
        ▼
   Wan VAE.encode(full 33-frame clip)  ──►  z  [B,48,9,14,28]   (T:33→9, /16 spatial)
        │            │                                  T5 text cache [128,4096] ─┐
        │            │  z[:,:,0:1] = frame-0 latent     proprio[8]→Linear(8→4096)─┤(+1 ctx tok)
        │            ▼                                                            │
        │   video_expert.pre_dit(x=z0, t=0, action=None, fuse=True)  ◄───────────┘
        │        │   (single-frame f=1; seperated_timestep path, frame-0 t=0)
        │   MoT.extract_video_hidden  (frame-0 self-attn only)
        │        │
        │        ├──► TAP @ layer L ─────────►  Z_teacher  [B,7,14,3072]   ─┐
        │                                                                   ├─► write cache/
        └──► Fut_real = z[:,:,1:9]  ─► [B,8,48,14,28]  (same clip, no re-encode) ─┘
                                                          key = (suite,task,ep,frame)
══════════════════════════════════════════════════════════════════════════════════════
 PHASE B — STUDENT DISTILLATION TRAINING        (conda env: starVLA · 8×GPU · accelerate)
══════════════════════════════════════════════════════════════════════════════════════
  cur frame (agentview,wrist 224×224, NO flip) + instruction + state[7]
        │
   build_qwenvl_inputs → Qwen3-VL  (output_hidden_states=True, return_dict=True)
        │   ├─ image tokens: locate IMAGE_TOKEN_INDEX spans in input_ids, segment by
        │   │        │        image_grid_thw per camera → per-cam [hs,ws] grid
        │   │        │ tap @ mid layer ─► RepaProjector(2048→3072)
        │   │        │                         │
        │   │        │            ╔════════════▼═══════════╗   Z_teacher (cache, resize→grid)
        │   │        │            ║  L_REPA = 1-cos        ║◄──────── stop-grad
        │   │        │            ╚════════════════════════╝
        │   └─ last_hidden [B,seq,2048] = vl_embs (conditioning) ─────────┬───────────┐
        │                                                                 │           │
        │   ┌── FUTURE FLOW HEAD  (training-only · conditional denoiser) ─┤           ▼
        │   │   Fut_real z0 [B,8,48,14,28] (cache)   sample ε~N(0,1), t~U │   GR00T action head
        │   │            └─► z_t = (1−t)·ε + t·z0 ──► [z_t , t] ──► DiT ◄─┘   (flow-matching,
        │   │                                          (cross-attn vl_embs)    chunk 8, cond
        │   │                                              │ pred v̂            on vl_embs)
        │   │                          L_future = ‖ v̂ − (ε − z0) ‖²            │
        │   └──────────────────────────────────────────────────────────┘     │
        │                              ╔═══════════════╗             ╔═════════▼════════╗
        │                              ║    L_future   ║             ║    L_action      ║
        │                              ╚═══════════════╝             ╚══════════════════╝
        │   (note: head RECEIVES noised future z_t as input — NOT predicting from current
        │    alone; same as teacher's video-DiT denoiser. Whole head dropped at inference.)
   TOTAL  L = L_action + 0.5·L_REPA + 0.5·L_future  [+0.5·L_futureREPA, gated]
══════════════════════════════════════════════════════════════════════════════════════
 PHASE C — INFERENCE   (conda env: libero · student only · fast · single frame)
══════════════════════════════════════════════════════════════════════════════════════
  frame(+180° flip, see §3)+instruction+state ─► Qwen-VL ─► action head (4 flow steps)
  NO 5B video DiT.  NO REPA/future heads.  replan every 8 steps.
══════════════════════════════════════════════════════════════════════════════════════
```

---

## 1. Teacher facts (verified — FastWAM uncond)

- **Inference** (`fastwam.py:905-1048`): requires `video_attention_mask_mode=="first_frame_causal"`
  (asserted `fastwam.py:923`). VAE-encode current frame (`_encode_input_image_latents_tensor`,
  `fastwam.py:253-265`) → `video_expert.pre_dit(action=None, fuse=True)` (`:998-1005`) →
  `mot.prefill_video_cache` (`mot.py:257-341`) using `attention_mask[:Sv,:Sv]` (`fastwam.py:1021`)
  → action expert × `num_inference_steps` (eval default 10) attending the cached frame-0 K/V.
  No future at test time.
- **Training** (`fastwam.py:448-568`): `loss = lambda_video*loss_video + lambda_action*loss_action`,
  `loss_lambda_video` default **1.0** (`fastwam.py:39`, `runtime.py:156`), not zeroed by uncond
  config (`configs/model/fastwam.yaml:31` sets `action_conditioned: false` on the *video* expert,
  not the loss weight). Denoises the 8 future latent frames (frame-0 overwritten clean,
  `fastwam.py:467-468`, sliced out of the loss `:535-537`); whole MoT incl. 5B DiT finetuned.
- **Frame-0 invariance (verified — REPA-critical):** with `first_frame_causal`,
  `video_mask[:tokens_per_frame, tokens_per_frame:] = False`
  (`wan_video_dit.py:501-505`, helper `build_video_to_video_mask`), so frame-0 query rows
  **do not attend to any future tokens**. In the `seperated_timestep` path
  (`wan_video_dit.py:537-550`) `token_timesteps[:, 0, :] = 0`, so frame-0 is always at
  clean t=0 regardless of the future noise level. Frame-0's cross-attention context is
  text-only in both training (action mask starts at `tokens_per_frame:`, `wan_video_dit.py:589`)
  and single-frame extraction (`action=None, f=1` → text-only mask, `:591-598`). **⇒ frame-0
  hidden states are bit-exact whether or not future tokens are present** → single-frame
  extraction == training-time features. The cache is faithful. (Requires `fuse_vae_embedding_in_latents=True`
  — config sets it, `configs/model/fastwam.yaml:28`; the non-fused path is `NotImplementedError`,
  `wan_video_dit.py:552`.)
- **I/O:** per-cam resize 224×224 then **horizontal concat → 224×448** (left = `image`/agentview,
  right = `wrist_image`; `robot_video_dataset.py:180-181`); pixels normalized to **[-1,1]**
  via `Normalize(mean=0.5,std=0.5)` (`robot_video_dataset.py:82-84`). T5 `[128,4096]`; proprio
  **[8]** (frame-0 only, `fastwam.py:361`) → `Linear(8→text_dim=4096)` appended as one context
  token (`fastwam.py:59,219-240`); action **[32,7]**.
- **Frame-0 latent grid (verified):** Wan2.2-TI2V-5B VAE spatial factor **16**
  (`wan_video_vae.py:1382 upsampling_factor=16`), temporal factor **4**
  (`:1383 temporal_downsample_factor=4`). 224×448 → latent **14×28**; DiT `patch_size=[1,2,2]`
  (`configs/model/fastwam.yaml:14`) → tokens_per_frame **= (14/2)·(28/2) = 7×14 = 98**,
  `hidden_dim=3072` (`:16`). The 448 width = 2 cams → teacher grid
  `[7×7 agentview | 7×7 wrist]` (cols 0:7 / 7:14). ✅ (v1 grid claim correct.)
- **Future-frame definition (corrected):** a training clip is `num_frames=33` pixel frames
  (`configs/data/libero_2cam.yaml:28`), VAE-encoded to **9 latent frames**
  ((33−1)/4 + 1 = 9). Frame-0 = current obs; frames **1..8 = future**. So
  `Fut_real = z[:,:,1:9]` = `[8,48,14,28]` from the **same clip's** VAE encode — NOT eight
  separately VAE-encoded strided pixel frames. (v1's "strides +4..+32 → separate encodes"
  was a misread; the strides are the pixel→latent temporal subsampling done once.)

## 2. Student facts (verified — starVLA `QwenGR00T`)

- Framework `starVLA/model/framework/VLM4A/QwenGR00T.py`. Backbone selected by name
  (`vlm/__init__.py:9-10` → `_QWen3_VL_Interface` for `"Qwen3-VL"`). **Config mismatch to
  resolve:** the QwenGR00T dataclass default is `Qwen3-VL-4B-Instruct` (`QwenGR00T.py:63`,
  `vl_hidden_dim=2048`), but the shipped LIBERO YAML points at `Qwen2.5-VL-3B-Instruct`
  (`starvla_cotrain_libero.yaml:13`). Pin the distill YAML to **one** backbone and stick to it
  for both M0 baseline and distill; `vl_hidden_dim` must match the chosen model. (Plan assumes
  Qwen3-VL-4B; switch all hidden dims if you choose 2.5-VL-3B — its `hidden_size` differs.)
- Action head `GR00T_ActionHeader.py:312-363` (velocity-MSE flow matching, loss `:362`).
  `action_dim=7`, `action_horizon=8`, `state_dim=7`, `num_inference_timesteps=4`,
  `num_target_vision_tokens=32` (learnable future query tokens). Inference = 4 Euler steps
  (`:376-415`). DiT cross-attends `vl_embs` (`:351-357`).
- **`forward` (verified `QwenGR00T.py:170-218`):** signature is `forward(self, examples: List[dict])`
  — a **list of per-sample dicts**, not a batched tensor dict. It builds `qwen_inputs =
  build_qwenvl_inputs(images, instructions)` (`:183`), calls
  `qwenvl_outputs = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True, return_dict=True)`
  (`:185-190`), takes `last_hidden = qwenvl_outputs.hidden_states[-1]` (`:192`), and returns
  `{"action_loss": action_loss}` (`:218`). **This is where aux losses are added.**
  `hidden_states` is a tuple of length `1 + num_text_layers`.
  **UNVERIFIED — check `playground/Pretrained_models/<backbone>/config.json` `text_config.num_hidden_layers`**
  for the exact count (v1's "tuple(37)" assumes 36 layers; do not hard-code). Mid tap =
  `hidden_states[1 + num_layers//2]`.
- **Images (verified):** `build_qwenvl_inputs` (`QWen3.py:115-172`) builds one chat message per
  sample with `content = [{image_0}, {image_1}, {text}]` and calls
  `processor.apply_chat_template(..., return_dict=True, return_tensors="pt")`. Each camera is a
  **separate image** with its own grid (NOT concatenated); the returned dict includes
  `pixel_values`, `input_ids`, **`image_grid_thw`** `[num_images, 3]=(t,h,w)` in *pre-merge*
  patch units. Image tokens are marked by `IMAGE_TOKEN_INDEX = 151655` (`QWen3.py:15`) and sit
  **interleaved before the text**, in image order. Per-camera token count after spatial merge
  (Qwen merge_size=2) `= (h/2)·(w/2)`.
- Train: `starVLA/training/train_starvla.py` `_train_step` (`:360-368`) does
  `output_dict = self.model.forward(batch_vla); action_loss = output_dict["action_loss"]`,
  where `batch_vla` is a **list of example dicts** (collate returns the list,
  `lerobot_datasets.py:19-20`). `base_framework.compute_loss` (`:145-181`) is a thin
  dispatcher that calls `self.forward(batch)` and scales — the standard loop calls `forward`
  directly. Config `examples/LIBERO/train_files/starvla_cotrain_libero.yaml`:
  `action_type: delta_qpos`, `obs: ["image_0"]` (wrist auto-added), 224×224.
- Eval: `examples/LIBERO/eval_files/eval_libero.py` + `model2libero_interface.py`; replan every
  `action_horizon` (8); 4 flow steps. **Student keeps its OWN action normalization**
  (`dataset_statistics.json`); no action distillation, so no FastWAM action stats imported.

---

## 3. The critical integration: cache-key alignment + image contract

Teacher and student read **different LeRobot copies** of LIBERO (FastWAM
`configs/data/libero_2cam.yaml` dirs vs starVLA `playground/Datasets/LEROBOT_LIBERO_DATA`,
gr00t format). Global indices will **not** match. Key every cache entry on canonical identity:

```
KEY = (task_suite, task_id, episode_id, frame_id)     # e.g. "libero_spatial/3/000012/000045"
```

**Reality check on derivability (revised — harder than v1 claimed):**
- **Teacher (Phase A):** LeRobot `__getitem__` exposes `episode_index`, `frame_index`, and
  `task` text (`lerobot_dataset.py:748-791`), plus `dataset_index` from `MultiLeRobotDataset`
  (`:1285`). suite/task come from the dataset dir + `meta/tasks.jsonl`. Derivable.
- **Student (Phase B):** `gr00t_lerobot/datasets.py` indexes by `(trajectory_id, base_index)`
  (`:1366`) = `(episode_id, frame_id)`, **but `_pack_sample` (`:1371-1404`) returns only
  `image/lang/action/state` — it does NOT surface episode/frame/suite/task.** ⚠️ You must add a
  small read-only hook that, for each `index`, also emits `(trajectory_id, base_index)` plus the
  suite/task resolved from `trajectory_ids_to_metadata` (`:913`) and the `data_mix` name
  (`starvla_cotrain_libero.yaml:66`). Episode/frame ordering may also differ between the two
  LeRobot copies, so do not assume `frame_id` matches without the pixel check below.

**Image-content contract (verified, REPA-critical):**
- **Both training paths use resize-only, NO geometric flip.** Teacher: per-cam resize→hcat
  (`robot_video_dataset.py:180-181`, transforms `libero_2cam.yaml:58-65`). Student training:
  `Image.fromarray(image).resize((224,224))` (`datasets.py:1377`); the optional
  `VideoHorizontalFlip` augmentation (`transform/video.py:380-397`) **must be disabled on the
  distill path** (use val transforms). ⇒ teacher cache frames and student training frames are
  geometrically identical — REPA correspondence holds. ✅
- **⚠️ Eval-only 180° rotation:** `eval_libero.py:139-140` applies `obs[...][::-1,::-1]`
  ("to match train preprocessing"). Training does **not** flip, so this comment is misleading —
  the rotation exists to undo LIBERO's rendered orientation. **This does not affect REPA**
  (REPA is train-only). It *does* affect whether the trained policy sees test images in the same
  orientation as training. M0 must confirm the baseline student already evals correctly with this
  flip in place; do not change it.
- Per-encoder pixel normalization differs (teacher [-1,1]; Qwen-VL its own mean/std). That is
  fine — REPA aligns *features*. What must match is the underlying crop/content per camera (same
  source frame, same agentview/wrist, same 224 resize), which it does.

**Action item M1 (hard gate):** add `canonical_key(sample)` in *each* repo (teacher reads it
from LeRobot fields; student needs the new metadata hook above), then assert on a ≥100-sample
overlap that decoding the same KEY yields the *same RGB frame* (pixel MSE ≈ 0 after identical
224 resize, no flip). **This is the single highest-risk step — do it first; it can fail purely
because the two LeRobot copies number episodes/frames differently.**

---

## 4. Channel 1 — REPA (concrete)

- **Teacher target:** `Z_teacher[KEY]` = frame-0 hidden at tap layer L (start L=15/30),
  reshaped `[7,14,3072]` → per-cam `[7,7,3072]` halves (cols 0:7 agentview, 7:14 wrist).
- **Student source:** Qwen image tokens at the mid hidden layer
  (`hidden_states[1+num_layers//2]`), sliced **per camera** via the
  `IMAGE_TOKEN_INDEX`-span + `image_grid_thw` recipe (§7.4), reshaped `[B, hs, ws, 2048]`.
  **⚠️ Top student-side unknown — must verify on one batch before trusting REPA:**
  whether `image_grid_thw` gives a clean per-camera `(hs,ws)` and whether the merged grid for a
  224×224 image is the expected ~7×7 (depends on the processor's `smart_resize` /
  `min_pixels`/`max_pixels`). If the merged grid is not ~7×7 it still works (we bilinear-resize
  the teacher to `(hs,ws)`), but the per-camera *span segmentation* must be exact.
  **Fallback if spatial alignment proves infeasible** (dynamic resolution scrambles the grid, or
  text/image positions can't be cleanly segmented): switch REPA to a **pooled / global** target —
  mean-pool teacher `Z_teacher` over its 98 tokens (or per-cam over 49) to a single vector, and
  align it to the mean-pooled student image tokens per camera (1-cos on pooled vectors). This
  drops spatial structure but keeps the world-feature signal and is robust to grid mismatch.
  Decide spatial-vs-pooled at M2 based on the one-batch overlay.
- **Projector:** `RepaProjector: 2048 → 2048 → 3072` (SiLU). Project student up; **stop-grad
  teacher**.
- **Loss:** resize teacher per-cam map `[7,7]→[hs,ws]` (bilinear), per-patch negative cosine,
  averaged over both cams + batch. `λ_repa = 0.5`, constant, from step 0.
- **v2 option (M5):** distill the action-expert K/V the policy actually reads
  (`mot.py:412-419` — action queries attend cached frame-0 K/V), or multi-layer.

## 5. Channel 2 — future-latent head (concrete) — ship after C1

- **Target:** `Fut_real[KEY]` = the **8 non-first latent frames** `z[:,:,1:9]`
  (`[8,48,14,28]`) from the same 33-frame clip the teacher VAE-encodes once. Cached offline in
  Phase A (no Wan VAE in the student env).
- **Head I/O (important — it is a conditional denoiser, NOT a regressor from the current
  frame):** input = `(z_t, t, vl_embs)` where `z_t = (1−t)·ε + t·z0` is the **noised future
  latent** (z0 = `Fut_real`), `t` the flow timestep, `vl_embs = last_hidden` the current-frame
  conditioning; output = predicted velocity `v̂`; target = `ε − z0`. Identical structure to the
  teacher's video-DiT future denoiser — both receive noised future as input and condition on the
  current observation, so the tasks match. The head exists only to push world-predictive gradient
  into the Qwen trunk; it is dropped at inference, so its sample quality is irrelevant.
- **Head (corrected feasibility):** reuse `flow_matching_head/cross_attention_dit.py:DiT`
  (`__init__` takes `cross_attention_dim`, `output_dim`, `num_layers`; `forward(hidden_states,
  encoder_hidden_states, timestep)`; `cross_attention_dim=2048` to match Qwen). **But that DiT
  operates on a flat token sequence `[B,T,inner_dim]`, not 2D spatial latents** — so you must
  patchify/flatten the `[8,48,14,28]` future target into tokens (e.g. patch the 14×28 spatial
  grid like the teacher's `[1,2,2]` → 7×14=98 tokens/frame × 8 = 784 tokens × 48ch, project
  48→inner_dim in, inner_dim→48 out) and add a small un-patchify head. This is the main C2
  engineering cost and is non-trivial; budget for it. Dimensionality/compute is fine
  (784 tokens, d~768–1024, 4–6 layers — comparable to the action DiT).
- **Loss:** velocity flow-matching to match the teacher's objective:
  `L_future = E_{t,ε} || head(z_t,t,vl_embs) − (ε − z0) ||²` over the 8 future latents; same
  shift-5.0 continuous scheduler family (`WanContinuousFlowMatchScheduler`,
  `schedulers/scheduler_continuous.py`). `λ_fut = 0.5` with ~5% warmup; optional stop-grad into
  the vision encoder for the first ~2k steps.
- **Rejected:** `Fut_teacher` (teacher *denoised* latents) as target — it is
  `real + teacher_error`; real frames dominate.
- **Cheaper alternative to weigh at M3:** before building the full pixel-space C2 head, try a
  **feature-space future match** — predict the teacher's *future-frame hidden states* from
  `last_hidden` (an MLP/cross-attn head, MSE on hidden features) instead of reconstructing
  48-ch VAE latents. Much smaller target, no un-patchify head, and it directly transfers the
  world representation rather than appearance. If it matches C2's eval gain it replaces C2.
- **future-REPA (gated, §M5):** align future-head hidden states to the teacher's *future-frame*
  hidden states. Needs the teacher's full 9-frame forward (≈ extraction cost ×, much larger
  cache) → enable only after M3. `λ_futrepa` (~0.5), off by default.

## 6. Combined loss

```
L = L_action  +  0.5·L_REPA  +  0.5·L_future  [ + 0.5·L_futureREPA (gated) ]
```
Add the aux terms inside `QwenGR00T.forward` onto `action_loss` (the trainer reads only
`output_dict["action_loss"]`, `train_starvla.py:367`) and also return detached scalars for
logging. Monitor grad-norms; if an aux grad > 2× the action grad, halve its λ. Channel flags in
config so each is independently toggleable (M2/M3 ablations) and all vanish at inference.

---

## 7. Concrete code artifacts

### 7.1 Teacher: hidden-state tap (FastWAM repo)

`src/fastwam/models/wan22/mot.py` — add (non-breaking; mirrors `prefill_video_cache:257-341`,
uses the real helpers `_build_expert_attention_io`, `_mixed_attention`,
`_apply_post_with_optional_checkpoint` — all verified present):
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

`src/fastwam/models/wan22/fastwam.py` — add (mirrors `infer_action:993-1022`).
**Corrected:** the mask builder is `video_expert.build_video_to_video_mask` (NOT
`_build_video_attention_mask`); `meta` exposes `grid_size:(f,h,w)` and `tokens_per_frame`
(`wan_video_dit.py:615-618`); `fuse` must be True (config) or `pre_dit` raises:
```python
@torch.no_grad()
def extract_world_features(self, input_image, context, context_mask,
                           proprio=None, tap_layer=15, tiled=False):
    self.eval()
    if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
        raise ValueError("extract_world_features requires first_frame_causal (matches infer_action).")
    if input_image.ndim == 3: input_image = input_image.unsqueeze(0)
    input_image = input_image.to(self.device, self.torch_dtype)
    z0 = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)  # [B,48,1,14,28]
    fuse = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
    assert fuse, "fuse_vae_embedding_in_latents must be True (seperated_timestep path)."
    context = context.to(self.device, self.torch_dtype)
    context_mask = context_mask.to(self.device, torch.bool)
    if proprio is not None:
        context, context_mask = self._append_proprio_to_context(
            context, context_mask, proprio.to(self.device, self.torch_dtype))
    t0 = torch.zeros((z0.shape[0],), dtype=z0.dtype, device=self.device)
    vp = self.video_expert.pre_dit(x=z0, timestep=t0, context=context,
            context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=fuse)
    Sv  = vp["tokens"].shape[1]
    tpf = int(vp["meta"]["tokens_per_frame"])            # = 98 for 224×448
    _, h, w = vp["meta"]["grid_size"]                    # (f=1, h=7, w=14)
    vmask = self.video_expert.build_video_to_video_mask(  # [Sv,Sv], first_frame_causal
        video_seq_len=Sv, video_tokens_per_frame=tpf, device=vp["tokens"].device)
    x = self.mot.extract_video_hidden(
            video_tokens=vp["tokens"], video_freqs=vp["freqs"], video_t_mod=vp["t_mod"],
            video_context_payload={"context": vp["context"], "mask": vp["context_mask"]},
            video_attention_mask=vmask, tap_layer=tap_layer)
    return x[:, :tpf].reshape(x.shape[0], h, w, x.shape[-1])   # [B,7,14,3072]
```
*(For f=1 single-frame, `build_video_to_video_mask` returns an all-ones `[Sv,Sv]` since
`first_frame_tokens==Sv` — i.e. plain self-attn, exactly what the frame-0 rows experience in
the multi-frame forward. Faithfulness preserved.)*

### 7.2 Teacher: precompute script (FastWAM repo)

`scripts/precompute_distill_cache.py` — mirror `scripts/precompute_text_embeds.py` infra
(verified): `_init_distributed()` (`:31-44`), `_atomic_torch_save(payload, path)` (`:163-167`),
`@hydra.main(config_path="../configs", config_name="train", version_base="1.3")` (`:170`),
torchrun rank sharding `items = items[rank::world_size]` (`:252`). Per unique
`(suite,task,ep,frame)` window:
```
for KEY, sample in iterate_clips(dataset, dedupe_by=KEY, shard=rank::world):
    img   = sample.frame0_image            # [1,3,224,448] in [-1,1]
    ctx,m = load_text_cache(sample.prompt) # reuse existing T5 cache
    pro   = sample.proprio0                 # [1,8] normalized (frame-0)
    Z = model.extract_world_features(img, ctx, m, pro, tap_layer=L)   # [1,7,14,3072]
    z = model._encode_video_latents(sample.clip_33f)                  # [1,48,9,14,28]
    Fut = z[:, :, 1:9]                                                # [1,8,48,14,28]
    _atomic_torch_save({"Z": Z.to(torch.bfloat16).cpu(),
                        "Fut": Fut.to(torch.float16).cpu()}, cache_dir / f"{KEY}.pt")
```
Run `torchrun --standalone --nproc_per_node=8 ... --dtype bfloat16`. Encode each clip once.

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

`QwenGR00T.py:170-218` — note the **real** variable names (`examples` list, `qwen_inputs`,
`qwenvl_outputs`, `last_hidden`, `last_hidden_repeated`). Add aux losses onto `action_loss`:
```python
qwen_inputs   = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images,
                                                           instructions=instructions)
qwenvl_outputs = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True,
                                        return_dict=True)
last_hidden = qwenvl_outputs.hidden_states[-1]
# ... existing action_loss = self.action_model(last_hidden_repeated, ...) ...

if self.cfg.distill.repa:
    n_layers = self.qwen_vl_interface.model.config.text_config.num_hidden_layers
    mid = qwenvl_outputs.hidden_states[1 + n_layers // 2]                 # [B, seq, 2048]
    tok_a, grid_a = slice_image_tokens(mid, qwen_inputs, cam=0)
    tok_w, grid_w = slice_image_tokens(mid, qwen_inputs, cam=1)
    Z = batch_Z_teacher                       # [B,7,14,3072] from cache loader
    lr = 0.5*(repa_loss(tok_a, Z[:, :, :7, :], grid_a, self.repa_proj)
              + repa_loss(tok_w, Z[:, :, 7:, :], grid_w, self.repa_proj))
    action_loss = action_loss + self.cfg.distill.lambda_repa * lr
if self.cfg.distill.future:
    lf = self.future_head.flow_loss(last_hidden, batch_Fut_real)         # cached targets
    action_loss = action_loss + self.cfg.distill.lambda_fut * lf
return {"action_loss": action_loss, "repa_loss": lr.detach(), "future_loss": lf.detach()}
```
**`slice_image_tokens` (the tricky part — verify on one batch, M2):**
```python
def slice_image_tokens(hidden, qwen_inputs, cam):
    # input_ids: [B, L]; image tokens marked by IMAGE_TOKEN_INDEX (151655).
    # image_grid_thw: [num_images, 3] = (t,h,w) in PRE-merge patch units; merge_size=2.
    ids   = qwen_inputs["input_ids"]                       # [B, L]
    thw   = qwen_inputs["image_grid_thw"]                  # [B*num_cams, 3]
    merge = self.qwen_vl_interface.processor.image_processor.merge_size  # =2
    # For each sample: find contiguous IMAGE_TOKEN_INDEX spans (cam0 then cam1 in order),
    # take span #cam, reshape to (h//merge, w//merge), return tokens + grid_hw.
    ...
    return tokens, (h // merge, w // merge)
```
Because batch padding side is **left** (`QWen3.py:68`), index image-token positions per row,
not by a global offset. If span segmentation is unreliable under dynamic resolution, fall back
to the pooled REPA target (§4). Build `repa_proj` / `future_head` in `__init__` gated by
`cfg.distill`.

### 7.5 Student: cache loader + metadata hook (starVLA repo)

In `gr00t_lerobot/datasets.py`: (a) extend the sample with canonical identity — for each
`index`, also emit `(trajectory_id, base_index)` (`:1366`) + suite/task resolved from
`trajectory_ids_to_metadata` (`:913`) and the `data_mix` name; (b) in `_pack_sample` compute
`KEY`, then `blob = torch.load(cache_dir/f"{KEY}.pt")` and add `sample["Z_teacher"]=blob["Z"]`,
`sample["Fut_real"]=blob["Fut"]`. Collate (currently returns the raw list,
`lerobot_datasets.py:19-20`) → stack `Z_teacher`/`Fut_real` into tensors keyed alongside the
example list, and read them in `forward` (the list-of-dicts contract means you can attach them
per-example and stack inside `forward`).

### 7.6 Config additions (starVLA YAML)

`examples/LIBERO/train_files/starvla_distill_libero.yaml` (extends the cotrain yaml; **pin the
backbone explicitly**):
```yaml
framework:
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3-VL-4B-Instruct   # MUST match vl_hidden_dim
distill:
  repa: true
  future: false            # OFF for M2; turn on at M3
  future_repa: false
  lambda_repa: 0.5
  lambda_fut: 0.5
  lambda_futrepa: 0.5
  teacher_tap_layer: 15
  student_tap_layer: auto  # = 1 + num_hidden_layers//2, resolved at runtime
  cache_dir: ./playground/Datasets/LIBERO_distill_cache
  disable_train_image_aug: true   # no random HFlip on the distill path (REPA correspondence)
```

---

## 8. Caching sizing (Phase A output)

- `Z_teacher`: `7·14·3072·2 B ≈ 0.59 MB/frame` (bf16, 1 tap layer).
- `Fut_real`: `8·48·14·28·2 B ≈ 2.9 MB/frame` (fp16).
- LIBERO ≈ 300k unique frames ⇒ ~175 GB (Z) + ~880 GB (Fut). **Decision (revised): do NOT cache
  `Fut` for all 300k frames up front.** For C2 you only need it after C1 is proven, and only for
  the suite you ablate first. Recommended: cache `Z` at every frame (~175 GB, cheap, needed for
  C1) **now**; cache `Fut` lazily — one task-suite subset for the M3 ablation (≈ tens of GB),
  expand only if C2 beats baseline. The "every 4th anchor" stride (~220 GB) is a fallback if you
  insist on full coverage. Keep one tap layer for v1.

---

## 9. Remote GPU runbook

Three conda envs: **fastwam** (teacher/preprocess), **starVLA** (train), **libero** (eval).
`cd` paths absolute.

**0) One-time setup**
```bash
conda activate fastwam   # torch 2.7.1+cu128; pip install -e /…/FastWAM
conda create -n starVLA python=3.10 -y && conda activate starVLA
cd /…/starVLA && pip install -r requirements.txt && pip install flash-attn --no-build-isolation && pip install -e .
bash examples/LIBERO/data_preparation.sh          # downloads starVLA LIBERO + Qwen ckpt
conda create -n libero python=3.10 -y && conda activate libero && pip install libero mujoco==3.3.2
```

**1) M0 baselines (sanity before any distill)**
```bash
# student baseline (no distill) — confirm it evals correctly WITH the eval [::-1,::-1] flip in place
conda activate starVLA && cd /…/starVLA
bash examples/LIBERO/train_files/run_libero_train.sh
# teacher eval reproduces on 1 suite
conda activate fastwam && cd /…/FastWAM
python experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

**2) M1 key-alignment check** — run the `canonical_key` overlap assert (pixel-MSE≈0, no flip)
across both dataloaders **before** extraction. **Hard gate.**

**3) Phase A — precompute teacher cache** (fastwam env, multi-GPU)
```bash
conda activate fastwam && cd /…/FastWAM
torchrun --standalone --nproc_per_node=8 scripts/precompute_distill_cache.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  +distill.tap_layer=15 +distill.cache_future=false \
  +distill.cache_dir=/…/starVLA/playground/Datasets/LIBERO_distill_cache
# (re-run with +distill.cache_future=true for the M3 subset only)
```

**4) Phase B — distillation training** (starVLA env, 8×GPU; C1 first)
```bash
conda activate starVLA && cd /…/starVLA
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/LIBERO/train_files/starvla_distill_libero.yaml \
  --framework.name QwenGR00T --run_root_dir ./playground/Checkpoints --run_id libero_distill_c1 \
  --distill.repa true --distill.future false
```

**5) Phase C — eval** (libero env)
```bash
conda activate libero && cd /…/starVLA
python examples/LIBERO/eval_files/eval_libero.py --task_suite_name libero_spatial \
  --num_trials_per_task 10 --pretrained_path ./playground/Checkpoints/libero_distill_c1
# report success/suite + per-step latency & flow-step count vs teacher
```

---

## 10. Milestones (go/no-go)

1. **M0** baselines reproduce (student ±2% of published, evaluated with the eval-time 180° flip;
   teacher 1-suite OK).
2. **M1** key alignment verified (pixel-MSE≈0 on overlap; canonical key derivable on BOTH sides,
   incl. the new starVLA metadata hook). *Hard gate — blocks everything.*
3. **M2** Channel 1 alone: first prove `slice_image_tokens` on one batch (overlay image-token
   grid on the picture); then +REPA neutral-or-better on `libero_spatial`; teacher `Z_teacher`
   PCA is object-aligned; student cosine rises. If spatial slicing is unreliable, switch to
   pooled REPA and re-run.
4. **M3** Channel 2 alone (built on top of working C1 infra): future head trains (L_future↓,
   action loss intact), ≥ baseline. Compare against the **feature-space future-match** cheaper
   variant (§5) — keep whichever wins per GPU-hour.
5. **M4** both, full LIBERO: success ≥ FastWAM−3% AND ≥2× faster (count heavy passes +
   wall-clock).
6. **M5** (gated) future-REPA / K/V target / multi-layer — only if it beats M4 enough to justify
   the extra extraction + cache.

## 11. Risks

- **Canonical-key derivability on the student side** — `_pack_sample` doesn't expose
  episode/frame/suite/task; needs a metadata hook, and the two LeRobot copies may number
  episodes/frames differently. → M1 hard gate (§3).
- **Qwen image-token spatial slicing** — dynamic resolution + left-padding + interleaved
  image/text tokens. → verify `image_grid_thw`→grid on one batch (overlay) before trusting REPA;
  pooled-REPA fallback ready.
- **Backbone mismatch** — dataclass default (Qwen3-VL-4B) vs shipped YAML (Qwen2.5-VL-3B). →
  pin one backbone in the distill YAML; keep `vl_hidden_dim` consistent.
- **C2 head must consume 2D 48-ch latents** — the reusable DiT is a 1D-token block; needs
  patchify/un-patchify wiring. → real engineering cost; build only after C1.
- **Aux losses corrupt action head** → grad-norm caps, channel-toggle ablations (M2/M3).
- **Disk blowup (Fut_real)** → cache lazily, subset-first (§8).
- **Speed must actually improve** → by construction: student = 1 Qwen prefill + 4 action flow
  steps; teacher = 1 VAE encode + 1×5B video-DiT prefill + 10×1.1B action steps. Verify by
  counting heavy passes + wall-clock, not assuming.

## 12. File reference index (corrected)

- **Teacher:** `fastwam.py:905-1048` (`infer_action`; asserts `first_frame_causal` `:923`;
  prefill mask slice `:1021`), `448-568` (training loss; `loss_lambda_video` default 1.0 `:39`),
  `467-468/535-537` (frame-0 clean + sliced out of loss), `361` (frame-0 proprio),
  `219-240` (`_append_proprio_to_context`), `253-265` (`_encode_input_image_latents_tensor`),
  `386-407` (`_build_mot_attention_mask`); `mot.py:257-341` (`prefill_video_cache`, has the
  `_build_expert_attention_io`/`_mixed_attention`/`_apply_post_with_optional_checkpoint`
  helpers `:124-255`), `412-419` (action reads frame-0 K/V); `wan_video_dit.py:473-507`
  (`build_video_to_video_mask`, `first_frame_causal` `:501-505`), `:509-620` (`pre_dit`;
  `seperated_timestep` frame-0 t=0 `:537-550`; `meta` keys `:615-618`),
  `wan_video_vae.py:1382-1383` (spatial 16 / temporal 4); `runtime.py:156`;
  `configs/model/fastwam.yaml:14,16,28,30,31`; `configs/data/libero_2cam.yaml:13-17,28,30,42-43`;
  `robot_video_dataset.py:82-84,180-181` (normalize, hcat order);
  `scripts/precompute_text_embeds.py:31-44,163-167,170,252` (infra to mirror).
- **Student:** `QwenGR00T.py:170-218` (`forward(examples: List[dict])`, returns
  `{"action_loss"}`, hidden tap `:192`), `predict_action:220-270`;
  `QWen3.py:15` (`IMAGE_TOKEN_INDEX`), `:61-72` (Qwen3-VL load), `:68` (left padding),
  `:115-172` (`build_qwenvl_inputs`, `return_dict=True`); `vlm/__init__.py:9-10`;
  `GR00T_ActionHeader.py:312-363` (flow loss `:362`), `:366-415` (`predict_action`, 4 Euler
  steps), `:248` (`num_inference_timesteps`); `cross_attention_dit.py:188-271` (DiT
  `__init__`/`forward`, `cross_attention_dim`); `train_starvla.py:360-368` (`_train_step`,
  reads `output_dict["action_loss"]`); `base_framework.py:145-181` (`compute_loss` dispatcher);
  `gr00t_lerobot/datasets.py:1366` (index→`(trajectory_id, base_index)`), `:1371-1404`
  (`_pack_sample`, does NOT expose suite/task — needs hook), `:913`
  (`trajectory_ids_to_metadata`), `:1377` (resize-only, no flip);
  `transform/video.py:380-397` (optional HFlip — disable on distill path);
  `examples/LIBERO/eval_files/eval_libero.py:139-140` (eval `[::-1,::-1]` flip),
  `model2libero_interface.py:156` (eval resize); cfg
  `examples/LIBERO/train_files/starvla_cotrain_libero.yaml:13,27,34,66` (backbone, horizon,
  inference steps, data_mix). **UNVERIFIED — check `playground/Pretrained_models/<backbone>/config.json`**
  for `text_config.num_hidden_layers` and vision `merge_size`/`patch_size`.

---

## Changelog vs v1

1. **Mask builder name fixed.** v1's `extract_world_features`/`extract_video_hidden` skeletons
   called `self.video_expert._build_video_attention_mask(...)`, which does not exist. The real
   API is `video_expert.build_video_to_video_mask(video_seq_len, video_tokens_per_frame, device)`
   (`wan_video_dit.py:473`); the MoT-level joint mask is `fastwam._build_mot_attention_mask`
   (`fastwam.py:386`). Corrected the skeleton and added the `first_frame_causal` assertion that
   `infer_action` itself enforces.
2. **`meta` keys fixed.** `pre_dit` returns `meta = {grid_size:(f,h,w), tokens_per_frame,
   batch_size}` (`wan_video_dit.py:615-618`) — there is no direct `h/w` key; v1 hard-coded
   `h,w = 7,14`. Now read `grid_size` from `meta`.
3. **`fuse_vae_embedding_in_latents` is mandatory.** The `seperated_timestep` path in `pre_dit`
   raises `NotImplementedError` unless `fuse=True` (`wan_video_dit.py:537-552`). Added an assert;
   config already sets it True.
4. **Future-frame definition corrected.** v1 said `Fut_real` = 8 *separately strided pixel
   frames* re-encoded by the VAE. Reality: a clip is 33 px frames → **9 latent frames** via the
   VAE (temporal /4); frame-0 is current, frames 1..8 are future, so `Fut_real = z[:,:,1:9]`
   from a **single** clip encode (`libero_2cam.yaml:28,30`; `wan_video_vae.py:1383`;
   `fastwam.py:467-468,535-537`). Cache script and §5 rewritten accordingly.
5. **Grid claim verified (kept).** 224×448 →/16→ 14×28 →patch/2→ 7×14=98 tokens, dim 3072, split
   7×7|7×7. Confirmed VAE `upsampling_factor=16`, `patch_size=[1,2,2]`, `hidden_dim=3072`.
6. **Train/inference invariance verified (kept, now justified).** Frame-0 hidden states are
   bit-exact regardless of future tokens, because `first_frame_causal` masks frame-0→future
   (`wan_video_dit.py:501-505`) and `token_timesteps[:,0,:]=0` keeps frame-0 at t=0
   (`:537-550`), with text-only cross-attn for frame-0 in both paths. Stated the proof.
7. **Student `forward` signature corrected.** It is `forward(self, examples: List[dict])`
   returning `{"action_loss": ...}` (`QwenGR00T.py:170-218`), and the trainer reads
   `output_dict["action_loss"]` (`train_starvla.py:367`); batches are **lists of dicts**, not
   tensor dicts. v1's `forward(batch)` / `losses["action_loss"]=...+...` pattern adjusted; aux
   losses are folded into `action_loss`.
8. **`image_grid_thw` availability confirmed but slicing flagged as the #1 student risk.**
   `build_qwenvl_inputs` uses `apply_chat_template(return_dict=True)` which yields
   `image_grid_thw`; images are *separate* per camera (not concatenated), interleaved before
   text, marked by `IMAGE_TOKEN_INDEX=151655`, with **left padding**. Added a concrete
   `slice_image_tokens` recipe and a **pooled-REPA fallback** if spatial recovery proves
   infeasible under dynamic resolution.
9. **Hidden-state tuple length de-hard-coded.** v1 assumed `tuple(37)`. Read
   `text_config.num_hidden_layers` at runtime; marked the exact count UNVERIFIED with the file
   to check.
10. **Backbone mismatch surfaced.** QwenGR00T default = Qwen3-VL-4B (`QwenGR00T.py:63`) but the
    shipped LIBERO YAML = Qwen2.5-VL-3B (`starvla_cotrain_libero.yaml:13`). Plan now pins the
    backbone in the distill YAML and warns to keep `vl_hidden_dim` consistent.
11. **Canonical-key derivation made honest.** v1 implied the student key was readily available;
    in fact `_pack_sample` returns only image/lang/action/state (`datasets.py:1371-1404`) — a
    metadata hook is required, and the two LeRobot copies may number episodes/frames
    differently. Promoted to the top hard-gate risk.
12. **Image flip contract pinned.** Verified both *training* paths are resize-only (no flip;
    teacher `robot_video_dataset.py:180-181`, student `datasets.py:1377`), the optional student
    `VideoHorizontalFlip` must be disabled on the distill path, and the eval-only `[::-1,::-1]`
    180° rotation (`eval_libero.py:139-140`) does **not** affect REPA. Camera order confirmed
    consistent (left=agentview, right=wrist on both sides).
13. **Channel 2 feasibility tempered.** The reusable `cross_attention_dit.DiT` is a **1D-token**
    block (`cross_attention_dit.py:188-271`), so the 2D 48-ch future latents need
    patchify/un-patchify wiring — a real cost. Added a cheaper **feature-space future-match**
    alternative to evaluate against C2 at M3.
14. **Sequencing + caching strategy revised.** Ship Channel 1 before Channel 2 (M2 before M3);
    cache `Z` for all frames now (~175 GB) but cache `Fut` lazily on a single suite first
    (avoid the ~880 GB up-front blowup until C2 is justified).
15. **Tap layer default lowered to 15/30** (mid-depth, matching the student mid-layer tap) and
    `student_tap_layer` made `auto = 1 + num_layers//2`.

**Net:** the architecture/thesis is sound and the frame-0 invariance (the load-bearing
assumption for REPA) is genuinely exact. The real execution risks are (1) deriving the canonical
key on the starVLA side and (2) recovering a per-camera spatial grid from Qwen3-VL — both now
hard-gated with concrete verification and fallbacks. The future head is correct in principle but
costlier to wire than v1 implied, so it is deferred behind a working Channel 1 and a cheaper
feature-space alternative.
