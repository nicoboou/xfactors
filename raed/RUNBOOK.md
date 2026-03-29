# RAED Runbook (Simple)

This runbook lists the minimal commands to train/evaluate each stage.

## GPU selection (YAML)

Each train config has:

- `runtime.gpu_ids: [0]`
- `runtime.use_ddp: false`

Examples:

- Single GPU `2`: `runtime.gpu_ids=[2] runtime.use_ddp=false`
- Multi-GPU DDP on GPUs `0,1,2`: `runtime.gpu_ids=[0,1,2] runtime.use_ddp=true`

For true DDP, launch with `torchrun` and set `CUDA_VISIBLE_DEVICES` consistently.

## Dataset choice (mandatory)

You must always set `data.dataset` explicitly:

- `data.dataset=celeba` -> requires `data.root=/path/to/celeba_root`
- `data.dataset=shapes` -> requires `data.train_path`, `data.val_path` (NPZ) and `data.image_key`

Sampler mode:

- `data.sampler_mode=shuffle` (recommended, no dataset pre-scan bottleneck)
- `data.sampler_mode=balanced` (currently uses shuffle fallback without pre-scan)

## 0) Setup

```bash
uv sync
source .venv/bin/activate
python -m compileall raed
```

---

## 1) Stage A — Disentanglement in DINO space

### Train
```bash
python ./raed/src/train/train_stage_a.py --config raed/configs/stage_a_base.yaml
```

### Train with specific GPU id (example GPU 2)
```bash
python -m raed.src.train.train_stage_a --config raed/configs/stage_a_base.yaml data.dataset=celeba data.root=/projects/compures/alexandre/PyTorch-VAE/Data/ runtime.gpu_ids=[2] runtime.use_ddp=false
```

### Train on Shapes3D NPZ (single GPU)
```bash
python -m raed.src.train.train_stage_a --config raed/configs/stage_a_base.yaml data.dataset=shapes data.train_path=/projects/compures/alexandre/disdiff_adapters/disdiff_adapters/data/3dshapes/shapes3d_train.npz data.val_path=/projects/compures/alexandre/disdiff_adapters/disdiff_adapters/data/3dshapes/shapes3d_val.npz data.image_key=images
```

### Train with true DDP on GPUs 0,1,2
```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 -m raed.src.train.train_stage_a --config raed/configs/stage_a_base.yaml data.dataset=celeba data.root=/projects/compures/alexandre/PyTorch-VAE/Data/ runtime.gpu_ids=[0,1,2] runtime.use_ddp=true
```

### Evaluate (all)
```bash
python -m raed.src.eval.eval_probes --config raed/configs/stage_a_base.yaml
python -m raed.src.eval.eval_swap --config raed/configs/stage_a_base.yaml
python -m raed.src.eval.eval_geometry --config raed/configs/stage_a_base.yaml
```

### Ablations
```bash
python -m raed.src.train.train_stage_a --config raed/configs/ablations/ablation_no_nce.yaml
python -m raed.src.train.train_stage_a --config raed/configs/ablations/ablation_no_kl_t.yaml
```

---

## 2) Stage B1 — Pixel Decoder (RAE-like decoder)

## Options
- `decoder_mode=plain`
- `decoder_mode=dinotok`

### Train (default config)
```bash
python -m raed.src.train.train_stage_b_decoder --config raed/configs/stage_b_decoder.yaml data.dataset=celeba data.root=/projects/compures/alexandre/PyTorch-VAE/Data/
```

### Train with true DDP on GPUs 0,1,2
```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 -m raed.src.train.train_stage_b_decoder --config raed/configs/stage_b_decoder.yaml data.dataset=celeba data.root=/projects/compures/alexandre/PyTorch-VAE/Data/ runtime.gpu_ids=[0,1,2] runtime.use_ddp=true
```

### Evaluate restoration
```bash
python -m raed.src.eval.eval_restoration --config raed/configs/stage_b_decoder.yaml
```

### Decoder ablations
```bash
python -m raed.src.train.train_stage_b_decoder --config raed/configs/ablations/ablation_plain_decoder.yaml
python -m raed.src.train.train_stage_b_decoder --config raed/configs/ablations/ablation_dinotok_decoder.yaml
```

---

## 3) Stage B2 — Diffusion (single toggle, two options)

## Option 1
- `diffusion.option=option1`
- Uses only semantic `s` conditioning
- No Stage B1 decoder at generation time

## Option 2
- `diffusion.option=option2`
- Diffuses on `[s,t]`
- Uses Stage B1 decoder
- Inference does `t_bad -> t_clean_anchor` then refine/decode

### Train Option 1
```bash
python -m raed.src.train.train_stage_b_diffusion --config raed/configs/stage_b2_diffusion.yaml data.dataset=celeba data.root=/projects/compures/alexandre/PyTorch-VAE/Data/ diffusion.option=option1
```

### Train Option 1 with true DDP on GPUs 0,1,2
```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 -m raed.src.train.train_stage_b_diffusion --config raed/configs/stage_b2_diffusion.yaml data.dataset=celeba data.root=/projects/compures/alexandre/PyTorch-VAE/Data/ diffusion.option=option1 runtime.gpu_ids=[0,1,2] runtime.use_ddp=true
```

### Train Option 2
```bash
python -m raed.src.train.train_stage_b_diffusion --config raed/configs/stage_b2_diffusion.yaml data.dataset=celeba data.root=/projects/compures/alexandre/PyTorch-VAE/Data/ diffusion.option=option2
```

### Train Option 2 with true DDP on GPUs 0,1,2
```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 -m raed.src.train.train_stage_b_diffusion --config raed/configs/stage_b2_diffusion.yaml data.dataset=celeba data.root=/projects/compures/alexandre/PyTorch-VAE/Data/ diffusion.option=option2 runtime.gpu_ids=[0,1,2] runtime.use_ddp=true
```

### Evaluate Option 1
```bash
python -m raed.src.eval.eval_diffusion_refinement --config raed/configs/stage_b2_diffusion.yaml diffusion.option=option1 --out_dir raed/outputs/eval_diffusion_refinement_option1
```

### Evaluate Option 2
```bash
python -m raed.src.eval.eval_diffusion_refinement --config raed/configs/stage_b2_diffusion.yaml diffusion.option=option2 --out_dir raed/outputs/eval_diffusion_refinement_option2
```

### B2 ablation configs (optional)
```bash
python -m raed.src.train.train_stage_b_diffusion --config raed/configs/ablations/ablation_b2_option1.yaml
python -m raed.src.train.train_stage_b_diffusion --config raed/configs/ablations/ablation_b2_option2.yaml
```

---

## 4) Aggregate final ablation comparison

```bash
python -m raed.src.eval.eval_ablation \
  --decoder_metrics raed/outputs/eval_restoration/restoration_metrics.json \
  --option1_metrics raed/outputs/eval_diffusion_refinement_option1/option1_metrics.json \
  --option2_metrics raed/outputs/eval_diffusion_refinement_option2/option2_metrics.json
```

Output:
- `raed/outputs/eval_ablation/ablation_metrics.csv`
- `raed/outputs/eval_ablation/ablation_metrics.json`

---

## 5) Quick smoke commands (fast checks)

```bash
python -m raed.src.train.train_stage_a --config raed/configs/stage_a_base.yaml data.dataset=celeba data.root=/projects/compures/alexandre/PyTorch-VAE/Data/ train.epochs=1
python -m raed.src.train.train_stage_b_decoder --config raed/configs/stage_b_decoder.yaml data.dataset=celeba data.root=/projects/compures/alexandre/PyTorch-VAE/Data/ train.epochs=1
python -m raed.src.train.train_stage_b_diffusion --config raed/configs/stage_b2_diffusion.yaml data.dataset=celeba data.root=/projects/compures/alexandre/PyTorch-VAE/Data/ diffusion.option=option1 train.epochs=1
python -m raed.src.train.train_stage_b_diffusion --config raed/configs/stage_b2_diffusion.yaml data.dataset=celeba data.root=/projects/compures/alexandre/PyTorch-VAE/Data/ diffusion.option=option2 train.epochs=1
```
