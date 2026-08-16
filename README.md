# Medical Image Super-Resolution with Diffusion Models

**Final Year Project** — a denoising diffusion probabilistic model (DDPM) that reconstructs high-resolution medical scans from low-resolution inputs.

Medical imaging is often captured at reduced resolution to keep scan times and radiation dose down. Recovering detail afterwards is a super-resolution problem, and the failure mode that matters clinically is *hallucinated* detail — structure that looks plausible but isn't in the patient. This project takes the diffusion route because its iterative denoising gives a better-behaved reconstruction than the adversarial approach explored first in the companion repo, [medical-image-super-resolution-srgan](https://github.com/maliawan0/medical-image-super-resolution-srgan).

## Architecture

A conditional U-Net denoiser with sinusoidal-style time conditioning:

- **`TimeEmbedding`** — projects the diffusion timestep `t` into a learned embedding injected at each stage
- **`DoubleConv`** blocks — 3×3 convolutions with GroupNorm and ReLU
- **U-Net** encoder/decoder with skip connections, `BASE_CHANNELS = 64`
- Single-channel in and out (`IN_CHANNELS = OUT_CHANNELS = 1`) — grayscale medical modalities

## Configuration

All hyperparameters live in [`config.py`](config.py):

| Setting | Value | Note |
|---|---|---|
| `IMAGE_SIZE` | 512 | Full-resolution target |
| `BASE_CHANNELS` | 64 | U-Net width |
| `TIMESTEPS` | 50 | Balances sample quality against inference cost |
| `BETA_START` / `BETA_END` | 1e-4 / 0.02 | Linear noise schedule |
| `EPOCHS` | 200 | |
| `BATCH_SIZE` | 2 | Sized for 6 GB VRAM at 512×512 |
| `LR` | 2e-4 | Converges faster than 1e-4 in practice |
| `SAVE_EVERY` | 5 | Checkpoint interval, in epochs |

## Layout

```
config.py      # all hyperparameters and paths
model.py       # TimeEmbedding, DoubleConv, U-Net denoiser
diffusion.py   # forward noising + reverse sampling loop
dataset.py     # paired HR/LR loader
train.py       # training loop with checkpointing
test.py        # inference / sample generation
```

## Usage

Place paired data under `data/HR` and `data/LR`, then:

```bash
pip install -r requirements.txt
python train.py
```

Checkpoints are written to `checkpoints/` every 5 epochs and samples to `samples/`. To run inference with a trained checkpoint:

```bash
python test.py
```

## Requirements

PyTorch with CUDA (`DEVICE = "cuda"` in config). The default batch size targets a 6 GB card; raise it if you have more headroom.
