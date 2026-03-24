# FLUX Image Generator

A local AI image generation app using FLUX Schnell + IP-Adapter for personalized image generation. Upload a reference photo and describe a scene — the model will generate an image incorporating the reference face/style.

Built with a Gradio web UI that runs locally on Apple Silicon Macs.

## Features

- **Text-to-image** generation with FLUX Schnell
- **IP-Adapter support** for face/style transfer from a reference photo
- **Gradio web UI** — type a prompt, upload an image, and hit Generate
- **Runs locally** — no cloud API needed, all models on disk

## Requirements

- macOS with Apple Silicon (MPS backend)
- Python 3.12+
- ~41GB disk space for models
- 36GB+ unified memory recommended

## Model Setup

Download models to `~/flux-setup/ComfyUI/models/`:

| Model | Path | Size |
|-------|------|------|
| FLUX Schnell transformer | `unet/flux1-schnell.safetensors` | 23.8GB |
| VAE | `vae/ae.safetensors` | 335MB |
| CLIP text encoder | `clip/clip_l.safetensors` | — |
| T5 text encoder | `clip/t5xxl_fp8_e4m3fn.safetensors` | — |
| IP-Adapter | `ipadapter/ip-adapter.bin` | 5.29GB |
| SigLIP vision encoder | Downloaded from `google/siglip-so400m-patch14-384` | 3.51GB |

## Installation

```bash
# Activate your venv (must have PyTorch with MPS support)
source ~/flux-setup/ComfyUI/venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web UI (Gradio)

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

### Command Line

```bash
# Text-only generation
python generate.py --prompt "a sunset over the ocean"

# With reference image (IP-Adapter)
python generate.py --prompt "a wedding portrait" --reference photo.jpg --scale 0.8
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt`, `-p` | Varanasi wedding prompt | Text prompt |
| `--reference`, `-r` | None | Reference image for IP-Adapter |
| `--scale` | 0.8 | IP-Adapter influence strength |
| `--steps` | 4 | Inference steps (4 is good for Schnell) |
| `--guidance` | 1.0 | Guidance scale |
| `--width` | 1024 | Image width |
| `--height` | 1024 | Image height |
| `--seed` | Random | Seed for reproducibility |
| `--output`, `-o` | `~/Desktop/generated.png` | Output path |

## Stack

- [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) — fast text-to-image model
- [InstantX FLUX IP-Adapter](https://github.com/InstantX-Research/FLUX.1-dev-IP-Adapter) — reference image conditioning
- [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) — vision encoder for IP-Adapter
- [Gradio](https://gradio.app/) — web UI
- [diffusers](https://github.com/huggingface/diffusers) — pipeline framework
