"""
FLUX Image Generator with IP-Adapter Support
Uses InstantX FLUX pipeline with local model files.
"""

import argparse
import os
from pathlib import Path

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5TokenizerFast
from PIL import Image

# Model paths (local ComfyUI models)
MODELS_DIR = Path.home() / "flux-setup" / "ComfyUI" / "models"
UNET_PATH = MODELS_DIR / "unet" / "flux1-schnell.safetensors"
VAE_PATH = MODELS_DIR / "vae" / "ae.safetensors"

# HF cache paths (downloaded once, reused forever)
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--black-forest-labs--FLUX.1-schnell" / "snapshots" / "741f7c3ce8b383c54771c7003378a50191e9efe9"
TEXT_ENCODER_PATH = HF_CACHE / "text_encoder"
TEXT_ENCODER_2_PATH = HF_CACHE / "text_encoder_2"
TOKENIZER_PATH = HF_CACHE / "tokenizer"
TOKENIZER_2_PATH = HF_CACHE / "tokenizer_2"
SCHEDULER_PATH = HF_CACHE / "scheduler"

# Default output directory
OUTPUT_DIR = Path.home() / "Desktop"

# Default prompt
DEFAULT_PROMPT = (
    "A romantic couple wedding portrait at the ancient ghats of Varanasi India, "
    "sacred Ganges river glowing at golden hour, ornate Hindu mandap decorated with "
    "marigold garlands and roses, sacred havan fire in foreground, "
    "cinematic depth of field, photorealistic wedding photography, 8k ultra detailed, masterpiece"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with FLUX")
    parser.add_argument("--prompt", "-p", type=str, default=DEFAULT_PROMPT,
                        help="Text prompt for image generation")
    parser.add_argument("--output", "-o", type=str, default=str(OUTPUT_DIR / "generated.png"),
                        help="Output file path")
    parser.add_argument("--steps", type=int, default=4,
                        help="Number of inference steps (default: 4 for Schnell)")
    parser.add_argument("--guidance", type=float, default=1.0,
                        help="Guidance scale (default: 1.0 for Schnell)")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width (default: 1024)")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height (default: 1024)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def load_pipeline():
    """Load the FLUX Schnell pipeline from local model files."""
    print("Loading FLUX Schnell pipeline from local files...")
    print(f"  Using device: MPS (Apple Silicon)")

    # Load the transformer (23.8GB) from local file
    print(f"  Loading transformer from {UNET_PATH.name}...")
    transformer = FluxTransformer2DModel.from_single_file(
        str(UNET_PATH),
        config="black-forest-labs/FLUX.1-schnell",
        subfolder="transformer",
        torch_dtype=torch.float16,
    )

    # Load the VAE from local file
    print(f"  Loading VAE from {VAE_PATH.name}...")
    vae = AutoencoderKL.from_single_file(
        str(VAE_PATH),
        config="black-forest-labs/FLUX.1-schnell",
        subfolder="vae",
        torch_dtype=torch.float16,
    )

    # Load text encoders from HF cache (already downloaded)
    print(f"  Loading CLIP text encoder...")
    text_encoder = CLIPTextModel.from_pretrained(
        str(TEXT_ENCODER_PATH), torch_dtype=torch.float16, local_files_only=True,
    )

    print(f"  Loading T5 text encoder...")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        str(TEXT_ENCODER_2_PATH), torch_dtype=torch.float16, local_files_only=True,
    )

    # Load tokenizers from HF cache
    tokenizer = CLIPTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)
    tokenizer_2 = T5TokenizerFast.from_pretrained(str(TOKENIZER_2_PATH), local_files_only=True)

    # Assemble pipeline — everything from local files, zero downloads
    print("  Assembling pipeline...")
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(SCHEDULER_PATH), local_files_only=True,
    )

    pipe = FluxPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        scheduler=scheduler,
    )
    pipe.to("mps")

    print("Pipeline loaded successfully.")
    return pipe


def generate(pipe, prompt, steps, guidance, width, height, seed=None):
    """Generate an image from a text prompt."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device="mps").manual_seed(seed)
        print(f"  Seed: {seed}")

    print(f"  Steps: {steps}")
    print(f"  Guidance: {guidance}")
    print(f"  Size: {width}x{height}")
    print(f"  Prompt: {prompt[:80]}...")
    print("Generating...")

    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
    ).images[0]

    return image


def main():
    args = parse_args()

    pipe = load_pipeline()

    image = generate(
        pipe=pipe,
        prompt=args.prompt,
        steps=args.steps,
        guidance=args.guidance,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved to: {output_path}")

    # Open the image
    image.show()


if __name__ == "__main__":
    main()
