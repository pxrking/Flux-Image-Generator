"""
FLUX Image Generator with IP-Adapter Support
Uses InstantX FLUX pipeline with local model files.
"""

import argparse
import os
from pathlib import Path

import torch
from diffusers import FluxPipeline
from PIL import Image

# Model paths (local ComfyUI models)
MODELS_DIR = Path.home() / "flux-setup" / "ComfyUI" / "models"
UNET_PATH = MODELS_DIR / "unet" / "flux1-schnell.safetensors"
VAE_PATH = MODELS_DIR / "vae" / "ae.safetensors"
CLIP_L_PATH = MODELS_DIR / "clip" / "clip_l.safetensors"
T5_PATH = MODELS_DIR / "clip" / "t5xxl_fp8_e4m3fn.safetensors"

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
    """Load the FLUX Schnell pipeline from local files."""
    print("Loading FLUX Schnell pipeline...")
    print(f"  Using device: MPS (Apple Silicon)")

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.float16,
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
