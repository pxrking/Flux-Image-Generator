"""
FLUX Image Generator with IP-Adapter Support
Uses InstantX custom pipeline for IP-Adapter reference photos.
Falls back to standard diffusers for text-only generation.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

# Model paths (local ComfyUI models)
MODELS_DIR = Path.home() / "flux-setup" / "ComfyUI" / "models"
UNET_PATH = MODELS_DIR / "unet" / "flux1-schnell.safetensors"
VAE_PATH = MODELS_DIR / "vae" / "ae.safetensors"
IP_ADAPTER_PATH = MODELS_DIR / "ipadapter" / "ip-adapter.bin"
IMAGE_ENCODER_PATH = "google/siglip-so400m-patch14-384"

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
    parser.add_argument("--reference", "-r", type=str, default=None,
                        help="Path to reference image for IP-Adapter (face/style transfer)")
    parser.add_argument("--scale", type=float, default=0.8,
                        help="IP-Adapter scale — how strongly the reference influences output (default: 0.8)")
    return parser.parse_args()


def load_pipeline_text_only():
    """Load the standard FLUX Schnell pipeline for text-only generation."""
    from diffusers import FluxPipeline as DiffusersFluxPipeline
    from diffusers import FluxTransformer2DModel as DiffusersTransformer
    from diffusers import AutoencoderKL
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5TokenizerFast

    print("Loading FLUX Schnell pipeline (text-only)...")
    print("  Using device: MPS (Apple Silicon)")

    print(f"  Loading transformer from {UNET_PATH.name}...")
    transformer = DiffusersTransformer.from_single_file(
        str(UNET_PATH),
        config="black-forest-labs/FLUX.1-schnell",
        subfolder="transformer",
        torch_dtype=torch.float16,
    )

    print(f"  Loading VAE from {VAE_PATH.name}...")
    vae = AutoencoderKL.from_single_file(
        str(VAE_PATH),
        config="black-forest-labs/FLUX.1-schnell",
        subfolder="vae",
        torch_dtype=torch.float16,
    )

    print("  Loading CLIP text encoder...")
    text_encoder = CLIPTextModel.from_pretrained(
        str(TEXT_ENCODER_PATH), torch_dtype=torch.float16, local_files_only=True,
    )

    print("  Loading T5 text encoder...")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        str(TEXT_ENCODER_2_PATH), torch_dtype=torch.float16, local_files_only=True,
    )

    tokenizer = CLIPTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)
    tokenizer_2 = T5TokenizerFast.from_pretrained(str(TOKENIZER_2_PATH), local_files_only=True)

    print("  Assembling pipeline...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(SCHEDULER_PATH), local_files_only=True,
    )

    pipe = DiffusersFluxPipeline(
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


def load_pipeline_ip_adapter(scale=0.8):
    """Load the InstantX custom FLUX pipeline with IP-Adapter support."""
    from diffusers import FluxTransformer2DModel as DiffusersTransformer
    from transformer_flux import FluxTransformer2DModel as InstantXTransformer
    from pipeline_flux_ipa import FluxPipeline as InstantXFluxPipeline
    from infer_flux_ipa_siglip import IPAdapter

    print("Loading FLUX Schnell pipeline (with IP-Adapter)...")
    print("  Using device: MPS (Apple Silicon)")

    # Load weights via standard diffusers (from_single_file), then transfer
    # to the InstantX custom transformer that supports IP-Adapter attention.
    print(f"  Loading transformer from {UNET_PATH.name}...")
    diffusers_transformer = DiffusersTransformer.from_single_file(
        str(UNET_PATH),
        config="black-forest-labs/FLUX.1-schnell",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    print("  Converting to InstantX transformer...")
    import inspect
    valid_params = set(inspect.signature(InstantXTransformer.__init__).parameters.keys()) - {"self"}
    config = {k: v for k, v in diffusers_transformer.config.items() if k in valid_params}
    transformer = InstantXTransformer(**config)
    transformer.load_state_dict(diffusers_transformer.state_dict(), strict=True)
    transformer = transformer.to(dtype=torch.bfloat16)
    del diffusers_transformer
    import gc
    gc.collect()
    torch.mps.empty_cache()

    print("  Building InstantX pipeline...")
    pipe = InstantXFluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    gc.collect()
    torch.mps.empty_cache()

    print(f"  Loading IP-Adapter and SigLIP image encoder...")
    ip_model = IPAdapter(
        pipe, IMAGE_ENCODER_PATH, str(IP_ADAPTER_PATH),
        device="mps", num_tokens=128,
    )

    print("Pipeline with IP-Adapter loaded successfully.")
    return ip_model


def main():
    args = parse_args()

    if args.reference:
        # IP-Adapter mode — use InstantX custom pipeline
        ref_path = Path(args.reference).expanduser()
        if not ref_path.exists():
            print(f"Error: Reference image not found: {ref_path}")
            return

        reference_image = Image.open(ref_path).convert("RGB")
        print(f"  Reference image: {ref_path.name} ({reference_image.size[0]}x{reference_image.size[1]})")

        ip_model = load_pipeline_ip_adapter(scale=args.scale)

        print(f"  Steps: {args.steps}")
        print(f"  Guidance: {args.guidance}")
        print(f"  Scale: {args.scale}")
        print(f"  Size: {args.width}x{args.height}")
        print(f"  Prompt: {args.prompt[:80]}...")
        print("Generating with IP-Adapter...")

        images = ip_model.generate(
            pil_image=reference_image,
            prompt=args.prompt,
            scale=args.scale,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
        )
        image = images[0]

    else:
        # Text-only mode — use standard diffusers pipeline
        pipe = load_pipeline_text_only()

        generator = None
        if args.seed is not None:
            generator = torch.Generator(device="mps").manual_seed(args.seed)
            print(f"  Seed: {args.seed}")

        print(f"  Steps: {args.steps}")
        print(f"  Guidance: {args.guidance}")
        print(f"  Size: {args.width}x{args.height}")
        print(f"  Prompt: {args.prompt[:80]}...")
        print("Generating...")

        image = pipe(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            width=args.width,
            height=args.height,
            generator=generator,
        ).images[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved to: {output_path}")
    image.show()


if __name__ == "__main__":
    main()
