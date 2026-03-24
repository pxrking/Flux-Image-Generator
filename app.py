"""
FLUX Image Generator — Gradio Web UI
Run with: python app.py
Opens at http://localhost:7860
"""

import torch
import gradio as gr
from PIL import Image
from pathlib import Path

from generate import (
    load_pipeline_text_only,
    load_pipeline_ip_adapter,
)

# Global pipeline cache so we don't reload 40GB every time
_text_pipe = None
_ip_model = None


def get_text_pipe():
    global _text_pipe
    if _text_pipe is None:
        _text_pipe = load_pipeline_text_only()
    return _text_pipe


def get_ip_model(scale=0.8):
    global _ip_model
    if _ip_model is None:
        _ip_model = load_pipeline_ip_adapter(scale=scale)
    return _ip_model


def generate_image(prompt, reference_image, steps, guidance, scale, width, height, seed):
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    seed = int(seed) if seed else None

    if reference_image is not None:
        # IP-Adapter mode
        ref = reference_image.convert("RGB")
        ip_model = get_ip_model(scale=scale)
        images = ip_model.generate(
            pil_image=ref,
            prompt=prompt,
            scale=scale,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=seed,
        )
        return images[0]
    else:
        # Text-only mode
        pipe = get_text_pipe()
        generator = None
        if seed is not None:
            generator = torch.Generator(device="mps").manual_seed(seed)
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
        ).images[0]
        return image


with gr.Blocks(title="FLUX Image Generator") as demo:
    gr.Markdown("# FLUX Image Generator")
    gr.Markdown("Type a prompt and optionally upload a reference photo for face/style transfer via IP-Adapter.")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="e.g. A wedding photo of me and my gf in traditional Varanasi Indian attire",
                lines=3,
            )
            reference_image = gr.Image(label="Reference Photo (optional)", type="pil")

            with gr.Accordion("Settings", open=False):
                with gr.Row():
                    steps = gr.Slider(1, 50, value=4, step=1, label="Steps")
                    guidance = gr.Slider(0.0, 10.0, value=1.0, step=0.1, label="Guidance Scale")
                with gr.Row():
                    scale = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="IP-Adapter Scale")
                    seed = gr.Number(label="Seed (blank = random)", precision=0)
                with gr.Row():
                    width = gr.Slider(512, 1024, value=512, step=64, label="Width")
                    height = gr.Slider(512, 1024, value=512, step=64, label="Height")

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Generated Image", type="pil")

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, reference_image, steps, guidance, scale, width, height, seed],
        outputs=output_image,
    )

if __name__ == "__main__":
    demo.launch()
