"""
FlowEdit on FLUX.2 [klein] 4B
=============================

Edit an image with a Source-Prompt -> Target-Prompt pair using the FlowEdit
ODE on top of Black Forest Labs' FLUX.2 [klein] 4B model.

Example
-------
    python run_flux2_klein.py \
        --input_img example_images/lighthouse.png \
        --src_prompt "a tall white lighthouse on a hill, blue sky" \
        --tar_prompt "Big Ben clock tower on a hill, blue sky" \
        --output edited.png

Notes
-----
* FLUX.2 [klein] 4B is a *distilled* rectified-flow model. It does not
  respond to classifier-free guidance, so the defaults below set both
  `src_guidance_scale` and `tar_guidance_scale` to 1.0 (CFG disabled), which
  also halves runtime/VRAM by skipping the unconditional forward.
* About 13 GB VRAM is needed in bf16 (use `--cpu_offload` if your GPU is
  smaller).
* The image side length is automatically rounded down to a multiple of 16
  to satisfy FLUX.2's patch grid; aspect ratio is preserved otherwise.
"""

import argparse
import os
import random

import numpy as np
import torch
from PIL import Image

from FlowEdit_utils import FlowEditFLUX2Klein


def parse_args():
    parser = argparse.ArgumentParser(
        description="FlowEdit with FLUX.2 [klein] 4B"
    )
    parser.add_argument("--input_img", type=str, required=True,
                        help="Path to the input image to edit.")
    parser.add_argument("--src_prompt", type=str, required=True,
                        help="Text describing the input image (source prompt).")
    parser.add_argument("--tar_prompt", type=str, required=True,
                        help="Text describing the desired edit (target prompt).")
    parser.add_argument("--output", type=str, default="flowedit_output.png",
                        help="Output image path.")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="Optional negative prompt (only used if CFG > 1).")

    # model
    parser.add_argument("--model_id", type=str,
                        default="black-forest-labs/FLUX.2-klein-4B",
                        help="HF model id or local path. "
                             "Fallback mirror: 'YuCollection/FLUX.2-klein-4B-Diffusers'.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--cpu_offload", action="store_true",
                        help="Enable model CPU offload to save VRAM.")

    # FlowEdit hyper-parameters (defaults follow the FLUX section of the paper)
    parser.add_argument("--T_steps", type=int, default=28,
                        help="Total number of ODE steps.")
    parser.add_argument("--n_max", type=int, default=24,
                        help="Largest noise level used for editing "
                             "(higher -> more deviation from source).")
    parser.add_argument("--n_min", type=int, default=0,
                        help="Number of final SDEdit-refinement steps.")
    parser.add_argument("--n_avg", type=int, default=1,
                        help="Number of velocity samples averaged per step.")
    parser.add_argument("--src_guidance_scale", type=float, default=1.0,
                        help="CFG scale for the source branch (1.0 = off, "
                             "recommended for distilled FLUX.2 [klein]).")
    parser.add_argument("--tar_guidance_scale", type=float, default=1.0,
                        help="CFG scale for the target branch.")

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_preprocess(image_path: str, multiple_of: int = 16) -> Image.Image:
    """Open the image and crop so both sides are divisible by `multiple_of`
    (FLUX.2 expects sides divisible by `vae_scale_factor * 2 = 16`)."""
    img = Image.open(image_path).convert("RGB")
    new_w = img.width - (img.width % multiple_of)
    new_h = img.height - (img.height % multiple_of)
    if (new_w, new_h) != img.size:
        img = img.crop((0, 0, new_w, new_h))
    return img


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    # ------------------------------------------------------------------
    # 1. Load the FLUX.2 [klein] pipeline
    # ------------------------------------------------------------------
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as e:
        raise ImportError(
            "Flux2KleinPipeline not found. Please install a recent diffusers:\n"
            "    pip install -U git+https://github.com/huggingface/diffusers.git\n"
            "(or `pip install -U diffusers>=0.36`)"
        ) from e

    print(f"[FlowEdit-FLUX2] Loading {args.model_id} ({args.dtype}) ...")
    pipe = Flux2KleinPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe = pipe.to(device)
    scheduler = pipe.scheduler

    # ------------------------------------------------------------------
    # 2. Encode the source image into VAE latents
    # ------------------------------------------------------------------
    image = load_and_preprocess(args.input_img, multiple_of=16)
    print(f"[FlowEdit-FLUX2] Input size after crop: {image.size}")

    image_src = pipe.image_processor.preprocess(image)            # (1, 3, H, W) in [-1, 1]
    image_src = image_src.to(device=device, dtype=dtype)

    with torch.inference_mode():
        latent_dist = pipe.vae.encode(image_src).latent_dist
        x0_src = latent_dist.mode()                               # raw VAE latents

    # ------------------------------------------------------------------
    # 3. Run FlowEdit
    # ------------------------------------------------------------------
    print(f"[FlowEdit-FLUX2] Running FlowEdit ({args.T_steps} steps, "
          f"n_max={args.n_max}, n_min={args.n_min}) ...")
    x0_tar = FlowEditFLUX2Klein(
        pipe=pipe,
        scheduler=scheduler,
        x_src=x0_src,
        src_prompt=args.src_prompt,
        tar_prompt=args.tar_prompt,
        negative_prompt=args.negative_prompt,
        T_steps=args.T_steps,
        n_avg=args.n_avg,
        src_guidance_scale=args.src_guidance_scale,
        tar_guidance_scale=args.tar_guidance_scale,
        n_min=args.n_min,
        n_max=args.n_max,
    )

    # ------------------------------------------------------------------
    # 4. Decode back to image
    # ------------------------------------------------------------------
    print("[FlowEdit-FLUX2] Decoding edited latents ...")
    with torch.inference_mode():
        image_tar = pipe.vae.decode(x0_tar.to(dtype), return_dict=False)[0]
    image_tar = pipe.image_processor.postprocess(image_tar, output_type="pil")

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    image_tar[0].save(out_path)
    print(f"[FlowEdit-FLUX2] Saved edited image to: {out_path}")


if __name__ == "__main__":
    main()
