"""
FlowEdit on Z-Image (Tongyi-MAI) 6B
===================================

Edit an image with a Source-Prompt -> Target-Prompt pair using the FlowEdit
ODE on top of Tongyi-MAI's Z-Image / Z-Image-Turbo model.

Example
-------
    python run_zimage.py \
        --input_img example_images/lighthouse.png \
        --src_prompt "a tall white lighthouse on a hill, blue sky" \
        --tar_prompt "Big Ben clock tower on a hill, blue sky" \
        --output edited.png

Notes
-----
* Z-Image-Turbo is a *distilled* rectified-flow model. It does not respond to
  classifier-free guidance, so the defaults below set both
  `src_guidance_scale` and `tar_guidance_scale` to 1.0 (CFG disabled), which
  also halves runtime/VRAM by skipping the unconditional forward.
* The image side length is automatically rounded down to a multiple of 16
  (= vae_scale_factor * 2 = 8 * 2) to satisfy Z-Image's patch grid; aspect
  ratio is preserved otherwise.
* Diffusers must include ZImagePipeline (PRs #12703 / #12715). If your
  installed version doesn't expose it, install from source:
      pip install -U git+https://github.com/huggingface/diffusers.git
"""

import argparse
import os
import random

import numpy as np
import torch
from PIL import Image

from FlowEdit_utils import FlowEditZImage


def parse_args():
    parser = argparse.ArgumentParser(description="FlowEdit with Z-Image")
    parser.add_argument("--input_img", type=str, required=True,
                        help="Path to the input image to edit.")
    parser.add_argument("--src_prompt", type=str, required=True,
                        help="Text describing the input image (source prompt).")
    parser.add_argument("--tar_prompt", type=str, required=True,
                        help="Text describing the desired edit (target prompt).")
    parser.add_argument("--output", type=str, default="flowedit_zimage_output.png",
                        help="Output image path.")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="Optional negative prompt (only used if CFG > 1).")

    # model
    parser.add_argument("--model_id", type=str,
                        default="Tongyi-MAI/Z-Image-Turbo",
                        help="HF model id or local path. Use 'Tongyi-MAI/Z-Image' "
                             "for the non-distilled base model.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--cpu_offload", action="store_true",
                        help="Enable model CPU offload to save VRAM.")

    # FlowEdit hyper-parameters
    parser.add_argument("--T_steps", type=int, default=28,
                        help="Total number of ODE steps. Z-Image-Turbo "
                             "officially uses 8 NFEs but FlowEdit benefits "
                             "from a denser schedule; 28 is a safe default.")
    parser.add_argument("--n_max", type=int, default=24,
                        help="Largest noise level used for editing "
                             "(higher -> more deviation from source).")
    parser.add_argument("--n_min", type=int, default=0,
                        help="Number of final SDEdit-refinement steps.")
    parser.add_argument("--n_avg", type=int, default=1,
                        help="Number of velocity samples averaged per step.")
    parser.add_argument("--src_guidance_scale", type=float, default=1.0,
                        help="CFG scale for the source branch (1.0 = off, "
                             "recommended for Z-Image-Turbo).")
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
    (Z-Image expects sides divisible by `vae_scale_factor * 2 = 16`)."""
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
    # 1. Load the Z-Image pipeline
    # ------------------------------------------------------------------
    try:
        from diffusers import ZImagePipeline
    except ImportError as e:
        raise ImportError(
            "ZImagePipeline not found. Please install a recent diffusers:\n"
            "    pip install -U git+https://github.com/huggingface/diffusers.git"
        ) from e

    print(f"[FlowEdit-ZImage] Loading {args.model_id} ({args.dtype}) ...")
    pipe = ZImagePipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe = pipe.to(device)
    scheduler = pipe.scheduler

    # ------------------------------------------------------------------
    # 2. Encode the source image into VAE latents (scaled space)
    # ------------------------------------------------------------------
    image = load_and_preprocess(args.input_img, multiple_of=16)
    print(f"[FlowEdit-ZImage] Input size after crop: {image.size}")

    image_src = pipe.image_processor.preprocess(image)            # (1, 3, H, W) in [-1, 1]
    image_src = image_src.to(device=device, dtype=dtype)

    with torch.inference_mode():
        latent_dist = pipe.vae.encode(image_src).latent_dist
        x0_raw = latent_dist.mode()                               # raw VAE latents
    # Scale into the transformer's working space:
    #   z = (raw - shift) * scaling
    shift_factor = pipe.vae.config.shift_factor
    scaling_factor = pipe.vae.config.scaling_factor
    x0_src = (x0_raw - shift_factor) * scaling_factor

    # ------------------------------------------------------------------
    # 3. Run FlowEdit
    # ------------------------------------------------------------------
    print(f"[FlowEdit-ZImage] Running FlowEdit ({args.T_steps} steps, "
          f"n_max={args.n_max}, n_min={args.n_min}) ...")
    x0_tar = FlowEditZImage(
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
    #    Inverse of (raw - shift) * scaling  ==>  raw = z / scaling + shift
    # ------------------------------------------------------------------
    print("[FlowEdit-ZImage] Decoding edited latents ...")
    x0_tar_decode = (x0_tar.to(dtype) / scaling_factor) + shift_factor
    with torch.inference_mode():
        image_tar = pipe.vae.decode(x0_tar_decode, return_dict=False)[0]
    image_tar = pipe.image_processor.postprocess(image_tar, output_type="pil")

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    image_tar[0].save(out_path)
    print(f"[FlowEdit-ZImage] Saved edited image to: {out_path}")


if __name__ == "__main__":
    main()
