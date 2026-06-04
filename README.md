# FlowEdit-Advanced

[Original Project](https://matankleiner.github.io/flowedit/) | [Arxiv](https://arxiv.org/abs/2412.08629) | [Original Repo](https://github.com/fallenshock/FlowEdit) | [Demo](https://huggingface.co/spaces/fallenshock/FlowEdit)

**Extended PyTorch implementation** of [FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](https://arxiv.org/abs/2412.08629) (ICCV 2025 Best Student Paper), with additional support for **FLUX.2 [klein] 4B** and **Z-Image / Z-Image-Turbo 6B**.

> This repo is built on top of [fallenshock/FlowEdit](https://github.com/fallenshock/FlowEdit). The original implementation supports Stable Diffusion 3 and FLUX.1; we extend it to two additional state-of-the-art rectified-flow models.

---

## What's New

| Model | Size | Type | CFG | Script |
|---|---|---|---|---|
| **FLUX.2 [klein]** | 4B | Distilled rectified-flow | Off (distilled) | `run_flux2_klein.py` |
| **Z-Image / Z-Image-Turbo** | 6B | Rectified-flow (base / distilled) | Optional (off for Turbo) | `run_zimage.py` |

Key differences from the original FlowEdit:

- **FLUX.2 [klein]**: No `pooled_projections` or guidance embeddings; uses 2×2 patchify + BatchNorm normalisation; supports optional real CFG for non-distilled variants.
- **Z-Image**: Transformer expects 5D latents with per-sample prompt embeddings; sign-flipped velocity output; uses VAE shift/scaling factor for latent space conversion.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/FlowEdit-Advanced.git
   cd FlowEdit-Advanced
   ```

2. Install dependencies:
   ```bash
   pip install torch diffusers transformers accelerate sentencepiece protobuf
   ```

   > **Note:** FLUX.2 [klein] and Z-Image require a recent `diffusers`. If the pipeline classes are not found, install from source:
   > ```bash
   > pip install -U git+https://github.com/huggingface/diffusers.git
   > ```
   > Tested with CUDA 12.4 and `diffusers >= 0.36`.

---

## Quick Start

### FLUX.2 [klein] 4B

```bash
python run_flux2_klein.py \
    --input_img example_images/lighthouse.png \
    --src_prompt "a tall white lighthouse on a hill, blue sky" \
    --tar_prompt "Big Ben clock tower on a hill, blue sky" \
    --output edited.png
```

### Z-Image / Z-Image-Turbo 6B

```bash
python run_zimage.py \
    --input_img example_images/lighthouse.png \
    --src_prompt "a tall white lighthouse on a hill, blue sky" \
    --tar_prompt "Big Ben clock tower on a hill, blue sky" \
    --output edited.png
```

To use the non-distilled Z-Image base model (supports CFG):

```bash
python run_zimage.py \
    --model_id Tongyi-MAI/Z-Image \
    --src_guidance_scale 3.0 \
    --tar_guidance_scale 7.0 \
    --input_img example_images/lighthouse.png \
    --src_prompt "a tall white lighthouse on a hill, blue sky" \
    --tar_prompt "Big Ben clock tower on a hill, blue sky" \
    --output edited.png
```

---

## Usage – Your Own Examples

1. Place your input image in the `example_images/` folder (or anywhere on disk).

2. Choose the appropriate script and provide the required arguments:

   | Argument | Description |
   |---|---|
   | `--input_img` | Path to the input image |
   | `--src_prompt` | Text describing the input image |
   | `--tar_prompt` | Text describing the desired edit |
   | `--output` | Output image path |
   | `--negative_prompt` | Optional negative prompt (only effective when CFG > 1) |
   | `--model_id` | HuggingFace model id or local path |
   | `--cpu_offload` | Enable CPU offload to save VRAM |

3. Tune FlowEdit hyper-parameters:

   | Parameter | Default (FLUX.2) | Default (Z-Image) | Description |
   |---|---|---|---|
   | `--T_steps` | 28 | 28 | Total ODE steps |
   | `--n_max` | 24 | 24 | Largest noise level used (higher → more deviation) |
   | `--n_min` | 0 | 0 | Final SDEdit-refinement steps |
   | `--n_avg` | 1 | 1 | Velocity samples averaged per step |
   | `--src_guidance_scale` | 1.0 | 1.0 | CFG scale for source (1.0 = off for distilled) |
   | `--tar_guidance_scale` | 1.0 | 1.0 | CFG scale for target |

   For a detailed discussion of hyper-parameters, refer to the [original paper](https://arxiv.org/abs/2412.08629).

---

## Supported Models Summary

| Model | Pipeline Class | VRAM (bf16) | CFG | Notes |
|---|---|---|---|---|
| SD3 | `StableDiffusion3Pipeline` | ~12 GB | Yes | Original FlowEdit |
| FLUX.1 | `FluxPipeline` | ~16 GB | Yes | Original FlowEdit |
| **FLUX.2 [klein] 4B** | `Flux2KleinPipeline` | ~13 GB | Optional | New ✨ |
| **Z-Image / Turbo 6B** | `ZImagePipeline` | ~14 GB | Optional | New ✨ |

---

## Project Structure

```
FlowEdit-Advanced/
├── FlowEdit_utils.py       # Core algorithms: FlowEditSD3, FlowEditFLUX, FlowEditFLUX2Klein, FlowEditZImage
├── run_flux2_klein.py      # Runner script for FLUX.2 [klein] 4B
├── run_zimage.py           # Runner script for Z-Image / Z-Image-Turbo
└── example_images/         # Sample images for testing
```

---

## Acknowledgements

- [FlowEdit](https://github.com/fallenshock/FlowEdit) — Original implementation by Kulikov et al.
- [FLUX.2 [klein]](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) — Black Forest Labs
- [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) — Tongyi-MAI (通义万相)

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT), consistent with the original FlowEdit repository.

---

## Citation

If you use this code, please cite the original FlowEdit paper:

```bibtex
@inproceedings{kulikov2025flowedit,
  title={Flowedit: Inversion-free text-based editing using pre-trained flow models},
  author={Kulikov, Vladimir and Kleiner, Matan and Huberman-Spiegelglas, Inbar and Michaeli, Tomer},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={19721--19730},
  year={2025}
}
```
