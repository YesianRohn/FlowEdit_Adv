from typing import Optional, Tuple, Union
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
import numpy as np

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps



def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Foward process in flow-matching

    Args:
        sample (`torch.FloatTensor`):
            The input sample.
        timestep (`int`, *optional*):
            The current timestep in the diffusion chain.

    Returns:
        `torch.FloatTensor`:
            A scaled input sample.
    """
    # if scheduler.step_index is None:
    scheduler._init_step_index(timestep)

    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample

    return sample


# for flux
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu



def calc_v_sd3(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(src_tar_latent_model_input.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i


    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # perform guidance source
        if pipe.do_classifier_free_guidance:
            src_noise_pred_uncond, src_noise_pred_text, tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_src_tar.chunk(4)
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)
            noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)

    return noise_pred_src, noise_pred_tar



def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(latents.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i


    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return noise_pred



@torch.no_grad()
def FlowEditSD3(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,):
    
    device = x_src.device

    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)

    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    pipe._guidance_scale = src_guidance_scale
    
    # src prompts
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )
 
    # CFG prep
    src_tar_prompt_embeds = torch.cat([src_negative_prompt_embeds, src_prompt_embeds, tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
    src_tar_pooled_prompt_embeds = torch.cat([src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)
    
    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps)):
        
        if T_steps - i > n_max:
            continue
        
        t_i = t/1000
        if i+1 < len(timesteps): 
            t_im1 = (timesteps[i+1])/1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)
        
        if T_steps - i > n_min:

            # Calculate the average of the V predictions
            V_delta_avg = torch.zeros_like(x_src)
            for k in range(n_avg):

                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                
                zt_src = (1-t_i)*x_src + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_model_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar]) if pipe.do_classifier_free_guidance else (zt_src, zt_tar) 

                Vt_src, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input,src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)

                V_delta_avg += (1/n_avg) * (Vt_tar - Vt_src) # - (hfg-1)*( x_src))

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else: # i >= T_steps-n_min # regular sampling for last n_min steps

            if i == T_steps-n_min:
                # initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src
                
            src_tar_latent_model_input = torch.cat([xt_tar, xt_tar, xt_tar, xt_tar]) if pipe.do_classifier_free_guidance else (xt_src, xt_tar)

            _, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input,src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)

            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(noise_pred_tar.dtype)

            xt_tar = prev_sample
        
    return zt_edit if n_min == 0 else xt_tar



@torch.no_grad()
def FlowEditFLUX(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,):

    device = x_src.device
    orig_height, orig_width = x_src.shape[2]*pipe.vae_scale_factor//2, x_src.shape[3]*pipe.vae_scale_factor//2
    num_channels_latents = pipe.transformer.config.in_channels // 4

    pipe.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=orig_height,
        width=orig_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    x_src, latent_src_image_ids = pipe.prepare_latents(batch_size= x_src.shape[0], num_channels_latents=num_channels_latents, height=orig_height, width=orig_width, dtype=x_src.dtype, device=x_src.device, generator=None,latents=x_src)
    x_src_packed = pipe._pack_latents(x_src, x_src.shape[0], num_channels_latents, x_src.shape[2], x_src.shape[3])
    latent_tar_image_ids = latent_src_image_ids

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
        )
    
    num_warmup_steps = max(len(timesteps) - T_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    
    # src prompts
    (
        src_prompt_embeds,
        src_pooled_prompt_embeds,
        src_text_ids,

    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=device,
    )

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_text_ids,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    # handle guidance
    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device)
        src_guidance = src_guidance.expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device)
        tar_guidance = tar_guidance.expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src_packed.clone()

    for i, t in tqdm(enumerate(timesteps)):
        
        if T_steps - i > n_max:
            continue
        
        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if i < len(timesteps):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = t_i
        
        if T_steps - i > n_min:

            # Calculate the average of the V predictions
            V_delta_avg = torch.zeros_like(x_src_packed)

            for k in range(n_avg):
                                    

                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                
                zt_src = (1-t_i)*x_src_packed + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src_packed

                # Merge in the future to avoid double computation
                Vt_src = calc_v_flux(pipe,
                                                    latents=zt_src,
                                                    prompt_embeds=src_prompt_embeds, 
                                                    pooled_prompt_embeds=src_pooled_prompt_embeds, 
                                                    guidance=src_guidance,
                                                    text_ids=src_text_ids, 
                                                    latent_image_ids=latent_src_image_ids, 
                                                    t=t)
                
                Vt_tar = calc_v_flux(pipe,
                                                    latents=zt_tar,
                                                    prompt_embeds=tar_prompt_embeds, 
                                                    pooled_prompt_embeds=tar_pooled_prompt_embeds, 
                                                    guidance=tar_guidance,
                                                    text_ids=tar_text_ids, 
                                                    latent_image_ids=latent_tar_image_ids, 
                                                    t=t)

                V_delta_avg += (1/n_avg) * (Vt_tar - Vt_src) # - (hfg-1)*( x_src))

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg

            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else: # i >= T_steps-n_min # regular sampling last n_min steps

            if i == T_steps-n_min:
                # initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed
                
            Vt_tar = calc_v_flux(pipe,
                                    latents=xt_tar,
                                    prompt_embeds=tar_prompt_embeds, 
                                    pooled_prompt_embeds=tar_pooled_prompt_embeds, 
                                    guidance=tar_guidance,
                                    text_ids=tar_text_ids, 
                                    latent_image_ids=latent_tar_image_ids, 
                                    t=t)


            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample
    out = zt_edit if n_min == 0 else xt_tar
    unpacked_out = pipe._unpack_latents(out, orig_height, orig_width, pipe.vae_scale_factor)
    return unpacked_out



# ---------------------------------------------------------------------------
# FLUX.2 (Klein) support
# ---------------------------------------------------------------------------

def calc_v_flux2(pipe, latents, prompt_embeds, text_ids, latent_image_ids, t):
    """
    Single forward pass through the FLUX.2 (Klein) transformer.

    NOTE - Differences vs FLUX.1:
        * No `pooled_projections` argument.
        * `guidance` is None (Klein uses real CFG via two passes; the
          distilled 4B variant is conditioned without a guidance embed).
        * `timestep` must be normalised to [0, 1] (divide by 1000).
    """
    timestep = t.expand(latents.shape[0]).to(latents.dtype)

    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep / 1000,
            img_ids=latent_image_ids,
            txt_ids=text_ids,
            guidance=None,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
    return noise_pred


@torch.no_grad()
def FlowEditFLUX2Klein(
    pipe,
    scheduler,
    x_src,                       # raw VAE latents, shape (B, C_lat, H_lat, W_lat)
    src_prompt,
    tar_prompt,
    negative_prompt: str = "",
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.0,
    tar_guidance_scale: float = 1.0,
    n_min: int = 0,
    n_max: int = 24,
):
    """
    FlowEdit adapted to the FLUX.2 (Klein) Diffusers pipeline.

    `x_src` is expected to be the *raw* VAE latents produced by
    `pipe.vae.encode(...).latent_dist.mode()`. This routine performs the
    FLUX.2-specific normalisation (BatchNorm running stats), 2x2 patchify,
    and packing internally, mirroring the official Flux2KleinPipeline.

    Returns un-packed un-patchified latents in the *normalised* space, so the
    caller must apply the inverse normalisation before VAE decoding.
    """
    device = x_src.device
    dtype = x_src.dtype

    # --------------------------------------------------------------
    # 1. 2x2 patchify, then normalise with VAE BN running stats
    #    NOTE: BN running stats live in the patchified (C*4) space, so the
    #    order MUST be patchify -> BN, mirroring Flux2KleinPipeline._encode_vae_image.
    # --------------------------------------------------------------
    x_src_patched = pipe._patchify_latents(x_src)            # (B, C*4, H/2, W/2)

    bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype)
    bn_std = torch.sqrt(
        pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
    ).to(device, dtype)

    x_src_patched = (x_src_patched - bn_mean) / bn_std       # (B, C*4, H/2, W/2)

    # --------------------------------------------------------------
    # 2. Build position ids and pack into sequence form
    # --------------------------------------------------------------
    latent_ids = pipe._prepare_latent_ids(x_src_patched).to(device)
    x_src_packed = pipe._pack_latents(x_src_patched)         # (B, H/2 * W/2, C*4)

    # --------------------------------------------------------------
    # 3. Encode prompts (Qwen3 -> (B, seq, hidden), text_ids (B, seq, 4))
    # --------------------------------------------------------------
    src_prompt_embeds, src_text_ids = pipe.encode_prompt(
        prompt=src_prompt, device=device,
    )
    tar_prompt_embeds, tar_text_ids = pipe.encode_prompt(
        prompt=tar_prompt, device=device,
    )

    do_cfg = (src_guidance_scale > 1.0) or (tar_guidance_scale > 1.0)
    if do_cfg:
        neg_prompt_embeds, neg_text_ids = pipe.encode_prompt(
            prompt=negative_prompt if negative_prompt is not None else "",
            device=device,
        )

    # --------------------------------------------------------------
    # 4. Build timesteps (FLUX-style sigma schedule with mu shift)
    # --------------------------------------------------------------
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]

    # FLUX.2 ships its own empirical mu; fall back to FLUX.1 shift if missing.
    try:
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=T_steps)
    except Exception:
        mu = calculate_shift(
            image_seq_len,
            getattr(scheduler.config, "base_image_seq_len", 256),
            getattr(scheduler.config, "max_image_seq_len", 4096),
            getattr(scheduler.config, "base_shift", 0.5),
            getattr(scheduler.config, "max_shift", 1.16),
        )

    timesteps, T_steps = retrieve_timesteps(
        scheduler, T_steps, device, timesteps=None, sigmas=sigmas, mu=mu,
    )
    pipe._num_timesteps = len(timesteps)

    # --------------------------------------------------------------
    # 5. FlowEdit ODE integration
    # --------------------------------------------------------------
    zt_edit = x_src_packed.clone()
    xt_tar = None

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

        if T_steps - i > n_max:
            continue

        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if scheduler.step_index + 1 < len(scheduler.sigmas):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = torch.zeros_like(t_i)

        if T_steps - i > n_min:

            V_delta_avg = torch.zeros_like(x_src_packed)
            for _ in range(n_avg):

                fwd_noise = torch.randn_like(x_src_packed)
                zt_src = (1 - t_i) * x_src_packed + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src_packed

                # cond forward (src)
                Vt_src = calc_v_flux2(
                    pipe, zt_src, src_prompt_embeds, src_text_ids, latent_ids, t,
                )
                # cond forward (tar)
                Vt_tar = calc_v_flux2(
                    pipe, zt_tar, tar_prompt_embeds, tar_text_ids, latent_ids, t,
                )

                # Optional real CFG (skip for distilled checkpoints)
                if do_cfg:
                    Vt_src_uncond = calc_v_flux2(
                        pipe, zt_src, neg_prompt_embeds, neg_text_ids, latent_ids, t,
                    )
                    Vt_tar_uncond = calc_v_flux2(
                        pipe, zt_tar, neg_prompt_embeds, neg_text_ids, latent_ids, t,
                    )
                    Vt_src = Vt_src_uncond + src_guidance_scale * (Vt_src - Vt_src_uncond)
                    Vt_tar = Vt_tar_uncond + tar_guidance_scale * (Vt_tar - Vt_tar_uncond)

                V_delta_avg += (1.0 / n_avg) * (Vt_tar - Vt_src)

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:  # last n_min steps -> SDEdit-style refinement on the target
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src_packed)
                xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed

            Vt_tar = calc_v_flux2(
                pipe, xt_tar, tar_prompt_embeds, tar_text_ids, latent_ids, t,
            )
            if do_cfg:
                Vt_tar_uncond = calc_v_flux2(
                    pipe, xt_tar, neg_prompt_embeds, neg_text_ids, latent_ids, t,
                )
                Vt_tar = Vt_tar_uncond + tar_guidance_scale * (Vt_tar - Vt_tar_uncond)

            xt_tar = xt_tar.to(torch.float32)
            xt_tar = xt_tar + (t_im1 - t_i) * Vt_tar
            xt_tar = xt_tar.to(Vt_tar.dtype)

    out_packed = zt_edit if n_min == 0 else xt_tar

    # --------------------------------------------------------------
    # 6. Unpack: (B, seq, C*4) -> (B, C*4, H/2, W/2),
    #    invert BN normalisation in patchified space, then unpatchify
    #    -> (B, C, H, W) raw VAE space.
    # --------------------------------------------------------------
    _, _, H, W = x_src_patched.shape
    out_2d = pipe._unpack_latents_with_ids(out_packed, latent_ids, H, W)   # (B, C*4, H/2, W/2)
    # invert BatchNorm normalisation (still in patchified C*4 space)
    out_2d = out_2d * bn_std + bn_mean
    out_raw = pipe._unpatchify_latents(out_2d)                             # (B, C, H, W)
    return out_raw


# ---------------------------------------------------------------------------
# Z-Image (Tongyi-MAI) support
# ---------------------------------------------------------------------------

def calc_v_zimage(pipe, latents, prompt_embeds_list, t):
    """
    Single conditional forward pass through the Z-Image transformer.

    NOTE - Differences vs FLUX:
        * Transformer is called with positional args: (latents_list, timestep, prompt_embeds_list).
        * No img_ids / txt_ids / guidance / pooled_projections.
        * latents must be 5D (unsqueeze(2)) and split to a list along batch dim.
        * prompt_embeds is a list of variable-length tensors (one per sample).
        * timestep is reverse-normalised: (1000 - t) / 1000.
        * The transformer's velocity is sign-flipped (noise_pred = -noise_pred).
    """
    B = latents.shape[0]
    # match the batch length of prompt_embeds_list (caller is responsible for that)
    assert len(prompt_embeds_list) == B, (
        f"prompt_embeds_list length ({len(prompt_embeds_list)}) "
        f"must match latents batch ({B})"
    )

    timestep = t.expand(B).to(latents.dtype)
    timestep = (1000 - timestep) / 1000

    latent_5d = latents.unsqueeze(2)                       # (B, C, 1, H, W)
    latent_list = list(latent_5d.unbind(dim=0))            # list of (C, 1, H, W)

    with torch.no_grad():
        out_list = pipe.transformer(
            latent_list,
            timestep,
            prompt_embeds_list,
            return_dict=False,
        )[0]

    noise_pred = torch.stack([o.float() for o in out_list], dim=0)   # (B, C, 1, H, W)
    noise_pred = noise_pred.squeeze(2)                                # (B, C, H, W)
    noise_pred = -noise_pred                                          # Z-Image sign convention
    return noise_pred.to(latents.dtype)


@torch.no_grad()
def FlowEditZImage(
    pipe,
    scheduler,
    x_src,                       # VAE latents AFTER (x - shift) * scaling, shape (B, C, H, W)
    src_prompt,
    tar_prompt,
    negative_prompt: str = "",
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.0,
    tar_guidance_scale: float = 1.0,
    n_min: int = 0,
    n_max: int = 24,
):
    """
    FlowEdit adapted to Tongyi-MAI's Z-Image pipeline.

    Conventions
    -----------
    * `x_src` is expected to already be in the *scaled* latent space the
      transformer operates in:  x_src = (vae.encode(image).mode() - shift) * scaling.
    * Returns latents in the same scaled space; the caller must invert the
      VAE scaling before decoding:  z = x_tar / scaling + shift.
    * For Z-Image-Turbo, set src_/tar_guidance_scale = 1.0 (CFG off) — the
      distilled checkpoint does not respond to CFG and disabling it halves
      runtime/VRAM.
    """
    device = x_src.device
    dtype = x_src.dtype

    # --------------------------------------------------------------
    # 1. Encode prompts.  pipe.encode_prompt returns (pos_list, neg_list);
    #    both are *lists* of variable-length tensors (one per sample).
    # --------------------------------------------------------------
    do_cfg = (src_guidance_scale > 1.0) or (tar_guidance_scale > 1.0)

    src_prompt_embeds, src_neg_embeds = pipe.encode_prompt(
        prompt=src_prompt,
        device=device,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt if do_cfg else None,
    )
    tar_prompt_embeds, tar_neg_embeds = pipe.encode_prompt(
        prompt=tar_prompt,
        device=device,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt if do_cfg else None,
    )
    # Force batch size 1 throughout this routine (single-image edit).
    B = x_src.shape[0]
    assert B == 1, "FlowEditZImage currently supports batch size 1."

    # --------------------------------------------------------------
    # 2. Build timesteps — Flux-style sigma schedule with mu shift.
    #    image_seq_len uses patch=2: (H//2) * (W//2)
    # --------------------------------------------------------------
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = (x_src.shape[2] // 2) * (x_src.shape[3] // 2)
    mu = calculate_shift(
        image_seq_len,
        getattr(scheduler.config, "base_image_seq_len", 256),
        getattr(scheduler.config, "max_image_seq_len", 4096),
        getattr(scheduler.config, "base_shift", 0.5),
        getattr(scheduler.config, "max_shift", 1.15),
    )
    scheduler.sigma_min = 0.0
    timesteps, T_steps = retrieve_timesteps(
        scheduler, T_steps, device, timesteps=None, sigmas=sigmas, mu=mu,
    )
    pipe._num_timesteps = len(timesteps)

    # --------------------------------------------------------------
    # 3. FlowEdit ODE integration
    # --------------------------------------------------------------
    zt_edit = x_src.clone()
    xt_tar = None

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
        if T_steps - i > n_max:
            continue

        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if scheduler.step_index + 1 < len(scheduler.sigmas):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = torch.zeros_like(t_i)

        if T_steps - i > n_min:
            V_delta_avg = torch.zeros_like(x_src)
            for _ in range(n_avg):
                fwd_noise = torch.randn_like(x_src)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src

                # cond forward (src / tar)
                Vt_src = calc_v_zimage(pipe, zt_src, src_prompt_embeds, t)
                Vt_tar = calc_v_zimage(pipe, zt_tar, tar_prompt_embeds, t)

                if do_cfg:
                    Vt_src_uncond = calc_v_zimage(pipe, zt_src, src_neg_embeds, t)
                    Vt_tar_uncond = calc_v_zimage(pipe, zt_tar, tar_neg_embeds, t)
                    Vt_src = Vt_src_uncond + src_guidance_scale * (Vt_src - Vt_src_uncond)
                    Vt_tar = Vt_tar_uncond + tar_guidance_scale * (Vt_tar - Vt_tar_uncond)

                V_delta_avg += (1.0 / n_avg) * (Vt_tar - Vt_src)

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:  # last n_min steps -> SDEdit-style refinement on the target
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            Vt_tar = calc_v_zimage(pipe, xt_tar, tar_prompt_embeds, t)
            if do_cfg:
                Vt_tar_uncond = calc_v_zimage(pipe, xt_tar, tar_neg_embeds, t)
                Vt_tar = Vt_tar_uncond + tar_guidance_scale * (Vt_tar - Vt_tar_uncond)

            xt_tar = xt_tar.to(torch.float32)
            xt_tar = xt_tar + (t_im1 - t_i) * Vt_tar
            xt_tar = xt_tar.to(Vt_tar.dtype)

    return zt_edit if n_min == 0 else xt_tar
