def calc_zhat_0(z_n, t, prompt_embeds, prompt_mask, guidance_scale_input,
    guidance_scale_post):
    use_cf_guidance = guidance_scale_post > 1.0
    z_n_input = torch.cat([z_n] * 2) if use_cf_guidance else z_n
    z_n_input = inference_scheduler.scale_model_input(z_n_input, t)
    unet = self.student_ema_unet if use_ema else self.student_target_unet
    zhat_0 = unet(z_n_input, t, guidance=guidance_scale_input,
        encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_mask
        ).sample
    if use_cf_guidance:
        zhat_0_uncond, zhat_0_cond = zhat_0.chunk(2)
        zhat_0 = (1 - guidance_scale_post
            ) * zhat_0_uncond + guidance_scale_post * zhat_0_cond
    return zhat_0
