def calc_zhat_0(z_n: Tensor, t: int):
    """ Query the consistency model to get zhat_0, which is the denoised embedding.
            Args:
                z_n (Tensor):   The noisy embedding.
                t (int):        The time step.
            Returns:
                Tensor:         The denoised embedding.
            """
    z_n_input = torch.cat([z_n] * 2) if use_cf_guidance else z_n
    z_n_input = self.scheduler.scale_model_input(z_n_input, t)
    zhat_0 = self.unet(z_n_input, t, guidance=cfg_scale_input,
        encoder_hidden_states=encoder_states, encoder_attention_mask=
        encoder_att_mask).sample
    if use_cf_guidance:
        zhat_0_uncond, zhat_0_cond = zhat_0.chunk(2)
        zhat_0 = (1 - cfg_scale_post
            ) * zhat_0_uncond + cfg_scale_post * zhat_0_cond
    return zhat_0
