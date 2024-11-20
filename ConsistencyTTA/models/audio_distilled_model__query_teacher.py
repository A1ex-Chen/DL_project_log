def _query_teacher(self, z_scaled, t, prompt_embeds, prompt_mask,
    guidance_scale=None):
    """ This helper function takes care of classifier-free guidance
            The last argument (guidance_scale) is only effective when using variable
            guidance scale, i.e., self.teacher_guidance_scale is -1
        """
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    if len(t.reshape(-1)) != 1 and self.use_teacher_cf_guidance:
        t = torch.cat([t] * 2)
    z_scaled_cat = torch.cat([z_scaled] * 2
        ) if self.use_teacher_cf_guidance else z_scaled
    noise_pred = self.teacher_unet(z_scaled_cat, t, prompt_embeds,
        encoder_attention_mask=prompt_mask).sample.detach()
    if self.use_teacher_cf_guidance:
        if self.teacher_guidance_scale == -1:
            if not torch.is_tensor(guidance_scale):
                guidance_scale = torch.tensor(guidance_scale)
            cur_guidance_scale = guidance_scale.to(noise_pred.device)
            cur_guidance_scale = cur_guidance_scale.reshape(-1, 1, 1, 1)
        else:
            cur_guidance_scale = self.teacher_guidance_scale
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = (1 - cur_guidance_scale
            ) * noise_pred_uncond + cur_guidance_scale * noise_pred_cond
    assert not noise_pred.isnan().any(), f'noise_pred is NaN at t={t}'
    return noise_pred
