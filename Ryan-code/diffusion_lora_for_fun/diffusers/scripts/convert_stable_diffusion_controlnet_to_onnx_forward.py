def forward(self, sample, timestep, encoder_hidden_states, controlnet_conds,
    controlnet_scales, text_embeds, time_ids):
    added_cond_kwargs = {'text_embeds': text_embeds, 'time_ids': time_ids}
    for i, (controlnet_cond, conditioning_scale, controlnet) in enumerate(zip
        (controlnet_conds, controlnet_scales, self.controlnets)):
        down_samples, mid_sample = controlnet(sample, timestep,
            encoder_hidden_states=encoder_hidden_states, controlnet_cond=
            controlnet_cond, conditioning_scale=conditioning_scale,
            added_cond_kwargs=added_cond_kwargs, return_dict=False)
        if i == 0:
            down_block_res_samples, mid_block_res_sample = (down_samples,
                mid_sample)
        else:
            down_block_res_samples = [(samples_prev + samples_curr) for 
                samples_prev, samples_curr in zip(down_block_res_samples,
                down_samples)]
            mid_block_res_sample += mid_sample
    noise_pred = self.unet(sample, timestep, encoder_hidden_states=
        encoder_hidden_states, down_block_additional_residuals=
        down_block_res_samples, mid_block_additional_residual=
        mid_block_res_sample, added_cond_kwargs=added_cond_kwargs,
        return_dict=False)[0]
    return noise_pred
