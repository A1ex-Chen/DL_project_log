def get_noise_pred(self, latents, t, context):
    latents_input = torch.cat([latents] * 2)
    guidance_scale = 7.5
    noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)[
        'sample']
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text -
        noise_pred_uncond)
    latents = self.prev_step(noise_pred, t, latents)
    return latents
