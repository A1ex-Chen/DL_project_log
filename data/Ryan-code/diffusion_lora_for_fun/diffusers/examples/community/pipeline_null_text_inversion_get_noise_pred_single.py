def get_noise_pred_single(self, latents, t, context):
    noise_pred = self.unet(latents, t, encoder_hidden_states=context)['sample']
    return noise_pred
