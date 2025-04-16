def pred_x0(self, latents, noise_pred, t, generator, device, prompt_embeds,
    output_type):
    pred_z0 = self.pred_z0(latents, noise_pred, t)
    pred_x0 = self.vae.decode(pred_z0 / self.vae.config.scaling_factor,
        return_dict=False, generator=generator)[0]
    pred_x0, ____ = self.run_safety_checker(pred_x0, device, prompt_embeds.
        dtype)
    do_denormalize = [True] * pred_x0.shape[0]
    pred_x0 = self.image_processor.postprocess(pred_x0, output_type=
        output_type, do_denormalize=do_denormalize)
    return pred_x0
