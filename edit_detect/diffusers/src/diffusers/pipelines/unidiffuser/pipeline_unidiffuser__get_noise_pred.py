def _get_noise_pred(self, mode, latents, t, prompt_embeds, img_vae,
    img_clip, max_timestep, data_type, guidance_scale, generator, device,
    height, width):
    """
        Gets the noise prediction using the `unet` and performs classifier-free guidance, if necessary.
        """
    if mode == 'joint':
        img_vae_latents, img_clip_latents, text_latents = self._split_joint(
            latents, height, width)
        img_vae_out, img_clip_out, text_out = self.unet(img_vae_latents,
            img_clip_latents, text_latents, timestep_img=t, timestep_text=t,
            data_type=data_type)
        x_out = self._combine_joint(img_vae_out, img_clip_out, text_out)
        if guidance_scale <= 1.0:
            return x_out
        img_vae_T = randn_tensor(img_vae.shape, generator=generator, device
            =device, dtype=img_vae.dtype)
        img_clip_T = randn_tensor(img_clip.shape, generator=generator,
            device=device, dtype=img_clip.dtype)
        text_T = randn_tensor(prompt_embeds.shape, generator=generator,
            device=device, dtype=prompt_embeds.dtype)
        _, _, text_out_uncond = self.unet(img_vae_T, img_clip_T,
            text_latents, timestep_img=max_timestep, timestep_text=t,
            data_type=data_type)
        img_vae_out_uncond, img_clip_out_uncond, _ = self.unet(img_vae_latents,
            img_clip_latents, text_T, timestep_img=t, timestep_text=
            max_timestep, data_type=data_type)
        x_out_uncond = self._combine_joint(img_vae_out_uncond,
            img_clip_out_uncond, text_out_uncond)
        return guidance_scale * x_out + (1.0 - guidance_scale) * x_out_uncond
    elif mode == 'text2img':
        img_vae_latents, img_clip_latents = self._split(latents, height, width)
        img_vae_out, img_clip_out, text_out = self.unet(img_vae_latents,
            img_clip_latents, prompt_embeds, timestep_img=t, timestep_text=
            0, data_type=data_type)
        img_out = self._combine(img_vae_out, img_clip_out)
        if guidance_scale <= 1.0:
            return img_out
        text_T = randn_tensor(prompt_embeds.shape, generator=generator,
            device=device, dtype=prompt_embeds.dtype)
        img_vae_out_uncond, img_clip_out_uncond, text_out_uncond = self.unet(
            img_vae_latents, img_clip_latents, text_T, timestep_img=t,
            timestep_text=max_timestep, data_type=data_type)
        img_out_uncond = self._combine(img_vae_out_uncond, img_clip_out_uncond)
        return guidance_scale * img_out + (1.0 - guidance_scale
            ) * img_out_uncond
    elif mode == 'img2text':
        img_vae_out, img_clip_out, text_out = self.unet(img_vae, img_clip,
            latents, timestep_img=0, timestep_text=t, data_type=data_type)
        if guidance_scale <= 1.0:
            return text_out
        img_vae_T = randn_tensor(img_vae.shape, generator=generator, device
            =device, dtype=img_vae.dtype)
        img_clip_T = randn_tensor(img_clip.shape, generator=generator,
            device=device, dtype=img_clip.dtype)
        img_vae_out_uncond, img_clip_out_uncond, text_out_uncond = self.unet(
            img_vae_T, img_clip_T, latents, timestep_img=max_timestep,
            timestep_text=t, data_type=data_type)
        return guidance_scale * text_out + (1.0 - guidance_scale
            ) * text_out_uncond
    elif mode == 'text':
        img_vae_out, img_clip_out, text_out = self.unet(img_vae, img_clip,
            latents, timestep_img=max_timestep, timestep_text=t, data_type=
            data_type)
        return text_out
    elif mode == 'img':
        img_vae_latents, img_clip_latents = self._split(latents, height, width)
        img_vae_out, img_clip_out, text_out = self.unet(img_vae_latents,
            img_clip_latents, prompt_embeds, timestep_img=t, timestep_text=
            max_timestep, data_type=data_type)
        img_out = self._combine(img_vae_out, img_clip_out)
        return img_out
