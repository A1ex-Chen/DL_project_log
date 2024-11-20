def __call__(self, img: Image.Image, prompt: str, kmin: float=0.3, kmax:
    float=0.6, mix_factor: float=0.5, seed: int=42, steps: int=50,
    guidance_scale: float=7.5) ->Image.Image:
    tmin = steps - int(kmin * steps)
    tmax = steps - int(kmax * steps)
    text_embeddings = self.prep_text(prompt)
    self.scheduler.set_timesteps(steps)
    width, height = img.size
    encoded = self.encode(img)
    torch.manual_seed(seed)
    noise = torch.randn((1, self.unet.config.in_channels, height // 8, 
        width // 8)).to(self.device)
    latents = self.scheduler.add_noise(encoded, noise, timesteps=self.
        scheduler.timesteps[tmax])
    input = torch.cat([latents] * 2)
    input = self.scheduler.scale_model_input(input, self.scheduler.
        timesteps[tmax])
    with torch.no_grad():
        pred = self.unet(input, self.scheduler.timesteps[tmax],
            encoder_hidden_states=text_embeddings).sample
    pred_uncond, pred_text = pred.chunk(2)
    pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)
    latents = self.scheduler.step(pred, self.scheduler.timesteps[tmax], latents
        ).prev_sample
    for i, t in enumerate(tqdm(self.scheduler.timesteps)):
        if i > tmax:
            if i < tmin:
                orig_latents = self.scheduler.add_noise(encoded, noise,
                    timesteps=t)
                input = mix_factor * latents + (1 - mix_factor) * orig_latents
                input = torch.cat([input] * 2)
            else:
                input = torch.cat([latents] * 2)
            input = self.scheduler.scale_model_input(input, t)
            with torch.no_grad():
                pred = self.unet(input, t, encoder_hidden_states=
                    text_embeddings).sample
            pred_uncond, pred_text = pred.chunk(2)
            pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)
            latents = self.scheduler.step(pred, t, latents).prev_sample
    return self.decode(latents)
