def _forward(self, latent, steps, t0, lora_scale_min, text_embeddings,
    generator):

    def scale_schedule(begin, end, n, length, type='linear'):
        if type == 'constant':
            return end
        elif type == 'linear':
            return begin + (end - begin) * n / length
        elif type == 'cos':
            factor = (1 - math.cos(n * math.pi / length)) / 2
            return (1 - factor) * begin + factor * end
        else:
            raise NotImplementedError(type)
    noises = []
    latents = []
    lora_scales = []
    cfg_scales = []
    latents.append(latent)
    t0 = int(t0 * steps)
    t_begin = steps - t0
    length = len(self.scheduler.timesteps[t_begin - 1:-1]) - 1
    index = 1
    for t in self.scheduler.timesteps[t_begin:].flip(dims=[0]):
        lora_scale = scale_schedule(1, lora_scale_min, index, length, type=
            'cos')
        cfg_scale = scale_schedule(1, 3.0, index, length, type='linear')
        latent, noise = self._forward_sde(t, latent, cfg_scale,
            text_embeddings, steps, lora_scale=lora_scale, generator=generator)
        noises.append(noise)
        latents.append(latent)
        lora_scales.append(lora_scale)
        cfg_scales.append(cfg_scale)
        index += 1
    return latent, noises, latents, lora_scales, cfg_scales
