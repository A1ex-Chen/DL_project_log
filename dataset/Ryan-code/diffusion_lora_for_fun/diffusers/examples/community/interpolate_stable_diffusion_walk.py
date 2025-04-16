def walk(self, prompts: List[str], seeds: List[int],
    num_interpolation_steps: Optional[int]=6, output_dir: Optional[str]=
    './dreams', name: Optional[str]=None, batch_size: Optional[int]=1,
    height: Optional[int]=512, width: Optional[int]=512, guidance_scale:
    Optional[float]=7.5, num_inference_steps: Optional[int]=50, eta:
    Optional[float]=0.0) ->List[str]:
    """
        Walks through a series of prompts and seeds, interpolating between them and saving the results to disk.

        Args:
            prompts (`List[str]`):
                List of prompts to generate images for.
            seeds (`List[int]`):
                List of seeds corresponding to provided prompts. Must be the same length as prompts.
            num_interpolation_steps (`int`, *optional*, defaults to 6):
                Number of interpolation steps to take between prompts.
            output_dir (`str`, *optional*, defaults to `./dreams`):
                Directory to save the generated images to.
            name (`str`, *optional*, defaults to `None`):
                Subdirectory of `output_dir` to save the generated images to. If `None`, the name will
                be the current time.
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate at once.
            height (`int`, *optional*, defaults to 512):
                Height of the generated images.
            width (`int`, *optional*, defaults to 512):
                Width of the generated images.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.

        Returns:
            `List[str]`: List of paths to the generated images.
        """
    if not len(prompts) == len(seeds):
        raise ValueError(
            f'Number of prompts and seeds must be equalGot {len(prompts)} prompts and {len(seeds)} seeds'
            )
    name = name or time.strftime('%Y%m%d-%H%M%S')
    save_path = Path(output_dir) / name
    save_path.mkdir(exist_ok=True, parents=True)
    frame_idx = 0
    frame_filepaths = []
    for prompt_a, prompt_b, seed_a, seed_b in zip(prompts, prompts[1:],
        seeds, seeds[1:]):
        embed_a = self.embed_text(prompt_a)
        embed_b = self.embed_text(prompt_b)
        noise_dtype = embed_a.dtype
        noise_a = self.get_noise(seed_a, noise_dtype, height, width)
        noise_b = self.get_noise(seed_b, noise_dtype, height, width)
        noise_batch, embeds_batch = None, None
        T = np.linspace(0.0, 1.0, num_interpolation_steps)
        for i, t in enumerate(T):
            noise = slerp(float(t), noise_a, noise_b)
            embed = torch.lerp(embed_a, embed_b, t)
            noise_batch = noise if noise_batch is None else torch.cat([
                noise_batch, noise], dim=0)
            embeds_batch = embed if embeds_batch is None else torch.cat([
                embeds_batch, embed], dim=0)
            batch_is_ready = embeds_batch.shape[0
                ] == batch_size or i + 1 == T.shape[0]
            if batch_is_ready:
                outputs = self(latents=noise_batch, text_embeddings=
                    embeds_batch, height=height, width=width,
                    guidance_scale=guidance_scale, eta=eta,
                    num_inference_steps=num_inference_steps)
                noise_batch, embeds_batch = None, None
                for image in outputs['images']:
                    frame_filepath = str(save_path /
                        f'frame_{frame_idx:06d}.png')
                    image.save(frame_filepath)
                    frame_filepaths.append(frame_filepath)
                    frame_idx += 1
    return frame_filepaths
