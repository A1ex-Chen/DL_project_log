def reconstruct_x0_n_steps(pipeline: DiffusionPipeline, x0: torch.Tensor,
    noise_scale: Union[int, torch.FloatTensor], noise: torch.Tensor=None,
    generator: torch.Generator=0) ->torch.Tensor:
    if x0.dim() != 4:
        x0 = x0.unsqueeze(0)
    num: int = x0.shape[0]
    generator = set_generator(generator=generator)
    if noise is None:
        noise = torch.randn(size=x0.shape, generator=generator)
    xt = get_xt(x0=x0, noise_scale=noise_scale, noise=noise)
    latents = pipeline.invert(init=xt).latents
    images = pipeline(init=latents).latents
    return images
