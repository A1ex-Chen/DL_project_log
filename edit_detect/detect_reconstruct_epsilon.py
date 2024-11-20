def reconstruct_epsilon(pipeline: DiffusionPipeline, x0: torch.Tensor, t:
    Union[int, torch.IntTensor, torch.LongTensor], noise: torch.Tensor=None,
    generator: torch.Generator=0) ->torch.Tensor:
    if x0.dim() != 4:
        x0 = x0.unsqueeze(0)
    num: int = x0.shape[0]
    if isinstance(t, int):
        t = torch.LongTensor([t] * num)
    alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(pipeline.device)
    alpha_prod_t = alphas_cumprod[t].reshape((-1, *([1] * len(x0.shape[1:]))))
    beta_prod_t = 1 - alpha_prod_t
    generator = set_generator(generator=generator)
    if noise is None:
        noise = torch.randn(size=x0.shape, generator=generator)
    scaled_noise = noise * beta_prod_t ** 0.5
    pred_epsilon = ddpm_pred(pipeline=pipeline, xt=alpha_prod_t ** 0.5 * x0 +
        scaled_noise, t=t)
    return pred_epsilon
