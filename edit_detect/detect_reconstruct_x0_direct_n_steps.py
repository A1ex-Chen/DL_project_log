def reconstruct_x0_direct_n_steps(pipeline: DiffusionPipeline, x0: torch.
    Tensor, timestep: int, noise: torch.Tensor=None, generator: torch.
    Generator=0) ->torch.Tensor:
    if x0.dim() != 4:
        x0 = x0.unsqueeze(0)
    num: int = x0.shape[0]
    generator = set_generator(generator=generator)
    if noise is None:
        noise = torch.randn(size=x0.shape, generator=generator)
    xt = get_xt_by_t(pipeline=pipeline, x0=x0, t=timestep, noise=noise)
    print(f'pipeline xt: {xt.shape}')
    pipeline_output = pipeline(init=xt, start_ratio_inference_steps=
        timestep / 1000)
    print(
        f'pipeline_output.pred_orig_samples: {pipeline_output.pred_orig_samples.shape}'
        )
    return pipeline_output.latents, pipeline_output.pred_orig_samples
