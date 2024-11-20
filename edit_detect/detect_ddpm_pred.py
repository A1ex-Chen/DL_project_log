def ddpm_pred(pipeline: DiffusionPipeline, xt: torch.Tensor, t: torch.Tensor
    ) ->torch.Tensor:
    model_output = pipeline.unet(xt, t).sample
    return model_output
