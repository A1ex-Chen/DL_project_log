def single_step_denoise(pipeline, latents: torch.Tensor, from_t: int=0):
    return (pipeline.inference(batch_size=len(latents), num_inference_steps
        =1, class_labels=None, latents=latents, output_type='pt', is_pbar=
        False).images - 0.5) * 2
