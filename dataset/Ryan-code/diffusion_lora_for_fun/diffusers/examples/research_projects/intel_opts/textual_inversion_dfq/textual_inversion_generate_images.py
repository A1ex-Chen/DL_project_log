def generate_images(pipeline, prompt='', guidance_scale=7.5,
    num_inference_steps=50, num_images_per_prompt=1, seed=42):
    generator = torch.Generator(pipeline.device).manual_seed(seed)
    images = pipeline(prompt, guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps, generator=generator,
        num_images_per_prompt=num_images_per_prompt).images
    _rows = int(math.sqrt(num_images_per_prompt))
    grid = make_image_grid(images, rows=_rows, cols=num_images_per_prompt //
        _rows)
    return grid
