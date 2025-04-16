def test_stable_diffusion_text2img_pipeline_unflawed(self):
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-text2img/lion_galaxy.npy'
        )
    pipe = StableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-1')
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config,
        timestep_spacing='trailing', rescale_betas_zero_snr=True)
    pipe.to(torch_device)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    prompt = (
        'A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k'
        )
    generator = torch.Generator('cpu').manual_seed(0)
    output = pipe(prompt=prompt, guidance_scale=7.5, num_inference_steps=10,
        guidance_rescale=0.7, generator=generator, output_type='np')
    image = output.images[0]
    assert image.shape == (768, 768, 3)
    max_diff = numpy_cosine_similarity_distance(image.flatten(),
        expected_image.flatten())
    assert max_diff < 0.05
