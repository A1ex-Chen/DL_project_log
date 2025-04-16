def test_stable_diffusion_vae_tiling(self):
    torch.cuda.reset_peak_memory_stats()
    model_id = 'CompVis/stable-diffusion-v1-4'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, revision=
        'fp16', torch_dtype=torch.float16, safety_checker=None)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
    prompt = 'a photograph of an astronaut riding a horse'
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()
    generator = torch.Generator(device='cpu').manual_seed(0)
    output_chunked = pipe([prompt], width=1024, height=1024, generator=
        generator, guidance_scale=7.5, num_inference_steps=2, output_type='np')
    image_chunked = output_chunked.images
    mem_bytes = torch.cuda.max_memory_allocated()
    pipe.disable_vae_tiling()
    generator = torch.Generator(device='cpu').manual_seed(0)
    output = pipe([prompt], width=1024, height=1024, generator=generator,
        guidance_scale=7.5, num_inference_steps=2, output_type='np')
    image = output.images
    assert mem_bytes < 10000000000.0
    max_diff = numpy_cosine_similarity_distance(image_chunked.flatten(),
        image.flatten())
    assert max_diff < 0.01
