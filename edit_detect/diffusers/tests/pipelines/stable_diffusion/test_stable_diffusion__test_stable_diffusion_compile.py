def _test_stable_diffusion_compile(in_queue, out_queue, timeout):
    error = None
    try:
        inputs = in_queue.get(timeout=timeout)
        torch_device = inputs.pop('torch_device')
        seed = inputs.pop('seed')
        inputs['generator'] = torch.Generator(device=torch_device).manual_seed(
            seed)
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4', safety_checker=None)
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.unet.to(memory_format=torch.channels_last)
        sd_pipe.unet = torch.compile(sd_pipe.unet, mode='reduce-overhead',
            fullgraph=True)
        sd_pipe.set_progress_bar_config(disable=None)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.38019, 0.28647, 0.27321, 0.40377, 
            0.3829, 0.35446, 0.39218, 0.38165, 0.42239])
        assert np.abs(image_slice - expected_slice).max() < 0.005
    except Exception:
        error = f'{traceback.format_exc()}'
    results = {'error': error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()
