def _test_img2img_compile(in_queue, out_queue, timeout):
    error = None
    try:
        inputs = in_queue.get(timeout=timeout)
        torch_device = inputs.pop('torch_device')
        seed = inputs.pop('seed')
        inputs['generator'] = torch.Generator(device=torch_device).manual_seed(
            seed)
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4', safety_checker=None)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.unet.set_default_attn_processor()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.unet = torch.compile(pipe.unet, mode='reduce-overhead',
            fullgraph=True)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 768, 3)
        expected_slice = np.array([0.0606, 0.057, 0.0805, 0.0579, 0.0628, 
            0.0623, 0.0843, 0.1115, 0.0806])
        assert np.abs(expected_slice - image_slice).max() < 0.001
    except Exception:
        error = f'{traceback.format_exc()}'
    results = {'error': error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()
