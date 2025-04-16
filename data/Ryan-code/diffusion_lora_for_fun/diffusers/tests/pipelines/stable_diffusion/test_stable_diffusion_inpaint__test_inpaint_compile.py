def _test_inpaint_compile(in_queue, out_queue, timeout):
    error = None
    try:
        inputs = in_queue.get(timeout=timeout)
        torch_device = inputs.pop('torch_device')
        seed = inputs.pop('seed')
        inputs['generator'] = torch.Generator(device=torch_device).manual_seed(
            seed)
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            'runwayml/stable-diffusion-inpainting', safety_checker=None)
        pipe.unet.set_default_attn_processor()
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.unet = torch.compile(pipe.unet, mode='reduce-overhead',
            fullgraph=True)
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.0689, 0.0699, 0.079, 0.0536, 0.047, 
            0.0488, 0.041, 0.0508, 0.04179])
        assert np.abs(expected_slice - image_slice).max() < 0.003
    except Exception:
        error = f'{traceback.format_exc()}'
    results = {'error': error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()
