def _test_unidiffuser_compile(in_queue, out_queue, timeout):
    error = None
    try:
        inputs = in_queue.get(timeout=timeout)
        torch_device = inputs.pop('torch_device')
        seed = inputs.pop('seed')
        inputs['generator'] = torch.Generator(device=torch_device).manual_seed(
            seed)
        pipe = UniDiffuserPipeline.from_pretrained('thu-ml/unidiffuser-v1')
        pipe = pipe.to(torch_device)
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.unet = torch.compile(pipe.unet, mode='reduce-overhead',
            fullgraph=True)
        pipe.set_progress_bar_config(disable=None)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.2402, 0.2375, 0.2285, 0.2378, 0.2407, 
            0.2263, 0.2354, 0.2307, 0.252])
        assert np.abs(image_slice - expected_slice).max() < 0.1
    except Exception:
        error = f'{traceback.format_exc()}'
    results = {'error': error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()
