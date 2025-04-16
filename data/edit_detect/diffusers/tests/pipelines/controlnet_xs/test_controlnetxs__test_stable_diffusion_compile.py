def _test_stable_diffusion_compile(in_queue, out_queue, timeout):
    error = None
    try:
        _ = in_queue.get(timeout=timeout)
        controlnet = ControlNetXSAdapter.from_pretrained(
            'UmerHA/Testing-ConrolNetXS-SD2.1-canny', torch_dtype=torch.float16
            )
        pipe = StableDiffusionControlNetXSPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2-1-base', controlnet=controlnet,
            safety_checker=None, torch_dtype=torch.float16)
        pipe.to('cuda')
        pipe.set_progress_bar_config(disable=None)
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.unet = torch.compile(pipe.unet, mode='reduce-overhead',
            fullgraph=True)
        generator = torch.Generator(device='cpu').manual_seed(0)
        prompt = 'bird'
        image = load_image(
            'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png'
            ).resize((512, 512))
        output = pipe(prompt, image, num_inference_steps=10, generator=
            generator, output_type='np')
        image = output.images[0]
        assert image.shape == (512, 512, 3)
        expected_image = load_numpy(
            'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny_out_full.npy'
            )
        expected_image = np.resize(expected_image, (512, 512, 3))
        assert np.abs(expected_image - image).max() < 1.0
    except Exception:
        error = f'{traceback.format_exc()}'
    results = {'error': error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()
