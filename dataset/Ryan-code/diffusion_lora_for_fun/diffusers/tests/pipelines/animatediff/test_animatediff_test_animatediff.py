def test_animatediff(self):
    adapter = MotionAdapter.from_pretrained(
        'guoyww/animatediff-motion-adapter-v1-5-2')
    pipe = AnimateDiffPipeline.from_pretrained('frankjoshua/toonyou_beta6',
        motion_adapter=adapter)
    pipe = pipe.to(torch_device)
    pipe.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='linear', steps_offset=1, clip_sample=False)
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    prompt = (
        'night, b&w photo of old house, post apocalypse, forest, storm weather, wind, rocks, 8k uhd, dslr, soft lighting, high quality, film grain'
        )
    negative_prompt = 'bad quality, worse quality'
    generator = torch.Generator('cpu').manual_seed(0)
    output = pipe(prompt, negative_prompt=negative_prompt, num_frames=16,
        generator=generator, guidance_scale=7.5, num_inference_steps=3,
        output_type='np')
    image = output.frames[0]
    assert image.shape == (16, 512, 512, 3)
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = np.array([0.11357737, 0.11285847, 0.11180121, 
        0.11084166, 0.11414117, 0.09785956, 0.10742754, 0.10510018, 0.08045256]
        )
    assert numpy_cosine_similarity_distance(image_slice.flatten(),
        expected_slice.flatten()) < 0.001
