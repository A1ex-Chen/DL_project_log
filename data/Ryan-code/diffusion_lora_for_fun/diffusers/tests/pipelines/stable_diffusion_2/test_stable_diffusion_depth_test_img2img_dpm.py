def test_img2img_dpm(self):
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-depth')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler
        .config)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs()
    inputs['num_inference_steps'] = 30
    image = pipe(**inputs).images[0]
    expected_image = load_numpy(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_depth2img/stable_diffusion_2_0_dpm_multi.npy'
        )
    max_diff = np.abs(expected_image - image).max()
    assert max_diff < 0.001
