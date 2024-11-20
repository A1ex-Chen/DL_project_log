def test_stable_diffusion_diffedit_dpm(self):
    generator = torch.manual_seed(0)
    pipe = StableDiffusionDiffEditPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-1', safety_checker=None,
        torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler
        .config)
    pipe.inverse_scheduler = DPMSolverMultistepInverseScheduler.from_config(
        pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    source_prompt = 'a bowl of fruit'
    target_prompt = 'a bowl of pears'
    mask_image = pipe.generate_mask(image=self.raw_image, source_prompt=
        source_prompt, target_prompt=target_prompt, generator=generator)
    inv_latents = pipe.invert(prompt=source_prompt, image=self.raw_image,
        inpaint_strength=0.7, generator=generator, num_inference_steps=25
        ).latents
    image = pipe(prompt=target_prompt, mask_image=mask_image, image_latents
        =inv_latents, generator=generator, negative_prompt=source_prompt,
        inpaint_strength=0.7, num_inference_steps=25, output_type='np').images[
        0]
    expected_image = np.array(load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/diffedit/pears.png'
        ).resize((768, 768))) / 255
    assert np.abs((expected_image - image).max()) < 0.5
