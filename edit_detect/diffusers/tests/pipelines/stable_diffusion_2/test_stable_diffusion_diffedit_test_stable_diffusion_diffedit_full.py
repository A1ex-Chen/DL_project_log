def test_stable_diffusion_diffedit_full(self):
    generator = torch.manual_seed(0)
    pipe = StableDiffusionDiffEditPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-1-base', safety_checker=None,
        torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.clip_sample = True
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.
        scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    source_prompt = 'a bowl of fruit'
    target_prompt = 'a bowl of pears'
    mask_image = pipe.generate_mask(image=self.raw_image, source_prompt=
        source_prompt, target_prompt=target_prompt, generator=generator)
    inv_latents = pipe.invert(prompt=source_prompt, image=self.raw_image,
        inpaint_strength=0.7, generator=generator, num_inference_steps=5
        ).latents
    image = pipe(prompt=target_prompt, mask_image=mask_image, image_latents
        =inv_latents, generator=generator, negative_prompt=source_prompt,
        inpaint_strength=0.7, num_inference_steps=5, output_type='np').images[0
        ]
    expected_image = np.array(load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/diffedit/pears.png'
        ).resize((256, 256))) / 255
    assert numpy_cosine_similarity_distance(expected_image.flatten(), image
        .flatten()) < 0.2
