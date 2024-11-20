def test_sdv1_5_lcm_lora_img2img(self):
    pipe = AutoPipelineForImage2Image.from_pretrained(
        'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
    pipe.to(torch_device)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    init_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/fantasy_landscape.png'
        )
    generator = torch.Generator('cpu').manual_seed(0)
    lora_model_id = 'latent-consistency/lcm-lora-sdv1-5'
    pipe.load_lora_weights(lora_model_id)
    image = pipe('snowy mountain', generator=generator, image=init_image,
        strength=0.5, num_inference_steps=4, guidance_scale=0.5).images[0]
    expected_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/lcm_lora/sdv15_lcm_lora_img2img.png'
        )
    image_np = pipe.image_processor.pil_to_numpy(image)
    expected_image_np = pipe.image_processor.pil_to_numpy(expected_image)
    max_diff = numpy_cosine_similarity_distance(image_np.flatten(),
        expected_image_np.flatten())
    assert max_diff < 0.0001
    pipe.unload_lora_weights()
    release_memory(pipe)
