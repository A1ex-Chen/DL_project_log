def test_stable_diffusion_lcm(self):
    torch.manual_seed(0)
    unet = UNet2DConditionModel.from_pretrained('latent-consistency/lcm-ssd-1b'
        , torch_dtype=torch.float16, variant='fp16')
    sd_pipe = StableDiffusionXLPipeline.from_pretrained('segmind/SSD-1B',
        unet=unet, torch_dtype=torch.float16, variant='fp16').to(torch_device)
    sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe.set_progress_bar_config(disable=None)
    prompt = 'a red car standing on the side of the street'
    image = sd_pipe(prompt, num_inference_steps=4, guidance_scale=8.0).images[0
        ]
    expected_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/lcm_full/stable_diffusion_ssd_1b_lcm.png'
        )
    image = sd_pipe.image_processor.pil_to_numpy(image)
    expected_image = sd_pipe.image_processor.pil_to_numpy(expected_image)
    max_diff = numpy_cosine_similarity_distance(image.flatten(),
        expected_image.flatten())
    assert max_diff < 0.01
