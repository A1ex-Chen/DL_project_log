def test_stable_diffusion_lcm(self):
    unet = UNet2DConditionModel.from_pretrained('SimianLuo/LCM_Dreamshaper_v7',
        subfolder='unet')
    sd_pipe = StableDiffusionPipeline.from_pretrained('Lykon/dreamshaper-7',
        unet=unet).to(torch_device)
    sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    inputs['num_inference_steps'] = 6
    inputs['output_type'] = 'pil'
    image = sd_pipe(**inputs).images[0]
    expected_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/lcm_full/stable_diffusion_lcm.png'
        )
    image = sd_pipe.image_processor.pil_to_numpy(image)
    expected_image = sd_pipe.image_processor.pil_to_numpy(expected_image)
    max_diff = numpy_cosine_similarity_distance(image.flatten(),
        expected_image.flatten())
    assert max_diff < 0.01
