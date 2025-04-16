def test_inpaint(self):
    controlnet = ControlNetModel.from_pretrained(
        'lllyasviel/control_v11p_sd15_inpaint')
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', safety_checker=None, controlnet=
        controlnet)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    generator = torch.Generator(device='cpu').manual_seed(33)
    init_image = load_image(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy.png'
        )
    init_image = init_image.resize((512, 512))
    mask_image = load_image(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy_mask.png'
        )
    mask_image = mask_image.resize((512, 512))
    prompt = 'a handsome man with ray-ban sunglasses'

    def make_inpaint_condition(image, image_mask):
        image = np.array(image.convert('RGB')).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert('L')).astype(np.float32
            ) / 255.0
        assert image.shape[0:1] == image_mask.shape[0:1
            ], 'image and image_mask must have the same image size'
        image[image_mask > 0.5] = -1.0
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image
    control_image = make_inpaint_condition(init_image, mask_image)
    output = pipe(prompt, image=init_image, mask_image=mask_image,
        control_image=control_image, guidance_scale=9.0, eta=1.0, generator
        =generator, num_inference_steps=20, output_type='np')
    image = output.images[0]
    assert image.shape == (512, 512, 3)
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/boy_ray_ban.npy'
        )
    assert numpy_cosine_similarity_distance(expected_image.flatten(), image
        .flatten()) < 0.01
