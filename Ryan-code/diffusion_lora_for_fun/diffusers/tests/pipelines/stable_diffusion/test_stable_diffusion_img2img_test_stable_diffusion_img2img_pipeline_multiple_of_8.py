def test_stable_diffusion_img2img_pipeline_multiple_of_8(self):
    init_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/sketch-mountains-input.jpg'
        )
    init_image = init_image.resize((760, 504))
    model_id = 'CompVis/stable-diffusion-v1-4'
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id,
        safety_checker=None)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    prompt = 'A fantasy landscape, trending on artstation'
    generator = torch.manual_seed(0)
    output = pipe(prompt=prompt, image=init_image, strength=0.75,
        guidance_scale=7.5, generator=generator, output_type='np')
    image = output.images[0]
    image_slice = image[255:258, 383:386, -1]
    assert image.shape == (504, 760, 3)
    expected_slice = np.array([0.9393, 0.95, 0.9399, 0.9438, 0.9458, 0.94, 
        0.9455, 0.9414, 0.9423])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.005
