def test_canny(self):
    controlnet = ControlNetModel.from_pretrained(
        'lllyasviel/sd-controlnet-canny')
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-inpainting', safety_checker=None,
        controlnet=controlnet)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    generator = torch.Generator(device='cpu').manual_seed(0)
    image = load_image(
        'https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/images/bird.png'
        ).resize((512, 512))
    mask_image = load_image(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/input_bench_mask.png'
        ).resize((512, 512))
    prompt = 'pitch black hole'
    control_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png'
        ).resize((512, 512))
    output = pipe(prompt, image=image, mask_image=mask_image, control_image
        =control_image, generator=generator, output_type='np',
        num_inference_steps=3)
    image = output.images[0]
    assert image.shape == (512, 512, 3)
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/inpaint.npy'
        )
    assert np.abs(expected_image - image).max() < 0.09
