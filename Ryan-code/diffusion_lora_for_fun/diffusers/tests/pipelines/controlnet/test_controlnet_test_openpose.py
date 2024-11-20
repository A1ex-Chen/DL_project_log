def test_openpose(self):
    controlnet = ControlNetModel.from_pretrained(
        'lllyasviel/sd-controlnet-openpose')
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', safety_checker=None, controlnet=
        controlnet)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    generator = torch.Generator(device='cpu').manual_seed(0)
    prompt = 'Chef in the kitchen'
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/pose.png'
        )
    output = pipe(prompt, image, generator=generator, output_type='np',
        num_inference_steps=3)
    image = output.images[0]
    assert image.shape == (768, 512, 3)
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/chef_pose_out.npy'
        )
    assert np.abs(expected_image - image).max() < 0.08
