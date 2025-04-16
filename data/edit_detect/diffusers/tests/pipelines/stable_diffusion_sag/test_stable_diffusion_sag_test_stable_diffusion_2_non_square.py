def test_stable_diffusion_2_non_square(self):
    sag_pipe = StableDiffusionSAGPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-1-base')
    sag_pipe = sag_pipe.to(torch_device)
    sag_pipe.set_progress_bar_config(disable=None)
    prompt = '.'
    generator = torch.manual_seed(0)
    output = sag_pipe([prompt], width=768, height=512, generator=generator,
        guidance_scale=7.5, sag_scale=1.0, num_inference_steps=20,
        output_type='np')
    image = output.images
    assert image.shape == (1, 512, 768, 3)
