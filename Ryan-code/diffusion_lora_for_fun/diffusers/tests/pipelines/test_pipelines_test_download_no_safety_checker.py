def test_download_no_safety_checker(self):
    prompt = 'hello'
    pipe = StableDiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-torch', safety_checker=None)
    pipe = pipe.to(torch_device)
    generator = torch.manual_seed(0)
    out = pipe(prompt, num_inference_steps=2, generator=generator,
        output_type='np').images
    pipe_2 = StableDiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-torch')
    pipe_2 = pipe_2.to(torch_device)
    generator = torch.manual_seed(0)
    out_2 = pipe_2(prompt, num_inference_steps=2, generator=generator,
        output_type='np').images
    assert np.max(np.abs(out - out_2)) < 0.001
