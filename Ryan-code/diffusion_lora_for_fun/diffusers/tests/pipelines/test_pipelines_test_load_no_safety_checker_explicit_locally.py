def test_load_no_safety_checker_explicit_locally(self):
    prompt = 'hello'
    pipe = StableDiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-torch', safety_checker=None)
    pipe = pipe.to(torch_device)
    generator = torch.manual_seed(0)
    out = pipe(prompt, num_inference_steps=2, generator=generator,
        output_type='np').images
    with tempfile.TemporaryDirectory() as tmpdirname:
        pipe.save_pretrained(tmpdirname)
        pipe_2 = StableDiffusionPipeline.from_pretrained(tmpdirname,
            safety_checker=None)
        pipe_2 = pipe_2.to(torch_device)
        generator = torch.manual_seed(0)
        out_2 = pipe_2(prompt, num_inference_steps=2, generator=generator,
            output_type='np').images
    assert np.max(np.abs(out - out_2)) < 0.001
