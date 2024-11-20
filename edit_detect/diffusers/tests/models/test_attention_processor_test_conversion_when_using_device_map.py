def test_conversion_when_using_device_map(self):
    pipe = DiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-torch', safety_checker=None)
    pre_conversion = pipe('foo', num_inference_steps=2, generator=torch.
        Generator('cpu').manual_seed(0), output_type='np').images
    pipe = DiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-torch', device_map=
        'balanced', safety_checker=None)
    conversion = pipe('foo', num_inference_steps=2, generator=torch.
        Generator('cpu').manual_seed(0), output_type='np').images
    with tempfile.TemporaryDirectory() as tmpdir:
        pipe.save_pretrained(tmpdir)
        pipe = DiffusionPipeline.from_pretrained(tmpdir, device_map=
            'balanced', safety_checker=None)
    after_conversion = pipe('foo', num_inference_steps=2, generator=torch.
        Generator('cpu').manual_seed(0), output_type='np').images
    self.assertTrue(np.allclose(pre_conversion, conversion, atol=0.001))
    self.assertTrue(np.allclose(conversion, after_conversion, atol=0.001))
