@require_flax
def test_from_flax_from_pt(self):
    pipe_pt = StableDiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-torch', safety_checker=None)
    pipe_pt.to(torch_device)
    from diffusers import FlaxStableDiffusionPipeline
    with tempfile.TemporaryDirectory() as tmpdirname:
        pipe_pt.save_pretrained(tmpdirname)
        pipe_flax, params = FlaxStableDiffusionPipeline.from_pretrained(
            tmpdirname, safety_checker=None, from_pt=True)
    with tempfile.TemporaryDirectory() as tmpdirname:
        pipe_flax.save_pretrained(tmpdirname, params=params)
        pipe_pt_2 = StableDiffusionPipeline.from_pretrained(tmpdirname,
            safety_checker=None, from_flax=True)
        pipe_pt_2.to(torch_device)
    prompt = 'Hello'
    generator = torch.manual_seed(0)
    image_0 = pipe_pt([prompt], generator=generator, num_inference_steps=2,
        output_type='np').images[0]
    generator = torch.manual_seed(0)
    image_1 = pipe_pt_2([prompt], generator=generator, num_inference_steps=
        2, output_type='np').images[0]
    assert np.abs(image_0 - image_1).sum(
        ) < 1e-05, "Models don't give the same forward pass"
