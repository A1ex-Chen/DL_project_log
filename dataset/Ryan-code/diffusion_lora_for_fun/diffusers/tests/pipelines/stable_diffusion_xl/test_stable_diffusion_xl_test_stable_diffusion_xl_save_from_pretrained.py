def test_stable_diffusion_xl_save_from_pretrained(self):
    pipes = []
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionXLPipeline(**components).to(torch_device)
    pipes.append(sd_pipe)
    with tempfile.TemporaryDirectory() as tmpdirname:
        sd_pipe.save_pretrained(tmpdirname)
        sd_pipe = StableDiffusionXLPipeline.from_pretrained(tmpdirname).to(
            torch_device)
    pipes.append(sd_pipe)
    image_slices = []
    for pipe in pipes:
        pipe.unet.set_default_attn_processor()
        inputs = self.get_dummy_inputs(torch_device)
        image = pipe(**inputs).images
        image_slices.append(image[0, -3:, -3:, -1].flatten())
    assert np.abs(image_slices[0] - image_slices[1]).max() < 0.001
