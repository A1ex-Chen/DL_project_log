@require_torch_gpu
def test_stable_diffusion_xl_offloads(self):
    pipes = []
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionXLPipeline(**components).to(torch_device)
    pipes.append(sd_pipe)
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionXLPipeline(**components)
    sd_pipe.enable_model_cpu_offload()
    pipes.append(sd_pipe)
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionXLPipeline(**components)
    sd_pipe.enable_sequential_cpu_offload()
    pipes.append(sd_pipe)
    image_slices = []
    for pipe in pipes:
        pipe.unet.set_default_attn_processor()
        inputs = self.get_dummy_inputs(torch_device)
        image = pipe(**inputs).images
        image_slices.append(image[0, -3:, -3:, -1].flatten())
    assert np.abs(image_slices[0] - image_slices[1]).max() < 0.001
    assert np.abs(image_slices[0] - image_slices[2]).max() < 0.001
