@require_torch_gpu
def test_offloads(self):
    pipes = []
    components = self.get_dummy_components()
    sd_pipe = self.pipeline_class(**components).to(torch_device)
    pipes.append(sd_pipe)
    components = self.get_dummy_components()
    sd_pipe = self.pipeline_class(**components)
    sd_pipe.enable_sequential_cpu_offload()
    pipes.append(sd_pipe)
    components = self.get_dummy_components()
    sd_pipe = self.pipeline_class(**components)
    sd_pipe.enable_model_cpu_offload()
    pipes.append(sd_pipe)
    image_slices = []
    for pipe in pipes:
        inputs = self.get_dummy_inputs(torch_device)
        image = pipe(**inputs).images
        image_slices.append(image[0, -3:, -3:, -1].flatten())
    assert np.abs(image_slices[0] - image_slices[1]).max() < 0.001
    assert np.abs(image_slices[0] - image_slices[2]).max() < 0.001
