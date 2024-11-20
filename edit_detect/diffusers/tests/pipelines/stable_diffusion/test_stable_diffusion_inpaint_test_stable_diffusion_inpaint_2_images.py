def test_stable_diffusion_inpaint_2_images(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = self.pipeline_class(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    gen1 = torch.Generator(device=device).manual_seed(0)
    gen2 = torch.Generator(device=device).manual_seed(0)
    for name in ['prompt', 'image', 'mask_image']:
        inputs[name] = [inputs[name]] * 2
    inputs['generator'] = [gen1, gen2]
    images = sd_pipe(**inputs).images
    assert images.shape == (2, 64, 64, 3)
    image_slice1 = images[0, -3:, -3:, -1]
    image_slice2 = images[1, -3:, -3:, -1]
    assert np.abs(image_slice1.flatten() - image_slice2.flatten()).max(
        ) < 0.0001
    inputs = self.get_dummy_inputs_2images(device)
    images = sd_pipe(**inputs).images
    assert images.shape == (2, 64, 64, 3)
    image_slice1 = images[0, -3:, -3:, -1]
    image_slice2 = images[1, -3:, -3:, -1]
    assert np.abs(image_slice1.flatten() - image_slice2.flatten()).max() > 0.01
