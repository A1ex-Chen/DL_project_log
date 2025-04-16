def test_stable_diffusion_inpaint_image_tensor(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionInpaintPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    output = sd_pipe(**inputs)
    out_pil = output.images
    inputs = self.get_dummy_inputs(device)
    inputs['image'] = torch.tensor(np.array(inputs['image']) / 127.5 - 1
        ).permute(2, 0, 1).unsqueeze(0)
    inputs['mask_image'] = torch.tensor(np.array(inputs['mask_image']) / 255
        ).permute(2, 0, 1)[:1].unsqueeze(0)
    output = sd_pipe(**inputs)
    out_tensor = output.images
    assert out_pil.shape == (1, 64, 64, 3)
    assert np.abs(out_pil.flatten() - out_tensor.flatten()).max() < 0.05
