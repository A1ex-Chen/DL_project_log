def test_stable_diffusion_xl_negative_conditions(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionXLPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice_with_no_neg_cond = image[0, -3:, -3:, -1]
    image = sd_pipe(**inputs, negative_original_size=(512, 512),
        negative_crops_coords_top_left=(0, 0), negative_target_size=(1024, 
        1024)).images
    image_slice_with_neg_cond = image[0, -3:, -3:, -1]
    self.assertTrue(np.abs(image_slice_with_no_neg_cond -
        image_slice_with_neg_cond).max() > 0.01)
