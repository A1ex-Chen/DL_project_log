def test_negative_conditions(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(torch_device)
    inputs = self.get_dummy_inputs(torch_device)
    image = pipe(**inputs).images
    image_slice_without_neg_cond = image[0, -3:, -3:, -1]
    image = pipe(**inputs, negative_original_size=(512, 512),
        negative_crops_coords_top_left=(0, 0), negative_target_size=(1024, 
        1024)).images
    image_slice_with_neg_cond = image[0, -3:, -3:, -1]
    self.assertTrue(np.abs(image_slice_without_neg_cond -
        image_slice_with_neg_cond).max() > 0.01)
