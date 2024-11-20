def test_stable_diffusion_pix2pix_pipeline_multiple_of_8(self):
    inputs = self.get_inputs()
    inputs['image'] = inputs['image'].resize((504, 504))
    model_id = 'timbrooks/instruct-pix2pix'
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id,
        safety_checker=None)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    output = pipe(**inputs)
    image = output.images[0]
    image_slice = image[255:258, 383:386, -1]
    assert image.shape == (504, 504, 3)
    expected_slice = np.array([0.2726, 0.2529, 0.2664, 0.2655, 0.2641, 
        0.2642, 0.2591, 0.2649, 0.259])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.005
