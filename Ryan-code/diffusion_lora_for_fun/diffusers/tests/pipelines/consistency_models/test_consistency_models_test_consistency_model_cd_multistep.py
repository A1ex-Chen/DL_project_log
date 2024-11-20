def test_consistency_model_cd_multistep(self):
    unet = UNet2DModel.from_pretrained('diffusers/consistency_models',
        subfolder='diffusers_cd_imagenet64_l2')
    scheduler = CMStochasticIterativeScheduler(num_train_timesteps=40,
        sigma_min=0.002, sigma_max=80.0)
    pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
    pipe.to(torch_device=torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs()
    image = pipe(**inputs).images
    assert image.shape == (1, 64, 64, 3)
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = np.array([0.0146, 0.0158, 0.0092, 0.0086, 0.0, 0.0, 
        0.0, 0.0, 0.0058])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
