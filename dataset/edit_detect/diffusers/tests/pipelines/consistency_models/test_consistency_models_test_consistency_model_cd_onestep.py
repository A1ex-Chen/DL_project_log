def test_consistency_model_cd_onestep(self):
    unet = UNet2DModel.from_pretrained('diffusers/consistency_models',
        subfolder='diffusers_cd_imagenet64_l2')
    scheduler = CMStochasticIterativeScheduler(num_train_timesteps=40,
        sigma_min=0.002, sigma_max=80.0)
    pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
    pipe.to(torch_device=torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs()
    inputs['num_inference_steps'] = 1
    inputs['timesteps'] = None
    image = pipe(**inputs).images
    assert image.shape == (1, 64, 64, 3)
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = np.array([0.0059, 0.0003, 0.0, 0.0023, 0.0052, 0.0007,
        0.0165, 0.0081, 0.0095])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
