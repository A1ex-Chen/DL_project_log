@require_torch_2
def test_consistency_model_cd_onestep_flash_attn(self):
    unet = UNet2DModel.from_pretrained('diffusers/consistency_models',
        subfolder='diffusers_cd_imagenet64_l2')
    scheduler = CMStochasticIterativeScheduler(num_train_timesteps=40,
        sigma_min=0.002, sigma_max=80.0)
    pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
    pipe.to(torch_device=torch_device, torch_dtype=torch.float16)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(get_fixed_latents=True, device=torch_device)
    inputs['num_inference_steps'] = 1
    inputs['timesteps'] = None
    with sdp_kernel(enable_flash=True, enable_math=False,
        enable_mem_efficient=False):
        image = pipe(**inputs).images
    assert image.shape == (1, 64, 64, 3)
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = np.array([0.1623, 0.2009, 0.2387, 0.1731, 0.1168, 
        0.1202, 0.2031, 0.1327, 0.2447])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
