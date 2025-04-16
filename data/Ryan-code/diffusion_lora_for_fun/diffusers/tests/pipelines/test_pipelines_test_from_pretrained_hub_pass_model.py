def test_from_pretrained_hub_pass_model(self):
    model_path = 'google/ddpm-cifar10-32'
    scheduler = DDPMScheduler(num_train_timesteps=10)
    unet = UNet2DModel.from_pretrained(model_path)
    ddpm_from_hub_custom_model = DiffusionPipeline.from_pretrained(model_path,
        unet=unet, scheduler=scheduler)
    ddpm_from_hub_custom_model = ddpm_from_hub_custom_model.to(torch_device)
    ddpm_from_hub_custom_model.set_progress_bar_config(disable=None)
    ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path, scheduler
        =scheduler)
    ddpm_from_hub = ddpm_from_hub.to(torch_device)
    ddpm_from_hub_custom_model.set_progress_bar_config(disable=None)
    generator = torch.Generator(device=torch_device).manual_seed(0)
    image = ddpm_from_hub_custom_model(generator=generator,
        num_inference_steps=5, output_type='np').images
    generator = torch.Generator(device=torch_device).manual_seed(0)
    new_image = ddpm_from_hub(generator=generator, num_inference_steps=5,
        output_type='np').images
    assert np.abs(image - new_image).max(
        ) < 1e-05, "Models don't give the same forward pass"
