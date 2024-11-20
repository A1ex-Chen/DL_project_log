def test_ddpm_ddim_equality_batched(self):
    seed = 0
    model_id = 'google/ddpm-cifar10-32'
    unet = UNet2DModel.from_pretrained(model_id)
    ddpm_scheduler = DDPMScheduler()
    ddim_scheduler = DDIMScheduler()
    ddpm = DDPMPipeline(unet=unet, scheduler=ddpm_scheduler)
    ddpm.to(torch_device)
    ddpm.set_progress_bar_config(disable=None)
    ddim = DDIMPipeline(unet=unet, scheduler=ddim_scheduler)
    ddim.to(torch_device)
    ddim.set_progress_bar_config(disable=None)
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    ddpm_images = ddpm(batch_size=2, generator=generator, output_type='np'
        ).images
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    ddim_images = ddim(batch_size=2, generator=generator,
        num_inference_steps=1000, eta=1.0, output_type='np',
        use_clipped_model_output=True).images
    assert np.abs(ddpm_images - ddim_images).max() < 0.1
