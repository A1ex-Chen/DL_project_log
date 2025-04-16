def test_from_save_pretrained(self):
    model = UNet2DModel(block_out_channels=(32, 64), layers_per_block=2,
        sample_size=32, in_channels=3, out_channels=3, down_block_types=(
        'DownBlock2D', 'AttnDownBlock2D'), up_block_types=('AttnUpBlock2D',
        'UpBlock2D'))
    scheduler = DDPMScheduler(num_train_timesteps=10)
    ddpm = DDPMPipeline(model, scheduler)
    ddpm.to(torch_device)
    ddpm.set_progress_bar_config(disable=None)
    with tempfile.TemporaryDirectory() as tmpdirname:
        ddpm.save_pretrained(tmpdirname)
        new_ddpm = DDPMPipeline.from_pretrained(tmpdirname)
        new_ddpm.to(torch_device)
    generator = torch.Generator(device=torch_device).manual_seed(0)
    image = ddpm(generator=generator, num_inference_steps=5, output_type='np'
        ).images
    generator = torch.Generator(device=torch_device).manual_seed(0)
    new_image = new_ddpm(generator=generator, num_inference_steps=5,
        output_type='np').images
    assert np.abs(image - new_image).max(
        ) < 1e-05, "Models don't give the same forward pass"
