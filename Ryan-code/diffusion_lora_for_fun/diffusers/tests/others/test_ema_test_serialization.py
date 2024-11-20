@skip_mps
def test_serialization(self):
    unet, ema_unet = self.get_models()
    noisy_latents, timesteps, encoder_hidden_states = self.get_dummy_inputs()
    with tempfile.TemporaryDirectory() as tmpdir:
        ema_unet.save_pretrained(tmpdir)
        loaded_unet = UNet2DConditionModel.from_pretrained(tmpdir,
            model_cls=UNet2DConditionModel)
        loaded_unet = loaded_unet.to(unet.device)
    output = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    output_loaded = loaded_unet(noisy_latents, timesteps, encoder_hidden_states
        ).sample
    assert torch.allclose(output, output_loaded, atol=0.0001)
