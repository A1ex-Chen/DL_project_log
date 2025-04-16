@slow
def test_unet_1d_maestro(self):
    model_id = 'harmonai/maestro-150k'
    model = UNet1DModel.from_pretrained(model_id, subfolder='unet')
    model.to(torch_device)
    sample_size = 65536
    noise = torch.sin(torch.arange(sample_size)[None, None, :].repeat(1, 2, 1)
        ).to(torch_device)
    timestep = torch.tensor([1]).to(torch_device)
    with torch.no_grad():
        output = model(noise, timestep).sample
    output_sum = output.abs().sum()
    output_max = output.abs().max()
    assert (output_sum - 224.0896).abs() < 0.5
    assert (output_max - 0.0607).abs() < 0.0004
