def test_sd(self):
    vae = ConsistencyDecoderVAE.from_pretrained('openai/consistency-decoder')
    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', vae=vae, safety_checker=None)
    pipe.to(torch_device)
    out = pipe('horse', num_inference_steps=2, output_type='pt', generator=
        torch.Generator('cpu').manual_seed(0)).images[0]
    actual_output = out[:2, :2, :2].flatten().cpu()
    expected_output = torch.tensor([0.7686, 0.8228, 0.6489, 0.7455, 0.8661,
        0.8797, 0.8241, 0.8759])
    assert torch_all_close(actual_output, expected_output, atol=0.005)
