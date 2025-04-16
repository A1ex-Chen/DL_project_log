def test_sd_f16(self):
    vae = ConsistencyDecoderVAE.from_pretrained('openai/consistency-decoder',
        torch_dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16, vae=
        vae, safety_checker=None)
    pipe.to(torch_device)
    out = pipe('horse', num_inference_steps=2, output_type='pt', generator=
        torch.Generator('cpu').manual_seed(0)).images[0]
    actual_output = out[:2, :2, :2].flatten().cpu()
    expected_output = torch.tensor([0.0, 0.0249, 0.0, 0.0, 0.1709, 0.2773, 
        0.0471, 0.1035], dtype=torch.float16)
    assert torch_all_close(actual_output, expected_output, atol=0.005)
