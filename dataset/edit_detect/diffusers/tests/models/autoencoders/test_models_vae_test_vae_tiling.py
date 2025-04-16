def test_vae_tiling(self):
    vae = ConsistencyDecoderVAE.from_pretrained('openai/consistency-decoder',
        torch_dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', vae=vae, safety_checker=None,
        torch_dtype=torch.float16)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    out_1 = pipe('horse', num_inference_steps=2, output_type='pt',
        generator=torch.Generator('cpu').manual_seed(0)).images[0]
    pipe.enable_vae_tiling()
    out_2 = pipe('horse', num_inference_steps=2, output_type='pt',
        generator=torch.Generator('cpu').manual_seed(0)).images[0]
    assert torch_all_close(out_1, out_2, atol=0.005)
    shapes = [(1, 4, 73, 97), (1, 4, 97, 73), (1, 4, 49, 65), (1, 4, 65, 49)]
    with torch.no_grad():
        for shape in shapes:
            image = torch.zeros(shape, device=torch_device, dtype=pipe.vae.
                dtype)
            pipe.vae.decode(image)
