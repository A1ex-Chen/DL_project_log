def test_latent_upscaler_fp16(self):
    generator = torch.manual_seed(33)
    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16)
    pipe.to('cuda')
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
        'stabilityai/sd-x2-latent-upscaler', torch_dtype=torch.float16)
    upscaler.to('cuda')
    prompt = (
        'a photo of an astronaut high resolution, unreal engine, ultra realistic'
        )
    low_res_latents = pipe(prompt, generator=generator, output_type='latent'
        ).images
    image = upscaler(prompt=prompt, image=low_res_latents,
        num_inference_steps=20, guidance_scale=0, generator=generator,
        output_type='np').images[0]
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/latent-upscaler/astronaut_1024.npy'
        )
    assert np.abs((expected_image - image).mean()) < 0.05
