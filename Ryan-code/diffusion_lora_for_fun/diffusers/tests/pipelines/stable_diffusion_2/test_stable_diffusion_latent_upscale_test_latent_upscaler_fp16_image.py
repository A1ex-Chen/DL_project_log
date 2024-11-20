def test_latent_upscaler_fp16_image(self):
    generator = torch.manual_seed(33)
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
        'stabilityai/sd-x2-latent-upscaler', torch_dtype=torch.float16)
    upscaler.to('cuda')
    prompt = (
        'the temple of fire by Ross Tran and Gerardo Dottori, oil on canvas')
    low_res_img = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/latent-upscaler/fire_temple_512.png'
        )
    image = upscaler(prompt=prompt, image=low_res_img, num_inference_steps=
        20, guidance_scale=0, generator=generator, output_type='np').images[0]
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/latent-upscaler/fire_temple_1024.npy'
        )
    assert np.abs((expected_image - image).max()) < 0.05
