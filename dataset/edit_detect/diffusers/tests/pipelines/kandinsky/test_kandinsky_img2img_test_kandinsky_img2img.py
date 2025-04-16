def test_kandinsky_img2img(self):
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/kandinsky_img2img_frog.npy'
        )
    init_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png'
        )
    prompt = 'A red cartoon frog, 4k'
    pipe_prior = KandinskyPriorPipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-1-prior', torch_dtype=torch.float16)
    pipe_prior.to(torch_device)
    pipeline = KandinskyImg2ImgPipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-1', torch_dtype=torch.float16)
    pipeline = pipeline.to(torch_device)
    pipeline.set_progress_bar_config(disable=None)
    generator = torch.Generator(device='cpu').manual_seed(0)
    image_emb, zero_image_emb = pipe_prior(prompt, generator=generator,
        num_inference_steps=5, negative_prompt='').to_tuple()
    output = pipeline(prompt, image=init_image, image_embeds=image_emb,
        negative_image_embeds=zero_image_emb, generator=generator,
        num_inference_steps=100, height=768, width=768, strength=0.2,
        output_type='np')
    image = output.images[0]
    assert image.shape == (768, 768, 3)
    assert_mean_pixel_difference(image, expected_image)
