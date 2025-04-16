def test_kandinsky_inpaint(self):
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/kandinsky_inpaint_cat_with_hat_fp16.npy'
        )
    init_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png'
        )
    mask = np.zeros((768, 768), dtype=np.float32)
    mask[:250, 250:-250] = 1
    prompt = 'a hat'
    pipe_prior = KandinskyPriorPipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-1-prior', torch_dtype=torch.float16)
    pipe_prior.to(torch_device)
    pipeline = KandinskyInpaintPipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-1-inpaint', torch_dtype=torch.float16)
    pipeline = pipeline.to(torch_device)
    pipeline.set_progress_bar_config(disable=None)
    generator = torch.Generator(device='cpu').manual_seed(0)
    image_emb, zero_image_emb = pipe_prior(prompt, generator=generator,
        num_inference_steps=5, negative_prompt='').to_tuple()
    output = pipeline(prompt, image=init_image, mask_image=mask,
        image_embeds=image_emb, negative_image_embeds=zero_image_emb,
        generator=generator, num_inference_steps=100, height=768, width=768,
        output_type='np')
    image = output.images[0]
    assert image.shape == (768, 768, 3)
    assert_mean_pixel_difference(image, expected_image)
